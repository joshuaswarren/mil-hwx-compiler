#import "HWXImage.h"

#import "HWXMachOFormat.h"

NSString *const HWXImageErrorDomain = @"ANE.HWX.Image";

static NSError *makeError(HWXImageErrorCode code, NSString *message) {
    return [NSError errorWithDomain:HWXImageErrorDomain code:code
                           userInfo:@{NSLocalizedDescriptionKey: message}];
}

static BOOL rangeFits(NSUInteger offset, NSUInteger length, NSUInteger limit) {
    return offset <= limit && length <= limit - offset;
}

static NSString *fixedName(const char bytes[16]) {
    size_t length = strnlen(bytes, 16);
    return [[NSString alloc] initWithBytes:bytes length:length
                                  encoding:NSASCIIStringEncoding] ?: @"";
}

@interface HWXSection ()
@property(nonatomic, readwrite, copy) NSString *name;
@property(nonatomic, readwrite, copy) NSString *segmentName;
@property(nonatomic, readwrite) uint64_t size;
@property(nonatomic, readwrite) uint32_t fileOffset;
@property(nonatomic, readwrite, copy) NSData *data;
@end
@implementation HWXSection
@end

@interface HWXImage ()
@property(nonatomic, readwrite, copy) NSData *data;
@property(nonatomic, readwrite) uint32_t magic;
@property(nonatomic, readwrite) int32_t cpuType;
@property(nonatomic, readwrite) int32_t cpuSubtype;
@property(nonatomic, readwrite, copy) NSArray<HWXSection *> *sections;
@end

@implementation HWXImage
+ (instancetype)imageWithData:(NSData *)data error:(NSError **)error {
    if (data.length < sizeof(struct mach_header_64)) {
        if (error) *error = makeError(HWXImageErrorTruncated,
                                      @"HWX header is truncated");
        return nil;
    }
    struct mach_header_64 header;
    memcpy(&header, data.bytes, sizeof(header));
    if (header.magic != 0xBEEFFACEu) {
        if (error) *error = makeError(HWXImageErrorBadMagic, @"bad HWX magic");
        return nil;
    }
    NSUInteger commandOffset = sizeof(header);
    if (!rangeFits(commandOffset, header.sizeofcmds, data.length)) {
        if (error) *error = makeError(HWXImageErrorMalformedCommands,
                                      @"load commands exceed the file");
        return nil;
    }
    NSUInteger commandEnd = commandOffset + header.sizeofcmds;
    NSMutableArray<HWXSection *> *sections = [NSMutableArray array];
    NSMutableArray<NSData *> *sectionRecords = [NSMutableArray array];
    NSMutableArray<NSData *> *symbolCommands = [NSMutableArray array];
    const uint8_t *bytes = (const uint8_t *)data.bytes;
    for (uint32_t commandIndex = 0; commandIndex < header.ncmds;
         ++commandIndex) {
        if (!rangeFits(commandOffset, sizeof(struct load_command), commandEnd)) {
            if (error) *error = makeError(HWXImageErrorMalformedCommands,
                                          @"load command is truncated");
            return nil;
        }
        struct load_command command;
        memcpy(&command, bytes + commandOffset, sizeof(command));
        if (command.cmdsize < sizeof(command) ||
            !rangeFits(commandOffset, command.cmdsize, commandEnd)) {
            if (error) *error = makeError(HWXImageErrorMalformedCommands,
                                          @"invalid load-command size");
            return nil;
        }
        if (command.cmd == LC_SEGMENT_64) {
            if (command.cmdsize < sizeof(struct segment_command_64)) {
                if (error) *error = makeError(HWXImageErrorMalformedCommands,
                                              @"segment command is truncated");
                return nil;
            }
            struct segment_command_64 segment;
            memcpy(&segment, bytes + commandOffset, sizeof(segment));
            NSUInteger sectionBytes = 0;
            if (__builtin_mul_overflow((NSUInteger)segment.nsects,
                                      sizeof(struct section_64), &sectionBytes) ||
                sizeof(segment) + sectionBytes > command.cmdsize) {
                if (error) *error = makeError(HWXImageErrorMalformedCommands,
                                              @"section table exceeds command");
                return nil;
            }
            NSUInteger sectionOffset = commandOffset + sizeof(segment);
            for (uint32_t sectionIndex = 0; sectionIndex < segment.nsects;
                 ++sectionIndex) {
                struct section_64 record;
                memcpy(&record, bytes + sectionOffset, sizeof(record));
                [sectionRecords addObject:[NSData dataWithBytes:&record
                                                         length:sizeof(record)]];
                HWXSection *section = [[HWXSection alloc] init];
                section.name = fixedName(record.sectname);
                section.segmentName = fixedName(record.segname);
                section.size = record.size;
                section.fileOffset = record.offset;
                BOOL fileBacked = segment.filesize != 0 && record.size != 0;
                if (fileBacked) {
                    if (!rangeFits(record.offset, (NSUInteger)record.size,
                                   data.length)) {
                        if (error) *error = makeError(HWXImageErrorRange,
                            [NSString stringWithFormat:@"section %@.%@ exceeds file",
                                                       section.segmentName,
                                                       section.name]);
                        return nil;
                    }
                    section.data = [data subdataWithRange:
                        NSMakeRange(record.offset, (NSUInteger)record.size)];
                } else {
                    section.data = [NSData data];
                }
                [sections addObject:section];
                sectionOffset += sizeof(record);
            }
        } else if (command.cmd == LC_SYMTAB) {
            if (command.cmdsize != sizeof(struct symtab_command)) {
                if (error) *error = makeError(HWXImageErrorMalformedCommands,
                                              @"symbol command has the wrong size");
                return nil;
            }
            [symbolCommands addObject:[data subdataWithRange:
                NSMakeRange(commandOffset, sizeof(struct symtab_command))]];
        }
        commandOffset += command.cmdsize;
    }
    if (commandOffset != commandEnd) {
        if (error) *error = makeError(HWXImageErrorMalformedCommands,
                                      @"command count does not fill envelope");
        return nil;
    }
    NSUInteger maximumSymbolCount = 0;
    for (NSData *commandData in symbolCommands) {
        struct symtab_command command;
        memcpy(&command, commandData.bytes, sizeof(command));
        NSUInteger symbolBytes = 0;
        if (__builtin_mul_overflow((NSUInteger)command.nsyms,
                                  sizeof(struct nlist_64), &symbolBytes) ||
            !rangeFits(command.symoff, symbolBytes, data.length) ||
            !rangeFits(command.stroff, command.strsize, data.length)) {
            if (error) *error = makeError(HWXImageErrorRange,
                                          @"symbol table exceeds the file");
            return nil;
        }
        maximumSymbolCount = MAX(maximumSymbolCount,
                                 (NSUInteger)command.nsyms);
        for (uint32_t index = 0; index < command.nsyms; ++index) {
            struct nlist_64 symbol;
            memcpy(&symbol, bytes + command.symoff +
                   index * sizeof(symbol), sizeof(symbol));
            if (symbol.n_un.n_strx >= command.strsize ||
                ((symbol.n_type & N_TYPE) == N_SECT &&
                 (symbol.n_sect == 0 ||
                  symbol.n_sect > sectionRecords.count))) {
                if (error) *error = makeError(HWXImageErrorRange,
                                              @"symbol refers outside its tables");
                return nil;
            }
        }
    }
    for (NSData *recordData in sectionRecords) {
        struct section_64 record;
        memcpy(&record, recordData.bytes, sizeof(record));
        if (record.nreloc == 0) continue;
        NSUInteger relocationBytes = 0;
        if (__builtin_mul_overflow((NSUInteger)record.nreloc,
                                  (NSUInteger)8, &relocationBytes) ||
            !rangeFits(record.reloff, relocationBytes, data.length)) {
            if (error) *error = makeError(HWXImageErrorRange,
                                          @"relocation table exceeds the file");
            return nil;
        }
        for (uint32_t index = 0; index < record.nreloc; ++index) {
            uint32_t relocation[2] = {};
            memcpy(relocation, bytes + record.reloff + index * 8,
                   sizeof(relocation));
            NSUInteger target = relocation[1] & 0x00ffffffu;
            BOOL external = (relocation[1] & 0x08000000u) != 0;
            if (relocation[0] > record.size ||
                sizeof(uint32_t) > record.size - relocation[0] ||
                (external ? target >= maximumSymbolCount
                          : (target == 0 || target > sectionRecords.count))) {
                if (error) *error = makeError(HWXImageErrorRange,
                                              @"relocation target is invalid");
                return nil;
            }
        }
    }
    HWXImage *image = [[HWXImage alloc] init];
    image.data = data;
    image.magic = header.magic;
    image.cpuType = header.cputype;
    image.cpuSubtype = header.cpusubtype;
    image.sections = sections;
    return image;
}

- (HWXSection *)firstSectionNamed:(NSString *)name
                       inSegment:(NSString *)segmentName {
    for (HWXSection *section in self.sections)
        if ([section.name isEqualToString:name] &&
            [section.segmentName isEqualToString:segmentName]) return section;
    return nil;
}
@end
