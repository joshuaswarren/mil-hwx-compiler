#import "HWXObjectWriter.h"

#import <mach-o/loader.h>
#import <mach-o/nlist.h>

static NSString *const HWXObjectWriterErrorDomain = @"ANE.HWX.ObjectWriter";

static NSUInteger alignUp(NSUInteger value, NSUInteger alignment) {
    return (value + alignment - 1) / alignment * alignment;
}
static void setName(char destination[16], const char *name) {
    memset(destination, 0, 16);
    strlcpy(destination, name, 16);
}
static void writeU32(NSMutableData *data, NSUInteger offset, uint32_t value) {
    [data replaceBytesInRange:NSMakeRange(offset,4) withBytes:&value];
}
static void writeU64(NSMutableData *data, NSUInteger offset, uint64_t value) {
    [data replaceBytesInRange:NSMakeRange(offset,8) withBytes:&value];
}
static void writeString(NSMutableData *data, NSUInteger offset, NSString *value) {
    NSData *bytes = [value dataUsingEncoding:NSUTF8StringEncoding];
    [data replaceBytesInRange:NSMakeRange(offset, bytes.length) withBytes:bytes.bytes];
}
static struct section_64 sectionRecord(const char *sectionName,
                                       const char *segmentName,
                                       uint64_t address, uint64_t size,
                                       uint32_t fileOffset, uint32_t alignment,
                                       uint32_t flags) {
    struct section_64 section = {};
    setName(section.sectname, sectionName); setName(section.segname, segmentName);
    section.addr=address; section.size=size; section.offset=fileOffset;
    section.align=alignment; section.flags=flags; return section;
}
static NSData *segmentRecord(const char *name, uint64_t vmAddress,
                             uint64_t vmSize, uint64_t fileOffset,
                             uint64_t fileSize, int32_t maxProtection,
                             int32_t initialProtection, uint32_t flags,
                             const struct section_64 *sections,
                             uint32_t sectionCount) {
    NSMutableData *data = [NSMutableData dataWithLength:
        sizeof(struct segment_command_64)+sectionCount*sizeof(struct section_64)];
    struct segment_command_64 segment = {};
    segment.cmd=LC_SEGMENT_64; segment.cmdsize=(uint32_t)data.length;
    setName(segment.segname,name); segment.vmaddr=vmAddress; segment.vmsize=vmSize;
    segment.fileoff=fileOffset; segment.filesize=fileSize;
    segment.maxprot=maxProtection; segment.initprot=initialProtection;
    segment.nsects=sectionCount; segment.flags=flags;
    [data replaceBytesInRange:NSMakeRange(0,sizeof(segment)) withBytes:&segment];
    if (sectionCount) [data replaceBytesInRange:NSMakeRange(sizeof(segment),
        sectionCount*sizeof(struct section_64)) withBytes:sections];
    return data;
}
static NSUInteger elementBytes(ANEElementType type) {
    switch(type) {
        case ANEElementTypeFP16:return 2; case ANEElementTypeFP32:return 4;
        case ANEElementTypeInt8:return 1; case ANEElementTypeInt32:return 4;
        case ANEElementTypeUInt64:return 8; case ANEElementTypeBool:return 1;
        case ANEElementTypeString: case ANEElementTypeInvalid:return 0;
    }
}
static uint32_t elementCode(ANEElementType type) {
    if(type==ANEElementTypeFP16)return 5; if(type==ANEElementTypeInt8)return 2;
    if(type==ANEElementTypeFP32)return 6; return 0;
}
static void shapeDimensions(NSArray<NSNumber *> *s, NSUInteger *n,
                            NSUInteger *c, NSUInteger *h, NSUInteger *w) {
    *n = 1; *c = 1; *h = 1; *w = 1;
    if (s.count == 4) {
        *n=s[0].unsignedIntegerValue; *c=s[1].unsignedIntegerValue;
        *h=s[2].unsignedIntegerValue; *w=s[3].unsignedIntegerValue;
    } else if (s.count == 3) {
        *n=s[0].unsignedIntegerValue; *h=s[1].unsignedIntegerValue;
        *w=s[2].unsignedIntegerValue;
    } else if (s.count == 2) {
        *h=s[0].unsignedIntegerValue; *w=s[1].unsignedIntegerValue;
    } else if (s.count == 1) {
        *w=s[0].unsignedIntegerValue;
    }
}
static NSUInteger logicalBytes(HWXObjectBinding *binding) {
    NSUInteger count=1; for(NSNumber *d in binding.shape)count*=d.unsignedIntegerValue;
    return count*elementBytes(binding.elementType);
}
static NSData *bufferReference(HWXObjectBinding *binding, uint64_t vmAddress) {
    NSMutableData *data=[NSMutableData dataWithLength:0x20];
    writeU32(data,0,0x40); writeU32(data,4,0x20); writeU64(data,8,0x18);
    writeU64(data,0x10,vmAddress);
    NSData *name=[binding.shortName dataUsingEncoding:NSUTF8StringEncoding];
    NSUInteger length=MIN((NSUInteger)8,name.length);
    [data replaceBytesInRange:NSMakeRange(0x18,length) withBytes:name.bytes];
    return data;
}
static NSData *programDescriptor(uint64_t textAddress,uint64_t textConstAddress,
                                 uint64_t scratchAddress,
                                 NSArray<NSNumber *> *resourceAddresses,
                                 uint32_t taskWordCount,
                                 HWXObjectProgramInfo *info) {
    BOOL mixed = info.descriptorLayout ==
        HWXProgramDescriptorLayoutScratchBackedMixed;
    NSUInteger length = mixed ? 0x8c0 : 0x8a0;
    NSMutableData *d=[NSMutableData dataWithLength:length];
    writeU32(d,0,0x04);writeU32(d,4,(uint32_t)length);writeU32(d,8,4);
    writeU32(d,0xc,mixed ? 0x22a : 0x222);
    writeU64(d,0x10,textAddress);writeU64(d,0x20,textConstAddress);
    if (mixed) writeU64(d,0x40,scratchAddress);
    for (NSUInteger index = 0; index < MIN((NSUInteger)5,
                                           resourceAddresses.count); ++index)
        writeU64(d,0x70 + index * 0x10,
                 resourceAddresses[index].unsignedLongLongValue);
    writeU64(d,0x810,textAddress);
    writeU32(d,0x820,4);writeU32(d,0x824,taskWordCount);
    writeU32(d,0x830,(uint32_t)info.taskCount);
    writeU32(d,0x838,UINT32_MAX);writeU32(d,0x83c,UINT32_MAX);
    writeU32(d,0x844,0xffff);
    writeU32(d,0x860,(uint32_t)info.recordCount);
    if (mixed) {
        writeU32(d,0x848,0x8b8);writeU32(d,0x858,(uint32_t)info.scratchByteLength);
        writeU32(d,0x868,4);writeU32(d,0x870,4);writeU32(d,0x878,2);
        writeU32(d,0x87c,8);writeU32(d,0x880,(uint32_t)info.scratchByteLength);
        writeU32(d,0x888,3);writeU32(d,0x88c,8);writeU32(d,0x890,0x08020004);
        writeU32(d,0x894,8);writeU32(d,0x898,7);writeU32(d,0x89c,8);
        writeU32(d,0x8a0,0x461c4000);writeU32(d,0x8a8,9);writeU32(d,0x8ac,8);
        writeU32(d,0x8b0,info.formatCode);writeString(d,0x8b8,@"main");
    } else {
        writeU32(d,0x848,0x898);writeU32(d,0x868,4);writeU32(d,0x870,2);
        writeU32(d,0x878,7);writeU32(d,0x87c,8);writeU32(d,0x880,0x461c4000);
        writeU32(d,0x888,9);writeU32(d,0x88c,8);writeU32(d,0x890,info.formatCode);
        writeString(d,0x898,@"main");
    }
    return d;
}
static NSData *tensorDescriptor(HWXObjectBinding *binding, uint32_t bindingIndex) {
    NSUInteger n=1,c=1,h=1,w=1;
    shapeDimensions(binding.shape, &n, &c, &h, &w);
    NSUInteger eb=elementBytes(binding.elementType);
    NSUInteger row=binding.rowStrideBytes, plane=binding.planeStrideBytes;
    NSUInteger batch=binding.batchStrideBytes, total=binding.storageByteLength;
    NSData *symbol=[binding.symbol dataUsingEncoding:NSUTF8StringEncoding];
    NSData *shortName=[binding.shortName dataUsingEncoding:NSUTF8StringEncoding];
    uint32_t symbolOffset=0xd3d;
    uint32_t shortOffset=(uint32_t)(symbolOffset+symbol.length+1);
    NSUInteger recordBytes=MAX((NSUInteger)0xd48,
        alignUp(shortOffset+shortName.length+1,8));
    NSMutableData *d=[NSMutableData dataWithLength:recordBytes];
    writeU32(d,0,4);writeU32(d,4,(uint32_t)recordBytes);writeU32(d,8,3);writeU32(d,0xc,0x34a);
    writeU32(d,0x10,3);writeU32(d,0x14,bindingIndex);writeU32(d,0x18,0xd38);
    writeU32(d,0x20,symbolOffset);writeU32(d,0x24,elementCode(binding.elementType));
    writeU32(d,0x28,(uint32_t)n);writeU32(d,0x2c,(uint32_t)c);
    writeU32(d,0x30,(uint32_t)h);writeU32(d,0x34,(uint32_t)w);writeU32(d,0x38,1);
    writeU64(d,0x50,batch);writeU64(d,0x58,plane);writeU64(d,0x60,row);
    writeU64(d,0x68,eb);writeU64(d,0x70,total);writeU32(d,0x78,1);
    writeU32(d,0x7c,shortOffset);writeU64(d,0x80,total);
    writeString(d,0xd38,@"main");writeString(d,symbolOffset,binding.symbol);
    writeString(d,shortOffset,binding.shortName);return d;
}
static NSData *compilerMetadata(void) {
    NSString *text = @"ANEC v1\nresearch_hwx_compiler v1\n\n Module ANEC:\n"
                     @"\t Target: h16g\n";
    NSData *bytes = [text dataUsingEncoding:NSUTF8StringEncoding];
    NSMutableData *record = [NSMutableData dataWithLength:
        alignUp(8 + bytes.length + 1, 8)];
    writeU32(record, 0, 0x08);
    writeU32(record, 4, (uint32_t)record.length);
    [record replaceBytesInRange:NSMakeRange(8, bytes.length)
                      withBytes:bytes.bytes];
    return record;
}
static uint32_t appendSymbolName(NSMutableData *strings, NSString *name) {
    uint32_t offset = (uint32_t)strings.length;
    NSData *bytes = [name dataUsingEncoding:NSUTF8StringEncoding];
    [strings appendData:bytes];
    const uint8_t zero = 0;
    [strings appendBytes:&zero length:1];
    return offset;
}
static void appendSymbol(NSMutableData *symbols, NSMutableData *strings,
                         NSString *name, uint8_t section, uint16_t descriptor,
                         uint64_t value) {
    struct nlist_64 symbol = {};
    symbol.n_un.n_strx = appendSymbolName(strings, name);
    symbol.n_type = N_SECT | N_EXT;
    symbol.n_sect = section;
    symbol.n_desc = descriptor;
    symbol.n_value = value;
    [symbols appendBytes:&symbol length:sizeof(symbol)];
}
static void buildSymbolTables(NSArray<HWXObjectBinding *> *bindings,
                              NSArray<NSNumber *> *bindingAddresses,
                              uint64_t kernelAddress,
                              NSArray<NSNumber *> *kernelSymbolOffsets,
                              uint8_t firstBindingSection,
                              uint8_t kernelSection,
                              NSMutableData **symbolsOut,
                              NSMutableData **stringsOut) {
    NSMutableData *symbols = [NSMutableData data];
    NSMutableData *strings = [NSMutableData dataWithBytes:"\0" length:1];
    for (NSUInteger i = 0; i < bindings.count; ++i)
        appendSymbol(symbols, strings, bindings[i].shortName,
                     (uint8_t)(firstBindingSection + i), 0,
                     bindingAddresses[i].unsignedLongLongValue);
    for (NSUInteger i = 0; i < kernelSymbolOffsets.count; ++i) {
        NSUInteger kernelOffset = kernelSymbolOffsets[i].unsignedIntegerValue;
        appendSymbol(symbols, strings,
            [NSString stringWithFormat:@"__kern_0_tile_%lu", (unsigned long)i],
            kernelSection, i == 0 ? 2 : 0, kernelAddress + kernelOffset);
    }
    *symbolsOut = symbols;
    *stringsOut = strings;
}
static NSData *symbolTableCommand(uint32_t symbolOffset,
                                  uint32_t symbolCount,
                                  uint32_t stringOffset,
                                  uint32_t stringBytes) {
    struct symtab_command command = {};
    command.cmd = LC_SYMTAB;
    command.cmdsize = sizeof(command);
    command.symoff = symbolOffset;
    command.nsyms = symbolCount;
    command.stroff = stringOffset;
    command.strsize = stringBytes;
    return [NSData dataWithBytes:&command length:sizeof(command)];
}

@implementation HWXObjectBinding
- (instancetype)initWithSymbol:(NSString *)symbol shortName:(NSString *)shortName
                           role:(HWXObjectBindingRole)role
                    elementType:(ANEElementType)elementType
                          shape:(NSArray<NSNumber *> *)shape {
    NSUInteger n=1,c=1,h=1,w=1;
    shapeDimensions(shape, &n, &c, &h, &w);
    NSUInteger elementByteCount = elementBytes(elementType);
    NSUInteger row = w * elementByteCount;
    NSUInteger plane = h * row;
    NSUInteger batch = c * plane;
    return [self initWithSymbol:symbol shortName:shortName role:role
        elementType:elementType shape:shape rowStrideBytes:row
        planeStrideBytes:plane batchStrideBytes:batch
        storageByteLength:n * batch];
}

- (instancetype)initWithSymbol:(NSString *)symbol shortName:(NSString *)shortName
                           role:(HWXObjectBindingRole)role
                    elementType:(ANEElementType)elementType
                          shape:(NSArray<NSNumber *> *)shape
                 rowStrideBytes:(NSUInteger)rowStrideBytes
               planeStrideBytes:(NSUInteger)planeStrideBytes
               batchStrideBytes:(NSUInteger)batchStrideBytes
              storageByteLength:(NSUInteger)storageByteLength {
    self=[super init];
    if(self){
        _symbol=[symbol copy]; _shortName=[shortName copy]; _role=role;
        _elementType=elementType; _shape=[shape copy];
        _rowStrideBytes=rowStrideBytes; _planeStrideBytes=planeStrideBytes;
        _batchStrideBytes=batchStrideBytes; _storageByteLength=storageByteLength;
    }
    return self;
}
@end

@implementation HWXObjectProgramInfo
- (instancetype)initWithTaskCount:(NSUInteger)taskCount
                       recordCount:(NSUInteger)recordCount
                        formatCode:(uint32_t)formatCode
                 scratchByteLength:(NSUInteger)scratchByteLength
                  descriptorLayout:(HWXProgramDescriptorLayout)descriptorLayout {
    return [self initWithTaskCount:taskCount recordCount:recordCount
        formatCode:formatCode scratchByteLength:scratchByteLength
        scratchAllocationByteLength:scratchByteLength
        descriptorLayout:descriptorLayout];
}

- (instancetype)initWithTaskCount:(NSUInteger)taskCount
                       recordCount:(NSUInteger)recordCount
                        formatCode:(uint32_t)formatCode
                 scratchByteLength:(NSUInteger)scratchByteLength
       scratchAllocationByteLength:(NSUInteger)scratchAllocationByteLength
                  descriptorLayout:(HWXProgramDescriptorLayout)descriptorLayout {
    self = [super init];
    if (self) {
        _taskCount = taskCount; _recordCount = recordCount;
        _formatCode = formatCode; _scratchByteLength = scratchByteLength;
        _scratchAllocationByteLength = scratchAllocationByteLength;
        _descriptorLayout = descriptorLayout;
    }
    return self;
}
@end

@implementation HWXObjectWriter
+ (NSData *)buildObjectWithTaskDescriptor:(NSData *)taskDescriptor
                             constantRegion:(NSData *)constantRegion
                                    bindings:(NSArray<HWXObjectBinding *> *)bindings
                                       error:(NSError **)error {
    return [self buildObjectWithTaskDescriptor:taskDescriptor
        constantRegion:constantRegion bindings:bindings
        kernelRelocationOffsets:@[@0x1a8] programRecordCount:0x10
        programFormatCode:0x0f error:error];
}

+ (NSData *)buildObjectWithTaskDescriptor:(NSData *)taskDescriptor
                             constantRegion:(NSData *)constantRegion
                                    bindings:(NSArray<HWXObjectBinding *> *)bindings
                     kernelRelocationOffsets:(NSArray<NSNumber *> *)kernelRelocationOffsets
                          programRecordCount:(NSUInteger)programRecordCount
                           programFormatCode:(uint32_t)programFormatCode
                                       error:(NSError **)error {
    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:kernelRelocationOffsets.count
        recordCount:programRecordCount formatCode:programFormatCode
        scratchByteLength:0 descriptorLayout:HWXProgramDescriptorLayoutLinear];
    return [self buildObjectWithTaskDescriptor:taskDescriptor
        constantRegion:constantRegion bindings:bindings
        kernelRelocationOffsets:kernelRelocationOffsets programInfo:info
        error:error];
}

+ (NSData *)buildObjectWithTaskDescriptor:(NSData *)taskDescriptor
                             constantRegion:(NSData *)constantRegion
                                    bindings:(NSArray<HWXObjectBinding *> *)bindings
                     kernelRelocationOffsets:(NSArray<NSNumber *> *)kernelRelocationOffsets
                                 programInfo:(HWXObjectProgramInfo *)programInfo
                                       error:(NSError **)error {
    NSMutableArray<HWXObjectBinding *> *inputs = [NSMutableArray array];
    NSMutableArray<HWXObjectBinding *> *outputs = [NSMutableArray array];
    for (HWXObjectBinding *binding in bindings) {
        if (binding.role == HWXObjectBindingRoleInput)
            [inputs addObject:binding];
        else
            [outputs addObject:binding];
    }
    BOOL hasKernel = constantRegion.length != 0;
    BOOL isTwoSurfaceProgram = bindings.count == 2 && inputs.count == 1 &&
        outputs.count == 1;
    BOOL isThreeSurfaceProgram = bindings.count == 3 && inputs.count == 2 &&
        outputs.count == 1;
    BOOL bindingLayoutsValid = YES;
    for (HWXObjectBinding *binding in bindings) {
        NSUInteger n=1,c=1,h=1,w=1;
        shapeDimensions(binding.shape, &n, &c, &h, &w);
        NSUInteger denseRow = w * elementBytes(binding.elementType);
        bindingLayoutsValid = bindingLayoutsValid && denseRow != 0 &&
            binding.rowStrideBytes >= denseRow &&
            binding.planeStrideBytes >= h * binding.rowStrideBytes &&
            binding.batchStrideBytes >= c * binding.planeStrideBytes &&
            binding.storageByteLength >= n * binding.batchStrideBytes;
    }
    BOOL relocationsValid = hasKernel || kernelRelocationOffsets.count == 0;
    for (NSNumber *offset in kernelRelocationOffsets) {
        NSUInteger value = offset.unsignedIntegerValue;
        relocationsValid = relocationsValid && hasKernel && value % 4 == 0 &&
            value + sizeof(uint32_t) <= taskDescriptor.length;
    }
    if((!isTwoSurfaceProgram && !isThreeSurfaceProgram)||taskDescriptor.length==0||
       taskDescriptor.length>0x3fc0||constantRegion.length>0x100000||
       !programInfo || programInfo.taskCount == 0 ||
       (programInfo.descriptorLayout == HWXProgramDescriptorLayoutLinear &&
        programInfo.scratchByteLength != 0) ||
       (programInfo.descriptorLayout == HWXProgramDescriptorLayoutScratchBackedMixed &&
        (programInfo.scratchByteLength == 0 ||
         (!isThreeSurfaceProgram &&
          programInfo.scratchAllocationByteLength < programInfo.scratchByteLength))) ||
       !relocationsValid || !bindingLayoutsValid){
        if(error)*error=[NSError errorWithDomain:HWXObjectWriterErrorDomain code:1
            userInfo:@{NSLocalizedDescriptionKey:@"writer requires one-output programs with one or two inputs, plus bounded payloads and valid relocations"}];
        return nil;
    }
    uint64_t scratchVM=0x30000000;
    uint64_t nextBindingVM=scratchVM+programInfo.scratchAllocationByteLength;
    NSMutableArray<NSNumber *> *bindingVMs = [NSMutableArray array];
    NSMutableArray<NSNumber *> *bindingSizes = [NSMutableArray array];
    for (HWXObjectBinding *binding in bindings) {
        uint64_t size = alignUp(binding.storageByteLength, 0x4000);
        [bindingVMs addObject:@(nextBindingVM)];
        [bindingSizes addObject:@(size)];
        nextBindingVM += size;
    }
    uint64_t textVM=nextBindingVM;
    uint64_t kernelVM=hasKernel ? textVM+0x8000 : 0;
    uint64_t textConstVM=alignUp(textVM+taskDescriptor.length,0x40);
    BOOL compactThreeSurface = bindings.count == 3 &&
        programInfo.descriptorLayout == HWXProgramDescriptorLayoutLinear &&
        logicalBytes(bindings[0]) <= 0x100000;
    uint32_t textFileOffset = (kernelRelocationOffsets.count <= 1 &&
        (bindings.count == 2 || compactThreeSurface)) ? 0x4000 : 0x8000;
    uint32_t kernelFileOffset = textFileOffset + 0x8000;
    uint64_t kernelSegmentSize = hasKernel
        ? alignUp(constantRegion.length, 0x4000) : 0;
    uint32_t textConstFile=(uint32_t)alignUp(
        textFileOffset+taskDescriptor.length,0x40);
    NSMutableArray<NSData *> *commands=[NSMutableArray array];
    [commands addObject:segmentRecord("__PAGEZERO",0,0x4000,0,0,0,0,4,NULL,0)];
    BOOL hasScratch = programInfo.scratchAllocationByteLength != 0;
    if (hasScratch) {
        struct section_64 scratchSection=sectionRecord("__bss","__DATA",
            scratchVM,programInfo.scratchAllocationByteLength,0,6,0x25);
        [commands addObject:segmentRecord("__DATA",scratchVM,
            programInfo.scratchAllocationByteLength,0,0,3,3,4,&scratchSection,1)];
    }
    for (NSUInteger i = 0; i < bindings.count; ++i) {
        HWXObjectBinding *binding = bindings[i];
        BOOL isInput = binding.role == HWXObjectBindingRoleInput;
        uint64_t address = bindingVMs[i].unsignedLongLongValue;
        uint64_t size = bindingSizes[i].unsignedLongLongValue;
        struct section_64 section=sectionRecord(
            isInput ? "__const" : "__data", "__FVMLIB",
            address,size,0,14,isInput ? 0x21 : 0x23);
        [commands addObject:segmentRecord("__FVMLIB",address,size,0,0,
            isInput ? 1 : 2,isInput ? 1 : 2,6,&section,1)];
    }
    uint32_t textFlags = kernelRelocationOffsets.count == 0 ? 0x28 : 0x128;
    struct section_64 textSecs[2]={
        sectionRecord("__text","__TEXT",textVM,taskDescriptor.length,textFileOffset,14,textFlags),
        sectionRecord("__const","__TEXT",textConstVM,0x4000,textConstFile,6,0x26)};
    uint32_t textSegmentFlags = hasScratch ? 4 : 0;
    [commands addObject:segmentRecord("__TEXT",textVM,0x8000,textFileOffset,0x8000,
        5,5,textSegmentFlags,textSecs,2)];
    if (hasKernel) {
        struct section_64 kernSec=sectionRecord("__kern_0","__KERN_0",kernelVM,
            constantRegion.length,kernelFileOffset,6,0x26);
        [commands addObject:segmentRecord("__KERN_0",kernelVM,kernelSegmentSize,
            kernelFileOffset,kernelSegmentSize,1,1,4,&kernSec,1)];
    }
    for (NSUInteger i = 0; i < bindings.count; ++i)
        [commands addObject:bufferReference(bindings[i],
            bindingVMs[i].unsignedLongLongValue)];
    NSMutableArray<NSNumber *> *resourceAddresses = [bindingVMs mutableCopy];
    if (hasKernel) [resourceAddresses addObject:@(kernelVM)];
    [commands addObject:programDescriptor(textVM,textConstVM,scratchVM,
        resourceAddresses,(uint32_t)(taskDescriptor.length / 4),programInfo)];
    for (HWXObjectBinding *input in inputs)
        [commands addObject:tensorDescriptor(input,1)];
    for (HWXObjectBinding *output in outputs)
        [commands addObject:tensorDescriptor(output,2)];
    [commands addObject:compilerMetadata()];

    NSMutableData *symbols = nil;
    NSMutableData *strings = nil;
    NSMutableOrderedSet<NSNumber *> *kernelOffsets =
        [NSMutableOrderedSet orderedSet];
    if (hasKernel) [kernelOffsets addObject:@0];
    for (NSNumber *relocationOffsetNumber in kernelRelocationOffsets) {
        uint32_t addend = 0;
        [taskDescriptor getBytes:&addend range:NSMakeRange(
            relocationOffsetNumber.unsignedIntegerValue, sizeof(addend))];
        [kernelOffsets addObject:@(addend)];
    }
    uint8_t firstBindingSection = hasScratch ? 2 : 1;
    uint8_t kernelSection = (uint8_t)((hasScratch ? 1 : 0) +
        bindings.count + 3);
    buildSymbolTables(bindings, bindingVMs, kernelVM,
                      kernelOffsets.array, firstBindingSection, kernelSection,
                      &symbols, &strings);
    NSUInteger commandBytes=sizeof(struct symtab_command);
    for(NSData *c in commands)commandBytes+=c.length;
    NSUInteger commandEnd = sizeof(struct mach_header_64) + commandBytes;
    uint32_t symbolOffset = (uint32_t)alignUp(commandEnd, 8);
    uint32_t stringOffset = symbolOffset + (uint32_t)symbols.length;
    uint32_t relocationOffset = (uint32_t)alignUp(
        stringOffset + strings.length, 8);
    [commands addObject:symbolTableCommand(symbolOffset,
        (uint32_t)(symbols.length / sizeof(struct nlist_64)), stringOffset,
        (uint32_t)strings.length)];
    textSecs[0].reloff = relocationOffset;
    textSecs[0].nreloc = (uint32_t)kernelRelocationOffsets.count;
    NSUInteger textCommandIndex = 1 + (hasScratch ? 1 : 0) + bindings.count;
    commands[textCommandIndex] = segmentRecord("__TEXT",textVM,0x8000,textFileOffset,0x8000,
        5,5,textSegmentFlags,textSecs,2);
    if(sizeof(struct mach_header_64)+commandBytes>textFileOffset){
        if(error) {
            *error=[NSError errorWithDomain:HWXObjectWriterErrorDomain code:2
                userInfo:@{NSLocalizedDescriptionKey:@"generated commands cross __TEXT"}];
        }
        return nil;
    }
    NSUInteger relocationBytes = kernelRelocationOffsets.count * 8;
    if (relocationOffset > textFileOffset - relocationBytes) {
        if (error) {
            *error=[NSError errorWithDomain:HWXObjectWriterErrorDomain code:3
                userInfo:@{NSLocalizedDescriptionKey:
                    @"generated symbols and relocation cross __TEXT"}];
        }
        return nil;
    }
    NSUInteger imageLength = hasKernel
        ? kernelFileOffset + kernelSegmentSize : textFileOffset + 0x8000;
    NSMutableData *image=[NSMutableData dataWithLength:imageLength];
    struct mach_header_64 header={};header.magic=0xBEEFFACE;header.cputype=0x80;
    header.cpusubtype=0x07;header.filetype=2;header.ncmds=(uint32_t)commands.count;
    header.sizeofcmds=(uint32_t)commandBytes;header.flags=0x00200000;
    [image replaceBytesInRange:NSMakeRange(0,sizeof(header)) withBytes:&header];
    NSUInteger offset=sizeof(header);for(NSData *c in commands){
        [image replaceBytesInRange:NSMakeRange(offset,c.length) withBytes:c.bytes];offset+=c.length;}
    [image replaceBytesInRange:NSMakeRange(symbolOffset,symbols.length)
                     withBytes:symbols.bytes];
    [image replaceBytesInRange:NSMakeRange(stringOffset,strings.length)
                     withBytes:strings.bytes];
    for (NSUInteger i = 0; i < kernelRelocationOffsets.count; ++i) {
        const uint32_t relocation[2] = {
            kernelRelocationOffsets[i].unsignedIntValue,
            (uint32_t)(hasScratch ? 0x07000006 : 0x07000005)};
        [image replaceBytesInRange:NSMakeRange(relocationOffset + i * 8,
            sizeof(relocation)) withBytes:relocation];
    }
    [image replaceBytesInRange:NSMakeRange(textFileOffset,taskDescriptor.length)
                     withBytes:taskDescriptor.bytes];
    if (hasKernel) {
        [image replaceBytesInRange:NSMakeRange(kernelFileOffset,constantRegion.length)
                         withBytes:constantRegion.bytes];
    }
    return image;
}
@end
