#import "H16GConvChainEncoder.h"

#import "H16GTDWriter.h"

static BOOL put(H16GTDWriter *writer, NSUInteger base, NSUInteger offset,
                uint32_t value, NSString *field, NSError **error) {
    return [writer writeUInt32:value atOffset:base + offset
                         field:field error:error];
}

static BOOL writeTapTable(H16GTDWriter *writer, NSUInteger base,
                          NSUInteger offset, NSError **error) {
    for (NSUInteger i = 0; i < 31; ++i) {
        uint32_t value = i < 15 ? (uint32_t)(i + 1) * 0x100 : 0x100;
        if (!put(writer, base, offset + i * 4, value,
                 @"conv.weight_tap", error)) return NO;
    }
    return YES;
}

static BOOL writeIndexedChain(H16GTDWriter *writer, NSUInteger base,
                              NSUInteger first, NSUInteger last,
                              uint32_t value, NSError **error) {
    for (NSUInteger offset = first; offset <= last; offset += 4)
        if (!put(writer, base, offset, value,
                 @"conv.task_chain", error)) return NO;
    return YES;
}

static BOOL writeFirstBlock(H16GTDWriter *w, NSError **error) {
    const NSUInteger b = 0;
    if (!put(w,b,0x000,0x00000001,@"first.version",error) ||
        !put(w,b,0x010,0x00660000,@"first.op_class",error) ||
        !put(w,b,0x014,0x00000007,@"first.descriptor_count",error) ||
        !put(w,b,0x018,0x00000068,@"first.dispatch",error) ||
        !put(w,b,0x020,0x00fff868,@"first.dispatch_limit",error) ||
        !put(w,b,0x02c,0x0000001f,@"first.task_mask",error) ||
        !put(w,b,0x030,0x00050001,@"first.control",error) ||
        !put(w,b,0x034,0xffc01540,@"first.control_mask",error) ||
        !put(w,b,0x038,0x00010240,@"first.control_shape",error) ||
        !writeIndexedChain(w,b,0x03c,0x05c,0x21,error) ||
        !put(w,b,0x060,0x00031551,@"first.barrier",error) ||
        !writeIndexedChain(w,b,0x064,0x07c,0x21,error) ||
        !put(w,b,0x080,0x000f1559,@"first.tap_header",error) ||
        !writeTapTable(w,b,0x084,error)) return NO;
    const NSUInteger offsets[] = {
        0x100,0x104,0x108,0x10c,0x110,0x114,0x118,0x11c,0x120,0x124,
        0x128,0x12c,0x130,0x134,0x138,0x13c,0x140,0x144,0x148,0x14c,
        0x150,0x154,0x158,0x15c,0x160,0x164,0x168,0x16c,0x170,0x174,
        0x178,0x17c,0x180,0x184,0x188,0x194,0x1a0
    };
    const uint32_t values[] = {
        0x00018000,0x0000001a,0x40,0x40,0x40,0x93418005,0x40,0x40,
        0x40,0x40,0x00210000,0x2,0x00100000,0x800c1342,0xce,0x80,
        0x2000,0x135a,0x01002031,0x803e9041,0x00500172,0x00500130,
        0x00041400,0x80,0x2000,0x2000,0x2000,0x80031052,0x00500130,
        0x40,0x1050,0x80041240,0x81,0x3c00,0x24009544,0x23809344,
        0x22001340
    };
    for (NSUInteger i = 0; i < sizeof(offsets)/sizeof(offsets[0]); ++i)
        if (!put(w,b,offsets[i],values[i],@"first.command",error)) return NO;
    return YES;
}

static BOOL writeMiddleBlock(H16GTDWriter *w, NSUInteger base,
                             NSUInteger layer, NSError **error) {
    uint32_t chain = (uint32_t)(layer << 16) | 0x21;
    uint32_t identity = layer == 1
        ? 0x00590001 : 0x005a0000 | (uint32_t)layer;
    uint32_t historyMask = 0x1ffu << (4 * (layer - 1));
    uint32_t taskControl = (uint32_t)((layer + 1) << 16) | 0x240;
    if ((layer == 1 && !put(w,base,0x000,0x21,@"middle.link",error)) ||
        !put(w,base,0x00c,identity,@"middle.identity",error) ||
        !put(w,base,0x010,1,@"middle.version",error) ||
        !put(w,base,0x01c,0x00fff820,@"middle.dispatch_limit",error) ||
        !put(w,base,0x028,historyMask,@"middle.history_mask",error) ||
        !put(w,base,0x02c,0x00050001,@"middle.control",error) ||
        !put(w,base,0x030,0xffc01540,@"middle.control_mask",error) ||
        !put(w,base,0x034,taskControl,@"middle.task_control",error) ||
        !writeIndexedChain(w,base,0x038,0x058,chain,error) ||
        !put(w,base,0x05c,0x00031551,@"middle.barrier",error) ||
        !writeIndexedChain(w,base,0x060,0x078,chain,error) ||
        !put(w,base,0x07c,0x000f1559,@"middle.tap_header",error) ||
        !writeTapTable(w,base,0x080,error)) return NO;
    const NSUInteger offsets[] = {0x0fc,0x100,0x104,0x108,0x10c,0x110,
        0x114,0x118,0x11c,0x120,0x124,0x128,0x12c,0x130,0x134,0x138};
    const uint32_t values[] = {0x18000,0x19,0x40,0x40,0x40,0xb3418005,
        0x40,0x40,0x40,0x40,0x00222200,2,0x00100000,3,
        layer == 1 ? 0x800c1041u : 0x800e1041u,
        layer == 1 ? 0x120u : 0x104u};
    for (NSUInteger i = 0; i < sizeof(offsets)/sizeof(offsets[0]); ++i)
        if (!put(w,base,offsets[i],values[i],@"middle.command",error)) return NO;
    if (layer == 1) {
        const uint32_t stream[] = {0x40,0x1050,0x11053,0x41400,0x40,0x1050,
            0x80049240,0x81,0x04100000,0x3000,0x24009544,
            (uint32_t)(layer * 0x1000)};
        for (NSUInteger i = 0; i < sizeof(stream)/sizeof(stream[0]); ++i)
            if (!put(w,base,0x13c+i*4,stream[i],@"middle.stream",error)) return NO;
    } else {
        const uint32_t stream[] = {
            (uint32_t)((layer-1)*0x41400),0x40,0x1050,0x11053,
            (uint32_t)(layer*0x41400),0x40,0x1050,0x80049240,0x81,
            0x04100000,0x3000,0x24009544,(uint32_t)(layer*0x1000)};
        for (NSUInteger i = 0; i < sizeof(stream)/sizeof(stream[0]); ++i)
            if (!put(w,base,0x13c+i*4,stream[i],@"middle.stream",error)) return NO;
    }
    return YES;
}

static BOOL writeFinalBlock(H16GTDWriter *w, NSUInteger base,
                            NSUInteger layer, NSError **error) {
    uint32_t chain = (uint32_t)(layer << 16) | 0x21;
    if (!put(w,base,0x00c,0x00680000|(uint32_t)layer,@"final.identity",error) ||
        !put(w,base,0x010,7,@"final.version",error) ||
        !put(w,base,0x014,0x04000000,@"final.dispatch",error) ||
        !put(w,base,0x01c,0x00fff860,@"final.dispatch_limit",error) ||
        !put(w,base,0x02c,0x00050009,@"final.control",error) ||
        !put(w,base,0x030,0xffc01540,@"final.control_mask",error) ||
        !put(w,base,0x034,(uint32_t)((layer+1)<<16)|0x2c0,
             @"final.task_control",error) ||
        !writeIndexedChain(w,base,0x038,0x058,chain,error) ||
        !put(w,base,0x05c,0x00031551,@"final.barrier",error) ||
        !writeIndexedChain(w,base,0x060,0x078,chain,error) ||
        !put(w,base,0x07c,0x000f1559,@"final.tap_header",error) ||
        !writeTapTable(w,base,0x080,error)) return NO;
    const NSUInteger offsets[] = {0x0fc,0x100,0x104,0x108,0x10c,0x110,
        0x114,0x118,0x11c,0x120,0x124,0x128,0x12c,0x130,0x134,0x138,
        0x13c,0x140,0x144,0x148,0x14c,0x150,0x154,0x158,0x15c,0x160,
        0x164,0x168,0x16c,0x170,0x174,0x178,0x17c,0x184,0x188,0x18c,
        0x190,0x198,0x1a4,0x1a8};
    const uint32_t values[] = {0x18000,0x29,0x40,0x40,0x40,0x93418005,
        0x40,0x40,0x40,0x40,0x00220000,2,0x00100000,0x800e1041,0x104,
        (uint32_t)((layer-1)*0x41400),0x40,0x1050,0x9052,0x14a,
        (uint32_t)(layer*0x41400),0x80049240,0x81,0x04100000,0x2400,
        0x81009444,0x80,0x2000,0x01302031,0x83119640,0x12,0x00a000a0,
        (uint32_t)(layer<<16)|0xe0,0x007f0000,0x20010701,0x24009544,
        (uint32_t)(layer*0x1000),0x23009442,0x22001440,
        0x01000031|(uint32_t)(layer<<16)};
    for (NSUInteger i = 0; i < sizeof(offsets)/sizeof(offsets[0]); ++i)
        if (!put(w,base,offsets[i],values[i],@"final.command",error)) return NO;
    return YES;
}

@implementation H16GEncodedTDProgram
- (instancetype)initWithData:(NSData *)data
      kernelRelocationOffsets:(NSArray<NSNumber *> *)kernelRelocationOffsets
           programRecordCount:(NSUInteger)programRecordCount
            programFormatCode:(uint32_t)programFormatCode {
    return [self initWithData:data
        kernelRelocationOffsets:kernelRelocationOffsets
        programRecordCount:programRecordCount programFormatCode:programFormatCode
        scratchByteLength:0];
}
- (instancetype)initWithData:(NSData *)data
      kernelRelocationOffsets:(NSArray<NSNumber *> *)kernelRelocationOffsets
           programRecordCount:(NSUInteger)programRecordCount
            programFormatCode:(uint32_t)programFormatCode
            scratchByteLength:(NSUInteger)scratchByteLength {
    self = [super init];
    if (self) {
        _data = [data copy];
        _kernelRelocationOffsets = [kernelRelocationOffsets copy];
        _programRecordCount = programRecordCount;
        _programFormatCode = programFormatCode;
        _scratchByteLength = scratchByteLength;
    }
    return self;
}
@end

@implementation H16GConvChainEncoder
+ (H16GEncodedTDProgram *)encodeW8A8C64S64WithDepth:(NSUInteger)depth
                                               error:(NSError **)error {
    if (depth < 3 || depth > 6) {
        if (error) *error = [NSError errorWithDomain:H16GTDWriterErrorDomain
            code:3 userInfo:@{NSLocalizedDescriptionKey:
                @"decoded W8A8 chain grammar supports measured depths three through six"}];
        return nil;
    }
    NSUInteger finalBase = 0x1a4 + (depth - 2) * 0x170;
    H16GTDWriter *writer = [[H16GTDWriter alloc]
        initWithByteLength:finalBase + 0x1ac];
    NSMutableArray<NSNumber *> *relocations = [NSMutableArray array];
    if (!writeFirstBlock(writer,error)) return nil;
    [relocations addObject:@0x18c];
    for (NSUInteger layer = 1; layer + 1 < depth; ++layer) {
        NSUInteger base = 0x1a4 + (layer - 1) * 0x170;
        if (!writeMiddleBlock(writer,base,layer,error)) return nil;
        [relocations addObject:@(base + (layer == 1 ? 0x168 : 0x16c))];
    }
    if (!writeFinalBlock(writer,finalBase,depth-1,error)) return nil;
    [relocations addObject:@(finalBase + 0x190)];
    return [[H16GEncodedTDProgram alloc] initWithData:writer.data
        kernelRelocationOffsets:relocations
        programRecordCount:14 * depth + 3 programFormatCode:7];
}
@end
