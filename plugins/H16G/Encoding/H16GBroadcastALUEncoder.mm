#import "H16GBroadcastALUEncoder.h"

#import "H16GTDWriter.h"

#include <cmath>
#include <cstring>

static NSString *const H16GBroadcastALUErrorDomain =
    @"ANE.H16G.BroadcastALUEncoder";

static void setError(NSError **error, NSString *message) {
    if (error) *error = [NSError errorWithDomain:H16GBroadcastALUErrorDomain
        code:1 userInfo:@{NSLocalizedDescriptionKey:message}];
}

static BOOL put(H16GTDWriter *writer, NSUInteger offset, uint32_t value,
                NSString *field, NSError **error) {
    return [writer writeUInt32:value atOffset:offset field:field error:error];
}

#define WRITE(offset, value, field) do { \
    if (!put(writer, offset, value, field, error)) return nil; \
} while (0)

static H16GEncodedTDProgram *encodeRowState(uint32_t opcode,
                                            NSError **error) {
    H16GTDWriter *writer = [[H16GTDWriter alloc] initWithByteLength:0x130];
    WRITE(0x000,0x00000001,@"program.start");
    WRITE(0x010,0x00480000,@"program.packet_words");
    WRITE(0x018,0x04000068,@"program.records");
    WRITE(0x020,0x00fff868,@"program.dispatch_limit");
    WRITE(0x030,0x00000009,@"tensor.count");
    WRITE(0x034,0x00001540,@"tensor.control");
    WRITE(0x038,0x00000080,@"tensor.rows");
    WRITE(0x03c,0x00010001,@"tensor.rank");
    WRITE(0x040,0x00000001,@"tensor.channels");
    WRITE(0x044,0x00000080,@"tensor.height");
    WRITE(0x048,0x00000001,@"tensor.width");
    WRITE(0x04c,0x95418005,@"tensor.layout");
    WRITE(0x050,0x00000001,@"output.channels");
    WRITE(0x054,0x00000080,@"output.height");
    WRITE(0x058,0x00000001,@"output.width");
    WRITE(0x05c,0x00000080,@"output.rows");
    WRITE(0x060,0x00000030,@"alu.vector_width");
    WRITE(0x064,0x00000044,@"alu.geometry");
    WRITE(0x06c,0x81049342,@"dma.load");
    WRITE(0x070,0x000000ce,@"dma.left_flags");
    WRITE(0x074,0x000000ce,@"dma.right_flags");
    WRITE(0x078,0x00000040,@"dma.left_row_bytes");
    WRITE(0x07c,0x00000040,@"dma.right_row_bytes");
    WRITE(0x080,0x0000935a,@"dma.input_geometry");
    WRITE(0x084,0x01002031,@"dma.left_descriptor");
    WRITE(0x088,0x01002031,@"dma.right_descriptor");
    WRITE(0x08c,0x00009041,@"dma.read_command");
    WRITE(0x090,0x00500172,@"dma.left_stride");
    WRITE(0x094,0x00500172,@"dma.right_stride");
    WRITE(0x098,0x987f9045,@"dma.layout_command");
    WRITE(0x09c,0x00000010,@"dma.left_width");
    WRITE(0x0a0,0x00000020,@"dma.left_span");
    WRITE(0x0a4,0x00000010,@"dma.right_width");
    WRITE(0x0a8,0x00000010,@"dma.output_width");
    WRITE(0x0ac,0x00001040,@"dma.left_plane");
    WRITE(0x0b0,0x00000010,@"dma.right_row");
    WRITE(0x0b4,0x00000020,@"dma.right_span");
    WRITE(0x0b8,0x00000010,@"dma.output_row");
    WRITE(0x0bc,0x00000010,@"dma.output_span");
    WRITE(0x0c0,0x0050017a,@"dma.output_geometry");
    WRITE(0x0c4,0x00002040,@"dma.output_plane");
    WRITE(0x0c8,0x00001140,@"alu.mode");
    WRITE(0x0cc,opcode,@"alu.opcode");
    WRITE(0x0d0,0x81001444,@"dma.store");
    WRITE(0x0d4,0x00000040,@"dma.store_row_bytes");
    WRITE(0x0d8,0x01302031,@"dma.store_descriptor");
    WRITE(0x0dc,0x83109640,@"dma.store_command");
    WRITE(0x0e0,0x00000012,@"dma.store_flags");
    WRITE(0x0e4,0x00a000a0,@"dma.store_geometry");
    WRITE(0x0ec,0x007f0000,@"task.store_mask");
    WRITE(0x0f0,0x20010701,@"task.store_surface");
    WRITE(0x0f4,0x23809344,@"task.wait_left");
    WRITE(0x100,0x2300934a,@"task.wait_right");
    WRITE(0x10c,0x24009442,@"task.signal");
    WRITE(0x118,0x22001340,@"task.begin");
    WRITE(0x11c,0x00000021,@"task.begin_stamp");
    WRITE(0x120,0x22001341,@"task.complete");
    WRITE(0x124,0x00000021,@"task.complete_stamp");
    WRITE(0x128,0x22001440,@"task.end");
    WRITE(0x12c,0x01000031,@"task.end_stamp");
    return [[H16GEncodedTDProgram alloc] initWithData:writer.data
        kernelRelocationOffsets:@[] programRecordCount:16
        programFormatCode:0 scratchByteLength:0];
}

static H16GEncodedTDProgram *encodeMatrixRow(uint32_t opcode,
                                             NSError **error) {
    H16GTDWriter *writer = [[H16GTDWriter alloc] initWithByteLength:0x134];
    WRITE(0x000,0x00000001,@"program.start");
    WRITE(0x010,0x00490000,@"program.packet_words");
    WRITE(0x014,0x00000001,@"program.format");
    WRITE(0x018,0x04000068,@"program.records");
    WRITE(0x020,0x00fff868,@"program.dispatch_limit");
    WRITE(0x030,0x00000009,@"tensor.count");
    WRITE(0x034,0x00001540,@"tensor.control");
    WRITE(0x038,0x00000080,@"tensor.rows");
    WRITE(0x03c,0x00010001,@"tensor.rank");
    WRITE(0x040,0x00000080,@"tensor.height");
    WRITE(0x044,0x00000080,@"tensor.width");
    WRITE(0x048,0x00000001,@"tensor.channels");
    WRITE(0x04c,0x9d418005,@"tensor.broadcast_layout");
    WRITE(0x050,0x00000080,@"output.height");
    WRITE(0x054,0x00000080,@"output.width");
    WRITE(0x058,0x00000001,@"output.channels");
    WRITE(0x05c,0x00000080,@"output.rows");
    WRITE(0x060,0x00000030,@"alu.vector_width");
    WRITE(0x064,0x00000017,@"alu.geometry");
    WRITE(0x068,0x00000010,@"alu.broadcast_width");
    WRITE(0x070,0x81049342,@"dma.load");
    WRITE(0x074,0x000000ce,@"dma.left_flags");
    WRITE(0x078,0x000000ce,@"dma.right_flags");
    WRITE(0x07c,0x00000100,@"dma.left_row_bytes");
    WRITE(0x080,0x00000040,@"dma.right_row_bytes");
    WRITE(0x084,0x0000935a,@"dma.input_geometry");
    WRITE(0x088,0x01002031,@"dma.left_descriptor");
    WRITE(0x08c,0x01002031,@"dma.right_descriptor");
    WRITE(0x090,0x00009041,@"dma.read_command");
    WRITE(0x094,0x00500172,@"dma.left_stride");
    WRITE(0x098,0x00500172,@"dma.right_stride");
    WRITE(0x09c,0x987f9045,@"dma.layout_command");
    WRITE(0x0a0,0x00000100,@"dma.left_width");
    WRITE(0x0a4,0x00000100,@"dma.left_height");
    WRITE(0x0a8,0x00000100,@"dma.output_width");
    WRITE(0x0ac,0x00000100,@"dma.output_height");
    WRITE(0x0b0,0x00008000,@"dma.left_plane");
    WRITE(0x0b4,0x00000010,@"dma.right_width");
    WRITE(0x0b8,0x00000010,@"dma.right_height");
    WRITE(0x0bc,0x00000010,@"dma.right_plane");
    WRITE(0x0c0,0x00000010,@"dma.broadcast_span");
    WRITE(0x0c4,0x0050017a,@"dma.output_geometry");
    WRITE(0x0c8,0x00008800,@"dma.output_plane");
    WRITE(0x0cc,0x00001140,@"alu.mode");
    WRITE(0x0d0,opcode,@"alu.opcode");
    WRITE(0x0d4,0x81001444,@"dma.store");
    WRITE(0x0d8,0x00000100,@"dma.store_row_bytes");
    WRITE(0x0dc,0x01302031,@"dma.store_descriptor");
    WRITE(0x0e0,0x83109640,@"dma.store_command");
    WRITE(0x0e4,0x00000012,@"dma.store_flags");
    WRITE(0x0e8,0x00a000a0,@"dma.store_geometry");
    WRITE(0x0f0,0x007f0000,@"task.store_mask");
    WRITE(0x0f4,0x20010701,@"task.store_surface");
    WRITE(0x0f8,0x23809344,@"task.wait_left");
    WRITE(0x104,0x2300934a,@"task.wait_right");
    WRITE(0x110,0x24009442,@"task.signal");
    WRITE(0x11c,0x22001340,@"task.begin");
    WRITE(0x120,0x00000021,@"task.begin_stamp");
    WRITE(0x124,0x22001341,@"task.complete");
    WRITE(0x128,0x00000021,@"task.complete_stamp");
    WRITE(0x12c,0x22001440,@"task.end");
    WRITE(0x130,0x01000031,@"task.end_stamp");
    return [[H16GEncodedTDProgram alloc] initWithData:writer.data
        kernelRelocationOffsets:@[] programRecordCount:16
        programFormatCode:1 scratchByteLength:0];
}

uint32_t H16GALUScalarWord(double value) {
    float single = (float)value;
    uint32_t bits = 0;
    memcpy(&bits, &single, sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u)
        return bits & 0xffffe000u;
    // Round to nearest even on the 13 mantissa bits that are dropped.
    uint32_t dropped = bits & 0x1fffu;
    uint32_t kept = bits & 0xffffe000u;
    uint32_t halfway = 0x1000u;
    if (dropped > halfway || (dropped == halfway && (kept & 0x2000u)))
        kept += 0x2000u;
    return kept;
}

static H16GEncodedTDProgram *encodeScale(double scalar, NSError **error) {
    if (!std::isfinite(scalar)) {
        setError(error, @"scalar scale requires a finite operand");
        return nil;
    }
    uint32_t scalarWord = H16GALUScalarWord(scalar);
    H16GTDWriter *writer = [[H16GTDWriter alloc] initWithByteLength:0xfc];
    WRITE(0x000,0x00000001,@"program.start");
    WRITE(0x010,0x003b0000,@"program.packet_words");
    WRITE(0x018,0x04000068,@"program.records");
    WRITE(0x020,0x00fff868,@"program.dispatch_limit");
    WRITE(0x030,0x00000009,@"tensor.count");
    WRITE(0x034,0x00001540,@"tensor.control");
    WRITE(0x038,0x00000080,@"tensor.rows");
    WRITE(0x03c,0x00010001,@"tensor.rank");
    WRITE(0x040,0x00000080,@"tensor.height");
    WRITE(0x044,0x00000080,@"tensor.width");
    WRITE(0x048,0x00000001,@"tensor.channels");
    WRITE(0x04c,0x95418005,@"tensor.layout");
    WRITE(0x050,0x00000080,@"output.height");
    WRITE(0x054,0x00000080,@"output.width");
    WRITE(0x058,0x00000001,@"output.channels");
    WRITE(0x05c,0x00000080,@"output.rows");
    WRITE(0x060,0x00000020,@"alu.vector_width");
    WRITE(0x064,0x00000027,@"alu.geometry");
    WRITE(0x06c,0x80041342,@"dma.load");
    WRITE(0x070,0x000000ce,@"dma.load_flags");
    WRITE(0x074,0x00000100,@"dma.input_row_bytes");
    WRITE(0x078,0x0000135a,@"dma.input_geometry");
    WRITE(0x07c,0x01002031,@"dma.input_descriptor");
    WRITE(0x080,0x00009041,@"dma.read_command");
    WRITE(0x084,0x00500172,@"dma.input_stride");
    WRITE(0x088,0x00500130,@"dma.output_stride");
    WRITE(0x08c,0x98039045,@"dma.layout_command");
    WRITE(0x090,0x00000100,@"dma.input_width");
    WRITE(0x094,0x00000100,@"dma.input_height");
    WRITE(0x098,0x00000100,@"dma.output_width");
    WRITE(0x09c,0x00000100,@"dma.output_height");
    WRITE(0x0a0,0x0050017a,@"dma.output_geometry");
    WRITE(0x0a4,0x00008000,@"dma.output_plane");
    WRITE(0x0a8,0x00001142,@"alu.scalar_mode");
    WRITE(0x0ac,scalarWord,@"alu.scalar_operand");
    WRITE(0x0b0,0x81001444,@"dma.store");
    WRITE(0x0b4,0x00000100,@"dma.store_row_bytes");
    WRITE(0x0b8,0x01302031,@"dma.store_descriptor");
    WRITE(0x0bc,0x83109640,@"dma.store_command");
    WRITE(0x0c0,0x00000012,@"dma.store_flags");
    WRITE(0x0c4,0x00a000a0,@"dma.store_geometry");
    WRITE(0x0cc,0x007f0000,@"task.store_mask");
    WRITE(0x0d0,0x20010701,@"task.store_surface");
    WRITE(0x0d4,0x23009344,@"task.wait");
    WRITE(0x0e0,0x23809442,@"task.signal");
    WRITE(0x0ec,0x22001340,@"task.begin");
    WRITE(0x0f0,0x00000021,@"task.begin_stamp");
    WRITE(0x0f4,0x22001440,@"task.end");
    WRITE(0x0f8,0x01000031,@"task.end_stamp");
    return [[H16GEncodedTDProgram alloc] initWithData:writer.data
        kernelRelocationOffsets:@[] programRecordCount:16
        programFormatCode:0 scratchByteLength:0];
}

#undef WRITE

@implementation H16GBroadcastALUEncoder
+ (H16GEncodedTDProgram *)encodeScalarScaleForMatrixRows:(NSUInteger)rows
                                                  columns:(NSUInteger)columns
                                                   scalar:(double)scalar
                                                    error:(NSError **)error {
    if (rows != 128 || columns != 128) {
        setError(error,@"scalar scale requires the measured 128x128 geometry");
        return nil;
    }
    return encodeScale(scalar, error);
}

+ (H16GEncodedTDProgram *)encodeMatrixRowOperation:(NSString *)operationName
                                               rows:(NSUInteger)rows
                                            columns:(NSUInteger)columns
                                              error:(NSError **)error {
    uint32_t opcode = 0;
    if ([operationName isEqualToString:@"sub"]) opcode = 0x000c0000;
    else if ([operationName isEqualToString:@"mul"]) opcode = 0x00080004;
    else {
        setError(error,@"matrix-row broadcast requires measured sub or mul");
        return nil;
    }
    if (rows != 128 || columns != 128) {
        setError(error,@"matrix-row broadcast requires measured 128x128 geometry");
        return nil;
    }
    return encodeMatrixRow(opcode,error);
}

+ (H16GEncodedTDProgram *)encodeRowOperation:(NSString *)operationName
                                         rows:(NSUInteger)rows
                                        error:(NSError **)error {
    uint32_t opcode = 0;
    if ([operationName isEqualToString:@"add"]) opcode = 0x00080000;
    else if ([operationName isEqualToString:@"mul"]) opcode = 0x00080004;
    else if ([operationName isEqualToString:@"max"] ||
             [operationName isEqualToString:@"maximum"])
        opcode = 0x00080008;
    else {
        setError(error,@"row-state operation requires measured add, mul or max");
        return nil;
    }
    if (rows != 128) {
        setError(error,@"row-state operation requires measured 128-row geometry");
        return nil;
    }
    return encodeRowState(opcode,error);
}
@end
