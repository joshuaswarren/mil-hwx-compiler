#import "H16GMatrixRowDivisionEncoder.h"

#import "H16GLUTEncoder.h"
#import "H16GTDWriter.h"

static NSString *const H16GMatrixRowDivisionErrorDomain =
    @"ANE.H16G.MatrixRowDivisionEncoder";

typedef struct {
    NSUInteger offset;
    uint32_t value;
    const char *field;
} H16GNamedWord;

static const H16GNamedWord kDivision128Fields[] = {
    {0x000,0x00000001,"program.start"},
    {0x010,0x00370000,"reciprocal.packet_words"},
    {0x018,0x00000068,"reciprocal.records"},
    {0x020,0x00fff868,"reciprocal.dispatch_limit"},
    {0x02c,0x00000001,"reciprocal.task_index"},
    {0x030,0x00010001,"reciprocal.tensor_rank"},
    {0x034,0x00001540,"reciprocal.tensor_control"},
    {0x038,0x00010240,"reciprocal.table_control"},
    {0x03c,0x00009584,"reciprocal.table_mode"},
    {0x040,0x00000021,"reciprocal.table_flags"},
    {0x044,0x00000080,"reciprocal.rows"},
    {0x048,0x00010001,"reciprocal.input_rank"},
    {0x04c,0x00000001,"reciprocal.input_channels"},
    {0x050,0x00000080,"reciprocal.input_height"},
    {0x054,0x00000001,"reciprocal.input_width"},
    {0x058,0x93618005,"reciprocal.input_layout"},
    {0x05c,0x00000001,"reciprocal.output_channels"},
    {0x060,0x00000080,"reciprocal.output_height"},
    {0x064,0x00000001,"reciprocal.output_width"},
    {0x068,0x00014000,"reciprocal.output_layout"},
    {0x06c,0x00000080,"reciprocal.output_rows"},
    {0x070,0x00040000,"reciprocal.kernel_mode"},
    {0x078,0x00100000,"reciprocal.kernel_span"},
    {0x07c,0x80041342,"reciprocal.dma_load"},
    {0x080,0x000000ce,"reciprocal.dma_flags"},
    {0x084,0x00000040,"reciprocal.input_row_bytes"},
    {0x088,0x0000135a,"reciprocal.input_geometry"},
    {0x08c,0x01002031,"reciprocal.input_descriptor"},
    {0x090,0x803e9041,"reciprocal.read_command"},
    {0x094,0x00500172,"reciprocal.input_stride"},
    {0x098,0x00500130,"reciprocal.output_stride"},
    {0x09c,0x00001000,"reciprocal.input_storage"},
    {0x0a0,0x00000010,"reciprocal.input_width_bytes"},
    {0x0a4,0x00000020,"reciprocal.input_span"},
    {0x0a8,0x00000010,"reciprocal.output_width_bytes"},
    {0x0ac,0x00000010,"reciprocal.output_span"},
    {0x0b0,0x80021052,"reciprocal.table_load"},
    {0x0b4,0x00500170,"reciprocal.table_stride"},
    {0x0b8,0x00000020,"reciprocal.table_span"},
    {0x0bc,0x80049240,"reciprocal.table_descriptor"},
    {0x0c0,0x00010082,"reciprocal.table_index"},
    {0x0c4,0x00123c0c,"reciprocal.table_interpolation"},
    {0x0c8,0x00003c00,"reciprocal.table_scale"},
    {0x0cc,0x24809586,"reciprocal.kernel_reference"},
    {0x0d8,0x23009344,"reciprocal.wait"},
    {0x0e4,0x22001340,"reciprocal.begin"},
    {0x0e8,0x00000021,"reciprocal.begin_stamp"},
    {0x0f0,0x003f0001,"multiply.packet_words"},
    {0x0f8,0x04000000,"multiply.records"},
    {0x100,0x00fff860,"multiply.dispatch_limit"},
    {0x110,0x00000009,"multiply.tensor_count"},
    {0x114,0x00001540,"multiply.tensor_control"},
    {0x118,0x00000080,"multiply.rows"},
    {0x11c,0x00010001,"multiply.tensor_rank"},
    {0x120,0x00000080,"multiply.height"},
    {0x124,0x00000080,"multiply.width"},
    {0x128,0x00000001,"multiply.channels"},
    {0x12c,0x9d418005,"multiply.broadcast_layout"},
    {0x130,0x00000080,"multiply.output_height"},
    {0x134,0x00000080,"multiply.output_width"},
    {0x138,0x00000001,"multiply.output_channels"},
    {0x13c,0x00000080,"multiply.output_rows"},
    {0x140,0x00000030,"multiply.vector_width"},
    {0x144,0x00000017,"multiply.geometry"},
    {0x148,0x00000010,"multiply.broadcast_width"},
    {0x150,0x80041342,"multiply.dma_load"},
    {0x154,0x000000ce,"multiply.dma_flags"},
    {0x158,0x00000100,"multiply.input_row_bytes"},
    {0x15c,0x0000135a,"multiply.input_geometry"},
    {0x160,0x01002031,"multiply.input_descriptor"},
    {0x164,0x00009041,"multiply.read_command"},
    {0x168,0x00000152,"multiply.left_stride"},
    {0x16c,0x00000120,"multiply.right_stride"},
    {0x170,0xb0279044,"multiply.layout_command"},
    {0x174,0x00001000,"multiply.row_storage"},
    {0x178,0x00000100,"multiply.left_width"},
    {0x17c,0x00000100,"multiply.left_height"},
    {0x180,0x00000100,"multiply.output_width"},
    {0x184,0x00000100,"multiply.output_height"},
    {0x188,0x00000020,"multiply.broadcast_span"},
    {0x18c,0x0000014a,"multiply.output_descriptor"},
    {0x190,0x00009000,"multiply.output_storage"},
    {0x194,0x00001140,"multiply.alu_mode"},
    {0x198,0x00080004,"multiply.alu_opcode"},
    {0x19c,0x81001444,"multiply.dma_store"},
    {0x1a0,0x00000100,"multiply.store_row_bytes"},
    {0x1a4,0x01302031,"multiply.store_descriptor"},
    {0x1a8,0x83119640,"multiply.store_command"},
    {0x1ac,0x00000012,"multiply.store_flags"},
    {0x1b0,0x00a000a0,"multiply.store_geometry"},
    {0x1b4,0x000100e0,"multiply.store_stamp"},
    {0x1bc,0x007f0000,"multiply.store_mask"},
    {0x1c0,0x20010701,"multiply.store_surface"},
    {0x1c4,0x23809344,"multiply.wait"},
    {0x1d0,0x24009442,"multiply.signal"},
    {0x1dc,0x22001340,"multiply.begin"},
    {0x1e0,0x00010021,"multiply.begin_stamp"},
    {0x1e4,0x22001440,"multiply.end"},
    {0x1e8,0x01010031,"multiply.end_stamp"},
};

static void setError(NSError **error, NSString *message) {
    if (error) *error = [NSError errorWithDomain:H16GMatrixRowDivisionErrorDomain
        code:1 userInfo:@{NSLocalizedDescriptionKey:message}];
}

@interface H16GMatrixRowDivisionEncoding ()
@property(nonatomic, readwrite, strong) H16GEncodedTDProgram *tdProgram;
@property(nonatomic, readwrite, copy) NSData *constantRegion;
@property(nonatomic, readwrite) NSUInteger taskCount;
@end

@implementation H16GMatrixRowDivisionEncoding
@end

@implementation H16GMatrixRowDivisionEncoder
+ (H16GMatrixRowDivisionEncoding *)encodeRows:(NSUInteger)rows
                                      columns:(NSUInteger)columns
                                        error:(NSError **)error {
    if (rows != 128 || columns != 128) {
        setError(error, @"matrix-row division requires measured 128x128 geometry");
        return nil;
    }
    H16GTDWriter *writer = [[H16GTDWriter alloc] initWithByteLength:0x1ec];
    for (const H16GNamedWord &word : kDivision128Fields) {
        NSString *field = [NSString stringWithUTF8String:word.field];
        if (![writer writeUInt32:word.value atOffset:word.offset
                           field:field error:error]) return nil;
    }
    H16GEncodedTDProgram *program = [[H16GEncodedTDProgram alloc]
        initWithData:writer.data kernelRelocationOffsets:@[@0xd0]
        programRecordCount:31 programFormatCode:0 scratchByteLength:0];
    NSData *constants = [H16GLUTEncoder
        constantRegionForOperationName:@"reciprocal" error:error];
    if (!constants) return nil;
    H16GMatrixRowDivisionEncoding *encoding =
        [[H16GMatrixRowDivisionEncoding alloc] init];
    encoding.tdProgram = program;
    encoding.constantRegion = constants;
    encoding.taskCount = 2;
    return encoding;
}
@end
