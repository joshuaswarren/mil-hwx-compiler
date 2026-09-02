#import <Foundation/Foundation.h>

#import "ANEExecutableBundle.h"
#import "ANEHWXArtifact.h"
#import "H16GBroadcastALUEncoder.h"
#import "HWXObjectWriter.h"

static BOOL isRow(NSString *name) {
    return [name hasPrefix:@"row_"];
}

static HWXObjectBinding *objectBinding(NSString *name, BOOL row,
                                       HWXObjectBindingRole role) {
    NSArray<NSNumber *> *shape = row ? @[@1,@1,@128,@1]
                                      : @[@1,@1,@128,@128];
    NSUInteger rowBytes = row ? 64 : 256;
    NSUInteger storage = row ? 8192 : 32768;
    NSString *symbol = role == HWXObjectBindingRoleOutput
        ? [name stringByAppendingString:@"@output"] : name;
    return [[HWXObjectBinding alloc] initWithSymbol:symbol shortName:name
        role:role elementType:ANEElementTypeFP16 shape:shape
        rowStrideBytes:rowBytes planeStrideBytes:storage
        batchStrideBytes:storage storageByteLength:storage];
}

static ANEHWXBinding *runtimeBinding(NSString *name, BOOL row,
                                     ANESurfaceRole role, NSInteger index) {
    NSUInteger logical = row ? 256 : 32768;
    NSUInteger storage = row ? 8192 : 32768;
    NSUInteger rowBytes = row ? 64 : 256;
    return [[ANEHWXBinding alloc] initWithIdentifier:name role:role
        logicalByteLength:logical allocationByteLength:storage
        ioSurfaceIndex:index rowStrideBytes:rowBytes
        planeStrideBytes:storage batchStrideBytes:storage];
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) {
            fprintf(stderr,"usage: %s CASE OUTPUT_DIRECTORY\n",argv[0]);
            return 64;
        }
        NSString *name = [NSString stringWithUTF8String:argv[1]];
        NSString *output = [NSString stringWithUTF8String:argv[2]];
        NSError *error = nil;
        H16GEncodedTDProgram *program = nil;
        BOOL rowOutput = isRow(name);
        BOOL matrixRow = [name hasPrefix:@"matrix_"];
        if ([name isEqualToString:@"scale"])
            program = [H16GBroadcastALUEncoder
                encodeScalarScaleForMatrixRows:128 columns:128
                measuredScaleKind:H16GMeasuredScaleInverseSqrt128
                error:&error];
        else if (matrixRow)
            program = [H16GBroadcastALUEncoder encodeMatrixRowOperation:
                [name substringFromIndex:7] rows:128 columns:128 error:&error];
        else if (rowOutput)
            program = [H16GBroadcastALUEncoder encodeRowOperation:
                [name substringFromIndex:4] rows:128 error:&error];
        if (!program) {
            fprintf(stderr,"encode: %s\n",error.description.UTF8String);
            return 2;
        }
        BOOL twoInputs = ![name isEqualToString:@"scale"];
        BOOL leftRow = rowOutput;
        BOOL rightRow = twoInputs;
        NSMutableArray<HWXObjectBinding *> *objectBindings =
            [NSMutableArray array];
        if (twoInputs) [objectBindings addObject:objectBinding(
            @"right",rightRow,HWXObjectBindingRoleInput)];
        [objectBindings addObject:objectBinding(
            @"left",leftRow,HWXObjectBindingRoleInput)];
        [objectBindings addObject:objectBinding(
            @"output",rowOutput,HWXObjectBindingRoleOutput)];
        HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
            initWithTaskCount:1 recordCount:program.programRecordCount
            formatCode:program.programFormatCode scratchByteLength:0
            descriptorLayout:HWXProgramDescriptorLayoutLinear];
        NSData *image = [HWXObjectWriter buildObjectWithTaskDescriptor:
            program.data constantRegion:[NSData data] bindings:objectBindings
            kernelRelocationOffsets:@[] programInfo:info error:&error];
        if (!image) return 3;
        NSMutableArray<ANEHWXBinding *> *runtimeBindings =
            [NSMutableArray array];
        if (twoInputs) [runtimeBindings addObject:runtimeBinding(
            @"right",rightRow,ANESurfaceRoleInput,0)];
        [runtimeBindings addObject:runtimeBinding(
            @"left",leftRow,ANESurfaceRoleInput,twoInputs?1:0)];
        [runtimeBindings addObject:runtimeBinding(@"output",rowOutput,
            ANESurfaceRoleOutput,twoInputs?2:1)];
        ANEHWXArtifact *artifact = [[ANEHWXArtifact alloc]
            initWithImage:image bindings:runtimeBindings];
        ANEExecutableBundle *bundle = [[ANEExecutableBundle alloc]
            initWithTarget:@"H16G" artifacts:@[artifact]
            dispatchPlan:@[@0] passTrace:@[@"h16g.broadcast-alu"]];
        NSURL *directory = [NSURL fileURLWithPath:output isDirectory:YES];
        if (![bundle writeToDirectory:directory error:&error]) return 4;
        printf("PREPARE broadcast case=%s bytes=%lu\n",name.UTF8String,
               (unsigned long)image.length);
        return 0;
    }
}
