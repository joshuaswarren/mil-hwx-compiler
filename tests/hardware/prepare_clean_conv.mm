#import <Foundation/Foundation.h>

#import "ANEExecutableBundle.h"
#import "ANEHWXArtifact.h"
#import "H16GConstantPacker.h"
#import "H16GConvEncoder.h"
#import "HWXObjectWriter.h"

#include <stdio.h>

static NSData *readWeightPayload(NSString *path, NSError **error) {
    NSData *file = [NSData dataWithContentsOfFile:path options:0 error:error];
    const NSUInteger chunkOffset = 64;
    if (!file || file.length < chunkOffset + 24) return nil;
    const uint8_t *bytes = (const uint8_t *)file.bytes;
    uint32_t magic = 0;
    uint64_t payloadLength = 0;
    uint64_t payloadOffset = 0;
    memcpy(&magic, bytes + chunkOffset, sizeof(magic));
    memcpy(&payloadLength, bytes + chunkOffset + 8, sizeof(payloadLength));
    memcpy(&payloadOffset, bytes + chunkOffset + 16, sizeof(payloadOffset));
    const NSUInteger expected = 64 * 64 * sizeof(uint16_t);
    BOOL fits = payloadOffset <= NSUIntegerMax &&
        (NSUInteger)payloadOffset <= file.length &&
        expected <= file.length - (NSUInteger)payloadOffset;
    if (magic != 0xDEADBEEF || payloadLength < expected || !fits) {
        if (error) {
            *error = [NSError errorWithDomain:@"ANE.CleanConvProbe" code:1
                userInfo:@{NSLocalizedDescriptionKey:
                    @"weight file does not contain the expected BLOBFILE chunk"}];
        }
        return nil;
    }
    return [file subdataWithRange:NSMakeRange((NSUInteger)payloadOffset,
                                               expected)];
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) {
            fprintf(stderr, "usage: %s WEIGHT_BLOB OUTPUT_DIR\n", argv[0]);
            return 64;
        }
        NSError *error = nil;
        NSData *weights = readWeightPayload(
            [NSString stringWithUTF8String:argv[1]], &error);
        NSData *packed = weights ? [H16GConstantPacker
            packConv1x1Weights:weights inputChannels:64 outputChannels:64
            bytesPerWeight:sizeof(uint16_t) error:&error] : nil;
        NSData *td = packed ? [H16GConvEncoder
            encodeConv1x1WithInputChannels:64 outputChannels:64 spatial:64
            bytesPerWeight:sizeof(uint16_t)
            numericMode:ANELegalNumericModeFP16 reluEpilogue:YES
            error:&error] : nil;
        NSArray<HWXObjectBinding *> *objectBindings = @[
            [[HWXObjectBinding alloc] initWithSymbol:@"x" shortName:@"x"
                role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16
                shape:@[@1,@64,@64,@64]],
            [[HWXObjectBinding alloc] initWithSymbol:@"y@output" shortName:@"y"
                role:HWXObjectBindingRoleOutput elementType:ANEElementTypeFP16
                shape:@[@1,@64,@64,@64]],
        ];
        NSData *image = td ? [HWXObjectWriter
            buildObjectWithTaskDescriptor:td constantRegion:packed
            bindings:objectBindings error:&error] : nil;
        if (!image) {
            fprintf(stderr, "clean Conv construction failed: %s\n",
                    error.description.UTF8String);
            return 2;
        }

        const NSUInteger tensorBytes = 64 * 64 * 64 * sizeof(uint16_t);
        NSArray<ANEHWXBinding *> *runtimeBindings = @[
            [[ANEHWXBinding alloc] initWithIdentifier:@"input"
                role:ANESurfaceRoleInput logicalByteLength:tensorBytes
                allocationByteLength:tensorBytes ioSurfaceIndex:0
                rowStrideBytes:64 * sizeof(uint16_t)
                planeStrideBytes:64 * 64 * sizeof(uint16_t)
                batchStrideBytes:tensorBytes],
            [[ANEHWXBinding alloc] initWithIdentifier:@"weight"
                role:ANESurfaceRoleWeight logicalByteLength:packed.length
                allocationByteLength:packed.length ioSurfaceIndex:-1],
            [[ANEHWXBinding alloc] initWithIdentifier:@"output"
                role:ANESurfaceRoleOutput logicalByteLength:tensorBytes
                allocationByteLength:tensorBytes ioSurfaceIndex:1
                rowStrideBytes:64 * sizeof(uint16_t)
                planeStrideBytes:64 * 64 * sizeof(uint16_t)
                batchStrideBytes:tensorBytes],
        ];
        ANEHWXArtifact *artifact = [[ANEHWXArtifact alloc]
            initWithImage:image bindings:runtimeBindings];
        ANEExecutableBundle *bundle = [[ANEExecutableBundle alloc]
            initWithTarget:@"H16G" artifacts:@[artifact]
            dispatchPlan:@[@0]
            passTrace:@[@"probe.encode-conv", @"probe.write-clean-hwx"]];
        NSURL *output = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:argv[2]] isDirectory:YES];
        if (![bundle writeToDirectory:output error:&error]) {
            fprintf(stderr, "bundle write failed: %s\n",
                    error.description.UTF8String);
            return 3;
        }
        printf("clean Conv HWX bytes=%lu td=%lu constants=%lu\n",
               (unsigned long)image.length, (unsigned long)td.length,
               (unsigned long)packed.length);
        return 0;
    }
}
