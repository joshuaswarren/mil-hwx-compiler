#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEExecutableBundle.h"
#import "ANEStagedCompiler.h"

static NSString *layoutConvMIL(NSUInteger channels) {
    NSUInteger packed = channels * 16;
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, %lu, 128, 128]> x) {\n"
         "    int32 b = const()[name = string(\"b\"), val = int32(4)];\n"
         "    string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
         "    tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
         "    tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
         "    tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
         "    int32 gp = const()[name = string(\"gp\"), val = int32(1)];\n"
         "    tensor<fp16, [1, %lu, 32, 32]> packed = space_to_depth(x = x, block_size = b)[name = string(\"packed\")];\n"
         "    tensor<fp16, [%lu, %lu, 1, 1]> w = const()[name = string(\"w\"), val = tensor<fp16, [%lu, %lu, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
         "    tensor<fp16, [1, %lu, 32, 32]> c = conv(dilations = dl, groups = gp, pad = pd, pad_type = pt, strides = st, weight = w, x = packed)[name = string(\"c\")];\n"
         "    tensor<fp16, [1, %lu, 128, 128]> y = depth_to_space(x = c, block_size = b)[name = string(\"y\")];\n"
         "  } -> (y);\n}\n",
        (unsigned long)channels, (unsigned long)packed,
        (unsigned long)packed, (unsigned long)packed,
        (unsigned long)packed, (unsigned long)packed,
        (unsigned long)packed, (unsigned long)channels];
}

static BOOL writeIdentityBlob(NSString *path, NSUInteger channels,
                              NSError **error) {
    NSUInteger bytes = channels * channels * sizeof(_Float16);
    NSMutableData *blob = [NSMutableData dataWithLength:128 + bytes];
    uint8_t *data = (uint8_t *)blob.mutableBytes;
    const uint32_t fileVersion = 1, chunkCount = 2, magic = 0xDEADBEEF;
    const uint64_t payloadLength = bytes, payloadOffset = 128;
    memcpy(data, &fileVersion, 4); memcpy(data + 4, &chunkCount, 4);
    memcpy(data + 64, &magic, 4);
    memcpy(data + 72, &payloadLength, 8);
    memcpy(data + 80, &payloadOffset, 8);
    _Float16 *weights = (_Float16 *)(data + payloadOffset);
    for (NSUInteger i = 0; i < channels; ++i)
        weights[i * channels + i] = (_Float16)1.0f;
    return [blob writeToFile:path options:NSDataWritingAtomic error:error];
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) {
            fprintf(stderr, "usage: %s OUTPUT_DIR NATURAL_CHANNELS\n", argv[0]);
            return 64;
        }
        NSUInteger channels = (NSUInteger)strtoull(argv[2], NULL, 10);
        if (channels < 8 || channels > 32 || channels % 8 != 0) return 65;
        NSString *modelRoot = [NSString stringWithFormat:
            @"build/layout-conv-model-c%lu", (unsigned long)channels];
        NSString *weightDirectory =
            [modelRoot stringByAppendingPathComponent:@"weights"];
        NSError *error = nil;
        if (![NSFileManager.defaultManager createDirectoryAtPath:weightDirectory
            withIntermediateDirectories:YES attributes:nil error:&error] ||
            !writeIdentityBlob([weightDirectory
                stringByAppendingPathComponent:@"weight.bin"], channels * 16,
                &error)) {
            fprintf(stderr, "model: %s\n", error.description.UTF8String);
            return 2;
        }
        ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
        NSData *source = [layoutConvMIL(channels)
            dataUsingEncoding:NSUTF8StringEncoding];
        ANEExecutableBundle *bundle = [ANEStagedCompiler compileMILData:source
            modelRoot:[NSURL fileURLWithPath:modelRoot isDirectory:YES]
            target:@"H16G" diagnostics:diagnostics];
        for (ANEDiagnostic *diagnostic in diagnostics.diagnostics)
            fprintf(stderr, "%s: %s\n", diagnostic.code.UTF8String,
                    diagnostic.message.UTF8String);
        if (!bundle) return 3;
        NSURL *output = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:argv[1]] isDirectory:YES];
        if (![bundle writeToDirectory:output error:&error]) {
            fprintf(stderr, "bundle: %s\n", error.description.UTF8String);
            return 4;
        }
        printf("LAYOUT_CONV_BUNDLE C=%lu hwx_bytes=%lu\n",
               (unsigned long)channels,
               (unsigned long)bundle.artifacts[0].image.length);
        return 0;
    }
}
