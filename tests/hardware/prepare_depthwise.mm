#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEExecutableBundle.h"
#import "ANEStagedCompiler.h"

#include <stdio.h>

static NSString *milSource(NSUInteger channels) {
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, %lu, 64, 64]> x) {\n"
         "    string pt = const()[name = string(\"pt\"), val = string(\"same\")];\n"
         "    tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
         "    tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
         "    tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
         "    int32 gp = const()[name = string(\"gp\"), val = int32(%lu)];\n"
         "    tensor<fp16, [%lu, 1, 3, 3]> w = const()[name = string(\"w\"), val = tensor<fp16, [%lu, 1, 3, 3]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
         "    tensor<fp16, [1, %lu, 64, 64]> y = conv(dilations = dl, groups = gp, pad = pd, pad_type = pt, strides = st, weight = w, x = x)[name = string(\"y\")];\n"
         "  } -> (y);\n}\n",
        (unsigned long)channels,(unsigned long)channels,
        (unsigned long)channels,(unsigned long)channels,
        (unsigned long)channels];
}

static BOOL writeIdentityModel(NSString *root, NSUInteger channels,
                               NSError **error) {
    NSString *weightsDirectory = [root stringByAppendingPathComponent:@"weights"];
    if (![NSFileManager.defaultManager createDirectoryAtPath:weightsDirectory
        withIntermediateDirectories:YES attributes:nil error:error]) return NO;
    NSUInteger weightBytes = channels * 9 * sizeof(_Float16);
    NSMutableData *blob = [NSMutableData dataWithLength:128 + weightBytes];
    uint8_t *bytes = (uint8_t *)blob.mutableBytes;
    uint32_t version = 1, chunks = 2, magic = 0xDEADBEEF;
    uint64_t payloadLength = weightBytes, payloadOffset = 128;
    memcpy(bytes,&version,4); memcpy(bytes+4,&chunks,4);
    memcpy(bytes+64,&magic,4); memcpy(bytes+72,&payloadLength,8);
    memcpy(bytes+80,&payloadOffset,8);
    _Float16 *weights = (_Float16 *)(bytes + payloadOffset);
    for (NSUInteger channel = 0; channel < channels; ++channel)
        weights[channel * 9 + 4] = (_Float16)1.0f;
    if (![blob writeToFile:[weightsDirectory
        stringByAppendingPathComponent:@"weight.bin"]
        options:NSDataWritingAtomic error:error]) return NO;
    return [milSource(channels) writeToFile:[root
        stringByAppendingPathComponent:@"model.mil"] atomically:YES
        encoding:NSUTF8StringEncoding error:error];
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) {
            fprintf(stderr,"usage: %s CHANNELS OUTPUT_ROOT\n",argv[0]);
            return 64;
        }
        NSUInteger channels = (NSUInteger)strtoull(argv[1],NULL,10);
        if (![@[@64,@128,@256,@512] containsObject:@(channels)]) return 65;
        NSString *outputRoot = [NSString stringWithUTF8String:argv[2]];
        NSString *modelRoot = [outputRoot stringByAppendingPathComponent:@"model"];
        NSError *error = nil;
        if (!writeIdentityModel(modelRoot,channels,&error)) {
            fprintf(stderr,"model write: %s\n",error.description.UTF8String);
            return 2;
        }
        ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
        ANEExecutableBundle *bundle = [ANEStagedCompiler
            compileMILData:[milSource(channels) dataUsingEncoding:NSUTF8StringEncoding]
            modelRoot:[NSURL fileURLWithPath:modelRoot isDirectory:YES]
            target:@"H16G" diagnostics:diagnostics];
        for (ANEDiagnostic *diagnostic in diagnostics.diagnostics)
            fprintf(stderr,"%s: %s\n",diagnostic.code.UTF8String,
                diagnostic.message.UTF8String);
        if (!bundle) return 3;
        NSURL *bundleDirectory = [NSURL fileURLWithPath:
            [outputRoot stringByAppendingPathComponent:@"bundle"] isDirectory:YES];
        if (![bundle writeToDirectory:bundleDirectory error:&error]) {
            fprintf(stderr,"bundle write: %s\n",error.description.UTF8String);
            return 4;
        }
        printf("PREPARE depthwise channels=%lu hwx_bytes=%lu passes=%s\n",
            (unsigned long)channels,(unsigned long)bundle.artifacts[0].image.length,
            [bundle.passTrace componentsJoinedByString:@","].UTF8String);
        return 0;
    }
}
