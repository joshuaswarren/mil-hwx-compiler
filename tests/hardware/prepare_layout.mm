#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEExecutableBundle.h"
#import "ANEStagedCompiler.h"

static NSString *layoutMIL(BOOL depthToSpace, NSUInteger channels,
                           NSUInteger spatial, NSUInteger block) {
    NSUInteger outputChannels = depthToSpace
        ? channels / (block * block) : channels * block * block;
    NSUInteger outputSpatial = depthToSpace
        ? spatial * block : spatial / block;
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, %lu, %lu, %lu]> x) {\n"
         "    int32 block = const()[name = string(\"block\"), val = int32(%lu)];\n"
         "    tensor<fp16, [1, %lu, %lu, %lu]> y = %@(x = x, block_size = block)[name = string(\"layout\")];\n"
         "  } -> (y);\n}\n",
        (unsigned long)channels, (unsigned long)spatial,
        (unsigned long)spatial, (unsigned long)block,
        (unsigned long)outputChannels, (unsigned long)outputSpatial,
        (unsigned long)outputSpatial,
        depthToSpace ? @"depth_to_space" : @"space_to_depth"];
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 6) {
            fprintf(stderr, "usage: %s OUTPUT_DIR s2d|d2s C S B\n", argv[0]);
            return 64;
        }
        BOOL depthToSpace = strcmp(argv[2], "d2s") == 0;
        BOOL spaceToDepth = strcmp(argv[2], "s2d") == 0;
        NSUInteger channels = (NSUInteger)strtoull(argv[3], NULL, 10);
        NSUInteger spatial = (NSUInteger)strtoull(argv[4], NULL, 10);
        NSUInteger block = (NSUInteger)strtoull(argv[5], NULL, 10);
        if ((!depthToSpace && !spaceToDepth) || channels == 0 ||
            spatial == 0 || block == 0 ||
            (depthToSpace && channels % (block * block) != 0) ||
            (!depthToSpace && spatial % block != 0)) return 65;
        NSData *source = [layoutMIL(depthToSpace, channels, spatial, block)
            dataUsingEncoding:NSUTF8StringEncoding];
        ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
        ANEExecutableBundle *bundle = [ANEStagedCompiler compileMILData:source
            modelRoot:[NSURL fileURLWithPath:@"tests/models/conv_relu"
                                 isDirectory:YES]
            target:@"H16G" diagnostics:diagnostics];
        for (ANEDiagnostic *diagnostic in diagnostics.diagnostics)
            fprintf(stderr, "%s: %s\n", diagnostic.code.UTF8String,
                    diagnostic.message.UTF8String);
        if (!bundle) return 2;
        NSError *error = nil;
        NSURL *output = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:argv[1]] isDirectory:YES];
        if (![bundle writeToDirectory:output error:&error]) {
            fprintf(stderr, "bundle write: %s\n", error.description.UTF8String);
            return 3;
        }
        printf("LAYOUT_BUNDLE op=%s C=%lu S=%lu B=%lu hwx_bytes=%lu\n",
               depthToSpace ? "d2s" : "s2d", (unsigned long)channels,
               (unsigned long)spatial, (unsigned long)block,
               (unsigned long)bundle.artifacts[0].image.length);
        return 0;
    }
}
