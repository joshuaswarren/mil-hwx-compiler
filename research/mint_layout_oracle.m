#import <Foundation/Foundation.h>
#import <dlfcn.h>

typedef void (^ANECCompileBlock)(int status, CFDictionaryRef result);
typedef void (*ANECCompileFn)(CFDictionaryRef, CFDictionaryRef,
                              ANECCompileBlock);
typedef CFDictionaryRef (*ANECCreateInputFn)(const char *);

static NSString *layoutMIL(BOOL depthToSpace, NSUInteger channels,
                           NSUInteger spatial, NSUInteger block) {
    NSUInteger outputChannels = depthToSpace
        ? channels / (block * block) : channels * block * block;
    NSUInteger outputSpatial = depthToSpace
        ? spatial * block : spatial / block;
    NSString *operation = depthToSpace
        ? @"depth_to_space" : @"space_to_depth";
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, %lu, %lu, %lu]> x) {\n"
         "    int32 b = const()[name = string(\"b\"), val = int32(%lu)];\n"
         "    tensor<fp16, [1, %lu, %lu, %lu]> y = %@(x = x, block_size = b)[name = string(\"y\")];\n"
         "  } -> (y);\n}\n",
        (unsigned long)channels, (unsigned long)spatial,
        (unsigned long)spatial, (unsigned long)block,
        (unsigned long)outputChannels, (unsigned long)outputSpatial,
        (unsigned long)outputSpatial, operation];
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 7) {
            fprintf(stderr,
                "usage: %s OUTPUT_DIR NAME s2d|d2s C S B\n", argv[0]);
            return 64;
        }
        NSString *outputDirectory = [NSString stringWithUTF8String:argv[1]];
        NSString *name = [NSString stringWithUTF8String:argv[2]];
        BOOL depthToSpace = strcmp(argv[3], "d2s") == 0;
        BOOL spaceToDepth = strcmp(argv[3], "s2d") == 0;
        NSUInteger channels = (NSUInteger)strtoull(argv[4], NULL, 10);
        NSUInteger spatial = (NSUInteger)strtoull(argv[5], NULL, 10);
        NSUInteger block = (NSUInteger)strtoull(argv[6], NULL, 10);
        if ((!depthToSpace && !spaceToDepth) || channels == 0 ||
            spatial == 0 || block == 0 ||
            (depthToSpace && channels % (block * block) != 0) ||
            (!depthToSpace && spatial % block != 0)) return 65;

        NSFileManager *files = NSFileManager.defaultManager;
        NSError *error = nil;
        if (![files createDirectoryAtPath:outputDirectory
              withIntermediateDirectories:YES attributes:nil error:&error]) {
            fprintf(stderr, "mkdir: %s\n", error.description.UTF8String);
            return 2;
        }
        NSString *milPath =
            [outputDirectory stringByAppendingPathComponent:@"model.mil"];
        NSString *source = layoutMIL(
            depthToSpace, channels, spatial, block);
        if (![source writeToFile:milPath atomically:YES
                        encoding:NSUTF8StringEncoding error:&error]) {
            fprintf(stderr, "MIL write: %s\n", error.description.UTF8String);
            return 3;
        }
        void *library = dlopen(
            "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler",
            RTLD_NOW | RTLD_LOCAL);
        ANECCompileFn compile =
            (ANECCompileFn)dlsym(library, "ANECCompile");
        ANECCreateInputFn createInput = (ANECCreateInputFn)dlsym(
            library, "ANECCreateCompilerInputDictionary");
        if (!compile || !createInput) return 4;
        CFDictionaryRef raw = createInput(milPath.fileSystemRepresentation);
        if (!raw) return 5;
        CFMutableDictionaryRef input =
            CFDictionaryCreateMutableCopy(kCFAllocatorDefault, 0, raw);
        CFDictionarySetValue(input, CFSTR("OutputFileName"),
                             (__bridge CFStringRef)name);
        CFDictionarySetValue(input, CFSTR("OutputFilePath"),
            (__bridge CFStringRef)[outputDirectory stringByAppendingString:@"/"]);
        const void *keys[] = {CFSTR("TargetArchitecture")};
        const void *values[] = {CFSTR("H16G")};
        CFDictionaryRef options = CFDictionaryCreate(kCFAllocatorDefault,
            keys, values, 1, &kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks);
        __block int status = INT_MIN;
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        compile(input, options, ^(int result, CFDictionaryRef details) {
            (void)details;
            status = result;
            dispatch_semaphore_signal(done);
        });
        long wait = dispatch_semaphore_wait(done,
            dispatch_time(DISPATCH_TIME_NOW, 60 * NSEC_PER_SEC));
        CFRelease(options);
        CFRelease(input);
        printf("ANEC name=%s status=%d wait=%ld\n",
               name.UTF8String, status, wait);
        return status == 0 && wait == 0 ? 0 : 6;
    }
}
