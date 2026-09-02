#import <Foundation/Foundation.h>
#import <dlfcn.h>

typedef void (^ANECCompileBlock)(int, CFDictionaryRef);
typedef void (*ANECCompileFn)(CFDictionaryRef, CFDictionaryRef, ANECCompileBlock);
typedef CFDictionaryRef (*ANECCreateInputFn)(const char *);

static BOOL isHWX(NSData *data) {
    return data.length >= 4 && *(const uint32_t *)data.bytes == 0xBEEFFACEu;
}

static NSArray<NSString *> *filesBelow(NSString *root) {
    NSMutableArray<NSString *> *files = [NSMutableArray array];
    NSFileManager *fm = NSFileManager.defaultManager;
    for (NSString *relative in [fm enumeratorAtPath:root]) {
        NSString *path = [root stringByAppendingPathComponent:relative];
        BOOL directory = NO;
        if ([fm fileExistsAtPath:path isDirectory:&directory] && !directory)
            [files addObject:path];
    }
    return files;
}

static NSString *unaryMIL(NSString *expression, NSUInteger size) {
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, 1, %lu, %lu]> x) {\n"
         "    tensor<fp16, [1, 1, %lu, %lu]> y = %@;\n"
         "  } -> (y);\n}\n",
        (unsigned long)size, (unsigned long)size,
        (unsigned long)size, (unsigned long)size, expression];
}

static NSString *reduceMIL(NSString *operation, NSUInteger channels,
                           NSUInteger height, NSUInteger width, NSUInteger axis) {
    NSUInteger outC = axis == 1 ? 1 : channels;
    NSUInteger outH = axis == 2 ? 1 : height;
    NSUInteger outW = axis == 3 ? 1 : width;
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, %lu, %lu, %lu]> x) {\n"
         "    tensor<int32, [1]> ax = const()[name = string(\"ax\"), "
         "val = tensor<int32, [1]>([%lu])];\n"
         "    tensor<fp16, [1, %lu, %lu, %lu]> y = %@"
         "(x = x, axes = ax, keep_dims = bool(true));\n"
         "  } -> (y);\n}\n",
        (unsigned long)channels, (unsigned long)height, (unsigned long)width,
        (unsigned long)axis, (unsigned long)outC, (unsigned long)outH,
        (unsigned long)outW, operation];
}

static BOOL compileMIL(NSString *name, NSString *mil, NSString *root,
                       ANECCompileFn compile, ANECCreateInputFn createInput) {
    NSFileManager *fm = NSFileManager.defaultManager;
    NSString *work = [root stringByAppendingPathComponent:
        [name stringByAppendingString:@".work"]];
    if ([fm fileExistsAtPath:work]) {
        fprintf(stderr, "%s already exists; refusing to overwrite\n", work.UTF8String);
        return NO;
    }
    [fm createDirectoryAtPath:work withIntermediateDirectories:YES
                   attributes:nil error:nil];
    NSString *milPath = [work stringByAppendingPathComponent:@"model.mil"];
    [[mil dataUsingEncoding:NSUTF8StringEncoding] writeToFile:milPath atomically:YES];
    CFDictionaryRef raw = createInput(milPath.fileSystemRepresentation);
    if (!raw) return NO;
    CFMutableDictionaryRef input = CFDictionaryCreateMutableCopy(NULL, 0, raw);
    CFDictionarySetValue(input, CFSTR("OutputFileName"),
                         (__bridge CFStringRef)name);
    NSString *prefix = [work stringByAppendingString:@"/"];
    CFDictionarySetValue(input, CFSTR("OutputFilePath"),
                         (__bridge CFStringRef)prefix);
    const void *keys[] = {CFSTR("TargetArchitecture")};
    const void *values[] = {CFSTR("H16G")};
    CFDictionaryRef options = CFDictionaryCreate(NULL, keys, values, 1,
        &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    __block int status = INT_MIN;
    __block NSString *message = nil;
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    compile(input, options, ^(int value, CFDictionaryRef result) {
        status = value;
        NSArray *errors = ((__bridge NSDictionary *)result)[@"ErrorList"];
        if (errors.count) message = errors.description;
        dispatch_semaphore_signal(done);
    });
    dispatch_semaphore_wait(done,
        dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC));
    CFRelease(options);
    CFRelease(input);
    if (status != 0) {
        printf("MINT %s status=%d error=%s\n", name.UTF8String, status,
               message ? message.UTF8String : "(none)");
        return NO;
    }
    for (NSString *path in filesBelow(work)) {
        NSData *data = [NSData dataWithContentsOfFile:path];
        if (!isHWX(data)) continue;
        NSString *destination = [root stringByAppendingPathComponent:
            [name stringByAppendingPathExtension:@"hwx"]];
        if ([fm fileExistsAtPath:destination]) {
            fprintf(stderr, "%s already exists; refusing to overwrite\n",
                    destination.UTF8String);
            return NO;
        }
        BOOL copied = [fm copyItemAtPath:path toPath:destination error:nil];
        printf("MINT %s bytes=%lu copied=%d\n", name.UTF8String,
               (unsigned long)data.length, copied);
        return copied;
    }
    return NO;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) {
            fprintf(stderr, "usage: %s OUTPUT_DIRECTORY\n", argv[0]);
            return 64;
        }
        NSString *root = [NSString stringWithUTF8String:argv[1]];
        [NSFileManager.defaultManager createDirectoryAtPath:root
            withIntermediateDirectories:YES attributes:nil error:nil];
        void *library = dlopen("/System/Library/PrivateFrameworks/"
            "ANECompiler.framework/ANECompiler", RTLD_NOW);
        ANECCompileFn compile = (ANECCompileFn)dlsym(library, "ANECCompile");
        ANECCreateInputFn createInput = (ANECCreateInputFn)dlsym(
            library, "ANECCreateCompilerInputDictionary");
        if (!compile || !createInput) return 2;
        NSUInteger failures = 0;
        NSDictionary<NSString *,NSString *> *unaryExpressions = @{
            @"sigmoid": @"sigmoid(x = x)",
            @"exp": @"exp(x = x)",
            @"relu": @"relu(x = x)",
            @"rsqrt": @"rsqrt(x = x, epsilon = fp32(0.000001))",
            @"reciprocal": @"inverse(x = x, epsilon = fp32(0.000001))",
        };
        for (NSString *operation in unaryExpressions) {
            for (NSNumber *size in @[@128, @256, @512, @1024, @2048]) {
                NSString *name = [NSString stringWithFormat:@"%@_n%@",
                    operation, size];
                failures += !compileMIL(name,
                    unaryMIL(unaryExpressions[operation],
                        size.unsignedIntegerValue), root, compile, createInput);
            }
        }
        NSArray<NSArray<NSNumber *> *> *geometries = @[
            @[@32, @8, @8, @1], @[@64, @8, @8, @1], @[@128, @8, @8, @1],
            @[@64, @16, @16, @1], @[@64, @32, @32, @1],
            @[@1, @64, @64, @3], @[@1, @128, @128, @3],
            @[@32, @64, @16, @2],
        ];
        for (NSString *operation in @[@"reduce_sum", @"reduce_mean", @"reduce_max"]) {
            for (NSArray<NSNumber *> *g in geometries) {
                NSString *name = [NSString stringWithFormat:@"%@_c%@_h%@_w%@_a%@",
                    operation, g[0], g[1], g[2], g[3]];
                failures += !compileMIL(name, reduceMIL(operation,
                    g[0].unsignedIntegerValue, g[1].unsignedIntegerValue,
                    g[2].unsignedIntegerValue, g[3].unsignedIntegerValue),
                    root, compile, createInput);
            }
        }
        printf("SUMMARY failures=%lu\n", (unsigned long)failures);
        return failures ? 1 : 0;
    }
}
