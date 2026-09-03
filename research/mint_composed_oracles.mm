#import <Foundation/Foundation.h>
#import <dlfcn.h>

typedef void (^ANECCompileBlock)(int, CFDictionaryRef);
typedef void (*ANECCompileFn)(CFDictionaryRef, CFDictionaryRef, ANECCompileBlock);
typedef CFDictionaryRef (*ANECCreateInputFn)(const char *);

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

static BOOL isHWX(NSData *data) {
    return data.length >= 4 && *(const uint32_t *)data.bytes == 0xBEEFFACEu;
}

static NSString *binaryMIL(NSString *operation, NSUInteger rows,
                           NSUInteger columns, NSUInteger rightColumns) {
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, 1, %lu, %lu]> x, tensor<fp16, [1, 1, %lu, %lu]> rhs) {\n"
         "    tensor<fp16, [1, 1, %lu, %lu]> y = %@(x = x, y = rhs)[name = string(\"y\")];\n"
         "  } -> (y);\n}\n",
        (unsigned long)rows, (unsigned long)columns,
        (unsigned long)rows, (unsigned long)rightColumns,
        (unsigned long)rows, (unsigned long)columns, operation];
}

static NSString *scalarMIL(NSUInteger size) {
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, 1, %lu, %lu]> x) {\n"
         "    fp16 scale = const()[name = string(\"scale\"), val = fp16(0.08838834764831845)];\n"
         "    tensor<fp16, [1, 1, %lu, %lu]> y = mul(x = x, y = scale)[name = string(\"y\")];\n"
         "  } -> (y);\n}\n",
        (unsigned long)size, (unsigned long)size,
        (unsigned long)size, (unsigned long)size];
}

static BOOL compileMIL(NSString *name, NSString *mil, NSString *root,
                       ANECCompileFn compile, ANECCreateInputFn createInput) {
    NSFileManager *fm = NSFileManager.defaultManager;
    NSString *work = [root stringByAppendingPathComponent:
        [name stringByAppendingString:@".work"]];
    if ([fm fileExistsAtPath:work]) return NO;
    [fm createDirectoryAtPath:work withIntermediateDirectories:YES
                   attributes:nil error:nil];
    NSString *milPath = [work stringByAppendingPathComponent:@"model.mil"];
    [[mil dataUsingEncoding:NSUTF8StringEncoding] writeToFile:milPath atomically:YES];
    CFDictionaryRef raw = createInput(milPath.fileSystemRepresentation);
    if (!raw) return NO;
    CFMutableDictionaryRef input = CFDictionaryCreateMutableCopy(NULL, 0, raw);
    CFRelease(raw);
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
    long wait = dispatch_semaphore_wait(done,
        dispatch_time(DISPATCH_TIME_NOW, 60 * NSEC_PER_SEC));
    CFRelease(options);
    CFRelease(input);
    if (wait != 0 || status != 0) {
        printf("MINT %s status=%d error=%s\n", name.UTF8String, status,
               message ? message.UTF8String : "(none)");
        return NO;
    }
    for (NSString *path in filesBelow(work)) {
        NSData *data = [NSData dataWithContentsOfFile:path];
        if (!isHWX(data)) continue;
        NSString *destination = [root stringByAppendingPathComponent:
            [name stringByAppendingPathExtension:@"hwx"]];
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
            "ANECompiler.framework/ANECompiler", RTLD_NOW | RTLD_LOCAL);
        ANECCompileFn compile = (ANECCompileFn)dlsym(library, "ANECCompile");
        ANECCreateInputFn createInput = (ANECCreateInputFn)dlsym(
            library, "ANECCreateCompilerInputDictionary");
        if (!compile || !createInput) return 2;
        NSUInteger failures = 0;
        for (NSNumber *sizeNumber in @[@128, @256]) {
            NSUInteger size = sizeNumber.unsignedIntegerValue;
            failures += !compileMIL([NSString stringWithFormat:@"mul_scalar_n%@",
                sizeNumber], scalarMIL(size), root, compile, createInput);
            for (NSString *operation in @[@"sub", @"mul", @"real_div"]) {
                NSString *name = [NSString stringWithFormat:@"%@_matrix_row_n%@",
                    operation, sizeNumber];
                failures += !compileMIL(name,
                    binaryMIL(operation, size, size, 1), root, compile,
                    createInput);
            }
            for (NSString *operation in @[@"add", @"mul", @"maximum"]) {
                NSString *name = [NSString stringWithFormat:@"%@_row_row_n%@",
                    operation, sizeNumber];
                failures += !compileMIL(name,
                    binaryMIL(operation, size, 1, 1), root, compile,
                    createInput);
            }
        }
        printf("SUMMARY failures=%lu\n", (unsigned long)failures);
        return failures ? 1 : 0;
    }
}
