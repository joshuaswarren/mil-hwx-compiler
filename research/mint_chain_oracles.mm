// Mint Apple ANECompiler oracles for every MIL file in a directory.
//
// Research tool only. The compiled objects are Apple-generated and must stay
// outside the release tree; they are studied by research/analyze_chain_oracles.py
// to learn how one program carries a value between two tasks.
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

static BOOL compileMIL(NSString *name, NSString *milPathIn, NSString *root,
                       ANECCompileFn compile, ANECCreateInputFn createInput) {
    NSFileManager *fm = NSFileManager.defaultManager;
    NSString *work = [root stringByAppendingPathComponent:
        [name stringByAppendingString:@".work"]];
    NSString *destination = [root stringByAppendingPathComponent:
        [name stringByAppendingPathExtension:@"hwx"]];
    if ([fm fileExistsAtPath:destination]) {
        printf("MINT %s exists\n", name.UTF8String);
        return YES;
    }
    [fm removeItemAtPath:work error:nil];
    [fm createDirectoryAtPath:work withIntermediateDirectories:YES
                   attributes:nil error:nil];
    NSString *milPath = [work stringByAppendingPathComponent:@"model.mil"];
    if (![fm copyItemAtPath:milPathIn toPath:milPath error:nil]) return NO;
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
        BOOL copied = [fm copyItemAtPath:path toPath:destination error:nil];
        printf("MINT %s bytes=%lu copied=%d\n", name.UTF8String,
               (unsigned long)data.length, copied);
        return copied;
    }
    printf("MINT %s status=0 but no HWX produced\n", name.UTF8String);
    return NO;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) {
            fprintf(stderr, "usage: %s MIL_DIRECTORY OUTPUT_DIRECTORY\n", argv[0]);
            return 64;
        }
        NSString *milRoot = [NSString stringWithUTF8String:argv[1]];
        NSString *root = [NSString stringWithUTF8String:argv[2]];
        [NSFileManager.defaultManager createDirectoryAtPath:root
            withIntermediateDirectories:YES attributes:nil error:nil];
        void *library = dlopen("/System/Library/PrivateFrameworks/"
            "ANECompiler.framework/ANECompiler", RTLD_NOW | RTLD_LOCAL);
        ANECCompileFn compile = (ANECCompileFn)dlsym(library, "ANECCompile");
        ANECCreateInputFn createInput = (ANECCreateInputFn)dlsym(
            library, "ANECCreateCompilerInputDictionary");
        if (!compile || !createInput) return 2;
        NSUInteger failures = 0;
        NSArray<NSString *> *entries = [[NSFileManager.defaultManager
            contentsOfDirectoryAtPath:milRoot error:nil]
            sortedArrayUsingSelector:@selector(compare:)];
        for (NSString *entry in entries) {
            if (![entry.pathExtension isEqualToString:@"mil"]) continue;
            NSString *name = entry.stringByDeletingPathExtension;
            failures += !compileMIL(name,
                [milRoot stringByAppendingPathComponent:entry], root,
                compile, createInput);
        }
        printf("SUMMARY failures=%lu\n", (unsigned long)failures);
        return failures ? 1 : 0;
    }
}
