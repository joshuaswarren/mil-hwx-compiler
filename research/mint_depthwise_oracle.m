#import <Foundation/Foundation.h>
#import <dlfcn.h>

typedef void (^ANECCompileBlock)(int status, CFDictionaryRef result);
typedef void (*ANECCompileFn)(CFDictionaryRef input, CFDictionaryRef options,
                              ANECCompileBlock completion);
typedef CFDictionaryRef (*ANECCreateInputFn)(const char *milPath);

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

static NSData *weightBlob(NSUInteger channels, NSString *pattern) {
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
        for (NSUInteger tap = 0; tap < 9; ++tap)
            weights[channel * 9 + tap] = (_Float16)(
                [pattern isEqualToString:@"channel"] ? channel + 1 : tap + 1);
    return blob;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 4) {
            fprintf(stderr,"usage: %s OUTPUT_DIR CHANNELS channel|tap\n",argv[0]);
            return 64;
        }
        NSString *outputDirectory = [NSString stringWithUTF8String:argv[1]];
        NSUInteger channels = (NSUInteger)strtoull(argv[2],NULL,10);
        NSString *pattern = [NSString stringWithUTF8String:argv[3]];
        if (![@[@"channel",@"tap"] containsObject:pattern] ||
            ![@[@64,@128,@256,@512] containsObject:@(channels)]) return 65;
        NSString *weightsDirectory =
            [outputDirectory stringByAppendingPathComponent:@"weights"];
        NSError *error = nil;
        [NSFileManager.defaultManager createDirectoryAtPath:weightsDirectory
            withIntermediateDirectories:YES attributes:nil error:&error];
        NSString *milPath = [outputDirectory stringByAppendingPathComponent:@"model.mil"];
        [milSource(channels) writeToFile:milPath atomically:YES
            encoding:NSUTF8StringEncoding error:&error];
        [weightBlob(channels,pattern) writeToFile:[weightsDirectory
            stringByAppendingPathComponent:@"weight.bin"]
            options:NSDataWritingAtomic error:&error];
        if (error) {
            fprintf(stderr,"input write: %s\n",error.description.UTF8String);
            return 2;
        }
        void *library = dlopen(
            "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler",
            RTLD_NOW | RTLD_LOCAL);
        ANECCompileFn compile = library ? (ANECCompileFn)dlsym(library,"ANECCompile") : NULL;
        ANECCreateInputFn createInput = library ? (ANECCreateInputFn)dlsym(
            library,"ANECCreateCompilerInputDictionary") : NULL;
        if (!compile || !createInput) return 3;
        CFDictionaryRef raw = createInput(milPath.fileSystemRepresentation);
        if (!raw) return 4;
        CFMutableDictionaryRef input = CFDictionaryCreateMutableCopy(
            kCFAllocatorDefault,0,raw);
        CFDictionarySetValue(input,CFSTR("OutputFileName"),CFSTR("oracle"));
        CFDictionarySetValue(input,CFSTR("OutputFilePath"),
            (__bridge CFStringRef)[outputDirectory stringByAppendingString:@"/"]);
        const void *keys[] = {CFSTR("TargetArchitecture")};
        const void *values[] = {CFSTR("H16G")};
        CFDictionaryRef options = CFDictionaryCreate(kCFAllocatorDefault,
            keys,values,1,&kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks);
        __block int status = INT_MIN;
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        compile(input,options,^(int result,CFDictionaryRef output) {
            (void)output; status = result; dispatch_semaphore_signal(done);
        });
        long wait = dispatch_semaphore_wait(done,
            dispatch_time(DISPATCH_TIME_NOW,60*NSEC_PER_SEC));
        CFRelease(options); CFRelease(input); CFRelease(raw);
        if (wait != 0 || status != 0) {
            fprintf(stderr,"ANEC wait=%ld status=%d\n",wait,status);
            return 5;
        }
        NSString *found = nil;
        NSDirectoryEnumerator *enumerator =
            [NSFileManager.defaultManager enumeratorAtPath:outputDirectory];
        for (NSString *relative in enumerator) {
            NSString *path = [outputDirectory stringByAppendingPathComponent:relative];
            NSData *data = [NSData dataWithContentsOfFile:path
                options:NSDataReadingMappedIfSafe error:nil];
            if (data.length >= 4 && *(const uint32_t *)data.bytes == 0xBEEFFACE) {
                found = path; break;
            }
        }
        if (!found) return 6;
        NSString *stable = [outputDirectory stringByAppendingPathComponent:@"oracle.hwx"];
        if (![found isEqualToString:stable])
            [[NSData dataWithContentsOfFile:found] writeToFile:stable atomically:YES];
        printf("depthwise oracle channels=%lu pattern=%s path=%s\n",
            (unsigned long)channels,pattern.UTF8String,stable.UTF8String);
        return 0;
    }
}
