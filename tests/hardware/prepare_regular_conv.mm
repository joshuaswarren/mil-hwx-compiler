#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEExecutableBundle.h"
#import "ANEStagedCompiler.h"

#include <stdio.h>

static NSString *milSource(NSUInteger channels, NSUInteger spatial,
                           NSUInteger kernel) {
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, %lu, %lu, %lu]> x) {\n"
         "    string pt = const()[name = string(\"pt\"), val = string(\"same\")];\n"
         "    tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
         "    tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
         "    tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
         "    int32 gp = const()[name = string(\"gp\"), val = int32(1)];\n"
         "    tensor<fp16, [%lu, %lu, %lu, %lu]> w = const()[name = string(\"w\"), val = tensor<fp16, [%lu, %lu, %lu, %lu]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
         "    tensor<fp16, [1, %lu, %lu, %lu]> y = conv(dilations = dl, groups = gp, pad = pd, pad_type = pt, strides = st, weight = w, x = x)[name = string(\"y\")];\n"
         "  } -> (y);\n}\n",
        (unsigned long)channels,(unsigned long)spatial,(unsigned long)spatial,
        (unsigned long)channels,(unsigned long)channels,
        (unsigned long)kernel,(unsigned long)kernel,
        (unsigned long)channels,(unsigned long)channels,
        (unsigned long)kernel,(unsigned long)kernel,
        (unsigned long)channels,(unsigned long)spatial,(unsigned long)spatial];
}

static BOOL writeIdentityModel(NSString *root, NSUInteger channels,
                               NSUInteger kernel, NSError **error) {
    NSString *weightsDirectory=[root stringByAppendingPathComponent:@"weights"];
    if (![NSFileManager.defaultManager createDirectoryAtPath:weightsDirectory
        withIntermediateDirectories:YES attributes:nil error:error]) return NO;
    NSUInteger taps=kernel*kernel;
    NSUInteger count=channels*channels*taps;
    NSUInteger weightBytes=count*sizeof(_Float16);
    NSMutableData *blob=[NSMutableData dataWithLength:128+weightBytes];
    uint8_t *bytes=(uint8_t *)blob.mutableBytes;
    uint32_t version=1,chunks=2,magic=0xDEADBEEF;
    uint64_t payloadLength=weightBytes,payloadOffset=128;
    memcpy(bytes,&version,4);memcpy(bytes+4,&chunks,4);
    memcpy(bytes+64,&magic,4);memcpy(bytes+72,&payloadLength,8);
    memcpy(bytes+80,&payloadOffset,8);
    _Float16 *weights=(_Float16 *)(bytes+payloadOffset);
    NSUInteger center=taps/2;
    for(NSUInteger channel=0;channel<channels;++channel)
        weights[(channel*channels+channel)*taps+center]=(_Float16)1.0f;
    return [blob writeToFile:[weightsDirectory
        stringByAppendingPathComponent:@"weight.bin"]
        options:NSDataWritingAtomic error:error];
}

int main(int argc,const char *argv[]){@autoreleasepool{
    if(argc!=5){fprintf(stderr,"usage: %s CHANNELS SPATIAL KERNEL OUTPUT_ROOT\n",argv[0]);return 64;}
    NSUInteger channels=strtoull(argv[1],NULL,10);
    NSUInteger spatial=strtoull(argv[2],NULL,10);
    NSUInteger kernel=strtoull(argv[3],NULL,10);
    NSString *outputRoot=[NSString stringWithUTF8String:argv[4]];
    NSString *modelRoot=[outputRoot stringByAppendingPathComponent:@"model"];
    NSError *error=nil;
    if(!writeIdentityModel(modelRoot,channels,kernel,&error)){
        fprintf(stderr,"model write: %s\n",error.description.UTF8String);return 2;}
    ANEDiagnosticEngine *diagnostics=[[ANEDiagnosticEngine alloc] init];
    ANEExecutableBundle *bundle=[ANEStagedCompiler
        compileMILData:[milSource(channels,spatial,kernel)
            dataUsingEncoding:NSUTF8StringEncoding]
        modelRoot:[NSURL fileURLWithPath:modelRoot isDirectory:YES]
        target:@"H16G" diagnostics:diagnostics];
    for(ANEDiagnostic *diagnostic in diagnostics.diagnostics)
        fprintf(stderr,"%s: %s\n",diagnostic.code.UTF8String,
            diagnostic.message.UTF8String);
    if(!bundle)return 3;
    NSURL *bundleDirectory=[NSURL fileURLWithPath:
        [outputRoot stringByAppendingPathComponent:@"bundle"] isDirectory:YES];
    if(![bundle writeToDirectory:bundleDirectory error:&error])return 4;
    printf("PREPARE regular_conv C=%lu S=%lu K=%lu hwx_bytes=%lu passes=%s\n",
        (unsigned long)channels,(unsigned long)spatial,(unsigned long)kernel,
        (unsigned long)bundle.artifacts[0].image.length,
        [bundle.passTrace componentsJoinedByString:@","].UTF8String);
    return 0;
}}
