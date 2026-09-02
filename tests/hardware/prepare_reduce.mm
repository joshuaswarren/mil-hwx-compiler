#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEExecutableBundle.h"
#import "ANEStagedCompiler.h"

static NSString *milSource(NSString *operation, NSUInteger channels,
                           NSUInteger height, NSUInteger width,
                           NSUInteger axis) {
    NSUInteger outChannels=axis == 1 ? 1 : channels;
    NSUInteger outHeight=axis == 2 ? 1 : height;
    NSUInteger outWidth=axis == 3 ? 1 : width;
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, %lu, %lu, %lu]> x) {\n"
         "    tensor<int32, [1]> ax = const()[name = string(\"ax\"), "
         "val = tensor<int32, [1]>([%lu])];\n"
         "    tensor<fp16, [1, %lu, %lu, %lu]> y = %@"
         "(x = x, axes = ax, keep_dims = bool(true))"
         "[name = string(\"reduce\")];\n"
         "  } -> (y);\n}\n",
        (unsigned long)channels,(unsigned long)height,(unsigned long)width,
        (unsigned long)axis,(unsigned long)outChannels,
        (unsigned long)outHeight,(unsigned long)outWidth,operation];
}

int main(int argc,const char *argv[]){@autoreleasepool{
    if(argc!=7){
        fprintf(stderr,"usage: %s OP C H W AXIS OUTPUT_ROOT\n",argv[0]);
        return 64;
    }
    NSString *operation=[NSString stringWithUTF8String:argv[1]];
    NSUInteger channels=strtoull(argv[2],NULL,10);
    NSUInteger height=strtoull(argv[3],NULL,10);
    NSUInteger width=strtoull(argv[4],NULL,10);
    NSUInteger axis=strtoull(argv[5],NULL,10);
    NSString *root=[NSString stringWithUTF8String:argv[6]];
    ANEDiagnosticEngine *diagnostics=[[ANEDiagnosticEngine alloc]init];
    ANEExecutableBundle *bundle=[ANEStagedCompiler compileMILData:
        [milSource(operation,channels,height,width,axis)
            dataUsingEncoding:NSUTF8StringEncoding]
        modelRoot:[NSURL fileURLWithPath:root isDirectory:YES]
        target:@"H16G" diagnostics:diagnostics];
    for(ANEDiagnostic *diagnostic in diagnostics.diagnostics)
        fprintf(stderr,"%s: %s\n",diagnostic.code.UTF8String,
            diagnostic.message.UTF8String);
    if(!bundle)return 2;
    NSError *error=nil;
    NSURL *directory=[NSURL fileURLWithPath:
        [root stringByAppendingPathComponent:@"bundle"]isDirectory:YES];
    if(![bundle writeToDirectory:directory error:&error])return 3;
    printf("PREPARE reduce op=%s C=%lu H=%lu W=%lu axis=%lu hwx_bytes=%lu passes=%s\n",
        argv[1],(unsigned long)channels,(unsigned long)height,
        (unsigned long)width,(unsigned long)axis,
        (unsigned long)bundle.artifacts[0].image.length,
        [bundle.passTrace componentsJoinedByString:@","].UTF8String);
    return 0;
}}
