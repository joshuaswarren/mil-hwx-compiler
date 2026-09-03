#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEExecutableBundle.h"
#import "ANEStagedCompiler.h"

#include <stdio.h>

static NSString *milSource(NSUInteger size) {
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, %lu, %lu]> a, tensor<fp16, [1, %lu, %lu]> b) {\n"
         "    bool f = const()[name = string(\"f\"), val = bool(false)];\n"
         "    tensor<fp16, [1, %lu, %lu]> y = matmul(transpose_x = f, transpose_y = f, x = a, y = b)[name = string(\"mm\")];\n"
         "  } -> (y);\n}\n",
        (unsigned long)size,(unsigned long)size,
        (unsigned long)size,(unsigned long)size,
        (unsigned long)size,(unsigned long)size];
}

int main(int argc,const char *argv[]){@autoreleasepool{
    if(argc!=3){fprintf(stderr,"usage: %s SIZE OUTPUT_ROOT\n",argv[0]);return 64;}
    NSUInteger size=strtoull(argv[1],NULL,10);
    NSString *outputRoot=[NSString stringWithUTF8String:argv[2]];
    ANEDiagnosticEngine *diagnostics=[[ANEDiagnosticEngine alloc] init];
    ANEExecutableBundle *bundle=[ANEStagedCompiler compileMILData:
        [milSource(size) dataUsingEncoding:NSUTF8StringEncoding]
        modelRoot:[NSURL fileURLWithPath:outputRoot isDirectory:YES]
        target:@"H16G" diagnostics:diagnostics];
    for(ANEDiagnostic *diagnostic in diagnostics.diagnostics)
        fprintf(stderr,"%s: %s\n",diagnostic.code.UTF8String,
            diagnostic.message.UTF8String);
    if(!bundle)return 2;
    NSError *error=nil;
    NSURL *bundleDirectory=[NSURL fileURLWithPath:
        [outputRoot stringByAppendingPathComponent:@"bundle"] isDirectory:YES];
    if(![bundle writeToDirectory:bundleDirectory error:&error]){
        fprintf(stderr,"bundle write: %s\n",error.description.UTF8String);return 3;}
    printf("PREPARE matmul N=%lu hwx_bytes=%lu passes=%s\n",
        (unsigned long)size,(unsigned long)bundle.artifacts[0].image.length,
        [bundle.passTrace componentsJoinedByString:@","].UTF8String);
    return 0;
}}
