#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEExecutableBundle.h"
#import "ANEStagedCompiler.h"

#include <stdio.h>

static NSString *milSource(NSString *operation,NSUInteger size){
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, 1, %lu, %lu]> x, tensor<fp16, [1, 1, %lu, %lu]> z) {\n"
         "    tensor<fp16, [1, 1, %lu, %lu]> y = %@(x = x, y = z)[name = string(\"op\")];\n"
         "  } -> (y);\n}\n",
        (unsigned long)size,(unsigned long)size,(unsigned long)size,
        (unsigned long)size,(unsigned long)size,(unsigned long)size,operation];
}

int main(int argc,const char *argv[]){@autoreleasepool{
    if(argc!=4){fprintf(stderr,"usage: %s OP SIZE OUTPUT_ROOT\n",argv[0]);return 64;}
    NSString *operation=[NSString stringWithUTF8String:argv[1]];
    NSUInteger size=strtoull(argv[2],NULL,10);
    NSString *outputRoot=[NSString stringWithUTF8String:argv[3]];
    ANEDiagnosticEngine *diagnostics=[[ANEDiagnosticEngine alloc]init];
    ANEExecutableBundle *bundle=[ANEStagedCompiler compileMILData:
        [milSource(operation,size)dataUsingEncoding:NSUTF8StringEncoding]
        modelRoot:[NSURL fileURLWithPath:outputRoot isDirectory:YES]
        target:@"H16G" diagnostics:diagnostics];
    for(ANEDiagnostic *diagnostic in diagnostics.diagnostics)
        fprintf(stderr,"%s: %s\n",diagnostic.code.UTF8String,
            diagnostic.message.UTF8String);
    if(!bundle)return 2;
    NSError *error=nil;
    NSURL *directory=[NSURL fileURLWithPath:
        [outputRoot stringByAppendingPathComponent:@"bundle"]isDirectory:YES];
    if(![bundle writeToDirectory:directory error:&error])return 3;
    printf("PREPARE alu op=%s N=%lu hwx_bytes=%lu passes=%s\n",argv[1],
        (unsigned long)size,(unsigned long)bundle.artifacts[0].image.length,
        [bundle.passTrace componentsJoinedByString:@","].UTF8String);
    return 0;
}}
