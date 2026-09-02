#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#import <math.h>

#import "ANEExecutableBundle.h"
#import "ANEProvisionedRuntime.h"

static NSString *const kModelHash=
    @"56EEB7C58DD30EA059D75A62C53809DA7E7B2C1CF2856B0B3B67FC6ED03BCED2_"
     "DF3F619804A92FDB4057192DC43DD748EA778ADC52BC498CE80524C014B81119_"
     "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";

static _Float16 expected(NSString *operation,_Float16 left,_Float16 right){
    if([operation isEqualToString:@"add"])return left+right;
    if([operation isEqualToString:@"mul"])return left*right;
    if([operation isEqualToString:@"max"])return left>right?left:right;
    return left<right?left:right;
}

static BOOL validate(IOSurfaceRef output,NSString *operation,NSUInteger count,
                     NSUInteger run){
    IOSurfaceLock(output,kIOSurfaceLockReadOnly,NULL);
    const _Float16 *values=(const _Float16 *)IOSurfaceGetBaseAddress(output);
    NSUInteger mismatches=0;float maximumError=0;
    for(NSUInteger index=0;index<count;++index){
        _Float16 left=(_Float16)((NSInteger)(index%17)-8)*(_Float16)0.125f;
        _Float16 right=(_Float16)((NSInteger)(index%11)-5)*(_Float16)0.25f;
        float want=(float)expected(operation,left,right),actual=(float)values[index];
        float error=fabsf(actual-want);maximumError=fmaxf(maximumError,error);
        if(!isfinite(actual)||error!=0){
            if(mismatches<8){
                printf("MISMATCH run=%lu index=%lu expected=%g actual=%g\n",
                    (unsigned long)run,(unsigned long)index,want,actual);
            }
            ++mismatches;
        }
    }
    IOSurfaceUnlock(output,kIOSurfaceLockReadOnly,NULL);
    printf("HARDWARE alu op=%s run=%lu elements=%lu mismatches=%lu max_abs_error=%g\n",
        operation.UTF8String,(unsigned long)run,(unsigned long)count,
        (unsigned long)mismatches,maximumError);
    return mismatches==0;
}

int main(int argc,const char *argv[]){@autoreleasepool{
    if(argc!=4){fprintf(stderr,"usage: %s OP BUNDLE_DIR CACHE_HWX\n",argv[0]);return 64;}
    NSString *operation=[NSString stringWithUTF8String:argv[1]];NSError *error=nil;
    ANEExecutableBundle *bundle=[ANEExecutableBundle bundleWithContentsOfDirectory:
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[2]]isDirectory:YES]
        error:&error];
    ANEProvisionedRuntime *runtime=[[ANEProvisionedRuntime alloc]
        initWithBundle:bundle modelHash:kModelHash qos:21 error:&error];
    NSArray<ANEIOSurfaceBuffer *> *inputs=[runtime createInputBuffersWithError:&error];
    NSArray<ANEIOSurfaceBuffer *> *outputs=[runtime createOutputBuffersWithError:&error];
    if(!runtime||inputs.count!=2||outputs.count!=1)return 2;
    NSUInteger count=inputs[0].logicalByteLength/2;
    for(NSUInteger operand=0;operand<2;++operand)IOSurfaceLock(inputs[operand].ioSurface,0,NULL);
    _Float16 *left=(_Float16 *)IOSurfaceGetBaseAddress(inputs[0].ioSurface);
    _Float16 *right=(_Float16 *)IOSurfaceGetBaseAddress(inputs[1].ioSurface);
    for(NSUInteger index=0;index<count;++index){
        left[index]=(_Float16)((NSInteger)(index%17)-8)*(_Float16)0.125f;
        right[index]=(_Float16)((NSInteger)(index%11)-5)*(_Float16)0.25f;
    }
    for(NSUInteger operand=0;operand<2;++operand)IOSurfaceUnlock(inputs[operand].ioSurface,0,NULL);
    BOOL valid=[runtime loadWithError:&error];
    printf("LOAD alu op=%s result=%d cache=%s error=%s\n",argv[1],valid,argv[3],
        error?error.description.UTF8String:"(none)");
    @try{for(NSUInteger run=1;valid&&run<=2;++run){
        IOSurfaceLock(outputs[0].ioSurface,0,NULL);
        memset(IOSurfaceGetBaseAddress(outputs[0].ioSurface),0xff,
            outputs[0].allocationByteLength);
        IOSurfaceUnlock(outputs[0].ioSurface,0,NULL);error=nil;
        BOOL evaluated=[runtime evaluateInputs:inputs outputs:outputs error:&error];
        printf("EVAL alu op=%s run=%lu result=%d error=%s\n",argv[1],
            (unsigned long)run,evaluated,error?error.description.UTF8String:"(none)");
        valid=evaluated&&validate(outputs[0].ioSurface,operation,count,run);
    }}@finally{if(runtime.loaded)[runtime unloadWithError:nil];}
    printf("SUMMARY alu op=%s valid=%d runs=2\n",argv[1],valid);return valid?0:1;
}}
