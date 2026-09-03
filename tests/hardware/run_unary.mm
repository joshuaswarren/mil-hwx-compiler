#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#import <math.h>

#import "ANEExecutableBundle.h"
#import "ANEProvisionedRuntime.h"

static NSString *const kModelHash=
    @"56EEB7C58DD30EA059D75A62C53809DA7E7B2C1CF2856B0B3B67FC6ED03BCED2_"
     "DF3F619804A92FDB4057192DC43DD748EA778ADC52BC498CE80524C014B81119_"
     "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";

static float inputValue(NSString *operation,NSUInteger index){
    float unit=(float)(index%257)/256.0f;
    if([@[@"sqrt",@"rsqrt",@"reciprocal",@"log"]containsObject:operation])
        return 0.25f+3.75f*unit;
    if([operation isEqualToString:@"exp"])return -2.0f+4.0f*unit;
    return -4.0f+8.0f*unit;
}

static float expected(NSString *operation,float x){
    if([operation isEqualToString:@"relu"])return fmaxf(x,0.0f);
    if([operation isEqualToString:@"sigmoid"])return 1.0f/(1.0f+expf(-x));
    if([operation isEqualToString:@"tanh"])return tanhf(x);
    if([operation isEqualToString:@"gelu"])
        return 0.5f*x*(1.0f+erff(x*(float)M_SQRT1_2));
    if([operation isEqualToString:@"silu"])return x/(1.0f+expf(-x));
    if([operation isEqualToString:@"exp"])return expf(x);
    if([operation isEqualToString:@"log"])return logf(x);
    if([operation isEqualToString:@"sqrt"])return sqrtf(x);
    if([operation isEqualToString:@"rsqrt"])return 1.0f/sqrtf(x+0.000001f);
    return 1.0f/(x+0.000001f);
}

static float tolerance(NSString *operation){
    if([operation isEqualToString:@"relu"])return 0.0f;
    if([operation isEqualToString:@"sigmoid"]||
       [operation isEqualToString:@"tanh"])return 0.005f;
    if([operation isEqualToString:@"gelu"])return 0.01f;
    if([operation isEqualToString:@"silu"])return 0.016f;
    if([operation isEqualToString:@"exp"])return 0.01f;
    return 0.005f;
}

static BOOL validate(IOSurfaceRef output,NSString *operation,NSUInteger count,
                     NSUInteger run){
    IOSurfaceLock(output,kIOSurfaceLockReadOnly,NULL);
    const _Float16 *values=(const _Float16 *)IOSurfaceGetBaseAddress(output);
    NSUInteger mismatches=0;float maximumError=0,limit=tolerance(operation);
    for(NSUInteger index=0;index<count;++index){
        float x=(float)(_Float16)inputValue(operation,index);
        float want=(float)(_Float16)expected(operation,x);
        float actual=(float)values[index];
        float error=fabsf(actual-want);maximumError=fmaxf(maximumError,error);
        if(!isfinite(actual)||error>limit){
            if(mismatches<8)printf(
                "MISMATCH run=%lu index=%lu x=%g expected=%g actual=%g error=%g\n",
                (unsigned long)run,(unsigned long)index,x,want,actual,error);
            ++mismatches;
        }
    }
    IOSurfaceUnlock(output,kIOSurfaceLockReadOnly,NULL);
    printf("HARDWARE unary op=%s run=%lu elements=%lu mismatches=%lu max_abs_error=%g tolerance=%g\n",
        operation.UTF8String,(unsigned long)run,(unsigned long)count,
        (unsigned long)mismatches,maximumError,limit);
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
    if(!runtime||inputs.count!=1||outputs.count!=1)return 2;
    NSUInteger count=inputs[0].logicalByteLength/2;
    IOSurfaceLock(inputs[0].ioSurface,0,NULL);
    _Float16 *values=(_Float16 *)IOSurfaceGetBaseAddress(inputs[0].ioSurface);
    for(NSUInteger index=0;index<count;++index)
        values[index]=(_Float16)inputValue(operation,index);
    IOSurfaceUnlock(inputs[0].ioSurface,0,NULL);
    BOOL valid=[runtime loadWithError:&error];
    printf("LOAD unary op=%s result=%d cache=%s error=%s\n",argv[1],valid,argv[3],
        error?error.description.UTF8String:"(none)");
    @try{for(NSUInteger run=1;valid&&run<=2;++run){
        IOSurfaceLock(outputs[0].ioSurface,0,NULL);
        memset(IOSurfaceGetBaseAddress(outputs[0].ioSurface),0xff,
            outputs[0].allocationByteLength);
        IOSurfaceUnlock(outputs[0].ioSurface,0,NULL);error=nil;
        BOOL evaluated=[runtime evaluateInputs:inputs outputs:outputs error:&error];
        printf("EVAL unary op=%s run=%lu result=%d error=%s\n",argv[1],
            (unsigned long)run,evaluated,error?error.description.UTF8String:"(none)");
        valid=evaluated&&validate(outputs[0].ioSurface,operation,count,run);
    }}@finally{if(runtime.loaded)[runtime unloadWithError:nil];}
    printf("SUMMARY unary op=%s valid=%d runs=2\n",argv[1],valid);return valid?0:1;
}}
