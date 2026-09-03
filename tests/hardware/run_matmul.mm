#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#import <math.h>

#import "ANEExecutableBundle.h"
#import "ANEProvisionedRuntime.h"

static NSString *const kModelHash=
    @"56EEB7C58DD30EA059D75A62C53809DA7E7B2C1CF2856B0B3B67FC6ED03BCED2_"
     "DF3F619804A92FDB4057192DC43DD748EA778ADC52BC498CE80524C014B81119_"
     "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";

static _Float16 inputValue(NSUInteger row,NSUInteger column){
    return (_Float16)((row+column)&7u)*(_Float16)0.25f;
}

static BOOL validate(IOSurfaceRef output,NSUInteger size,NSUInteger run){
    IOSurfaceLock(output,kIOSurfaceLockReadOnly,NULL);
    const _Float16 *values=(const _Float16 *)IOSurfaceGetBaseAddress(output);
    NSUInteger mismatches=0;float maximumError=0;
    for(NSUInteger row=0;row<size;++row)
        for(NSUInteger column=0;column<size;++column){
            NSUInteger index=row*size+column;
            float expected=(float)inputValue(row,column),actual=(float)values[index];
            float error=fabsf(actual-expected);maximumError=fmaxf(maximumError,error);
            if(!isfinite(actual)||error!=0){
                if(mismatches<8)printf("MISMATCH run=%lu row=%lu col=%lu expected=%g actual=%g\n",
                    (unsigned long)run,(unsigned long)row,(unsigned long)column,
                    expected,actual);
                ++mismatches;
            }
        }
    IOSurfaceUnlock(output,kIOSurfaceLockReadOnly,NULL);
    printf("HARDWARE matmul N=%lu run=%lu elements=%lu mismatches=%lu max_abs_error=%g\n",
        (unsigned long)size,(unsigned long)run,(unsigned long)(size*size),
        (unsigned long)mismatches,maximumError);
    return mismatches==0;
}

int main(int argc,const char *argv[]){@autoreleasepool{
    if(argc!=3){fprintf(stderr,"usage: %s BUNDLE_DIR CACHE_HWX\n",argv[0]);return 64;}
    NSError *error=nil;
    ANEExecutableBundle *bundle=[ANEExecutableBundle bundleWithContentsOfDirectory:
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]
                   isDirectory:YES] error:&error];
    if(!bundle){fprintf(stderr,"bundle read: %s\n",error.description.UTF8String);return 2;}
    ANEProvisionedRuntime *runtime=[[ANEProvisionedRuntime alloc]
        initWithBundle:bundle modelHash:kModelHash qos:21 error:&error];
    NSArray<ANEIOSurfaceBuffer *> *inputs=[runtime createInputBuffersWithError:&error];
    NSArray<ANEIOSurfaceBuffer *> *outputs=[runtime createOutputBuffersWithError:&error];
    if(!runtime||inputs.count!=2||outputs.count!=1){
        fprintf(stderr,"runtime bindings: %s\n",error.description.UTF8String);return 3;}
    NSUInteger elements=inputs[0].logicalByteLength/sizeof(_Float16);
    NSUInteger size=(NSUInteger)sqrt((double)elements);
    if(size*size!=elements||inputs[1].logicalByteLength!=inputs[0].logicalByteLength||
       outputs[0].logicalByteLength!=inputs[0].logicalByteLength)return 4;
    IOSurfaceLock(inputs[0].ioSurface,0,NULL);
    IOSurfaceLock(inputs[1].ioSurface,0,NULL);
    _Float16 *left=(_Float16 *)IOSurfaceGetBaseAddress(inputs[0].ioSurface);
    _Float16 *right=(_Float16 *)IOSurfaceGetBaseAddress(inputs[1].ioSurface);
    memset(right,0,inputs[1].allocationByteLength);
    for(NSUInteger row=0;row<size;++row)
        for(NSUInteger column=0;column<size;++column)
            left[row*size+column]=inputValue(row,column);
    for(NSUInteger diagonal=0;diagonal<size;++diagonal)
        right[diagonal*size+diagonal]=(_Float16)1.0f;
    IOSurfaceUnlock(inputs[1].ioSurface,0,NULL);
    IOSurfaceUnlock(inputs[0].ioSurface,0,NULL);
    BOOL valid=[runtime loadWithError:&error];
    printf("LOAD matmul N=%lu result=%d cache=%s error=%s\n",
        (unsigned long)size,valid,argv[2],error?error.description.UTF8String:"(none)");
    @try{
        for(NSUInteger run=1;valid&&run<=2;++run){
            IOSurfaceLock(outputs[0].ioSurface,0,NULL);
            memset(IOSurfaceGetBaseAddress(outputs[0].ioSurface),0xff,
                outputs[0].allocationByteLength);
            IOSurfaceUnlock(outputs[0].ioSurface,0,NULL);
            error=nil;CFAbsoluteTime start=CFAbsoluteTimeGetCurrent();
            BOOL evaluated=[runtime evaluateInputs:inputs outputs:outputs error:&error];
            printf("EVAL matmul N=%lu run=%lu result=%d time_us=%.1f error=%s\n",
                (unsigned long)size,(unsigned long)run,evaluated,
                (CFAbsoluteTimeGetCurrent()-start)*1.0e6,
                error?error.description.UTF8String:"(none)");
            valid=evaluated&&validate(outputs[0].ioSurface,size,run);
        }
    }@finally{if(runtime.loaded)[runtime unloadWithError:nil];}
    printf("SUMMARY matmul N=%lu valid=%d runs=2\n",(unsigned long)size,valid);
    return valid?0:1;
}}
