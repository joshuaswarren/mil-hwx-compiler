#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#import <math.h>

#import "ANEExecutableBundle.h"
#import "ANEProvisionedRuntime.h"

static NSString *const kModelHash=
    @"A8FA2E340D752C36B7B5E31658D96FA6D6C85A8185A72B72CBE1305C3F4806AA_"
     "B9CB7B429B3392F8A4C6A77826C0698CF7719A8F59ED06176DE99C6D593F661B_"
     "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";

static _Float16 inputValue(NSUInteger index) {
    NSInteger centered=(NSInteger)(index%13)-6;
    return (_Float16)((float)centered/32.0f);
}

static _Float16 expectedValue(NSString *operation, NSUInteger channels,
                              NSUInteger height, NSUInteger width,
                              NSUInteger axis, NSUInteger outChannel,
                              NSUInteger outHeight, NSUInteger outWidth) {
    float value=[operation isEqualToString:@"reduce_max"] ? -INFINITY : 0.0f;
    NSUInteger count=axis==1?channels:(axis==2?height:width);
    for(NSUInteger reduced=0;reduced<count;++reduced){
        NSUInteger c=axis==1?reduced:outChannel;
        NSUInteger h=axis==2?reduced:outHeight;
        NSUInteger w=axis==3?reduced:outWidth;
        float sample=(float)inputValue((c*height+h)*width+w);
        if([operation isEqualToString:@"reduce_max"])
            value=fmaxf(value,sample);
        else value+=sample;
    }
    if([operation isEqualToString:@"reduce_mean"])
        value/=(float)count;
    return (_Float16)value;
}

static BOOL validate(IOSurfaceRef output, ANEHWXBinding *binding,
                     NSString *operation, NSUInteger channels,
                     NSUInteger height, NSUInteger width, NSUInteger axis,
                     NSUInteger run) {
    NSUInteger outChannels=axis==1?1:channels;
    NSUInteger outHeight=axis==2?1:height;
    NSUInteger outWidth=axis==3?1:width;
    IOSurfaceLock(output,kIOSurfaceLockReadOnly,NULL);
    const uint8_t *base=(const uint8_t *)IOSurfaceGetBaseAddress(output);
    NSUInteger mismatches=0; float maximumError=0.0f;
    for(NSUInteger c=0;c<outChannels;++c){
        for(NSUInteger h=0;h<outHeight;++h){
            for(NSUInteger w=0;w<outWidth;++w){
                NSUInteger offset=c*binding.planeStrideBytes+
                    h*binding.rowStrideBytes+w*sizeof(_Float16);
                _Float16 actualHalf=*(const _Float16 *)(base+offset);
                _Float16 expectedHalf=expectedValue(operation,channels,height,
                    width,axis,c,h,w);
                float actual=(float)actualHalf,want=(float)expectedHalf;
                float error=fabsf(actual-want);
                maximumError=fmaxf(maximumError,error);
                if(!isfinite(actual)||actualHalf!=expectedHalf){
                    if(mismatches<8)printf(
                        "MISMATCH run=%lu c=%lu h=%lu w=%lu expected=%g actual=%g error=%g\n",
                        (unsigned long)run,(unsigned long)c,(unsigned long)h,
                        (unsigned long)w,want,actual,error);
                    ++mismatches;
                }
            }
        }
    }
    IOSurfaceUnlock(output,kIOSurfaceLockReadOnly,NULL);
    printf("HARDWARE reduce op=%s C=%lu H=%lu W=%lu axis=%lu run=%lu outputs=%lu mismatches=%lu max_abs_error=%g\n",
        operation.UTF8String,(unsigned long)channels,(unsigned long)height,
        (unsigned long)width,(unsigned long)axis,(unsigned long)run,
        (unsigned long)(outChannels*outHeight*outWidth),
        (unsigned long)mismatches,maximumError);
    return mismatches==0;
}

int main(int argc,const char *argv[]){@autoreleasepool{
    if(argc!=8){
        fprintf(stderr,"usage: %s OP C H W AXIS BUNDLE_DIR CACHE_HWX\n",argv[0]);
        return 64;
    }
    NSString *operation=[NSString stringWithUTF8String:argv[1]];
    NSUInteger channels=strtoull(argv[2],NULL,10);
    NSUInteger height=strtoull(argv[3],NULL,10);
    NSUInteger width=strtoull(argv[4],NULL,10);
    NSUInteger axis=strtoull(argv[5],NULL,10);
    NSError *error=nil;
    ANEExecutableBundle *bundle=[ANEExecutableBundle bundleWithContentsOfDirectory:
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[6]]
                   isDirectory:YES] error:&error];
    ANEProvisionedRuntime *runtime=[[ANEProvisionedRuntime alloc]
        initWithBundle:bundle modelHash:kModelHash qos:21 error:&error];
    NSArray<ANEIOSurfaceBuffer *> *inputs=[runtime createInputBuffersWithError:&error];
    NSArray<ANEIOSurfaceBuffer *> *outputs=[runtime createOutputBuffersWithError:&error];
    if(!runtime||inputs.count!=1||outputs.count!=1)return 2;
    ANEHWXBinding *inputBinding=bundle.artifacts[0].bindings[0];
    IOSurfaceLock(inputs[0].ioSurface,0,NULL);
    uint8_t *inputBase=(uint8_t *)IOSurfaceGetBaseAddress(inputs[0].ioSurface);
    memset(inputBase,0,inputs[0].allocationByteLength);
    for(NSUInteger c=0;c<channels;++c){
        for(NSUInteger h=0;h<height;++h){
            for(NSUInteger w=0;w<width;++w){
                NSUInteger logical=(c*height+h)*width+w;
                NSUInteger physical=c*inputBinding.planeStrideBytes+
                    h*inputBinding.rowStrideBytes+w*sizeof(_Float16);
                *(_Float16 *)(inputBase+physical)=inputValue(logical);
            }
        }
    }
    IOSurfaceUnlock(inputs[0].ioSurface,0,NULL);
    BOOL valid=[runtime loadWithError:&error];
    printf("LOAD reduce op=%s result=%d cache=%s error=%s\n",argv[1],valid,argv[7],
        error?error.description.UTF8String:"(none)");
    ANEHWXBinding *outputBinding=bundle.artifacts[0].bindings[1];
    @try{for(NSUInteger run=1;valid&&run<=2;++run){
        IOSurfaceLock(outputs[0].ioSurface,0,NULL);
        memset(IOSurfaceGetBaseAddress(outputs[0].ioSurface),0xff,
            outputs[0].allocationByteLength);
        IOSurfaceUnlock(outputs[0].ioSurface,0,NULL);error=nil;
        BOOL evaluated=[runtime evaluateInputs:inputs outputs:outputs error:&error];
        printf("EVAL reduce op=%s run=%lu result=%d error=%s\n",argv[1],
            (unsigned long)run,evaluated,error?error.description.UTF8String:"(none)");
        valid=evaluated&&validate(outputs[0].ioSurface,outputBinding,operation,
            channels,height,width,axis,run);
    }}@finally{if(runtime.loaded)[runtime unloadWithError:nil];}
    printf("SUMMARY reduce op=%s valid=%d runs=2\n",argv[1],valid);
    return valid?0:1;
}}
