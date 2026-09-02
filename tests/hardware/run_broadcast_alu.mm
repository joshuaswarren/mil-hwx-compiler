#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#import <math.h>

#import "ANEExecutableBundle.h"
#import "ANEProvisionedRuntime.h"

static NSString *const kModelHash=
    @"56EEB7C58DD30EA059D75A62C53809DA7E7B2C1CF2856B0B3B67FC6ED03BCED2_"
     "DF3F619804A92FDB4057192DC43DD748EA778ADC52BC498CE80524C014B81119_"
     "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";

static _Float16 *rowAddress(IOSurfaceRef surface, NSUInteger row) {
    return (_Float16 *)((uint8_t *)IOSurfaceGetBaseAddress(surface)+row*64);
}

int main(int argc,const char *argv[]){@autoreleasepool{
    if(argc!=3)return 64;
    NSString *name=[NSString stringWithUTF8String:argv[1]];
    NSError *error=nil;
    ANEExecutableBundle *bundle=[ANEExecutableBundle bundleWithContentsOfDirectory:
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[2]]
                   isDirectory:YES] error:&error];
    ANEProvisionedRuntime *runtime=[[ANEProvisionedRuntime alloc]
        initWithBundle:bundle modelHash:kModelHash qos:21 error:&error];
    NSArray *inputs=[runtime createInputBuffersWithError:&error];
    NSArray *outputs=[runtime createOutputBuffersWithError:&error];
    if(!runtime||outputs.count!=1)return 2;
    BOOL row=[name hasPrefix:@"row_"];
    ANEIOSurfaceBuffer *leftBuffer=nil,*rightBuffer=nil;
    for(ANEIOSurfaceBuffer *buffer in inputs){
        if([buffer.identifier isEqualToString:@"left"])leftBuffer=buffer;
        if([buffer.identifier isEqualToString:@"right"])rightBuffer=buffer;
    }
    if(!leftBuffer||(inputs.count==2&&!rightBuffer))return 3;
    IOSurfaceLock(leftBuffer.ioSurface,0,NULL);
    for(NSUInteger r=0;r<128;++r){
        _Float16 *p=rowAddress(leftBuffer.ioSurface,row?r:4*r);
        NSUInteger width=row?1:128;
        for(NSUInteger c=0;c<width;++c)p[c]=(_Float16)(0.25f+(r%7)*0.1f+c*0.002f);
    }
    IOSurfaceUnlock(leftBuffer.ioSurface,0,NULL);
    if(inputs.count==2){
        IOSurfaceLock(rightBuffer.ioSurface,0,NULL);
        for(NSUInteger r=0;r<128;++r)
            rowAddress(rightBuffer.ioSurface,r)[0]=
                (_Float16)(0.5f+(r%5)*0.05f);
        IOSurfaceUnlock(rightBuffer.ioSurface,0,NULL);
    }
    BOOL valid=[runtime loadWithError:&error];
    for(NSUInteger run=0;valid&&run<2;++run){
        valid=[runtime evaluateInputs:inputs outputs:outputs error:&error];
        IOSurfaceRef out=((ANEIOSurfaceBuffer *)outputs[0]).ioSurface;
        IOSurfaceLock(out,kIOSurfaceLockReadOnly,NULL);
        float maxError=0;NSUInteger mismatches=0;
        for(NSUInteger r=0;r<128;++r){
            NSUInteger width=row?1:128;
            for(NSUInteger c=0;c<width;++c){
                float left=0.25f+(r%7)*0.1f+c*0.002f;
                float right=0.5f+(r%5)*0.05f;
                float expected=left;
                if([name isEqualToString:@"scale"]) expected=left*0.08838834764831845f;
                else if([name hasSuffix:@"sub"]) expected=left-right;
                else if([name hasSuffix:@"mul"]) expected=left*right;
                else if([name hasSuffix:@"add"]) expected=left+right;
                else if([name hasSuffix:@"max"]) expected=fmaxf(left,right);
                float actual=(float)rowAddress(out,row?r:4*r)[c];
                float e=fabsf(actual-(float)(_Float16)expected);
                maxError=fmaxf(maxError,e); if(!isfinite(actual)||e>0.002f)++mismatches;
            }
        }
        IOSurfaceUnlock(out,kIOSurfaceLockReadOnly,NULL);
        printf("HARDWARE broadcast case=%s run=%lu mismatches=%lu max_abs_error=%g\n",
            name.UTF8String,(unsigned long)(run+1),(unsigned long)mismatches,maxError);
        valid=valid&&mismatches==0;
    }
    if(runtime.loaded)[runtime unloadWithError:nil];
    return valid?0:1;
}}
