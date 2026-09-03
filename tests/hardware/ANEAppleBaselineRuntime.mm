#import "ANEAppleBaselineRuntime.h"

#import <IOSurface/IOSurface.h>
#import <dlfcn.h>
#import <objc/message.h>

#import "ANEProvisionedRuntime.h"

static NSString *const ANEAppleBaselineErrorDomain =
    @"dev.maderix.ANEAppleBaseline";

static BOOL baselineFailure(NSError **error, NSInteger code,
                            NSString *message) {
    if (error)
        *error = [NSError errorWithDomain:ANEAppleBaselineErrorDomain code:code
            userInfo:@{NSLocalizedDescriptionKey: message}];
    return NO;
}

@implementation ANEAppleBaselineRuntime {
    id _wrapper;
    NSString *_temporaryDirectory;
}

- (instancetype)initWithMILData:(NSData *)milData
                             qos:(unsigned int)qos
                           error:(NSError **)error {
    NSString *framework =
        @"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
         "AppleNeuralEngine";
    if (!dlopen(framework.fileSystemRepresentation, RTLD_NOW | RTLD_LOCAL)) {
        baselineFailure(error, 1,
                        @"AppleNeuralEngine framework is unavailable");
        return nil;
    }
    Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class wrapperClass = NSClassFromString(@"_ANEInMemoryModel");
    if (!descriptorClass || !wrapperClass) {
        baselineFailure(error, 2,
                        @"ANE in-memory compiler classes are unavailable");
        return nil;
    }
    id descriptor = ((id (*)(id, SEL, id, id, id))objc_msgSend)(
        descriptorClass, @selector(modelWithMILText:weights:optionsPlist:),
        milData, @{}, nil);
    id wrapper = ((id (*)(id, SEL, id))objc_msgSend)(
        wrapperClass, @selector(inMemoryModelWithDescriptor:), descriptor);
    NSString *identifier = ((id (*)(id, SEL))objc_msgSend)(
        wrapper, @selector(hexStringIdentifier));
    if (!wrapper || identifier.length == 0) {
        baselineFailure(error, 3, @"ANE in-memory model creation failed");
        return nil;
    }
    NSString *temporaryDirectory = [NSTemporaryDirectory()
        stringByAppendingPathComponent:identifier];
    if (![NSFileManager.defaultManager
            createDirectoryAtPath:temporaryDirectory
      withIntermediateDirectories:YES attributes:nil error:error] ||
        ![milData writeToFile:
            [temporaryDirectory stringByAppendingPathComponent:@"model.mil"]
                       options:NSDataWritingAtomic error:error])
        return nil;

    self = [super init];
    if (!self) return nil;
    _qos = qos;
    _wrapper = wrapper;
    _temporaryDirectory = [temporaryDirectory copy];

    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    NSError *compileError = nil;
    BOOL compiled = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))
                     objc_msgSend)(
        _wrapper, @selector(compileWithQoS:options:error:), _qos, @{},
        &compileError);
    _compileMicroseconds =
        (CFAbsoluteTimeGetCurrent() - started) * 1.0e6;
    if (!compiled) {
        if (error) *error = compileError;
        return nil;
    }
    return self;
}

- (BOOL)loadWithError:(NSError **)error {
    if (_loaded) return YES;
    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    BOOL loaded = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))
                   objc_msgSend)(
        _wrapper, @selector(loadWithQoS:options:error:), _qos, @{}, error);
    _loadMicroseconds = (CFAbsoluteTimeGetCurrent() - started) * 1.0e6;
    _loaded = loaded;
    return loaded;
}

- (BOOL)evaluateInputs:(NSArray<ANEIOSurfaceBuffer *> *)inputs
                outputs:(NSArray<ANEIOSurfaceBuffer *> *)outputs
                  error:(NSError **)error {
    if (!_loaded)
        return baselineFailure(error, 4,
                               @"load the Apple baseline before evaluation");
    Class requestClass = NSClassFromString(@"_ANERequest");
    Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
    if (!requestClass || !surfaceClass)
        return baselineFailure(error, 5,
                               @"ANE request classes are unavailable");
    @try {
        NSMutableArray *wrappedInputs = [NSMutableArray array];
        NSMutableArray<NSNumber *> *inputIndices = [NSMutableArray array];
        for (NSUInteger index = 0; index < inputs.count; ++index) {
            [wrappedInputs addObject:
                ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(
                    surfaceClass, @selector(objectWithIOSurface:),
                    inputs[index].ioSurface)];
            [inputIndices addObject:@(index)];
        }
        NSMutableArray *wrappedOutputs = [NSMutableArray array];
        NSMutableArray<NSNumber *> *outputIndices = [NSMutableArray array];
        for (NSUInteger index = 0; index < outputs.count; ++index) {
            [wrappedOutputs addObject:
                ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(
                    surfaceClass, @selector(objectWithIOSurface:),
                    outputs[index].ioSurface)];
            [outputIndices addObject:@(index)];
        }
        id request = ((id (*)(id, SEL, id, id, id, id, id, id, id))
                      objc_msgSend)(
            requestClass,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                      weightsBuffer:perfStats:procedureIndex:),
            wrappedInputs, inputIndices, wrappedOutputs, outputIndices,
            nil, nil, @0);
        return ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))
                objc_msgSend)(
            _wrapper, @selector(evaluateWithQoS:options:request:error:),
            _qos, @{}, request, error);
    } @catch (NSException *exception) {
        return baselineFailure(error, 6,
            [NSString stringWithFormat:@"Apple baseline evaluation raised %@: %@",
                                       exception.name, exception.reason]);
    }
}

- (BOOL)unloadWithError:(NSError **)error {
    if (!_loaded) return YES;
    BOOL unloaded = ((BOOL (*)(id, SEL, unsigned int, NSError **))
                     objc_msgSend)(
        _wrapper, @selector(unloadWithQoS:error:), _qos, error);
    if (unloaded) _loaded = NO;
    return unloaded;
}

- (void)dealloc {
    [self unloadWithError:nil];
    if (_temporaryDirectory.length != 0)
        [NSFileManager.defaultManager removeItemAtPath:_temporaryDirectory
                                                 error:nil];
}

@end
