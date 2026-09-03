#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, ANEIRLevel) {
    ANEIRLevelSyntax,
    ANEIRLevelGraph,
    ANEIRLevelPrimitive,
    ANEIRLevelMachine,
    ANEIRLevelExecutable,
};

@interface ANECompilationState : NSObject
@property(nonatomic) ANEIRLevel level;
@property(nonatomic, strong) id payload;
@property(nonatomic, readonly, copy) NSArray<NSString *> *passTrace;
- (instancetype)initWithLevel:(ANEIRLevel)level payload:(id)payload;
- (void)recordCompletedPass:(NSString *)identifier;
@end

typedef BOOL (^ANEIRVerifier)(id payload,
                              ANEDiagnosticEngine *diagnostics);

@interface ANEPassContext : NSObject
@property(nonatomic, readonly) ANEDiagnosticEngine *diagnostics;
- (instancetype)initWithDiagnostics:(ANEDiagnosticEngine *)diagnostics;
- (void)setVerifier:(ANEIRVerifier)verifier forLevel:(ANEIRLevel)level;
- (BOOL)verifyState:(ANECompilationState *)state;
@end

@protocol ANEPass <NSObject>
@property(nonatomic, readonly, copy) NSString *identifier;
@property(nonatomic, readonly) ANEIRLevel inputLevel;
@property(nonatomic, readonly) ANEIRLevel outputLevel;
- (BOOL)runOnState:(ANECompilationState *)state
            context:(ANEPassContext *)context;
@end

@interface ANEPassManager : NSObject
- (void)addPass:(id<ANEPass>)pass;
- (BOOL)runState:(ANECompilationState *)state context:(ANEPassContext *)context;
@end

NS_ASSUME_NONNULL_END

