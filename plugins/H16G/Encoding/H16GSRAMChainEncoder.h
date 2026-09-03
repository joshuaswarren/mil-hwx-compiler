#import <Foundation/Foundation.h>

#import "H16GConvChainEncoder.h"

NS_ASSUME_NONNULL_BEGIN

/// Encodes measured producer and consumer task forms that exchange their
/// intermediate through ANE SRAM inside one program.
@interface H16GSRAMChainEncoder : NSObject
+ (nullable H16GEncodedTDProgram *)encodeStageOperations:
    (NSArray<NSString *> *)stageOperations
    rows:(NSUInteger)rows
    columns:(NSUInteger)columns
    error:(NSError **)error;
+ (nullable NSData *)constantRegionForStageOperations:
    (NSArray<NSString *> *)stageOperations
    error:(NSError **)error;
+ (NSUInteger)taskCountForStageOperations:
    (NSArray<NSString *> *)stageOperations;
+ (nullable H16GEncodedTDProgram *)encodeProducerOperationName:
    (NSString *)producerOperationName
    consumerOperationName:(NSString *)consumerOperationName
    rows:(NSUInteger)rows
    columns:(NSUInteger)columns
    error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
