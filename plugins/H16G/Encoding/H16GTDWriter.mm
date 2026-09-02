#import "H16GTDWriter.h"

NSString *const H16GTDWriterErrorDomain = @"ANE.H16G.TDWriter";

@implementation H16GTDWriter {
    NSMutableData *_storage;
}
- (instancetype)initWithByteLength:(NSUInteger)byteLength {
    self = [super init];
    if (self) _storage = [NSMutableData dataWithLength:byteLength];
    return self;
}
- (NSData *)data { return [_storage copy]; }
- (BOOL)writeUInt32:(uint32_t)value atOffset:(NSUInteger)offset
              field:(NSString *)field error:(NSError **)error {
    if (offset % sizeof(uint32_t) != 0 || offset > _storage.length ||
        sizeof(uint32_t) > _storage.length - offset) {
        if (error) *error = [NSError errorWithDomain:H16GTDWriterErrorDomain code:1
            userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:
                @"field '%@' at 0x%lx exceeds or misaligns TD", field,
                (unsigned long)offset]}];
        return NO;
    }
    [_storage replaceBytesInRange:NSMakeRange(offset, sizeof(value))
                        withBytes:&value];
    return YES;
}
@end
