#import <Foundation/Foundation.h>

#include <stdio.h>

static int8_t weightValue(NSUInteger layer, NSUInteger outputChannel,
                          NSUInteger inputChannel) {
    NSUInteger sourceChannel = (outputChannel + layer + 1) % 64;
    return inputChannel == sourceChannel ? 8 : 0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) {
            fprintf(stderr, "usage: %s OUTPUT_MODEL_ROOT\n", argv[0]);
            return 64;
        }
        NSMutableData *blob = [[NSData dataWithContentsOfFile:
            @"tests/models/w8a8/weights/weight.bin"] mutableCopy];
        if (blob.length != 16704) return 2;
        int8_t *bytes = (int8_t *)blob.mutableBytes;
        const NSUInteger chunkBytes = 64 + 64 * 64;
        for (NSUInteger layer = 0; layer < 4; ++layer) {
            NSUInteger payloadOffset = 64 + layer * chunkBytes + 64;
            for (NSUInteger output = 0; output < 64; ++output)
                for (NSUInteger input = 0; input < 64; ++input)
                    bytes[payloadOffset + output * 64 + input] =
                        weightValue(layer, output, input);
        }
        NSString *root = [NSString stringWithUTF8String:argv[1]];
        NSString *weightsDirectory =
            [root stringByAppendingPathComponent:@"weights"];
        NSError *error = nil;
        if (![NSFileManager.defaultManager
                createDirectoryAtPath:weightsDirectory
          withIntermediateDirectories:YES attributes:nil error:&error]) {
            fprintf(stderr, "%s\n", error.description.UTF8String);
            return 3;
        }
        NSString *output =
            [weightsDirectory stringByAppendingPathComponent:@"weight.bin"];
        if (![blob writeToFile:output options:NSDataWritingAtomic error:&error]) {
            fprintf(stderr, "%s\n", error.description.UTF8String);
            return 4;
        }
        printf("prepared W8A8 hardware model: %s\n", output.UTF8String);
        return 0;
    }
}
