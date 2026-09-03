#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <objc/message.h>

static NSString *milSource(NSUInteger naturalChannels) {
    NSUInteger packedChannels = naturalChannels * 16;
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, %lu, 128, 128]> x) {\n"
         "    int32 b = const()[name = string(\"b\"), val = int32(4)];\n"
         "    string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
         "    tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
         "    tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
         "    tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
         "    int32 gp = const()[name = string(\"gp\"), val = int32(1)];\n"
         "    tensor<fp16, [1, %lu, 32, 32]> packed = space_to_depth(x = x, block_size = b)[name = string(\"packed\")];\n"
         "    tensor<fp16, [%lu, %lu, 1, 1]> w = const()[name = string(\"w\"), val = tensor<fp16, [%lu, %lu, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
         "    tensor<fp16, [1, %lu, 32, 32]> c = conv(dilations = dl, groups = gp, pad = pd, pad_type = pt, strides = st, weight = w, x = packed)[name = string(\"c\")];\n"
         "    tensor<fp16, [1, %lu, 128, 128]> y = depth_to_space(x = c, block_size = b)[name = string(\"y\")];\n"
         "  } -> (y);\n}\n",
        (unsigned long)naturalChannels, (unsigned long)packedChannels,
        (unsigned long)packedChannels, (unsigned long)packedChannels,
        (unsigned long)packedChannels, (unsigned long)packedChannels,
        (unsigned long)packedChannels, (unsigned long)naturalChannels];
}

static NSData *weightBlob(NSUInteger channels, NSString *pattern) {
    const NSUInteger weightBytes = channels * channels * sizeof(_Float16);
    NSMutableData *blob = [NSMutableData dataWithLength:128 + weightBytes];
    uint8_t *header = (uint8_t *)blob.mutableBytes;
    const uint32_t fileVersion = 1;
    const uint32_t chunkCount = 2;
    const uint32_t magic = 0xDEADBEEF;
    const uint64_t payloadLength = weightBytes;
    const uint64_t payloadOffset = 128;
    memcpy(header, &fileVersion, sizeof(fileVersion));
    memcpy(header + 4, &chunkCount, sizeof(chunkCount));
    memcpy(header + 64, &magic, sizeof(magic));
    memcpy(header + 72, &payloadLength, sizeof(payloadLength));
    memcpy(header + 80, &payloadOffset, sizeof(payloadOffset));
    _Float16 *weights = (_Float16 *)(header + payloadOffset);
    for (NSUInteger output = 0; output < channels; ++output)
        for (NSUInteger input = 0; input < channels; ++input)
            weights[output * channels + input] = (_Float16)(
                [pattern isEqualToString:@"output"] ? output + 1 :
                [pattern isEqualToString:@"input"] ? input + 1 :
                (output == input ? 1 : 0));
    return blob;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3 && argc != 4) {
            fprintf(stderr, "usage: %s OUTPUT_DIR NATURAL_CHANNELS [identity|output|input]\n", argv[0]);
            return 64;
        }
        NSString *outputDirectory = [NSString stringWithUTF8String:argv[1]];
        NSUInteger naturalChannels =
            (NSUInteger)strtoull(argv[2], NULL, 10);
        NSString *pattern = argc == 4
            ? [NSString stringWithUTF8String:argv[3]] : @"identity";
        if (naturalChannels == 0 || naturalChannels % 4 != 0) return 65;
        if (![@[@"identity",@"output",@"input"] containsObject:pattern])
            return 66;
        NSFileManager *files = NSFileManager.defaultManager;
        NSError *error = nil;
        if (![files createDirectoryAtPath:
                [outputDirectory stringByAppendingPathComponent:@"weights"]
                withIntermediateDirectories:YES attributes:nil error:&error]) {
            fprintf(stderr, "mkdir: %s\n", error.description.UTF8String);
            return 2;
        }
        NSString *milPath =
            [outputDirectory stringByAppendingPathComponent:@"model.mil"];
        [milSource(naturalChannels) writeToFile:milPath atomically:YES
                         encoding:NSUTF8StringEncoding error:&error];
        [weightBlob(naturalChannels * 16, pattern) writeToFile:[outputDirectory
            stringByAppendingPathComponent:@"weights/weight.bin"]
                         options:NSDataWritingAtomic error:&error];
        if (error) {
            fprintf(stderr, "input write: %s\n", error.description.UTF8String);
            return 3;
        }

        if (!dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
             "AppleNeuralEngine", RTLD_NOW | RTLD_LOCAL)) return 4;
        Class descriptorClass =
            NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        NSData *milData = [milSource(naturalChannels)
            dataUsingEncoding:NSUTF8StringEncoding];
        NSData *weights = weightBlob(naturalChannels * 16, pattern);
        NSDictionary *weightMap = @{
            @"@model_path/weights/weight.bin": @{
                @"offset": @0,
                @"data": weights,
            },
        };
        id descriptor = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:),
            milData, weightMap, nil);
        id model = descriptor ?
            ((id (*)(Class, SEL, id))objc_msgSend)(modelClass,
                @selector(inMemoryModelWithDescriptor:), descriptor) : nil;
        if (!model) return 5;
        NSString *identifier = ((id (*)(id, SEL))objc_msgSend)(
            model, @selector(hexStringIdentifier));
        NSString *staging =
            [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        [files createDirectoryAtPath:
            [staging stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:&error];
        [milData writeToFile:[staging stringByAppendingPathComponent:@"model.mil"]
                  atomically:YES];
        [weights writeToFile:[staging
            stringByAppendingPathComponent:@"weights/weight.bin"]
                  atomically:YES];
        BOOL compiled = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))
            objc_msgSend)(model, @selector(compileWithQoS:options:error:),
                          0, @{}, &error);
        if (!compiled) {
            fprintf(stderr, "compile: %s\n", error.description.UTF8String);
            return 6;
        }
        printf("ANEC identifier=%s cache_namespace=mint_layout_conv_chain "
               "natural_channels=%lu pattern=%s\n", identifier.UTF8String,
               (unsigned long)naturalChannels, pattern.UTF8String);
        return 0;
    }
}
