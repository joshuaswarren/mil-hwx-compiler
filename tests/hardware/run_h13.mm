#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#import <math.h>

#import "ANEExecutableBundle.h"
#import "ANEProvisionedRuntime.h"

static int fail(NSString *message) {
    fprintf(stderr, "error: %s\n", message.UTF8String);
    return 1;
}

static NSData *readFile(NSString *path, NSError **error) {
    return [NSData dataWithContentsOfFile:path options:0 error:error];
}

static NSData *decodeHex(NSString *text) {
    if (text.length % 2) return nil;
    NSMutableData *result = [NSMutableData dataWithLength:text.length / 2];
    uint8_t *bytes = static_cast<uint8_t *>(result.mutableBytes);
    for (NSUInteger index = 0; index < result.length; ++index) {
        unsigned int value = 0;
        NSString *pair = [text substringWithRange:NSMakeRange(index * 2, 2)];
        NSScanner *scanner = [NSScanner scannerWithString:pair];
        if (![scanner scanHexInt:&value] || !scanner.isAtEnd) return nil;
        bytes[index] = static_cast<uint8_t>(value);
    }
    return result;
}

static NSDictionary *sliceRecord(NSDictionary *binding) {
    NSDictionary *slice = binding[@"slice"];
    return slice ?: @{ @"tensor": binding[@"name"], @"elementOffset": @0,
                       @"elementCount": @([binding[@"logicalBytes"] unsignedIntegerValue] / 2) };
}

// Bindings are keyed by tensor name and ANEC channel: mul(x, x) binds one
// tensor to channels 5 and 6, and runtime identifiers must be unique.
static NSString *bindingIdentifier(NSDictionary *binding) {
    return [NSString stringWithFormat:@"%@#%@", binding[@"name"], binding[@"index"]];
}

static ANEHWXBinding *runtimeBinding(NSDictionary *binding, ANESurfaceRole role) {
    NSArray *nchw = binding[@"nchw"];
    NSUInteger plane = [nchw[4] unsignedIntegerValue];
    NSUInteger batch = [nchw[1] unsignedIntegerValue] * plane;
    return [[ANEHWXBinding alloc]
        initWithIdentifier:bindingIdentifier(binding) role:role
        logicalByteLength:[binding[@"logicalBytes"] unsignedIntegerValue]
        allocationByteLength:[binding[@"allocationBytes"] unsignedIntegerValue]
        ioSurfaceIndex:[binding[@"index"] integerValue]
        rowStrideBytes:[nchw[5] unsignedIntegerValue]
        planeStrideBytes:plane batchStrideBytes:batch];
}

/// The physical byte offset of one logical element on this surface: the
/// elementwise surface uses one 64-byte lane per element, the padded spatial
/// surface a `max(64, W * 2)` row, and the parity matmul surface dense rows,
/// so all three come from the binding's layout. A batched surface's IOSurface
/// is sized from `allocationBytes`, never from the tensor descriptor's own
/// size, which covers one batch element.
static NSUInteger physicalOffset(NSDictionary *binding, NSUInteger element) {
    NSArray *nchw = binding[@"nchw"];
    NSUInteger width = [nchw[3] unsignedIntegerValue];
    NSUInteger row = [nchw[5] unsignedIntegerValue];
    return (element / width) * row + (element % width) * 2;
}

static BOOL writePhysical(ANEIOSurfaceBuffer *buffer, NSData *dense,
                          NSDictionary *binding, BOOL localSlice,
                          NSError **error) {
    NSDictionary *slice = sliceRecord(binding);
    NSUInteger offset = localSlice ? 0 : [slice[@"elementOffset"] unsignedIntegerValue];
    NSUInteger count = [slice[@"elementCount"] unsignedIntegerValue];
    if (dense.length < (offset + count) * 2) {
        if (error) *error = [NSError errorWithDomain:@"H13Runner" code:1
            userInfo:@{NSLocalizedDescriptionKey: @"dense input is shorter than its manifest slice"}];
        return NO;
    }
    if (IOSurfaceLock(buffer.ioSurface, 0, nullptr) != kIOReturnSuccess) return NO;
    uint8_t *destination = static_cast<uint8_t *>(IOSurfaceGetBaseAddress(buffer.ioSurface));
    memset(destination, 0, buffer.allocationByteLength);
    const uint8_t *source = static_cast<const uint8_t *>(dense.bytes) + offset * 2;
    for (NSUInteger element = 0; element < count; ++element)
        memcpy(destination + physicalOffset(binding, element),
               source + element * 2, 2);
    IOSurfaceUnlock(buffer.ioSurface, 0, nullptr);
    return YES;
}

static BOOL readPhysical(ANEIOSurfaceBuffer *buffer, NSMutableData *dense,
                         NSDictionary *binding) {
    NSDictionary *slice = sliceRecord(binding);
    NSUInteger offset = [slice[@"elementOffset"] unsignedIntegerValue];
    NSUInteger count = [slice[@"elementCount"] unsignedIntegerValue];
    if (dense.length < (offset + count) * 2) return NO;
    if (IOSurfaceLock(buffer.ioSurface, kIOSurfaceLockReadOnly, nullptr) !=
        kIOReturnSuccess) return NO;
    const uint8_t *source = static_cast<const uint8_t *>(
        IOSurfaceGetBaseAddress(buffer.ioSurface));
    uint8_t *destination = static_cast<uint8_t *>(dense.mutableBytes) + offset * 2;
    for (NSUInteger element = 0; element < count; ++element)
        memcpy(destination + element * 2,
               source + physicalOffset(binding, element), 2);
    IOSurfaceUnlock(buffer.ioSurface, kIOSurfaceLockReadOnly, nullptr);
    return YES;
}

static ANEIOSurfaceBuffer *bufferNamed(NSArray<ANEIOSurfaceBuffer *> *buffers,
                                       NSString *name) {
    for (ANEIOSurfaceBuffer *buffer in buffers)
        if ([buffer.identifier isEqualToString:name]) return buffer;
    return nil;
}

static float fp16Value(uint16_t bits) {
    float sign = bits & 0x8000u ? -1.0f : 1.0f;
    unsigned exponent = (bits >> 10) & 0x1fu;
    unsigned fraction = bits & 0x03ffu;
    if (exponent == 0x1f) return fraction ? NAN : sign * INFINITY;
    if (exponent == 0) return sign * ldexpf((float)fraction, -24);
    return sign * ldexpf((float)(1024 + fraction), (int)exponent - 25);
}

static BOOL compareOutput(NSData *actual, NSData *expected, BOOL chunked,
                          NSString *name) {
    if (actual.length != expected.length || actual.length % 2) {
        printf("MISMATCH tensor=%s reason=byte-count actual=%lu expected=%lu\n",
            name.UTF8String, (unsigned long)actual.length,
            (unsigned long)expected.length);
        return NO;
    }
    const uint16_t *actualBits = static_cast<const uint16_t *>(actual.bytes);
    const uint16_t *expectedBits = static_cast<const uint16_t *>(expected.bytes);
    NSUInteger count = actual.length / 2;
    NSUInteger mismatches = 0;
    float maximumError = 0.0f;
    for (NSUInteger index = 0; index < count; ++index) {
        float got = fp16Value(actualBits[index]);
        float want = fp16Value(expectedBits[index]);
        float difference = fabsf(got - want);
        maximumError = fmaxf(maximumError, difference);
        BOOL equal = chunked
            ? isfinite(got) && difference <= 0.02f + 0.02f * fabsf(want)
            : actualBits[index] == expectedBits[index];
        if (!equal) {
            if (mismatches < 8)
                printf("MISMATCH tensor=%s index=%lu expected=%g actual=%g\n",
                    name.UTF8String, (unsigned long)index, want, got);
            ++mismatches;
        }
    }
    printf("COMPARE tensor=%s mode=%s elements=%lu mismatches=%lu max_abs_error=%g\n",
        name.UTF8String, chunked ? "abs0.02-rel0.02" : "bit-exact",
        (unsigned long)count, (unsigned long)mismatches, maximumError);
    return mismatches == 0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 4) {
            fprintf(stderr, "usage: %s PACKAGE --cache HASH... --input NAME=FILE... --expected NAME=FILE...\n", argv[0]);
            return 64;
        }
        NSString *package = [NSString stringWithUTF8String:argv[1]];
        NSMutableArray<NSString *> *hashes = [NSMutableArray array];
        NSMutableDictionary<NSString *, NSString *> *inputPaths = [NSMutableDictionary dictionary];
        NSMutableDictionary<NSString *, NSString *> *expectedPaths = [NSMutableDictionary dictionary];
        for (int index = 2; index < argc; index += 2) {
            if (index + 1 >= argc) return fail(@"an option is missing its value");
            NSString *option = [NSString stringWithUTF8String:argv[index]];
            NSString *value = [NSString stringWithUTF8String:argv[index + 1]];
            if ([option isEqualToString:@"--cache"]) {
                [hashes addObject:value];
                continue;
            }
            NSRange separator = [value rangeOfString:@"="];
            if (separator.location == NSNotFound || separator.location == 0)
                return fail(@"input and expected values must be NAME=FILE");
            NSString *name = [value substringToIndex:separator.location];
            NSString *path = [value substringFromIndex:separator.location + 1];
            if ([option isEqualToString:@"--input"]) inputPaths[name] = path;
            else if ([option isEqualToString:@"--expected"]) expectedPaths[name] = path;
            else return fail([NSString stringWithFormat:@"unknown option %@", option]);
        }

        NSError *error = nil;
        NSData *manifestBytes = readFile([package stringByAppendingPathComponent:@"manifest.json"], &error);
        NSDictionary *manifest = manifestBytes ?
            [NSJSONSerialization JSONObjectWithData:manifestBytes options:0 error:&error] : nil;
        if (!manifest || ![manifest[@"schema"] isEqualToString:@"mil-hwxc.h13-anec-package.v1"] ||
            ![manifest[@"target"] isEqualToString:@"H13"] ||
            ![manifest[@"artifactFormat"] isEqualToString:@"hwx"])
            return fail(error.localizedDescription ?: @"package is not an H13 HWX package");
        NSArray *programs = manifest[@"programs"];
        NSDictionary *tensors = manifest[@"tensors"];
        if (hashes.count != programs.count) return fail(@"one cache hash is required per program");

        NSMutableDictionary<NSString *, NSMutableData *> *dense = [NSMutableDictionary dictionary];
        for (NSString *name in tensors) {
            NSUInteger length = [tensors[name][@"logicalBytes"] unsignedIntegerValue];
            dense[name] = [NSMutableData dataWithLength:length];
        }
        for (NSString *name in inputPaths) {
            if (!dense[name] || ![tensors[name][@"role"] isEqualToString:@"input"])
                return fail([NSString stringWithFormat:@"unknown input tensor %@", name]);
            NSData *data = readFile(inputPaths[name], &error);
            if (!data || data.length != dense[name].length)
                return fail(error.localizedDescription ?: [NSString stringWithFormat:@"input %@ has the wrong byte count", name]);
            [dense[name] setData:data];
        }
        for (NSString *name in tensors) {
            NSString *role = tensors[name][@"role"];
            if ([role isEqualToString:@"input"] && !inputPaths[name])
                return fail([NSString stringWithFormat:@"missing input tensor %@", name]);
            if ([role isEqualToString:@"output"] && !expectedPaths[name])
                return fail([NSString stringWithFormat:@"missing expected tensor %@", name]);
        }
        for (NSString *name in expectedPaths)
            if (!dense[name] || ![tensors[name][@"role"] isEqualToString:@"output"])
                return fail([NSString stringWithFormat:@"unknown expected tensor %@", name]);

        BOOL valid = YES;
        for (NSUInteger programIndex = 0; valid && programIndex < programs.count; ++programIndex) {
            NSDictionary *program = programs[programIndex];
            NSMutableArray<ANEHWXBinding *> *bindings = [NSMutableArray array];
            for (NSDictionary *binding in program[@"inputs"])
                [bindings addObject:runtimeBinding(binding, ANESurfaceRoleInput)];
            for (NSDictionary *binding in program[@"outputs"])
                [bindings addObject:runtimeBinding(binding, ANESurfaceRoleOutput)];
            NSString *filename = program[@"file"];
            if (![filename.lastPathComponent isEqualToString:filename])
                return fail(@"program filename must be a basename");
            NSData *image = readFile([package stringByAppendingPathComponent:filename], &error);
            if (!image) return fail(error.localizedDescription);
            ANEHWXArtifact *artifact = [[ANEHWXArtifact alloc]
                initWithImage:image bindings:bindings operations:@[program[@"operation"]]];
            ANEExecutableBundle *bundle = [[ANEExecutableBundle alloc]
                initWithTarget:@"H13" artifacts:@[artifact] dispatchPlan:@[@0]
                passTrace:@[]];
            ANEProvisionedRuntime *runtime = [[ANEProvisionedRuntime alloc]
                initWithBundle:bundle modelHash:hashes[programIndex] qos:21 error:&error];
            NSArray<ANEIOSurfaceBuffer *> *inputs =
                [runtime createInputBuffersWithError:&error];
            NSArray<ANEIOSurfaceBuffer *> *outputs =
                [runtime createOutputBuffersWithError:&error];
            if (!runtime || !inputs || !outputs) return fail(error.localizedDescription);
            NSDictionary *constants = program[@"constantInputs"];
            for (NSDictionary *binding in program[@"inputs"]) {
                NSString *name = binding[@"name"];
                BOOL isConstant = constants[name] != nil;
                NSData *source = isConstant ? decodeHex(constants[name]) : dense[name];
                if (!source || !writePhysical(bufferNamed(inputs, bindingIdentifier(binding)), source,
                                              binding, isConstant, &error))
                    return fail(error.localizedDescription ?: @"failed to pack an input IOSurface");
            }
            BOOL loaded = [runtime loadWithError:&error];
            printf("LOAD h13 program=%lu result=%d cache=%s error=%s\n",
                (unsigned long)programIndex, loaded, hashes[programIndex].UTF8String,
                error ? error.description.UTF8String : "(none)");
            @try {
                if (loaded) {
                    error = nil;
                    valid = [runtime evaluateInputs:inputs outputs:outputs error:&error];
                    printf("EVAL h13 program=%lu result=%d error=%s\n",
                        (unsigned long)programIndex, valid,
                        error ? error.description.UTF8String : "(none)");
                    for (NSDictionary *binding in program[@"outputs"])
                        valid = valid && readPhysical(bufferNamed(outputs, bindingIdentifier(binding)),
                                                      dense[binding[@"name"]], binding);
                } else valid = NO;
            } @finally {
                if (runtime.loaded && ![runtime unloadWithError:&error]) valid = NO;
            }
        }

        for (NSString *name in expectedPaths) {
            NSDictionary *tensor = tensors[name];
            if (!tensor) return fail([NSString stringWithFormat:@"unknown expected tensor %@", name]);
            NSString *storedName = tensor[@"aliasOf"] ?: name;
            NSData *expected = readFile(expectedPaths[name], &error);
            if (!expected) return fail(error.localizedDescription);
            BOOL chunked = [tensor[@"accumulation"] isEqualToString:@"chunked-fp16"];
            valid = compareOutput(dense[storedName], expected, chunked, name) && valid;
        }
        printf("SUMMARY h13 valid=%d programs=%lu outputs=%lu\n", valid,
            (unsigned long)programs.count, (unsigned long)expectedPaths.count);
        return valid ? 0 : 1;
    }
}
