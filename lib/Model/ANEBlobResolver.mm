#import "ANEBlobResolver.h"

static ANEGraphArgument *findCall(ANEGraphArgument *argument,
                                  NSString *calleeName) {
    if (!argument) return nil;
    if (argument.kind == ANEGraphArgumentKindCall &&
        [argument.calleeName isEqualToString:calleeName]) return argument;
    for (ANEGraphNamedArgument *child in argument.callArguments) {
        ANEGraphArgument *found = findCall(child.value, calleeName);
        if (found) return found;
    }
    for (ANEGraphArgument *child in argument.elements) {
        ANEGraphArgument *found = findCall(child, calleeName);
        if (found) return found;
    }
    return nil;
}

static NSString *scalarText(ANEGraphArgument *argument) {
    if (!argument) return nil;
    if (argument.text) return argument.text;
    if (argument.kind == ANEGraphArgumentKindCall &&
        argument.callArguments.count == 1)
        return scalarText(argument.callArguments[0].value);
    return nil;
}

static void emitError(ANEDiagnosticEngine *diagnostics, NSString *code,
                      NSString *message, ANESourceRange range) {
    [diagnostics emitSeverity:ANEDiagnosticSeverityError code:code
                      message:message range:range];
}

static BOOL rangeFits(NSUInteger offset, NSUInteger length, NSUInteger limit) {
    return offset <= limit && length <= limit - offset;
}

static NSData *loadBlob(ANEGraphArgument *blob, NSUInteger expectedBytes,
                        NSURL *modelRoot,
                        ANEDiagnosticEngine *diagnostics) {
    NSString *path = scalarText(blob.namedArguments[@"path"].value);
    NSString *offsetText = scalarText(blob.namedArguments[@"offset"].value);
    if (![path hasPrefix:@"@model_path/"] || !offsetText) {
        emitError(diagnostics, @"ane.model.invalid-blob-reference",
                  @"BLOBFILE requires @model_path and an integer offset",
                  blob.range);
        return nil;
    }
    NSString *rootPath = modelRoot.URLByStandardizingPath.path;
    NSString *relative = [path substringFromIndex:@"@model_path/".length];
    NSString *candidate = [[rootPath stringByAppendingPathComponent:relative]
        stringByStandardizingPath];
    NSString *rootPrefix = [rootPath stringByAppendingString:@"/"];
    if (![candidate hasPrefix:rootPrefix]) {
        emitError(diagnostics, @"ane.model.path-escape",
                  @"BLOBFILE path escapes model root", blob.range);
        return nil;
    }
    NSData *file = [NSData dataWithContentsOfFile:candidate];
    if (!file) {
        emitError(diagnostics, @"ane.model.missing-blob",
            [NSString stringWithFormat:@"cannot read model blob '%@'", candidate],
            blob.range);
        return nil;
    }
    unsigned long long parsedOffset = strtoull(offsetText.UTF8String, NULL, 0);
    if (parsedOffset > NSUIntegerMax ||
        !rangeFits((NSUInteger)parsedOffset, 24, file.length)) {
        emitError(diagnostics, @"ane.model.blob-header-range",
                  @"BLOBFILE chunk header exceeds the file", blob.range);
        return nil;
    }
    const uint8_t *bytes = (const uint8_t *)file.bytes;
    NSUInteger chunkOffset = (NSUInteger)parsedOffset;
    uint32_t magic = 0;
    uint64_t payloadLength = 0;
    uint64_t payloadOffset = 0;
    memcpy(&magic, bytes + chunkOffset, sizeof(magic));
    memcpy(&payloadLength, bytes + chunkOffset + 8, sizeof(payloadLength));
    memcpy(&payloadOffset, bytes + chunkOffset + 16, sizeof(payloadOffset));
    if (magic != 0xDEADBEEFu || payloadLength < expectedBytes ||
        payloadOffset > NSUIntegerMax ||
        !rangeFits((NSUInteger)payloadOffset, expectedBytes, file.length)) {
        emitError(diagnostics, @"ane.model.invalid-blob-header",
                  @"BLOBFILE subheader has invalid magic, size, or payload offset",
                  blob.range);
        return nil;
    }
    return [file subdataWithRange:NSMakeRange((NSUInteger)payloadOffset,
                                               expectedBytes)];
}

@implementation ANEBlobResolver

+ (NSData *)loadConstantForOperation:(ANEGraphOperation *)operation
                        expectedBytes:(NSUInteger)expectedBytes
                             modelRoot:(NSURL *)modelRoot
                           diagnostics:(ANEDiagnosticEngine *)diagnostics {
    NSString *payload = [operation.operationName isEqualToString:@"constexpr_affine_dequantize"]
        ? @"quantized_data" : @"val";
    ANEGraphArgument *blob = findCall(operation.attributes[payload], @"BLOBFILE");
    if (!blob) {
        emitError(diagnostics, @"ane.model.missing-blob-reference",
                  @"constant has no BLOBFILE payload", operation.range);
        return nil;
    }
    return loadBlob(blob, expectedBytes, modelRoot, diagnostics);
}
@end
