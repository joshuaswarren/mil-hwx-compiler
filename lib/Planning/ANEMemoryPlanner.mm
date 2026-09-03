#import "ANEMemoryPlanner.h"

static NSUInteger alignUp(NSUInteger value, NSUInteger alignment) {
    return (value + alignment - 1) / alignment * alignment;
}
static BOOL lifetimesOverlap(ANEScheduledSurface *a, ANEScheduledSurface *b) {
    return !(a.lastTask < b.firstTask || b.lastTask < a.firstTask);
}
static BOOL rangesOverlap(NSUInteger a, NSUInteger as, NSUInteger b, NSUInteger bs) {
    return a < b + bs && b < a + as;
}

@implementation ANEMemoryPlanner
+ (NSUInteger)allocateSurfaces:(NSArray<ANEScheduledSurface *> *)surfaces
                         target:(H16GTarget *)target {
    NSMutableArray<ANEScheduledSurface *> *allocated = [NSMutableArray array];
    NSUInteger peak = 0;
    for (ANEScheduledSurface *surface in surfaces) {
        if (surface.role == ANEScheduledSurfaceRoleExternalInput ||
            surface.role == ANEScheduledSurfaceRoleConstant ||
            surface.role == ANEScheduledSurfaceRoleOutput) continue;
        NSUInteger size = alignUp(MAX((NSUInteger)1, surface.byteLength),
                                  target.sramGranuleBytes);
        NSUInteger candidate = 0;
        BOOL retry;
        do {
            retry = NO;
            for (ANEScheduledSurface *other in allocated) {
                NSUInteger otherSize = alignUp(MAX((NSUInteger)1, other.byteLength),
                    target.sramGranuleBytes);
                if (lifetimesOverlap(surface, other) &&
                    rangesOverlap(candidate, size, other.sramOffset, otherSize)) {
                    candidate = alignUp(other.sramOffset + otherSize,
                                        target.sramGranuleBytes);
                    retry = YES; break;
                }
            }
        } while (retry);
        [surface assignSRAMOffset:candidate bankCount:target.sramBankCount
                          granule:target.sramGranuleBytes];
        [allocated addObject:surface];
        peak = MAX(peak, candidate + size);
    }
    return peak;
}
@end
