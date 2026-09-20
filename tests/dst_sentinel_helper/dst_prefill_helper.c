/* Minimal test-only dst-BO prefill helper with safety guards.
 * Uses ane.h header types so all offsets come from the compiler.
 * Exports are extern "C" for stable dlsym names.
 * No existing symbol is changed; production ABI preserved.
 * Safety: ALL bounds checks precede pointer dereference.
 */
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include "ane.h"

#ifdef __cplusplus
extern "C" {
#endif

int __ane_dst_prefill(struct ane_nn *nn, const uint8_t byte,
                      const uint32_t idx)
{
	if (!nn) return -1;
	const uint32_t bdx = dst_bdx(nn, idx);
	if (bdx >= TILE_COUNT) return -1;
	struct ane_bo *bo = &nn->chans[bdx];
	if (!bo->map || bo->map == (void *)-1) return -1;
	if (!bo->size || bo->size > 0x100000000ULL) return -1;
	memset(bo->map, byte, bo->size);
	return 0;
}

uint64_t __ane_dst_surface_size(struct ane_nn *nn, const uint32_t idx)
{
	if (!nn) return 0;
	const uint32_t bdx = dst_bdx(nn, idx);
	if (bdx >= TILE_COUNT) return 0;
	struct ane_bo *bo = &nn->chans[bdx];
	if (!bo->map || bo->map == (void *)-1) return 0;
	if (!bo->size || bo->size > 0x100000000ULL) return 0;
	return bo->size;
}

void *__ane_dst_surface_map(struct ane_nn *nn, const uint32_t idx)
{
	if (!nn) return NULL;
	const uint32_t bdx = dst_bdx(nn, idx);
	if (bdx >= TILE_COUNT) return NULL;
	struct ane_bo *bo = &nn->chans[bdx];
	if (!bo->map || bo->map == (void *)-1) return NULL;
	return bo->map;
}

uint8_t __ane_dst_channel(struct ane_nn *nn, const uint32_t idx)
{
	if (!nn) return 0xFF;
	return dst_bdx(nn, idx);
}

uint8_t __ane_src_channel(struct ane_nn *nn, const uint32_t idx)
{
	if (!nn) return 0xFF;
	return src_bdx(nn, idx);
}

#ifdef __cplusplus
}
#endif
