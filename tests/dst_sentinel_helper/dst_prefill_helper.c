/* Minimal test-only dst-BO prefill helper with safety guards.
 * Uses ane.h types so offsets come from the compiler.
 * Exports are extern "C" for stable dlsym names.
 * No existing symbol is changed; production ABI preserved.
 */
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include "ane.h"

#ifdef __cplusplus
extern "C" {
#endif

void *__ane_dst_prefill(struct ane_nn *nn, const uint8_t byte,
                        const uint32_t idx)
{
	if (!nn) return NULL;
	const uint32_t bdx = dst_bdx(nn, idx);
	if (bdx >= TILE_COUNT) return NULL;
	struct ane_bo *bo = &nn->chans[bdx];
	if (!bo->map || bo->map == (void *)-1) return NULL;
	if (!bo->size || bo->size > 0x100000000ULL) return NULL;
	memset(bo->map, byte, bo->size);
	return bo->map;
}

uint64_t __ane_dst_surface_size(struct ane_nn *nn, const uint32_t idx)
{
	if (!nn) return 0;
	const uint32_t bdx = dst_bdx(nn, idx);
	if (bdx >= TILE_COUNT) return 0;
	return tile_size(nn, bdx);
}

void *__ane_dst_surface_map(struct ane_nn *nn, const uint32_t idx)
{
	if (!nn) return NULL;
	const uint32_t bdx = dst_bdx(nn, idx);
	if (bdx >= TILE_COUNT) return NULL;
	return nn->chans[bdx].map;
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
} /* extern "C" */
#endif
