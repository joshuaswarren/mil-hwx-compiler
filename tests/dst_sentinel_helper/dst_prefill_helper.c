/* Minimal test-only dst-BO prefill helper.
 *
 * Uses the ane.h header types (struct ane_nn, struct ane_bo, struct ane_bind)
 * so all offsets come from the compiler, not hard-coded ctypes guesses.
 * Preserves the production ABI: adds only __ane_dst_prefill and
 * __ane_dst_info; no existing symbol is changed.
 *
 * Built as a separate shared library (not the production libane).
 */
#include <string.h>
#include <stdint.h>
#include "ane.h"

void *__ane_dst_prefill(struct ane_nn *nn, const uint8_t byte,
                        const uint32_t idx)
{
	const uint32_t bdx = dst_bdx(nn, idx);
	struct ane_bo *bo = &nn->chans[bdx];
	memset(bo->map, byte, bo->size);
	return bo->map;
}

uint64_t __ane_dst_surface_size(struct ane_nn *nn, const uint32_t idx)
{
	return tile_size(nn, dst_bdx(nn, idx));
}

void *__ane_dst_surface_map(struct ane_nn *nn, const uint32_t idx)
{
	return nn->chans[dst_bdx(nn, idx)].map;
}

uint8_t __ane_dst_channel(struct ane_nn *nn, const uint32_t idx)
{
	return dst_bdx(nn, idx);
}

uint8_t __ane_src_channel(struct ane_nn *nn, const uint32_t idx)
{
	return src_bdx(nn, idx);
}
