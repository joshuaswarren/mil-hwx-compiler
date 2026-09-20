#ifndef __DST_PREFILL_HELPER_H__
#define __DST_PREFILL_HELPER_H__
#include <stdint.h>
#include "ane.h"
#ifdef __cplusplus
extern "C" {
#endif
void *__ane_dst_prefill(struct ane_nn *nn, const uint8_t byte, const uint32_t idx);
uint64_t __ane_dst_surface_size(struct ane_nn *nn, const uint32_t idx);
void *__ane_dst_surface_map(struct ane_nn *nn, const uint32_t idx);
uint8_t __ane_dst_channel(struct ane_nn *nn, const uint32_t idx);
uint8_t __ane_src_channel(struct ane_nn *nn, const uint32_t idx);
#ifdef __cplusplus
}
#endif
#endif
