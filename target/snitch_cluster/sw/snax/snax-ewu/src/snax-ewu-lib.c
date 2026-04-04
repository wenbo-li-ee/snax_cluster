// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "snax-ewu-lib.h"
#include "streamer_csr_addr_map.h"

void configure_streamer_a(uint32_t base_ptr_low, uint32_t base_ptr_high,
                          uint32_t spatial_stride, uint32_t temporal_bound,
                          uint32_t temporal_stride) {
    csrw_ss(BASE_PTR_READER_0_LOW, base_ptr_low);
    csrw_ss(BASE_PTR_READER_0_HIGH, base_ptr_high);
    csrw_ss(S_STRIDE_READER_0_0, spatial_stride);
    csrw_ss(T_BOUND_READER_0_0, temporal_bound);
    csrw_ss(T_STRIDE_READER_0_0, temporal_stride);
}

void configure_streamer_b(uint32_t base_ptr_low, uint32_t base_ptr_high,
                          uint32_t spatial_stride, uint32_t temporal_bound,
                          uint32_t temporal_stride) {
    csrw_ss(BASE_PTR_READER_1_LOW, base_ptr_low);
    csrw_ss(BASE_PTR_READER_1_HIGH, base_ptr_high);
    csrw_ss(S_STRIDE_READER_1_0, spatial_stride);
    csrw_ss(T_BOUND_READER_1_0, temporal_bound);
    csrw_ss(T_STRIDE_READER_1_0, temporal_stride);
}

void configure_streamer_o(uint32_t base_ptr_low, uint32_t base_ptr_high,
                          uint32_t spatial_stride, uint32_t temporal_bound,
                          uint32_t temporal_stride) {
    csrw_ss(BASE_PTR_WRITER_0_LOW, base_ptr_low);
    csrw_ss(BASE_PTR_WRITER_0_HIGH, base_ptr_high);
    csrw_ss(S_STRIDE_WRITER_0_0, spatial_stride);
    csrw_ss(T_BOUND_WRITER_0_0, temporal_bound);
    csrw_ss(T_STRIDE_WRITER_0_0, temporal_stride);
}

void start_streamer(void) { csrw_ss(STREAMER_START_CSR, 1); }

uint32_t read_busy_streamer(void) { return csrr_ss(STREAMER_BUSY_CSR); }

void configure_ewu(uint32_t mode, uint32_t data_len) {
    csrw_ss(EWU_RW_MODE, mode);
    csrw_ss(EWU_RW_DATALEN, data_len);
}

void start_ewu(void) { csrw_ss(EWU_RW_START, 1); }

uint32_t read_busy_ewu(void) { return csrr_ss(EWU_RO_BUSY); }
