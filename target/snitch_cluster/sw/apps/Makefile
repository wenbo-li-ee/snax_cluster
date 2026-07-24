# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

SUBDIRS  = lto
SUBDIRS += nop
# Tests below don't work with the generic runtime
ifneq ($(filter $(SELECT_RUNTIME),rtl banshee),)
SUBDIRS += blas/axpy
SUBDIRS += blas/gemm
SUBDIRS += dnn/batchnorm
SUBDIRS += dnn/conv2d
SUBDIRS += dnn/fusedconv
SUBDIRS += dnn/gelu
SUBDIRS += dnn/gemm
SUBDIRS += dnn/layernorm
SUBDIRS += dnn/linear
SUBDIRS += dnn/maxpool
SUBDIRS += dnn/softmax
endif
SUBDIRS += montecarlo/pi_estimation
SUBDIRS += snax-mac
SUBDIRS += snax-mac-simple/untiled
SUBDIRS += snax-mac-simple/tiled
ifeq ($(CFG_OVERRIDE), cfg/snax_alu_cluster.hjson)
SUBDIRS += snax-alu
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_multi_alu_cluster.hjson)
SUBDIRS += snax-alu
SUBDIRS += snax-multi-alu
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_KUL_cluster.hjson)
SUBDIRS += snax-xdma-memset
SUBDIRS += snax-xdma-reshape-comparison
SUBDIRS += snax-xdma-transpose
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_compiler_test_cluster.hjson)
SUBDIRS += snax-xdma-memset
SUBDIRS += snax-xdma-maxpool
SUBDIRS += snax-xdma-transpose
SUBDIRS += snax-xdma-reshape-comparison
SUBDIRS += snax-xdma-elementwise-add
SUBDIRS += snax-xdma-rescale-down
SUBDIRS += snax-xdma-rescale-up
SUBDIRS += snax-xdma-avgpool
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_versacore_to_cluster.hjson)
SUBDIRS += snax-versacore-to-matmul
SUBDIRS += snax-versacore-to-matmul-profile
SUBDIRS += snax-xdma-memset
SUBDIRS += snax-xdma-maxpool
SUBDIRS += snax-xdma-transpose
SUBDIRS += snax-xdma-reshape-comparison
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_versacore_dse_cluster.hjson)
SUBDIRS += snax-versacore-dse-matmul-profile
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_dual_versacore_swiglu_cluster.hjson)
SUBDIRS += snax-dual-versacore-swiglu-test
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_dual_versacore_int16x4_cluster.hjson)
SUBDIRS += snax-versacore-int16x4-sanity
SUBDIRS += snax-versacore-int16x4-scale16-batch
SUBDIRS += snax-versacore-int16x4-scale16-pingpong
SUBDIRS += snax-versacore-int16x4-fullsize-batch
SUBDIRS += snax-versacore-int16x4-fullsize-pingpong
SUBDIRS += snax-versacore-int16x4-swiglu-m1-batch
SUBDIRS += snax-versacore-int16x4-swiglu-m1-pingpong
SUBDIRS += snax-versacore-int16x4-swiglu-m4-batch
SUBDIRS += snax-versacore-int16x4-swiglu-m4-pingpong
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_dual_versacore_int16x4_rebalanced_cluster.hjson)
SUBDIRS += snax-versacore-int16x4-swiglu-m1-batch
SUBDIRS += snax-versacore-int16x4-swiglu-m1-pingpong
SUBDIRS += snax-versacore-int16x4-swiglu-m4-batch
SUBDIRS += snax-versacore-int16x4-swiglu-m4-pingpong
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_dual_versacore_int16x4_multidim_spatial.hjson)
SUBDIRS += snax-versacore-int16x4-multishape-batch
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_dual_versacore_int16x4_multidim_spatial_16x2.hjson)
SUBDIRS += snax-versacore-int16x4-multishape-16x2
SUBDIRS += snax-versacore-int16x4-multishape-16x2-mode0-single-writer
SUBDIRS += snax-versacore-int16x4-multishape-16x2-a-padded
SUBDIRS += snax-versacore-int16x4-multishape-16x2-layout-explore

endif
ifeq ($(CFG_OVERRIDE), cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson)
SUBDIRS += snax-versacore-int16x4-multishape-k8-8x4-mode0-single-writer
SUBDIRS += snax-versacore-int16x4-multishape-k8-8x4-mode1-pertoken
SUBDIRS += snax-versacore-int16x4-multishape-k8-8x4-l27-mode1-pertoken
SUBDIRS += snax-versacore-int16x4-multishape-k8-8x4-layout-explore
SUBDIRS += snax-versacore-int16x4-multishape-k8-8x4-mode1-pertoken-layout-explore
SUBDIRS += snax-versacore-int16x4-multishape-k8-8x4-mode1-contiguous-l15
SUBDIRS += snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15
SUBDIRS += snax-versacore-int16x4-moe-small-k8-highgran-layout-search
SUBDIRS += snax-versacore-int16x4-multishape-k8-8x4-dma-overlap
SUBDIRS += snax-versacore-int16x4-multishape-k8-common16-chunked
SUBDIRS += snax-versacore-int16x4-shape1-common16-weight-overlap
SUBDIRS += snax-xdma-gather-scatter
SUBDIRS += snax-dma-benchmark-4lane
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane_a4_b2_layout_v144.hjson)
SUBDIRS += snax-versacore-int16x4-multishape-k8-8x4-v144-layout
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane_bank_partition_g1.hjson)
SUBDIRS += snax-versacore-int16x4-multishape-k8-8x4-bank-partition-pingpong
endif
SUBDIRS += snax-multi-cluster-playground
ifeq ($(CFG_OVERRIDE), cfg/snax_hypercorex_cluster.hjson)
SUBDIRS += snax-hypercorex/test-csr
SUBDIRS += snax-hypercorex/char-recog
SUBDIRS += snax-hypercorex/high-dim-bind
SUBDIRS += snax-hypercorex/am-search
SUBDIRS += snax-hypercorex/isolet-recog
SUBDIRS += snax-hypercorex/isolet-recog-long
SUBDIRS += snax-hypercorex/ucihar-recog
SUBDIRS += snax-hypercorex/ucihar-recog-long
SUBDIRS += snax-hypercorex/digit-recog
SUBDIRS += snax-hypercorex/digit-recog-long
SUBDIRS += snax-hypercorex/dna-recog
SUBDIRS += snax-hypercorex/dna-recog-long
SUBDIRS += snax-hypercorex/lang-recog
SUBDIRS += snax-hypercorex/lang-recog-long
SUBDIRS += snax-hypercorex/vec-compress
endif


.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET)
