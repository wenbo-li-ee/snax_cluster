# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP = snax-versacore-int16x4-multishape-k8-8x4-v144-layout

SELECT_LAYOUT ?= 0
NUM_LAYOUTS_TO_RUN ?= 1
SELECT_SHAPE ?= -1
RUN_MODE1 ?= 1
FAST_BUILD ?= 0
RISCV_CFLAGS += -DSELECT_LAYOUT=$(SELECT_LAYOUT)
RISCV_CFLAGS += -DNUM_LAYOUTS_TO_RUN=$(NUM_LAYOUTS_TO_RUN)
RISCV_CFLAGS += -DSELECT_SHAPE=$(SELECT_SHAPE)
RISCV_CFLAGS += -DRUN_MODE1=$(RUN_MODE1)

INCDIRS = data \
          ../../snax/dual-versacore-swiglu/include

RISCV_LDFLAGS += ../../snax/dual-versacore-swiglu/build/snax-dual-versacore-swiglu-lib.o

# Reuse the validated V144 execution/checking harness.  This app owns its
# workload parameters and emits a single full-size V144 layout in data/data.h.
SRCS = ../snax-versacore-int16x4-moe-small-k8-highgran-layout-search/src/snax-versacore-int16x4-moe-small-k8-highgran-layout-search.c

include ../common.mk
include ./data/Makefile

ifeq ($(FAST_BUILD),1)
RISCV_CFLAGS += -O1 -g0
endif

$(DEP): $(DATA_H)
