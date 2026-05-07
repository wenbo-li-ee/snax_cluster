# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP = snax-versacore-int16x4-multishape-k8-8x4-mode1-pertoken-layout-explore

SELECT_LAYOUT ?= 0
SELECT_SHAPE ?= -1
RUN_MODE1 ?= 1
FAST_BUILD ?= 0
RISCV_CFLAGS += -DSELECT_LAYOUT=$(SELECT_LAYOUT)
RISCV_CFLAGS += -DSELECT_SHAPE=$(SELECT_SHAPE)
RISCV_CFLAGS += -DRUN_MODE1=$(RUN_MODE1)

INCDIRS = data \
          ../../snax/dual-versacore-swiglu/include

RISCV_LDFLAGS += ../../snax/dual-versacore-swiglu/build/snax-dual-versacore-swiglu-lib.o

SRCS = src/snax-versacore-int16x4-multishape-k8-8x4-mode1-pertoken-layout-explore.c

include ../common.mk
include ./data/Makefile

ifeq ($(FAST_BUILD),1)
RISCV_CFLAGS += -O1 -g0
endif

$(DEP): $(DATA_H)
