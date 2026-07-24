# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP = snax-versacore-int16x4-shape1-common16-weight-overlap
FAST_BUILD ?= 1
USE_XDMA ?= 0
CHUNK_COLS ?= 16

DATA_CFG ?= data/params.hjson
DATA_H := data/data.h

INCDIRS = data ../../snax/dual-versacore-swiglu/include
RISCV_LDFLAGS += ../../snax/dual-versacore-swiglu/build/snax-dual-versacore-swiglu-lib.o
ifneq ($(USE_XDMA),0)
INCDIRS += ../../snax/xdma/include
RISCV_LDFLAGS += ../../snax/xdma/build/snax-xdma-lib.o
endif
RISCV_CFLAGS += -DUSE_XDMA=$(USE_XDMA)
RISCV_CFLAGS += -DCHUNK_COLS=$(CHUNK_COLS)
SRCS = src/snax-versacore-int16x4-shape1-common16-weight-overlap.c

include ../common.mk

ifeq ($(FAST_BUILD),1)
RISCV_CFLAGS += -O1 -g0
endif

$(DATA_H): data/datagen.py $(DATA_CFG)
	data/datagen.py --swcfg $(DATA_CFG) \
	  --hwcfg $(ROOT)/target/snitch_cluster/$(CFG_OVERRIDE) > $@

$(DEP): $(DATA_H)
