# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP = snax-versacore-int16x4-multishape-k8-8x4-l27-mode1-pertoken

INCDIRS = data \
          ../../snax/dual-versacore-swiglu/include

RISCV_LDFLAGS += ../../snax/dual-versacore-swiglu/build/snax-dual-versacore-swiglu-lib.o

SRCS = src/snax-versacore-int16x4-multishape-k8-8x4-l27-mode1-pertoken.c

include ../common.mk
include ./data/Makefile

$(DEP): $(DATA_H)
