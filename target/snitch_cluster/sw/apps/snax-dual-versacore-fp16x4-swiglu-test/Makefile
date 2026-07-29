# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP = snax-dual-versacore-fp16x4-swiglu-test

INCDIRS = data \
          ../../snax/dual-versacore-swiglu/include

RISCV_LDFLAGS += ../../snax/dual-versacore-swiglu/build/snax-dual-versacore-swiglu-lib.o

SRCS = src/snax-dual-versacore-fp16x4-swiglu-test.c

include ../common.mk
include ./data/Makefile

$(DEP): $(DATA_H)
