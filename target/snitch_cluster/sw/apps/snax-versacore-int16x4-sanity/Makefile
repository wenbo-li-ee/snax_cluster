# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP = snax-versacore-int16x4-sanity

INCDIRS = data \
          ../../snax/dual-versacore-swiglu/include

# Link against the shared dual-versacore-swiglu library
RISCV_LDFLAGS += ../../snax/dual-versacore-swiglu/build/snax-dual-versacore-swiglu-lib.o

SRCS = src/snax-versacore-int16x4-sanity.c

include ../common.mk
include ./data/Makefile

$(DEP): $(DATA_H)
