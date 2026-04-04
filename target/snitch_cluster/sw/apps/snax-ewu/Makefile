# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP     = snax-ewu

INCDIRS = data
INCDIRS += ../../snax/snax-ewu/include

RISCV_LDFLAGS += ../../snax/snax-ewu/build/snax-ewu-lib.o

SRCS    = src/snax-ewu.c

include ../common.mk
include ./data/Makefile

$(DEP): $(DATA_H)
