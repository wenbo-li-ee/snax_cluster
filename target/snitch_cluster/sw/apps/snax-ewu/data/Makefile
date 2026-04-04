# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR))

LENGTH ?= 64
INPUT_WIDTH ?= 32
OUTPUT_WIDTH ?= 32
NUM_PE ?= 4

DATAGEN_PY = $(DATA_DIR)/datagen.py
DATA_H = $(DATA_DIR)/data.h
DATA_STAMP = $(DATA_DIR)/.data-$(LENGTH)-$(INPUT_WIDTH)-$(OUTPUT_WIDTH)-$(NUM_PE).stamp

.PHONY: clean clean-data

clean-data:
	rm -f $(DATA_H) $(DATA_DIR)/.data-*.stamp

clean: clean-data

$(DATA_STAMP):
	touch $@

$(DATA_H): $(DATAGEN_PY) $(DATA_STAMP)
	python3 $(DATAGEN_PY) --length=$(LENGTH) --input-width=$(INPUT_WIDTH) --output-width=$(OUTPUT_WIDTH) --num-pe=$(NUM_PE) > $@
