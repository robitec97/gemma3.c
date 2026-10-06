# Makefile for gemma3.c
#
# Usage:
#   make                - Optimized CPU build (native SIMD + thread pool)   [default]
#   make mps            - Metal GPU build (macOS Apple Silicon), CPU fallback included
#   make blas           - CPU build using BLAS sgemm for prompt processing
#                         (Accelerate on macOS, OpenBLAS elsewhere)
#   make portable       - CPU build without -march/-mcpu=native (for distributing binaries)
#   make debug          - Debug symbols, no optimization
#   make asan           - AddressSanitizer + UBSan build
#   make test           - Build and run unit tests (no model needed)
#   make test-model     - Run model-dependent tests (MODEL=path, default ./gemma-3-4b-it)
#   make bench          - Build the end-to-end benchmark (./gemma3-bench)
#   make bench-kernels  - Build and run kernel micro-benchmarks (no model needed)
#   make example        - Build the library API example (./gemma3-example)
#   make clean          - Remove all build artifacts
#
# Set EXTRA_CFLAGS=-DGEMMA3_NO_SIMD to build the plain C kernels only.

# --- Configuration ---

CC ?= cc
TARGET ?= gemma3
BUILD_DIR ?= build
MODEL ?= gemma-3-4b-it

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

LIB_SRCS = gemma3.c \
           gemma3_kernels.c \
           gemma3_safetensors.c \
           gemma3_threads.c \
           gemma3_tokenizer.c \
           gemma3_transformer.c

CFLAGS_BASE = -Wall -Wextra -Wpedantic -std=c11 -MMD -MP -I. -D_DEFAULT_SOURCE -D_DARWIN_C_SOURCE
LDFLAGS_BASE = -lm -lpthread

# Native SIMD: -mcpu=native on arm64 (NEON is always on), -march=native on x86-64
ifeq ($(UNAME_M),arm64)
    NATIVE_FLAGS = -mcpu=native
else ifeq ($(UNAME_M),aarch64)
    NATIVE_FLAGS = -mcpu=native
else
    NATIVE_FLAGS = -march=native
endif

# Default mode (set by the convenience targets below)
MODE ?= native

CFLAGS = $(CFLAGS_BASE)
LDFLAGS = $(LDFLAGS_BASE)
SRCS = $(LIB_SRCS)
SRCS_M =

# Extra user flags, e.g. EXTRA_CFLAGS=-DGEMMA3_NO_SIMD to test the scalar kernels
CFLAGS += $(EXTRA_CFLAGS)

ifeq ($(MODE),native)
    CFLAGS += -O3 -DNDEBUG $(NATIVE_FLAGS)
endif
ifeq ($(MODE),portable)
    CFLAGS += -O3 -DNDEBUG
endif
ifeq ($(MODE),debug)
    CFLAGS += -g -O0 -DDEBUG
endif
ifeq ($(MODE),asan)
    CFLAGS += -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer
    LDFLAGS += -fsanitize=address,undefined
endif
ifeq ($(MODE),blas)
    CFLAGS += -O3 -DNDEBUG $(NATIVE_FLAGS) -DUSE_BLAS
    ifeq ($(UNAME_S),Darwin)
        CFLAGS += -DACCELERATE_NEW_LAPACK
        LDFLAGS += -framework Accelerate
    else
        LDFLAGS += -lopenblas
    endif
endif
ifeq ($(MODE),mps)
    CFLAGS += -O3 -DNDEBUG $(NATIVE_FLAGS) -DUSE_MPS
    LDFLAGS += -framework Metal -framework Foundation
    SRCS_M += gemma3_metal.m
endif

OBJS_LIB = $(patsubst %.c, $(BUILD_DIR)/$(MODE)/%.o, $(SRCS)) \
           $(patsubst %.m, $(BUILD_DIR)/$(MODE)/%.o, $(SRCS_M))
OBJ_MAIN = $(BUILD_DIR)/$(MODE)/main.o

# --- Convenience Targets ---

.PHONY: all native portable debug asan blas mps fast threads blas-threads mps-threads \
        build_core test test-kernels test-model bench bench-kernels example clean help

all: native

native:
	@$(MAKE) --no-print-directory build_core MODE=native

portable:
	@$(MAKE) --no-print-directory build_core MODE=portable

debug:
	@$(MAKE) --no-print-directory build_core MODE=debug

asan:
	@$(MAKE) --no-print-directory build_core MODE=asan

blas:
	@$(MAKE) --no-print-directory build_core MODE=blas

mps:
	@$(MAKE) --no-print-directory build_core MODE=mps CC=clang

# Older target names (threads are now always enabled)
fast threads: native
blas-threads: blas
mps-threads: mps

build_core: $(TARGET)

$(TARGET): $(OBJS_LIB) $(OBJ_MAIN)
	@echo "Linking $(TARGET) [$(MODE)]"
	@$(CC) $(OBJS_LIB) $(OBJ_MAIN) -o $(TARGET) $(LDFLAGS)

$(BUILD_DIR)/$(MODE)/%.o: %.c
	@mkdir -p $(dir $@)
	@echo "CC $<"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/$(MODE)/%.o: %.m
	@mkdir -p $(dir $@)
	@echo "CC $<"
	@$(CC) $(CFLAGS) -Wno-overlength-strings -fobjc-arc -c $< -o $@

-include $(wildcard $(BUILD_DIR)/*/*.d)

# --- Tests and benchmarks ---

# Library objects of the current MODE (tests/benches link against these)
lib_objs: $(OBJS_LIB)

gemma3-test: tests/test_kernels.c $(OBJS_LIB)
	@echo "Linking gemma3-test [$(MODE)]"
	@$(CC) $(CFLAGS) tests/test_kernels.c $(OBJS_LIB) -o $@ $(LDFLAGS)

gemma3-bench: bench/bench_e2e.c $(OBJS_LIB)
	@echo "Linking gemma3-bench [$(MODE)]"
	@$(CC) $(CFLAGS) bench/bench_e2e.c $(OBJS_LIB) -o $@ $(LDFLAGS)

gemma3-example: examples/simple.c $(OBJS_LIB)
	@echo "Linking gemma3-example [$(MODE)]"
	@$(CC) $(CFLAGS) examples/simple.c $(OBJS_LIB) -o $@ $(LDFLAGS)

gemma3-bench-kernels: bench/bench_kernels.c $(OBJS_LIB)
	@echo "Linking gemma3-bench-kernels [$(MODE)]"
	@$(CC) $(CFLAGS) bench/bench_kernels.c $(OBJS_LIB) -o $@ $(LDFLAGS)

test test-kernels:
	@$(MAKE) --no-print-directory gemma3-test
	./gemma3-test

test-model:
	@$(MAKE) --no-print-directory build_core
	@$(MAKE) --no-print-directory gemma3-test-tokenizer
	./gemma3-test-tokenizer $(MODEL)/tokenizer.model tests
	@$(MAKE) --no-print-directory gemma3-test-cache
	./gemma3-test-cache $(MODEL)
	./tests/test_e2e.sh ./$(TARGET) $(MODEL)

gemma3-test-cache: tests/test_cache.c $(OBJS_LIB)
	@echo "Linking gemma3-test-cache [$(MODE)]"
	@$(CC) $(CFLAGS) tests/test_cache.c $(OBJS_LIB) -o $@ $(LDFLAGS)

gemma3-test-tokenizer: tests/test_tokenizer.c $(OBJS_LIB)
	@echo "Linking gemma3-test-tokenizer [$(MODE)]"
	@$(CC) $(CFLAGS) tests/test_tokenizer.c $(OBJS_LIB) -o $@ $(LDFLAGS)

bench:
	@$(MAKE) --no-print-directory gemma3-bench

example:
	@$(MAKE) --no-print-directory gemma3-example

bench-kernels:
	@$(MAKE) --no-print-directory gemma3-bench-kernels
	./gemma3-bench-kernels

clean:
	rm -rf $(TARGET) $(BUILD_DIR) gemma3-test gemma3-test-tokenizer gemma3-bench gemma3-bench-kernels gemma3-example gemma3-test-cache

help:
	@echo "Build targets:"
	@echo "  make               Optimized CPU build: native SIMD (NEON/AVX2) + threads [default]"
	@echo "  make mps           Metal GPU build for Apple Silicon (CPU fallback included)"
	@echo "  make blas          CPU build with BLAS prompt processing (Accelerate / OpenBLAS)"
	@echo "  make portable      CPU build without native CPU tuning"
	@echo "  make debug         Debug build"
	@echo "  make asan          AddressSanitizer + UndefinedBehaviorSanitizer build"
	@echo ""
	@echo "Testing and benchmarking:"
	@echo "  make test          Unit tests for kernels, sampler and thread pool (no model)"
	@echo "  make test-model    Tokenizer golden tests + end-to-end checks (needs model)"
	@echo "  make bench         Build ./gemma3-bench (end-to-end tokens/s, needs model)"
	@echo "  make bench-kernels Kernel micro-benchmarks (GB/s, GFLOP/s; no model)"
	@echo "  make example       Build ./gemma3-example (library API demo)"
	@echo ""
	@echo "Variables: MODE=native|portable|debug|asan|blas|mps  MODEL=<dir>  CC=<compiler>"
