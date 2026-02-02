# Makefile for gemma3.c
#
# Usage:
#   make                - Build release (default)
#   make debug          - Build with debug symbols
#   make fast           - Build with aggressive optimizations
#   make blas           - Build with OpenBLAS
#   make threads        - Build with thread pool
#   make blas-threads   - Build with OpenBLAS + thread pool
#   make clean          - Remove all build artifacts

# --- Configuration ---

CC ?= gcc
TARGET ?= gemma3
BUILD_DIR ?= build

# Base Sources
SRCS_BASE = gemma3.c \
            gemma3_kernels.c \
            gemma3_safetensors.c \
            gemma3_tokenizer.c \
            gemma3_transformer.c \
            main.c

# Base Flags
CFLAGS_BASE    = -Wall -Wextra -Wpedantic -std=c11 -MMD -MP
CFLAGS_RELEASE = -O3 -DNDEBUG
CFLAGS_DEBUG   = -g -O0 -DDEBUG
CFLAGS_FAST    = -O3 -march=native -ffast-math -DNDEBUG

LDFLAGS_BASE   = -lm

# --- Mode Logic (The Core Fix) ---

# Default mode is release
MODE ?= release

# Initialize variables based on defaults
CFLAGS = $(CFLAGS_BASE)
LDFLAGS = $(LDFLAGS_BASE)
SRCS = $(SRCS_BASE)

# Apply Mode-Specific configurations
# This runs only when the recursive make is called with MODE set

ifeq ($(MODE),release)
    CFLAGS += $(CFLAGS_RELEASE)
endif

ifeq ($(MODE),debug)
    CFLAGS += $(CFLAGS_DEBUG)
endif

ifeq ($(MODE),fast)
    CFLAGS += $(CFLAGS_FAST)
endif

# Check for BLAS in the mode string
ifneq (,$(findstring blas,$(MODE)))
    CFLAGS += $(CFLAGS_FAST) -DUSE_BLAS
    LDFLAGS += -lopenblas
endif

# Check for THREADS in the mode string
ifneq (,$(findstring threads,$(MODE)))
    CFLAGS += $(CFLAGS_FAST) -DUSE_THREADS
    SRCS += gemma3_threads.c
endif

# Calculate Objects based on the current MODE
# Result: build/blas-threads/gemma3.o, etc.
OBJS = $(patsubst %.c, $(BUILD_DIR)/$(MODE)/%.o, $(SRCS))

# --- Convenience Targets ---
# These targets just re-run make with a specific MODE

.PHONY: all debug fast blas threads blas-threads clean help

all:
	@$(MAKE) --no-print-directory build_core MODE=release

debug:
	@$(MAKE) --no-print-directory build_core MODE=debug

fast:
	@$(MAKE) --no-print-directory build_core MODE=fast

blas:
	@$(MAKE) --no-print-directory build_core MODE=blas

threads:
	@$(MAKE) --no-print-directory build_core MODE=threads

blas-threads:
	@$(MAKE) --no-print-directory build_core MODE=blas-threads

# --- The Real Build Target ---

# This target does the actual work. 
# It expects MODE to be set correctly by the calls above.
build_core: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking $(TARGET) [$(MODE)]..."
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

# The Compilation Rule
# Now we can explicitly use $(MODE) in the path because it is constant for this run
$(BUILD_DIR)/$(MODE)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Include dependencies
-include $(wildcard $(BUILD_DIR)/*/*.d)

clean:
	rm -rf $(TARGET) $(BUILD_DIR)

help:
	@echo "Available targets:"
	@echo "  make              : Release build"
	@echo "  make debug        : Debug build"
	@echo "  make fast         : Native optimizations"
	@echo "  make blas         : OpenBLAS"
	@echo "  make threads      : Thread pool"
	@echo "  make blas-threads : OpenBLAS + Threads"