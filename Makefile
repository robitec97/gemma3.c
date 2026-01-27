# Makefile for gemma3.c
#
# Build targets:
#   make          - Build with default settings
#   make debug    - Build with debug symbols
#   make fast     - Build with aggressive optimizations
#   make blas     - Build with OpenBLAS acceleration
#   make clean    - Remove build artifacts

# Compiler
CC ?= gcc

# Project name
TARGET = gemma3

# Source files
SRCS = gemma3.c \
       gemma3_kernels.c \
       gemma3_safetensors.c \
       gemma3_tokenizer.c \
       gemma3_transformer.c \
       main.c

# Object files
OBJS = $(SRCS:.c=.o)

# Header files
HDRS = gemma3.h gemma3_kernels.h

# Compiler flags
CFLAGS_BASE = -Wall -Wextra -Wpedantic -std=c11
CFLAGS_RELEASE = -O3 -DNDEBUG
CFLAGS_DEBUG = -g -O0 -DDEBUG
CFLAGS_FAST = -O3 -march=native -ffast-math -DNDEBUG

# Linker flags
LDFLAGS = -lm

# Platform detection
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    # macOS specific flags
    CFLAGS_BASE += -D_DARWIN_C_SOURCE
    # Uncomment to enable Metal support (requires gemma3_metal.m)
    # LDFLAGS += -framework Foundation -framework Metal -framework MetalKit
endif

ifeq ($(UNAME_S),Linux)
    # Linux specific flags
    CFLAGS_BASE += -D_GNU_SOURCE
    LDFLAGS += -lpthread
endif

# Default build
CFLAGS = $(CFLAGS_BASE) $(CFLAGS_RELEASE)

# Targets
.PHONY: all debug fast blas clean help

all: $(TARGET)

debug: CFLAGS = $(CFLAGS_BASE) $(CFLAGS_DEBUG)
debug: $(TARGET)

fast: CFLAGS = $(CFLAGS_BASE) $(CFLAGS_FAST)
fast: $(TARGET)

blas: CFLAGS = $(CFLAGS_BASE) $(CFLAGS_FAST) -DUSE_BLAS
blas: LDFLAGS += -lopenblas
blas: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

# Help target
help:
	@echo "Gemma 3 C Inference - Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all     - Build with release optimizations (default)"
	@echo "  debug   - Build with debug symbols (-g -O0)"
	@echo "  fast    - Build with aggressive optimizations (-O3 -march=native -ffast-math)"
	@echo "  blas    - Build with OpenBLAS acceleration (requires libopenblas-dev)"
	@echo "  clean   - Remove build artifacts"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  CC      - C compiler (default: gcc)"
	@echo ""
	@echo "Usage:"
	@echo "  make              # Build release version"
	@echo "  make debug        # Build debug version"
	@echo "  make fast         # Build optimized for local CPU"
	@echo "  make blas         # Build with OpenBLAS support"
	@echo "  make clean        # Clean build files"
	@echo ""
	@echo "After building, run:"
	@echo "  ./gemma3 -m <model_dir> -p \"Your prompt\""
	@echo "  ./gemma3 -m <model_dir> -i  # Interactive mode"

# Dependencies (auto-generated would be better, but keeping it simple)
gemma3.o: gemma3.c gemma3.h gemma3_kernels.h
gemma3_kernels.o: gemma3_kernels.c gemma3_kernels.h
gemma3_safetensors.o: gemma3_safetensors.c gemma3.h gemma3_kernels.h
gemma3_tokenizer.o: gemma3_tokenizer.c gemma3.h
gemma3_transformer.o: gemma3_transformer.c gemma3.h gemma3_kernels.h
main.o: main.c gemma3.h
