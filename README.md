# gemma3.c

`gemma3.c` is a **from-scratch inference engine** for the *Gemma 3 4B IT* model, written in pure C.

## Highlights

* **100% Pure C (C11)** - zero external dependencies
* **Full Gemma 3 architecture** - GQA, hybrid attention, SwiGLU
* **Metal GPU acceleration** - Apple Silicon via Metal Performance Shaders
* **Memory-mapped weights** - BF16 SafeTensors via `mmap`
* **Native SentencePiece tokenizer** - 262K vocab
* **Streaming output** - token-by-token callbacks
* **Interactive chat mode** - multi-turn conversations
* **CLI + Library API**
* **Multi-threaded inference** - thread pool for parallel computation
* **OpenBLAS support** (optional) - BLAS-accelerated matrix operations
* **Linux/macOS native**, Windows via **WSL** (recommended) or **MinGW**

---

## Quick Start

> POSIX-first: native on Linux/macOS. On Windows use **WSL** or **MinGW** (no `mmap`).

### 1. Download model

```bash
export HF_TOKEN=your_token_here
pip install huggingface_hub
python download_model.py
```

### 2. Build

```bash
make              # CPU release build
make mps          # Metal GPU (macOS Apple Silicon)
make blas-threads # OpenBLAS + threads (best CPU performance)
```

### 3. Run

```bash
# Single prompt
./gemma3 -m ./gemma-3-4b-it -p "Explain quantum computing simply."

# Interactive chat
./gemma3 -m ./gemma-3-4b-it -i
```

> **OpenBLAS builds:** `make blas` and `make blas-threads` require OpenBLAS:
> - Linux: `sudo apt install libopenblas-dev`
> - macOS: `brew install openblas`

---

## Build Targets

```bash
make              # Release build (default)
make debug        # Debug symbols
make fast         # Native optimizations (-march=native -ffast-math)
make threads      # Thread pool parallelization
make blas         # OpenBLAS acceleration (requires libopenblas)
make blas-threads # OpenBLAS + threads (best CPU performance)
make mps          # Metal GPU acceleration (macOS Apple Silicon)
make mps-threads  # Metal GPU + thread pool fallback
make clean        # Remove build artifacts
make help         # Show all targets
```

---

## CLI Options

### Basic

```
-m, --model <path>      Model directory (required)
-p, --prompt <text>     Input prompt
-i, --interactive       Interactive chat mode
-s, --system <text>     System prompt (default: "You are a helpful assistant.")
-c, --context <n>       Context size (default: 8192)
-v, --verbose           Verbose output
-h, --help              Show help
```

### Generation

```
-n, --max-tokens <n>    Max tokens to generate (default: 512)
-t, --temperature <f>   Sampling temperature (default: 0.7)
-k, --top-k <n>         Top-k sampling (default: 50, 0=disabled)
--top-p <f>             Top-p sampling (default: 0.9)
--seed <n>              Random seed (-1 for random)
--greedy                Force greedy decoding (deterministic)
```

### Debug / Utility

```
--verbose-tokens        Print token IDs during generation (to stderr)
--tokenize              Tokenize prompt and print token IDs
--detokenize            Detokenize comma-separated IDs from prompt
--logits                Run single forward pass and show top-20 logits
```

### Examples

```bash
# Standard generation
./gemma3 -m ./gemma-3-4b-it -p "Hello, how are you?"

# Custom system prompt
./gemma3 -m ./gemma-3-4b-it -p "Write a poem" -s "You are a poet."

# Greedy decoding with token IDs
./gemma3 -m ./gemma-3-4b-it -p "Say OK" --greedy --verbose-tokens

# Interactive chat
./gemma3 -m ./gemma-3-4b-it -i
./gemma3 -m ./gemma-3-4b-it -i -s "You are a pirate."

# Tokenize a string
./gemma3 -m ./gemma-3-4b-it --tokenize -p "Hello, world!"

# Inspect logits
./gemma3 -m ./gemma-3-4b-it --logits -p "The capital of France is"
```

---

## Interactive Mode

Start with `-i`:

```bash
./gemma3 -m ./gemma-3-4b-it -i -s "You are a helpful assistant."
```

**Commands:**

| Command        | Action                      |
| -------------- | --------------------------- |
| `quit` / `exit`| End the session             |
| `clear`        | Reset conversation history  |
| Ctrl+C         | Cancel current generation   |
| Ctrl+D         | Exit (EOF)                  |

Conversations are multi-turn - the model sees the full chat history. Use `clear` to start fresh without restarting.

---

## Library Example

```c
gemma3_ctx *ctx = gemma3_load_dir("./gemma-3-4b-it");

gemma3_gen_params params = gemma3_default_params();
char *out = gemma3_generate(ctx, "Hello!", &params, NULL, NULL);
printf("%s\n", out);
free(out);

gemma3_free(ctx);
```

---

## Model Specs

| Param          | Value              |
| -------------- | ------------------ |
| Vocab          | 262,208            |
| Layers         | 34                 |
| Hidden dim     | 2,560              |
| Intermediate   | 10,240             |
| Heads          | 8 (4 KV, GQA)     |
| Head dim       | 256                |
| Context        | 128K               |
| Attention      | 5 local : 1 global |
| Sliding window | 1,024              |
| RoPE theta     | 10K local / 1M global |

---

## Memory

* Weights: ~8 GB on disk (BF16)
* Runtime RAM: **~3 GB total**

The KV cache scales with context size. Reduce memory with a smaller context:

```bash
./gemma3 -m ./gemma-3-4b-it -c 512 -p "Hello"
```

---

For better CPU performance:

```bash
make fast          # Native optimizations (single-threaded)
make threads       # Multi-core parallelization
make blas-threads  # Best CPU performance (requires OpenBLAS)
```

### GPU (Metal)

On macOS with Apple Silicon, use the Metal backend for GPU-accelerated matrix operations:

```bash
make mps           # Metal GPU
make mps-threads   # Metal GPU + thread pool fallback
```

---

## Limitations

* Text only (no vision)
* No quantization (BF16 only)

---

## License

MIT License.
Model weights under Google's Gemma license.
