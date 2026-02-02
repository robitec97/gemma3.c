# gemma3.c

`gemma3.c` is a **from‑scratch CPU inference engine** for the *Gemma 3 4B IT* model.

## ✨ Highlights

* ⚙️ **100% Pure C (C11)** – zero external dependencies
* 🧠 **Full Gemma 3 architecture** – GQA, hybrid attention, SwiGLU
* 🗺️ **Memory‑mapped weights** – BF16 SafeTensors via `mmap`
* 🔤 **Native SentencePiece tokenizer** – 262K vocab
* 🌊 **Streaming output** – token‑by‑token callbacks
* 💬 **Interactive chat mode**
* 📦 **CLI + Library API**
* 🐧 **Linux/macOS native**, 🪟 Windows via **WSL** (recommended) or **MinGW**
* 🔗 **OpenBLAS support** (optional) – BLAS-accelerated matrix operations
* 🧵 **Multi-threaded inference** – Thread pool for parallel computation

---

## 🚀 Quick Start

> ⚠️ POSIX‑first: native on Linux/macOS. On Windows use **WSL** or **MinGW** (no `mmap`).

### 1️⃣ Download model (recommended)

```bash
export HF_TOKEN=your_token_here
python download_model.py
```

### 2️⃣ Build

```bash
make
```

### 3️⃣ Run

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

## 📥 Model Download

The included Python script:

* Handles HuggingFace auth
* Downloads all shards
* Resumes broken downloads
* Verifies integrity

```bash
python download_model.py --token YOUR_HF_TOKEN
```

Manual alternatives: `huggingface-cli` or `git lfs`.

---

## 🛠️ Build Targets

```bash
make              # Release build (default)
make debug        # Debug symbols
make fast         # Native optimizations (-march=native -ffast-math)
make threads      # Thread pool parallelization
make blas         # OpenBLAS acceleration (requires libopenblas)
make blas-threads # OpenBLAS + threads (best performance)
make clean        # Remove build artifacts
make help         # Show all targets
```

---

## 🧪 CLI Options

```
-m <path>    Model directory
-p <text>    Prompt
-i           Interactive mode
-s <text>    System prompt
-n <n>       Max tokens
-t <f>       Temperature
-k <n>       Top‑k
--top-p <f>  Top‑p
-c <n>       Context size
--seed <n>   RNG seed
-v           Verbose
```

---

## 📚 Library Example

```c
gemma3_ctx *ctx = gemma3_load_dir("./gemma-3-4b-it");

gemma3_gen_params params = gemma3_default_params();
char *out = gemma3_generate(ctx, "Hello!", &params, NULL, NULL);
printf("%s\n", out);
free(out);

gemma3_free(ctx);
```

---

## 🧠 Model Specs

| Param   | Value              |
| ------- | ------------------ |
| Vocab   | 262,208            |
| Layers  | 34                 |
| Hidden  | 2,560              |
| Heads   | 8 (4 KV, GQA)      |
| Context | 128K               |
| Pattern | 5 local : 1 global |

---

## 💾 Memory

* Weights: ~8 GB on disk (BF16)
* Runtime RAM: **~3 GB total**

Reduce usage:

```bash
./gemma3 -m ./gemma-3-4b-it -c 512 -p "Hello"
```

---

## ⚡ Performance (CPU)

* Prefill: ~2–5 tok/s
* Generation: ~1–3 tok/s

For better performance:

```bash
make fast          # Single-threaded with native optimizations
make threads       # Multi-core parallelization
make blas-threads  # Best performance (requires OpenBLAS)
```

---

## ⚠️ Limitations

* CPU only
* Text only
* No quantization (yet)

---

## 🪪 License

MIT License.
Model weights under Google’s Gemma license.

---

*If you ever wanted to see Gemma 3 breathe in pure C, this is it.*
