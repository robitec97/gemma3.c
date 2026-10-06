# Code review: findings and fixes

This document records a full review of gemma3.c (as of commit `76fe597`) and
what was changed as a result. Every number below was measured on a MacBook Air
M4 (10-core CPU, 10-core GPU, 16 GB) with `gemma-3-4b-it` in BF16; see
[the benchmark section of the README](../README.md#performance) for methodology.

## Correctness

| # | Problem | Impact | Fix |
|---|---------|--------|-----|
| 1 | The tokenizer did not recognise special tokens in text: `<start_of_turn>` became 7 ordinary tokens (`▁<`, `start`, `_`, `of`, `_`, `turn`, `>`). | The chat template never reached the model as intended, so it saw unfamiliar turn markers. Replies could end with a literal `<end_of_turn>` written as text, which also slipped past the end-of-turn stop check. | Added tokens (control and user-defined pieces) are matched leftmost-longest before BPE, exactly like Hugging Face `tokenizers`. |
| 2 | The tokenizer prepended a "dummy prefix" space and collapsed runs of spaces into one `▁`. | Encodings differed from the reference tokenizer, and code indentation was lost. | Every space now maps to its own `▁`, with no dummy prefix (Gemma's `add_dummy_prefix=false`). Verified against `tokenizer.json` on 454 golden cases plus a 23k-case stress run. |
| 3 | Linear RoPE scaling (`rope_scaling.factor = 8` for global layers in `config.json`) was not applied. | Logits differed from the reference by up to 1.3 (top-20 overlap 18/20 on a 1.5k-token prompt). | `rope_scaling` is read from `config.json` and applied to the global-layer RoPE tables. Logits now match an independent NumPy reference to within 1e-4. |
| 4 | Chat prompts had two BOS tokens, and the system prompt was sent as an extra `System:` user turn. | Did not follow the official Gemma 3 chat template. | Formatting now matches the official template: the system prompt is folded into the first user turn and there is a single BOS. |
| 5 | `-ffast-math` builds (threads, blas, mps) used `-INFINITY` in the sampler; the compiler warned this is undefined behaviour. | Possible NaN or wrong sampling under fast-math. | Dropped `-ffast-math` (explicit SIMD makes it unnecessary) and rewrote the sampler. |
| 6 | Missing tensors were only reported as warnings, which led to NULL dereferences later. Tensor offsets were never bounds-checked. | A truncated or wrong checkpoint crashed the program or read out of bounds. | The safetensors header and every tensor's bounds, dtype and shape are validated at load time. Three checkpoint naming schemes are recognised. |
| 7 | `gemma3_detokenize` stripped a leading space. | Text did not round-trip through encode and decode. | No stripping. Gemma has no dummy prefix to undo. |
| 8 | Ctrl+C at the chat prompt did nothing (BSD `signal()` restarts `fgets`), and Ctrl+C during prompt processing was ignored. | The interactive mode could not be interrupted as expected. | `sigaction` without `SA_RESTART`, plus `gemma3_abort()`, which is checked between prefill chunks. |
| 9 | Prompts longer than the context were silently truncated. | Wrong answers with no warning. | Prompts that don't fit produce a clear error. The chat mode drops the oldest turns when the conversation outgrows the context. |

The NumPy reference (`tools/reference_forward.py`) independently implements
the Hugging Face `modeling_gemma3` forward pass. `tools/compare_logits.py`
compares any build against it, and it was used to confirm items 3 and the
sliding-window handling.

## Performance

| Area | Before | After |
|------|--------|-------|
| Default `make` build | Scalar, single-threaded: **0.5 tok/s** | NEON/AVX2 with a thread pool (see the README for speeds) |
| Tokenizer | O(n²) BPE with a `malloc` per lookup: **389 bytes/s** (a 4 KB prompt took 10.4 s) | Heap-based BPE: 3-11 MB/s |
| Sampling | Softmax and `qsort` over all 262k logits every token: 9.2 ms/token | Top-k heap, then sort and softmax over the candidates only: 0.08 ms/token |
| CPU prefill | One token at a time (memory bound) | Batched BF16 GEMM over 128-token chunks (compute bound) |
| Multi-turn chat | Every turn re-processed the whole conversation | The KV cache is reused for the shared prefix, so only the new turn is processed |
| Thread pool | Main thread idle, condvar wake-up per job, static row split (E-cores straggle) | Caller participates, workers spin briefly, dynamic chunked scheduling |
| Metal weights | Copied into new buffers (~8 GB extra RAM, 2.9 s load) because tensors are rarely page-aligned | Zero-copy: the mapped safetensors files are wrapped directly |
| Metal prefill | One command buffer and a CPU sync per token | Batched GPU kernels per chunk |

## Usability

* `make` now produces the fastest portable CPU build. Old target names still work.
* New CLI options:
  * prompt from a file (`-f`), from stdin (`-p -`) or piped input
  * `--threads`, `--cpu`, `--stats`, `--min-p`, `-q`, `--version`
  * `GEMMA3_MODEL` / `GEMMA3_THREADS` environment variables
  * validated numeric arguments
* The interactive mode gains:
  * `/help`, `/clear`, `/system`, `/stats`, `/exit`
  * multi-line input and coloured output
  * per-reply timing stats
* New library API:
  * `gemma3_load_dir_opts()` (threads, GPU on/off, quiet loading)
  * `gemma3_get_stats()` (TTFT, prefill and decode tokens/s)
  * `gemma3_abort()`
  * `gemma3_token_to_bytes()` for streaming
  * `gemma3_backend_name()`
* Tests:
  * `make test`: unit tests for the NEON, AVX2 and scalar kernels, the sampler and the thread pool
  * `make test-model`: tokenizer golden tests and end-to-end checks
  * `tools/compare_logits.py`: comparison against the reference implementation
  * GitHub Actions CI: Linux x86-64 and arm64, macOS, and ASan/UBSan
* Benchmarks:
  * `make bench`: end-to-end tokens/s, TTFT and peak memory
  * `make bench-kernels`: GB/s, GFLOP/s and sampler cost, no model needed
