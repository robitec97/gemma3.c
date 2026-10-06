/*
 * bench_e2e.c - End-to-end inference benchmark for gemma3.c
 *
 * Uses only the public API in gemma3.h, so the same harness can be linked
 * against any version of the library for apples-to-apples comparisons.
 *
 * Measures:
 *   - model load time
 *   - prefill throughput (prompt tokens/s) for one or more prompt lengths
 *   - decode throughput (greedy, generated tokens/s), optionally after a long context (-d)
 *   - generation throughput with the default sampler (temperature/top-k/top-p)
 *   - tokenizer throughput
 *   - peak resident memory
 *
 * Usage:
 *   ./gemma3-bench [-m model_dir] [-p 64,256,1024] [-n 128] [-d 2048] [-r 1] [-c 4096]
 *                  [--json out.json] [--label name] [--no-sample] [--no-tokenizer]
 */

#if !defined(__APPLE__)
#define _POSIX_C_SOURCE 200809L
#endif
#include "../gemma3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double peak_rss_mb(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return (double)ru.ru_maxrss / (1024.0 * 1024.0);   /* bytes */
#else
    return (double)ru.ru_maxrss / 1024.0;              /* kilobytes */
#endif
}

static int argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

/* Public-domain style filler text used to build prompts of a given length. */
static const char *k_text =
    "The history of computing is a story of abstraction. Early machines were "
    "programmed by rewiring panels and flipping switches, and every program was "
    "tied to the physical layout of the hardware. Stored-program computers "
    "changed that: instructions became data, and data could be loaded, copied "
    "and modified like anything else. Assemblers turned mnemonic names into "
    "machine code, compilers turned structured languages into assembly, and "
    "operating systems turned the raw machine into a set of services that many "
    "programs could share. Each layer hid the details of the one below it while "
    "exposing a simpler, more general interface. Modern language models continue "
    "this tradition. A transformer is, at its core, a stack of matrix "
    "multiplications, normalizations and attention operations, yet from the "
    "outside it behaves like a function from text to text. Running one "
    "efficiently means paying attention to memory bandwidth, cache locality and "
    "parallelism, the same concerns that shaped every previous generation of "
    "software. ";

typedef struct {
    int prompt_len;
    double prefill_s;
} prefill_result;

static int parse_list(const char *s, int *out, int max) {
    int n = 0;
    while (*s && n < max) {
        char *end;
        long v = strtol(s, &end, 10);
        if (end == s) break;
        if (v > 0) out[n++] = (int)v;
        s = end;
        while (*s == ',' || *s == ' ') s++;
    }
    return n;
}

static int count_cb(int token_id, const char *s, void *ud) {
    (void)token_id; (void)s;
    (*(int *)ud)++;
    return 0;
}

int main(int argc, char **argv) {
    const char *model_dir = getenv("GEMMA3_MODEL") ? getenv("GEMMA3_MODEL") : "gemma-3-4b-it";
    const char *json_path = NULL;
    const char *label = "gemma3.c";
    int prompt_lens[16] = {64, 256, 1024};
    int n_prompts = 3;
    int depths[16] = {0};
    int n_depths = 0;
    int gen_tokens = 128;
    int repeats = 1;
    int context = 4096;
    int do_sample = 1;
    int do_tokenizer = 1;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-m") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(a, "-p") && i + 1 < argc) n_prompts = parse_list(argv[++i], prompt_lens, 16);
        else if (!strcmp(a, "-n") && i + 1 < argc) gen_tokens = atoi(argv[++i]);
        else if (!strcmp(a, "-d") && i + 1 < argc) n_depths = parse_list(argv[++i], depths, 16);
        else if (!strcmp(a, "-r") && i + 1 < argc) repeats = atoi(argv[++i]);
        else if (!strcmp(a, "-c") && i + 1 < argc) context = atoi(argv[++i]);
        else if (!strcmp(a, "--json") && i + 1 < argc) json_path = argv[++i];
        else if (!strcmp(a, "--label") && i + 1 < argc) label = argv[++i];
        else if (!strcmp(a, "--no-sample")) do_sample = 0;
        else if (!strcmp(a, "--no-tokenizer")) do_tokenizer = 0;
        else {
            fprintf(stderr,
                "Usage: %s [-m model_dir] [-p 64,256,1024] [-n gen_tokens] [-d 2048,4096] [-r repeats]\n"
                "          [-c context] [--json out.json] [--label name]\n"
                "          [--no-sample] [--no-tokenizer]\n", argv[0]);
            return 1;
        }
    }
    if (repeats < 1) repeats = 1;

    int max_prompt = 0;
    for (int i = 0; i < n_prompts; i++) if (prompt_lens[i] > max_prompt) max_prompt = prompt_lens[i];
    for (int i = 0; i < n_depths; i++) if (depths[i] > max_prompt) max_prompt = depths[i];
    if (max_prompt + gen_tokens + 8 > context) context = max_prompt + gen_tokens + 8;

    /* ---- Load ---- */
    double t0 = now_sec();
    gemma3_ctx *ctx = gemma3_load_dir_ex(model_dir, context);
    double load_s = now_sec() - t0;
    if (!ctx) {
        fprintf(stderr, "Failed to load model from %s: %s\n", model_dir, gemma3_get_error());
        return 1;
    }
    const gemma3_config *cfg = gemma3_get_config(ctx);
    gemma3_tokenizer *tok = gemma3_get_tokenizer(ctx);
    int vocab = cfg->vocab_size;

    float *logits = (float *)malloc((size_t)vocab * sizeof(float));
    int *base = (int *)malloc(4096 * sizeof(int));
    int *prompt = (int *)malloc((size_t)(max_prompt + 1) * sizeof(int));
    if (!logits || !base || !prompt) { fprintf(stderr, "OOM\n"); return 1; }

    int n_base = gemma3_tokenize(tok, k_text, base, 4096, 0, 0);
    if (n_base <= 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    /* Prompt = BOS + filler tokens repeated to the requested length. */
    prompt[0] = gemma3_bos_token(tok);
    for (int i = 1; i < max_prompt; i++) prompt[i] = base[(i - 1) % n_base];

    /* ---- Warm-up (page in weights, compile shaders, spin up threads) ---- */
    gemma3_reset_cache(ctx);
    gemma3_forward_batch(ctx, prompt, 8, 0, logits);
    gemma3_forward(ctx, argmax(logits, vocab), 8, logits);

    printf("\n%s  |  model: %s  |  context: %d\n", label, model_dir, context);
    printf("load: %.2f s\n\n", load_s);
    printf("%-28s %12s %12s\n", "test", "tokens/s", "ms/token");
    printf("%-28s %12s %12s\n", "----------------------------", "------------", "------------");

    /* ---- Prefill ---- */
    prefill_result pres[16];
    for (int pi = 0; pi < n_prompts; pi++) {
        int n = prompt_lens[pi];
        double best = 1e30;
        for (int r = 0; r < repeats; r++) {
            gemma3_reset_cache(ctx);
            double s = now_sec();
            if (gemma3_forward_batch(ctx, prompt, n, 0, logits) != 0) {
                fprintf(stderr, "prefill failed\n"); return 1;
            }
            double e = now_sec() - s;
            if (e < best) best = e;
        }
        pres[pi].prompt_len = n;
        pres[pi].prefill_s = best;
        char name[64];
        snprintf(name, sizeof(name), "prefill pp%d", n);
        printf("%-28s %12.2f %12.2f\n", name, n / best, 1000.0 * best / n);
        fflush(stdout);
    }

    /* ---- Decode (greedy, after a short prompt) ---- */
    int dec_prompt = prompt_lens[0] < 64 ? prompt_lens[0] : 64;
    double dec_best = 1e30;
    for (int r = 0; r < repeats; r++) {
        gemma3_reset_cache(ctx);
        gemma3_forward_batch(ctx, prompt, dec_prompt, 0, logits);
        int pos = dec_prompt;
        double s = now_sec();
        for (int i = 0; i < gen_tokens; i++) {
            int next = argmax(logits, vocab);
            if (gemma3_forward(ctx, next, pos++, logits) != 0) {
                fprintf(stderr, "decode failed\n"); return 1;
            }
        }
        double e = now_sec() - s;
        if (e < dec_best) dec_best = e;
    }
    char dname[64];
    snprintf(dname, sizeof(dname), "decode tg%d", gen_tokens);
    printf("%-28s %12.2f %12.2f\n", dname, gen_tokens / dec_best, 1000.0 * dec_best / gen_tokens);
    fflush(stdout);

    /* ---- Decode after a long context (-d): attention cost grows with depth ---- */
    double depth_tps[16] = {0};
    for (int di = 0; di < n_depths; di++) {
        int d = depths[di];
        double best = 1e30;
        for (int r = 0; r < repeats; r++) {
            gemma3_reset_cache(ctx);
            if (gemma3_forward_batch(ctx, prompt, d, 0, logits) != 0) {
                fprintf(stderr, "prefill for depth %d failed\n", d); return 1;
            }
            int pos = d;
            double s = now_sec();
            for (int i = 0; i < gen_tokens; i++) {
                if (gemma3_forward(ctx, argmax(logits, vocab), pos++, logits) != 0) {
                    fprintf(stderr, "decode failed\n"); return 1;
                }
            }
            double e = now_sec() - s;
            if (e < best) best = e;
        }
        depth_tps[di] = gen_tokens / best;
        char name[64];
        snprintf(name, sizeof(name), "decode tg%d @ depth %d", gen_tokens, d);
        printf("%-28s %12.2f %12.2f\n", name, depth_tps[di], 1000.0 * best / gen_tokens);
        fflush(stdout);
    }

    /* ---- Generation with default sampler (includes sampling cost) ---- */
    double gen_tps = 0.0, gen_s = 0.0;
    int gen_count = 0;
    if (do_sample) {
        gemma3_gen_params params = gemma3_default_params();
        params.max_tokens = gen_tokens;
        params.seed = 42;
        params.stop_on_eos = 0;
        double s = now_sec();
        char *out = gemma3_generate_tokens(ctx, prompt, dec_prompt, &params, count_cb, &gen_count);
        gen_s = now_sec() - s;
        free(out);
        gen_tps = gen_count > 0 ? gen_count / gen_s : 0.0;
        char gname[64];
        snprintf(gname, sizeof(gname), "generate (sampled) tg%d", gen_count);
        printf("%-28s %12.2f %12.2f   (incl. %d-token prefill)\n", gname, gen_tps,
               gen_count ? 1000.0 * gen_s / gen_count : 0.0, dec_prompt);
    }

    /* ---- Tokenizer throughput ---- */
    double tok_chars_per_s = 0.0;
    int tok_bytes = 0;
    if (do_tokenizer) {
        size_t tl = strlen(k_text);
        int reps = 4;
        char *big = (char *)malloc(tl * reps + 1);
        for (int i = 0; i < reps; i++) memcpy(big + i * tl, k_text, tl);
        big[tl * reps] = '\0';
        tok_bytes = (int)(tl * reps);
        int *tbuf = (int *)malloc((size_t)tok_bytes * sizeof(int));
        double s = now_sec();
        int nt = gemma3_tokenize(tok, big, tbuf, tok_bytes, 1, 0);
        double e = now_sec() - s;
        tok_chars_per_s = tok_bytes / e;
        printf("%-28s %12.0f %12s   (%d bytes -> %d tokens in %.1f ms)\n", "tokenizer (bytes/s)",
               tok_chars_per_s, "-", tok_bytes, nt, e * 1000.0);
        free(tbuf);
        free(big);
    }

    double rss = peak_rss_mb();
    printf("\npeak RSS: %.0f MB\n", rss);

    if (json_path) {
        FILE *f = fopen(json_path, "w");
        if (f) {
            fprintf(f, "{\n  \"label\": \"%s\",\n  \"version\": \"%s\",\n  \"context\": %d,\n",
                    label, gemma3_version(), context);
            fprintf(f, "  \"load_s\": %.4f,\n  \"prefill\": [", load_s);
            for (int i = 0; i < n_prompts; i++) {
                fprintf(f, "%s{\"tokens\": %d, \"seconds\": %.5f, \"tok_per_s\": %.3f}",
                        i ? ", " : "", pres[i].prompt_len, pres[i].prefill_s,
                        pres[i].prompt_len / pres[i].prefill_s);
            }
            fprintf(f, "],\n  \"decode\": {\"tokens\": %d, \"seconds\": %.5f, \"tok_per_s\": %.3f},\n",
                    gen_tokens, dec_best, gen_tokens / dec_best);
            fprintf(f, "  \"decode_at_depth\": [");
            for (int i = 0; i < n_depths; i++) {
                fprintf(f, "%s{\"depth\": %d, \"tok_per_s\": %.3f}", i ? ", " : "", depths[i], depth_tps[i]);
            }
            fprintf(f, "],\n");
            fprintf(f, "  \"generate\": {\"tokens\": %d, \"seconds\": %.5f, \"tok_per_s\": %.3f},\n",
                    gen_count, gen_s, gen_tps);
            fprintf(f, "  \"tokenizer_bytes_per_s\": %.1f,\n", tok_chars_per_s);
            fprintf(f, "  \"peak_rss_mb\": %.1f\n}\n", rss);
            fclose(f);
        }
    }

    free(logits);
    free(base);
    free(prompt);
    gemma3_free(ctx);
    return 0;
}
