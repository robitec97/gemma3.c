/*
 * test_cache.c - KV-cache reuse must not change results (needs the model)
 *
 * Generation reuses the longest prompt prefix already in the KV cache and
 * rewinds the cache when the new prompt diverges. This test checks that a
 * greedy continuation computed with reuse/rewind is token-for-token identical
 * to one computed from an empty cache, including after the sliding-window
 * ring buffers have wrapped (prompt > 1024 tokens).
 *
 *   make test-model     (or: ./gemma3-test-cache <model_dir>)
 */

#include "../gemma3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static int collect(int token_id, const char *piece, void *ud) {
    (void)piece;
    int *buf = (int *)ud;
    buf[1 + buf[0]++] = token_id;
    return 0;
}

/* Greedy-generate n tokens; returns them in out[1..out[0]] */
static void generate(gemma3_ctx *ctx, const int *prompt, int len, int n, int *out) {
    gemma3_gen_params p = gemma3_default_params();
    p.greedy = 1;
    p.max_tokens = n;
    p.stop_on_eos = 0;
    out[0] = 0;
    char *text = gemma3_generate_tokens(ctx, prompt, len, &p, collect, out);
    free(text);
}

static void expect_same(const char *name, const int *a, const int *b, int reused, int min_reused) {
    int same = a[0] == b[0] && memcmp(a + 1, b + 1, (size_t)a[0] * sizeof(int)) == 0;
    int ok = same && reused >= min_reused;
    printf("  %s %-46s reused %4d tokens, %s\n", ok ? "ok  " : "FAIL", name, reused,
           same ? "identical output" : "OUTPUT DIFFERS");
    if (!ok) failures++;
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : "gemma-3-4b-it";
    gemma3_load_options opts = gemma3_default_load_options();
    opts.max_context = 2048;
    gemma3_ctx *ctx = gemma3_load_dir_opts(model, &opts);
    if (!ctx) {
        fprintf(stderr, "load failed: %s\n", gemma3_get_error());
        return 1;
    }
    printf("KV-cache reuse tests (%s)\n", gemma3_backend_name(ctx));

    /* A long prompt from varied text */
    const char *para =
        "Lighthouses once relied on keepers who trimmed wicks and wound clockwork through the night. "
        "Fresnel lenses concentrated the flame into a beam visible for miles, and each station had a "
        "distinctive pattern of flashes so sailors could tell them apart. ";
    gemma3_tokenizer *tok = gemma3_get_tokenizer(ctx);
    int base[512];
    int nb = gemma3_tokenize(tok, para, base, 512, 0, 0);
    int L = 1300;
    int *prompt = malloc((size_t)(L + 64) * sizeof(int));
    prompt[0] = gemma3_bos_token(tok);
    for (int i = 1; i < L + 64; i++) prompt[i] = base[(i - 1) % nb];
    /* make every repetition slightly different */
    for (int i = 1; i < L + 64; i += 97) prompt[i] = 1000 + (i % 500);

    int fresh[64], reused[64];
    const int N = 12;

    /* 1. Extend a prompt that is already cached (short, no ring wrap) */
    gemma3_reset_cache(ctx);
    generate(ctx, prompt, 300, N, fresh);               /* from an empty cache */
    gemma3_reset_cache(ctx);
    generate(ctx, prompt, 200, 1, reused);              /* cache now holds 200 tokens */
    generate(ctx, prompt, 300, N, reused);
    expect_same("extend cached prompt 200 -> 300", fresh, reused,
                gemma3_get_stats(ctx)->reused_tokens, 200);

    /* 2. Same after the local-layer ring buffers have wrapped */
    gemma3_reset_cache(ctx);
    generate(ctx, prompt, L, N, fresh);
    gemma3_reset_cache(ctx);
    generate(ctx, prompt, L - 40, 1, reused);
    generate(ctx, prompt, L, N, reused);
    expect_same("extend after ring wrap 1260 -> 1300", fresh, reused,
                gemma3_get_stats(ctx)->reused_tokens, L - 40);

    /* 3. Rewind: the cache holds a diverging tail that must be discarded */
    int *alt = malloc((size_t)(L + 64) * sizeof(int));
    memcpy(alt, prompt, (size_t)(L + 64) * sizeof(int));
    for (int i = L - 30; i < L; i++) alt[i] = 2000 + i % 300;   /* diverge in the last 30 */
    gemma3_reset_cache(ctx);
    generate(ctx, prompt, L, N, fresh);
    gemma3_reset_cache(ctx);
    generate(ctx, alt, L, 8, reused);                   /* cache: alt + 7 generated */
    generate(ctx, prompt, L, N, reused);                /* rewind ~38 tokens */
    expect_same("rewind 38 tokens after ring wrap", fresh, reused,
                gemma3_get_stats(ctx)->reused_tokens, L - 30);

    /* 4. Unrelated prompt: must fall back to a full recompute */
    generate(ctx, prompt + 500, 400, N, reused);
    int r = gemma3_get_stats(ctx)->reused_tokens;
    gemma3_reset_cache(ctx);
    generate(ctx, prompt + 500, 400, N, fresh);
    expect_same("unrelated prompt (no reuse expected)", fresh, reused, 0, 0);
    if (r != 0) { printf("  FAIL expected no reuse, got %d\n", r); failures++; }

    /* 5. Two rewinds in a row: the second one must not trust ring rows that
     *    were overwritten before the first rewind (high-water mark) */
    int *a2 = malloc((size_t)(L + 64) * sizeof(int));
    int *a3 = malloc((size_t)(L + 64) * sizeof(int));
    memcpy(a2, prompt, (size_t)(L + 64) * sizeof(int));
    memcpy(a3, prompt, (size_t)(L + 64) * sizeof(int));
    a2[1080] = 3000;                                    /* A[:1080] + x */
    a3[50] = 3001;                                      /* A[:50] + y   */
    gemma3_reset_cache(ctx);
    generate(ctx, a3, 51, N, fresh);
    gemma3_reset_cache(ctx);
    generate(ctx, prompt, 1200, 1, reused);             /* ring wraps at 1152 */
    generate(ctx, a2, 1081, 1, reused);                 /* rewind 120: allowed */
    generate(ctx, a3, 51, N, reused);                   /* must recompute */
    expect_same("second rewind after an earlier rewind", fresh, reused,
                gemma3_get_stats(ctx)->reused_tokens, 0);
    free(a2);
    free(a3);

    free(prompt);
    free(alt);
    gemma3_free(ctx);
    printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
