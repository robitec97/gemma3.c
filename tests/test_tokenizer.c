/*
 * test_tokenizer.c - Golden tests for the Gemma 3 tokenizer
 *
 * Checks gemma3_tokenizer.c against cases generated from the Hugging Face
 * reference tokenizer by tests/gen_tokenizer_golden.py:
 *   - encode(text) == HF encode(text, add_special_tokens=False)
 *   - detokenize(ids) == HF decode(ids, skip_special_tokens=True)
 *   - concatenated gemma3_token_to_bytes() == detokenize(ids)
 *   - gemma3_format_chat() == the official chat template
 *
 * Build & run (from the repository root):
 *   cc -O2 -std=c11 -Wall -Wextra -Wpedantic -I. tests/test_tokenizer.c \
 *      gemma3_tokenizer.c -o build/test_tokenizer
 *   ./build/test_tokenizer gemma-3-4b-it/tokenizer.model [tests_dir] [--bench]
 */

#if !defined(__APPLE__)
#define _POSIX_C_SOURCE 200809L
#endif
#include "gemma3_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_failures = 0;
static int g_checks = 0;

/* ---------------------------------------------------------------------------
 * Golden file parsing
 * ------------------------------------------------------------------------- */

static char *read_text_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (buf && fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        buf = NULL;
    }
    if (buf) buf[n] = '\0';
    fclose(f);
    return buf;
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* Unescape \\ \n \t \r \xNN. Returns a malloc'd string. */
static char *unescape(const char *s, size_t len) {
    char *out = (char *)malloc(len + 1);
    size_t o = 0;
    for (size_t i = 0; i < len; i++) {
        if (s[i] == '\\' && i + 1 < len) {
            char c = s[++i];
            if (c == 'n') out[o++] = '\n';
            else if (c == 't') out[o++] = '\t';
            else if (c == 'r') out[o++] = '\r';
            else if (c == 'x' && i + 2 < len) {
                out[o++] = (char)(hexval(s[i + 1]) * 16 + hexval(s[i + 2]));
                i += 2;
            } else out[o++] = c;
        } else {
            out[o++] = s[i];
        }
    }
    out[o] = '\0';
    return out;
}

/* Iterate over lines: returns pointer to the next line, sets line and len */
static char *next_line(char *p, char **line, size_t *len) {
    if (!p || !*p) return NULL;
    char *nl = strchr(p, '\n');
    *line = p;
    *len = nl ? (size_t)(nl - p) : strlen(p);
    return nl ? nl + 1 : p + *len;
}

static int parse_ids(const char *s, size_t len, int *ids, int max) {
    int n = 0;
    size_t i = 0;
    while (i < len && n < max) {
        while (i < len && s[i] == ' ') i++;
        if (i >= len) break;
        int v = 0;
        while (i < len && s[i] >= '0' && s[i] <= '9') v = v * 10 + (s[i++] - '0');
        ids[n++] = v;
    }
    return n;
}

/* ---------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */

static void print_ids(const char *label, const int *ids, int n) {
    fprintf(stderr, "    %s (%d):", label, n);
    for (int i = 0; i < n && i < 40; i++) fprintf(stderr, " %d", ids[i]);
    if (n > 40) fprintf(stderr, " ...");
    fprintf(stderr, "\n");
}

static void check(int cond, const char *what, int case_no) {
    g_checks++;
    if (!cond) {
        g_failures++;
        if (g_failures <= 30) fprintf(stderr, "FAIL case %d: %s\n", case_no, what);
    }
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---------------------------------------------------------------------------
 * Tokenizer golden cases
 * ------------------------------------------------------------------------- */

typedef struct {
    char **texts;
    int n;
    int cap;
} text_list;

static int run_tokenizer_cases(gemma3_tokenizer *tok, const char *path, text_list *corpus) {
    char *data = read_text_file(path);
    if (!data) {
        fprintf(stderr, "cannot read %s\n", path);
        return -1;
    }

    char *p = data, *line;
    size_t len;
    char *text = NULL;
    int *expected = NULL;
    int n_expected = 0;
    int case_no = 0;

    while ((p = next_line(p, &line, &len)) != NULL) {
        if (len < 2 || line[0] == '#') continue;
        if (line[0] == 'T') {
            free(text);
            text = unescape(line + 2, len - 2);
        } else if (line[0] == 'I') {
            free(expected);
            expected = (int *)malloc((len + 1) * sizeof(int));
            n_expected = parse_ids(line + 2, len - 2, expected, (int)len + 1);
        } else if (line[0] == 'D') {
            char *dec_expected = unescape(line + 2, len - 2);
            case_no++;

            /* encode */
            int cap = (int)strlen(text) * 2 + 16;
            int *got = (int *)malloc((size_t)cap * sizeof(int));
            int n = gemma3_tokenize(tok, text, got, cap, 0, 0);
            int same = n == n_expected && memcmp(got, expected, (size_t)n * sizeof(int)) == 0;
            check(same, "encode mismatch", case_no);
            if (!same && g_failures <= 30) {
                fprintf(stderr, "    text: \"%.200s\"\n", text);
                print_ids("expected", expected, n_expected);
                print_ids("got     ", got, n);
            }

            /* encode with BOS/EOS */
            int n2 = gemma3_tokenize(tok, text, got, cap, 1, 1);
            check(n2 == n_expected + 2 && got[0] == gemma3_bos_token(tok) &&
                  got[n2 - 1] == gemma3_eos_token(tok), "bos/eos", case_no);

            /* truncation never overflows */
            if (n_expected > 3) {
                int small = n_expected / 2;
                int *buf = (int *)malloc((size_t)(small + 1) * sizeof(int));
                buf[small] = -12345;
                int nt = gemma3_tokenize(tok, text, buf, small, 0, 0);
                check(nt == small && buf[small] == -12345 &&
                      memcmp(buf, expected, (size_t)small * sizeof(int)) == 0,
                      "truncation", case_no);
                free(buf);
            }

            /* decode */
            char *dec = gemma3_detokenize(tok, expected, n_expected);
            int dec_ok = dec && strcmp(dec, dec_expected) == 0;
            check(dec_ok, "decode mismatch", case_no);
            if (!dec_ok && g_failures <= 30) {
                fprintf(stderr, "    expected: \"%.200s\"\n    got:      \"%.200s\"\n",
                        dec_expected, dec ? dec : "(null)");
            }

            /* streaming decode == full decode */
            size_t stream_cap = strlen(dec_expected) + 1;
            char *stream = (char *)malloc(stream_cap + 256);
            size_t sl = 0;
            int stream_ok = 1;
            for (int i = 0; i < n_expected; i++) {
                char piece[256];
                int k = gemma3_token_to_bytes(tok, expected[i], piece, sizeof(piece));
                if (k < 0 || sl + (size_t)k > stream_cap) { stream_ok = 0; break; }
                memcpy(stream + sl, piece, (size_t)k);
                sl += (size_t)k;
            }
            stream[sl] = '\0';
            check(stream_ok && strcmp(stream, dec_expected) == 0, "token_to_bytes mismatch", case_no);

            /* keep text for the benchmark corpus */
            if (corpus) {
                if (corpus->n == corpus->cap) {
                    corpus->cap = corpus->cap ? corpus->cap * 2 : 256;
                    corpus->texts = (char **)realloc(corpus->texts, (size_t)corpus->cap * sizeof(char *));
                }
                corpus->texts[corpus->n++] = text;
                text = NULL;
            }

            free(stream);
            free(dec);
            free(got);
            free(dec_expected);
        }
    }

    free(text);
    free(expected);
    free(data);
    return case_no;
}

/* ---------------------------------------------------------------------------
 * Chat template golden cases
 * ------------------------------------------------------------------------- */

static int run_chat_cases(gemma3_tokenizer *tok, const char *path) {
    char *data = read_text_file(path);
    if (!data) {
        fprintf(stderr, "cannot read %s\n", path);
        return -1;
    }

    char *p = data, *line;
    size_t len;
    gemma3_message msgs[32];
    char *contents[32];
    int n_msgs = 0;
    int case_no = 0;

    while ((p = next_line(p, &line, &len)) != NULL) {
        if (len < 2 || line[0] == '#') continue;
        if (line[0] == 'C') {
            n_msgs = 0;
        } else if (line[0] == 'M' && n_msgs < 32) {
            const char *r = line + 2;
            const char *sp = memchr(r, ' ', len - 2);
            size_t rlen = sp ? (size_t)(sp - r) : len - 2;
            gemma3_role role = GEMMA3_ROLE_USER;
            if (rlen == 6 && !strncmp(r, "system", 6)) role = GEMMA3_ROLE_SYSTEM;
            else if (rlen == 5 && !strncmp(r, "model", 5)) role = GEMMA3_ROLE_MODEL;
            const char *c = sp ? sp + 1 : line + len;
            contents[n_msgs] = unescape(c, (size_t)(line + len - c));
            msgs[n_msgs].role = role;
            msgs[n_msgs].content = contents[n_msgs];
            n_msgs++;
        } else if (line[0] == 'E') {
            case_no++;
            char *expected = unescape(line + 2, len - 2);
            char *got = gemma3_format_chat(tok, msgs, n_msgs);
            int ok = got && strcmp(got, expected) == 0;
            check(ok, "chat template mismatch", 1000 + case_no);
            if (!ok && g_failures <= 30) {
                fprintf(stderr, "    expected: \"%s\"\n    got:      \"%s\"\n", expected,
                        got ? got : "(null)");
            }
            /* The formatted prompt must start with the real BOS token and use the
             * single-token turn markers. */
            if (got) {
                int ids[4096];
                int n = gemma3_tokenize(tok, got, ids, 4096, 0, 0);
                check(n > 2 && ids[0] == gemma3_bos_token(tok) &&
                      ids[1] == gemma3_start_turn_token(tok), "chat tokens", 1000 + case_no);
            }
            free(got);
            free(expected);
            for (int i = 0; i < n_msgs; i++) free(contents[i]);
            n_msgs = 0;
        }
    }
    free(data);
    return case_no;
}

/* ---------------------------------------------------------------------------
 * Benchmark
 * ------------------------------------------------------------------------- */

static const char *k_filler =
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

static void bench_one(gemma3_tokenizer *tok, const char *name, const char *text, int iters) {
    size_t n = strlen(text);
    int cap = (int)n + 16;
    int *ids = (int *)malloc((size_t)cap * sizeof(int));
    int nt = 0;
    double best = 1e30;
    for (int it = 0; it < iters; it++) {
        double s = now_sec();
        nt = gemma3_tokenize(tok, text, ids, cap, 1, 0);
        double e = now_sec() - s;
        if (e < best) best = e;
    }
    printf("  %-28s %9zu bytes -> %8d tokens  %8.2f ms  %8.2f MB/s  %10.0f tok/s\n",
           name, n, nt, best * 1e3, n / best / 1e6, nt / best);
    free(ids);
}

static void run_bench(gemma3_tokenizer *tok, const text_list *corpus) {
    printf("\nTokenizer throughput (best of runs):\n");

    /* 4 KB English prose (same text as bench/bench_e2e.c) */
    size_t fl = strlen(k_filler);
    char *small = (char *)malloc(fl * 4 + 1);
    for (int i = 0; i < 4; i++) memcpy(small + i * fl, k_filler, fl);
    small[fl * 4] = '\0';
    bench_one(tok, "prose 4 KB", small, 20);

    /* ~1 MB English prose */
    size_t reps = (1u << 20) / fl;
    char *prose = (char *)malloc(fl * reps + 1);
    for (size_t i = 0; i < reps; i++) memcpy(prose + i * fl, k_filler, fl);
    prose[fl * reps] = '\0';
    bench_one(tok, "prose 1 MB", prose, 3);

    /* ~1 MB of the mixed golden corpus (code, multilingual, specials) */
    size_t total = 0;
    for (int i = 0; i < corpus->n; i++) total += strlen(corpus->texts[i]) + 1;
    if (total > 0) {
        size_t target = 1u << 20;
        char *mixed = (char *)malloc(target + total + 1);
        size_t o = 0;
        while (o < target) {
            for (int i = 0; i < corpus->n && o < target; i++) {
                size_t l = strlen(corpus->texts[i]);
                memcpy(mixed + o, corpus->texts[i], l);
                o += l;
                mixed[o++] = '\n';
            }
        }
        mixed[o] = '\0';
        bench_one(tok, "mixed golden corpus 1 MB", mixed, 3);
        free(mixed);
    }

    /* Single long line without newlines (worst case for segment length) */
    char *line = (char *)malloc(fl * 64 + 1);
    for (int i = 0; i < 64; i++) memcpy(line + i * fl, k_filler, fl);
    line[fl * 64] = '\0';
    bench_one(tok, "single 64 KB segment", line, 5);

    free(line);
    free(prose);
    free(small);
}

/* ---------------------------------------------------------------------------
 * Main
 * ------------------------------------------------------------------------- */

int main(int argc, char **argv) {
    const char *model_path = "gemma-3-4b-it/tokenizer.model";
    const char *tests_dir = "tests";
    int bench = 0;
    int positional = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bench")) bench = 1;
        else if (positional == 0) { model_path = argv[i]; positional++; }
        else if (positional == 1) { tests_dir = argv[i]; positional++; }
    }

    double t0 = now_sec();
    gemma3_tokenizer *tok = gemma3_tokenizer_load(model_path);
    double load_s = now_sec() - t0;
    if (!tok) {
        fprintf(stderr, "Failed to load tokenizer from %s\n", model_path);
        return 1;
    }
    printf("Loaded %s in %.1f ms\n", model_path, load_s * 1e3);

    char path[1024];
    text_list corpus = {0};

    snprintf(path, sizeof(path), "%s/tokenizer_golden.txt", tests_dir);
    int n_tok = run_tokenizer_cases(tok, path, &corpus);
    snprintf(path, sizeof(path), "%s/chat_template_golden.txt", tests_dir);
    int n_chat = run_chat_cases(tok, path);

    /* Misc API checks */
    {
        char buf[8];
        check(gemma3_token_to_bytes(tok, -1, buf, sizeof(buf)) < 0, "token_to_bytes(-1)", 0);
        check(gemma3_token_to_bytes(tok, gemma3_bos_token(tok), buf, sizeof(buf)) == 0,
              "token_to_bytes(<bos>) is empty", 0);
        char *empty = gemma3_detokenize(tok, NULL, 0);
        check(empty && empty[0] == '\0', "detokenize of zero tokens", 0);
        free(empty);
        int ids[4];
        check(gemma3_tokenize(tok, "", ids, 4, 1, 0) == 1 && ids[0] == gemma3_bos_token(tok),
              "empty text with BOS", 0);
        check(gemma3_end_turn_token(tok) == 106 && gemma3_start_turn_token(tok) == 105,
              "turn token ids", 0);
    }

    printf("Tokenizer cases: %d, chat cases: %d, checks: %d, failures: %d\n",
           n_tok, n_chat, g_checks, g_failures);

    if (bench) run_bench(tok, &corpus);

    for (int i = 0; i < corpus.n; i++) free(corpus.texts[i]);
    free(corpus.texts);
    gemma3_tokenizer_free(tok);

    if (n_tok <= 0 || n_chat <= 0) return 1;
    return g_failures == 0 ? 0 : 1;
}
