/*
 * simple.c - Minimal use of the gemma3.c library API
 *
 * Streams a two-turn conversation and prints timing statistics. The second
 * turn reuses the KV cache of the first, so only the new tokens are processed.
 *
 *   make example && ./gemma3-example ./gemma-3-4b-it
 */

#include "../gemma3.h"
#include <stdio.h>
#include <stdlib.h>

static int on_token(int token_id, const char *piece, void *user_data) {
    (void)piece;
    char buf[256];
    int n = gemma3_token_to_bytes((gemma3_tokenizer *)user_data, token_id, buf, sizeof(buf));
    if (n > 0) fwrite(buf, 1, (size_t)n, stdout);
    fflush(stdout);
    return 0;  /* return non-zero to stop generating */
}

static void print_stats(const gemma3_ctx *ctx) {
    const gemma3_stats *s = gemma3_get_stats(ctx);
    printf("\n  -> %d prompt tokens (%d reused from cache), first token after %.0f ms, "
           "%d tokens at %.1f tok/s\n\n",
           s->prompt_tokens, s->reused_tokens, s->ttft_ms, s->generated_tokens, s->decode_tok_per_s);
}

int main(int argc, char **argv) {
    const char *model_dir = argc > 1 ? argv[1] : "gemma-3-4b-it";

    gemma3_load_options opts = gemma3_default_load_options();
    opts.max_context = 4096;
    gemma3_ctx *ctx = gemma3_load_dir_opts(model_dir, &opts);
    if (!ctx) {
        fprintf(stderr, "load failed: %s\n", gemma3_get_error());
        return 1;
    }
    printf("backend: %s\n\n", gemma3_backend_name(ctx));

    gemma3_tokenizer *tok = gemma3_get_tokenizer(ctx);
    gemma3_gen_params params = gemma3_default_params();
    params.max_tokens = 128;
    params.seed = 42;

    gemma3_message chat[4] = {
        { GEMMA3_ROLE_SYSTEM, "You are a concise assistant." },
        { GEMMA3_ROLE_USER, "Name three famous C programmers." },
    };

    char *reply = gemma3_chat(ctx, chat, 2, &params, on_token, tok);
    if (!reply) {
        fprintf(stderr, "generation failed: %s\n", gemma3_get_error());
        gemma3_free(ctx);
        return 1;
    }
    print_stats(ctx);

    chat[2] = (gemma3_message){ GEMMA3_ROLE_MODEL, reply };
    chat[3] = (gemma3_message){ GEMMA3_ROLE_USER, "Which of them created C?" };
    char *reply2 = gemma3_chat(ctx, chat, 4, &params, on_token, tok);
    if (reply2) print_stats(ctx);

    free(reply);
    free(reply2);
    gemma3_free(ctx);
    return 0;
}
