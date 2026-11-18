/* Run the Qwen text encoder on one prompt (h3's T2VA path: tokenizer with the
 * chat template, 50 layers) and write the BF16 rows to a raw file.
 * Usage: h3_te_dump TEXT_ENCODER_DIR TOKENIZER_JSON "prompt" OUT.bf16 */
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s TEXT_ENCODER_DIR TOKENIZER_JSON PROMPT OUT\n",
                argv[0]);
        return 2;
    }
    char error[512] = {0};
    h3_tokenizer *tokenizer = h3_tokenizer_load(argv[2], error, sizeof(error));
    if (!tokenizer) { fprintf(stderr, "%s\n", error); return 1; }
    uint32_t *ids = NULL;
    size_t count = 0;
    if (!h3_tokenizer_encode(tokenizer, argv[3], 1, &ids, &count, error,
                             sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    h3_text_embedding text;
    double start = seconds();
    if (!h3_text_encode_bf16(argv[1], "h3_shaders.metal", ids, count, NULL,
                             NULL, &text, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    double elapsed = seconds() - start;
    FILE *out = fopen(argv[4], "wb");
    if (!out || fwrite(text.values, 2, text.tokens * text.width, out) !=
                    text.tokens * text.width) {
        fprintf(stderr, "cannot write %s\n", argv[4]);
        return 1;
    }
    fclose(out);
    printf("tokens %zu width %zu seconds %.3f\n", text.tokens, text.width,
           elapsed);
    h3_text_embedding_free(&text);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tokenizer);
    return 0;
}
