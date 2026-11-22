/* CPU-only checks of the conditioning disk cache (h3_cond_cache.c): round
 * trip, key sensitivity (prompt, media content, env), corruption -> miss,
 * LRU eviction. Usage: h3_cond_cache_test SCRATCH_DIR */
#include "h3_cond_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)

static void write_file(const char *path, const char *text) {
    FILE *f = fopen(path, "wb");
    fputs(text, f);
    fclose(f);
}

static h3_cond_disk_inputs inputs(const char *root, const char *prompt, const h3_params *params) {
    static char tok[1024], te[1024], vae[1024], avae[1024], sh[1024];
    snprintf(tok, sizeof tok, "%s/tokenizer.json", root);
    snprintf(te, sizeof te, "%s/te", root);
    snprintf(vae, sizeof vae, "%s/vae", root);
    snprintf(avae, sizeof avae, "%s/avae", root);
    snprintf(sh, sizeof sh, "%s/h3_shaders.metal", root);
    return (h3_cond_disk_inputs){prompt, params, 512, 512, 0, tok, te, vae, avae, sh, "test-gpu"};
}

static int roundtrip(h3_cond_disk *disk, size_t tokens, size_t width, size_t video) {
    h3_text_embedding text = {0};
    text.tokens = tokens; text.width = width;
    text.values = malloc(tokens * width * 2);
    text.tags = malloc(tokens);
    for (size_t i = 0; i < tokens * width; i++) text.values[i] = (uint16_t)(i * 2654435761u);
    for (size_t i = 0; i < tokens; i++) text.tags[i] = (uint8_t)(i & 1);
    float *rows = malloc((video ? video : 1) * sizeof(float));
    for (size_t i = 0; i < video; i++) rows[i] = (float)i * 0.25f - 3.0f;
    h3_layout_ref ref = {H3_LAYOUT_REF_IMAGE, 1, 2, 3, 4};
    int ok = h3_cond_disk_store(disk, &text, rows, video, NULL, 0, &ref, 1, 1);
    h3_text_embedding back; float *v, *a; size_t ve, ae, rc; h3_layout_ref *refs; int cond;
    int hit = ok && h3_cond_disk_load(disk, &back, &v, &ve, &a, &ae, &refs, &rc, &cond);
    int same = hit && back.tokens == tokens && back.width == width &&
        !memcmp(back.values, text.values, tokens * width * 2) && back.tags &&
        !memcmp(back.tags, text.tags, tokens) && ve == video &&
        (!video || !memcmp(v, rows, video * sizeof(float))) && ae == 0 && rc == 1 &&
        refs[0].latent_w == 3 && refs[0].audio_t == 4 && cond == 1;
    if (hit) { h3_text_embedding_free(&back); free(v); free(a); free(refs); }
    h3_text_embedding_free(&text);
    free(rows);
    return same;
}

static int loads(h3_cond_disk *disk) {
    h3_text_embedding back; float *v, *a; size_t ve, ae, rc; h3_layout_ref *refs; int cond;
    int hit = h3_cond_disk_load(disk, &back, &v, &ve, &a, &ae, &refs, &rc, &cond);
    if (hit) { h3_text_embedding_free(&back); free(v); free(a); free(refs); }
    return hit;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const char *root = argv[1];
    char path[1024], cache[1024];
    snprintf(path, sizeof path, "%s/te", root); mkdir(path, 0700);
    snprintf(path, sizeof path, "%s/te/model.safetensors", root); write_file(path, "weights");
    snprintf(path, sizeof path, "%s/vae", root); mkdir(path, 0700);
    snprintf(path, sizeof path, "%s/avae", root); mkdir(path, 0700);
    snprintf(path, sizeof path, "%s/tokenizer.json", root); write_file(path, "{}");
    snprintf(path, sizeof path, "%s/h3_shaders.metal", root); write_file(path, "kernel");
    snprintf(path, sizeof path, "%s/first.png", root); write_file(path, "image-a");
    snprintf(cache, sizeof cache, "%s/cache", root);
    setenv("H3_COND_CACHE_DIR", cache, 1);
    h3_params params = H3_PARAMS_DEFAULT;

    h3_cond_disk disk;
    h3_cond_disk_inputs in = inputs(root, "a cat", &params);
    CHECK(h3_cond_disk_open(&disk, &in));
    CHECK(!loads(&disk));                         /* cold: miss */
    CHECK(roundtrip(&disk, 7, 16, 0));            /* store + exact restore */
    char first_path[1024]; snprintf(first_path, sizeof first_path, "%s", disk.path);
    h3_cond_disk_close(&disk);

    in = inputs(root, "a dog", &params);          /* prompt changes the key */
    CHECK(h3_cond_disk_open(&disk, &in));
    CHECK(strcmp(disk.path, first_path) && !loads(&disk));
    h3_cond_disk_close(&disk);

    setenv("H3_SPARSE", "t0=1", 1);               /* DiT-only env: same key */
    setenv("H3_ATTN", "nax", 1);
    in = inputs(root, "a cat", &params);
    CHECK(h3_cond_disk_open(&disk, &in));
    CHECK(!strcmp(disk.path, first_path) && loads(&disk));
    h3_cond_disk_close(&disk);
    setenv("H3_TE", "int8", 1);                   /* other H3_ env: new key */
    CHECK(h3_cond_disk_open(&disk, &in));
    CHECK(strcmp(disk.path, first_path));
    h3_cond_disk_close(&disk);
    unsetenv("H3_TE");

    snprintf(path, sizeof path, "%s/first.png", root);
    params.first_frame = path;                    /* media: content-keyed */
    in = inputs(root, "a cat", &params);
    CHECK(h3_cond_disk_open(&disk, &in));
    CHECK(roundtrip(&disk, 5, 8, 96));
    char media_path[1024]; snprintf(media_path, sizeof media_path, "%s", disk.path);
    h3_cond_disk_close(&disk);
    char copy[1024]; snprintf(copy, sizeof copy, "%s/copy.png", root);
    write_file(copy, "image-a");                  /* same bytes, other path: hit */
    params.first_frame = copy;
    CHECK(h3_cond_disk_open(&disk, &in));
    CHECK(!strcmp(disk.path, media_path) && loads(&disk));
    h3_cond_disk_close(&disk);
    write_file(copy, "image-b");                  /* other bytes: miss */
    CHECK(h3_cond_disk_open(&disk, &in));
    CHECK(strcmp(disk.path, media_path) && !loads(&disk));
    h3_cond_disk_close(&disk);
    params.first_frame = NULL;

    in = inputs(root, "a cat", &params);          /* corruption: flip a payload byte */
    CHECK(h3_cond_disk_open(&disk, &in));
    FILE *f = fopen(disk.path, "r+b");
    fseek(f, -3, SEEK_END); int c = fgetc(f); fseek(f, -3, SEEK_END); fputc(c ^ 0x40, f); fclose(f);
    CHECK(!loads(&disk));
    CHECK(access(disk.path, F_OK) != 0);         /* bad entry removed */
    CHECK(roundtrip(&disk, 7, 16, 0));            /* and recomputed */
    truncate(disk.path, 100);                     /* truncated */
    CHECK(!loads(&disk));
    write_file(disk.path, "garbage that is not an entry at all, not even a header ......................................................................................................................................");
    CHECK(!loads(&disk));
    h3_cond_disk_close(&disk);

    setenv("H3_COND_CACHE_MAX_MB", "1", 1);       /* LRU: 1 MiB, 300 KiB entries */
    char prompts[8][16];
    for (int i = 0; i < 8; i++) {
        snprintf(prompts[i], sizeof prompts[i], "prompt %d", i);
        in = inputs(root, prompts[i], &params);
        CHECK(h3_cond_disk_open(&disk, &in));
        CHECK(roundtrip(&disk, 300, 512, 0));
        h3_cond_disk_close(&disk);
        if (i == 2) {   /* touch entry 0: now 1 is the least recently used */
            sleep(1);
            in = inputs(root, prompts[0], &params);
            CHECK(h3_cond_disk_open(&disk, &in) && loads(&disk));
            h3_cond_disk_close(&disk);
            sleep(1);
        }
        if (i == 3) {   /* storing 3 evicted 1, kept the touched 0 */
            in = inputs(root, prompts[1], &params);
            h3_cond_disk_open(&disk, &in);
            CHECK(access(disk.path, F_OK) != 0);
            h3_cond_disk_close(&disk);
            in = inputs(root, prompts[0], &params);
            h3_cond_disk_open(&disk, &in);
            CHECK(access(disk.path, F_OK) == 0);
            h3_cond_disk_close(&disk);
        }
    }
    int alive = 0;
    for (int i = 0; i < 8; i++) {
        in = inputs(root, prompts[i], &params);
        h3_cond_disk_open(&disk, &in);
        alive += access(disk.path, F_OK) == 0;
        h3_cond_disk_close(&disk);
    }
    CHECK(alive == 3);                            /* 3 x 300 KiB fit in 1 MiB */
    unsetenv("H3_COND_CACHE_DIR");
    CHECK(!h3_cond_disk_open(&disk, &in));        /* disabled */
    printf(failures ? "cond cache test: %d FAILED\n" : "cond cache test: OK\n", failures);
    return failures != 0;
}
