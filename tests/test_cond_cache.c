/* CPU-only checks of the conditioning disk cache (h3_cond_cache.c).
 * Usage: h3_cond_cache_test SCRATCH_DIR (an empty directory). */
#include "h3_cond_cache.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)

#define W H3_COND_TEXT_WIDTH
#define FL2VA_ROWS 384u   /* expected keyframe rows given to the cache in these tests */

static char root[1024];

static void write_file(const char *path, const char *text) {
    FILE *f = fopen(path, "wb");
    fputs(text, f);
    fclose(f);
}

static const char *at(const char *name) {
    static char buffers[8][1024];
    static int next;
    char *out = buffers[next++ & 7];
    snprintf(out, 1024, "%s/%s", root, name);
    return out;
}

static h3_cond_disk_inputs inputs(const char *prompt, const h3_params *params, int ref2va) {
    static char tok[1024], te[1024], vae[1024], avae[1024], sh[1024];
    snprintf(tok, sizeof tok, "%s/tokenizer.json", root);
    snprintf(te, sizeof te, "%s/te", root);
    snprintf(vae, sizeof vae, "%s/video_vae/source", root);
    snprintf(avae, sizeof avae, "%s/avae", root);
    snprintf(sh, sizeof sh, "%s/h3_shaders.metal", root);
    uint64_t rows = (!ref2va && (params->first_frame || params->last_frame)) ? FL2VA_ROWS : 0;
    h3_cond_disk_inputs in = {prompt, params, 512, 512, ref2va, tok, te, vae, avae, sh, "test-gpu", rows};
    return in;
}

typedef struct {
    size_t tokens;
    int tags;          /* 0: none; 1: valid (0/1); 3: one invalid value */
    size_t width;
    size_t video;
    size_t audio;
    h3_layout_ref refs[4];
    size_t ref_count;
    int conditioned;
} entry;

/* Store e, load it back: 1 = hit with identical bytes, 0 = miss, -1 = hit but different. */
static int roundtrip(h3_cond_disk *disk, const entry *e) {
    h3_text_embedding text = {0};
    text.tokens = e->tokens;
    text.width = e->width ? e->width : W;
    text.values = malloc(text.tokens * text.width * 2);
    for (size_t i = 0; i < text.tokens * text.width; i++) text.values[i] = (uint16_t)(i * 2654435761u);
    if (e->tags) {
        text.tags = malloc(text.tokens);
        for (size_t i = 0; i < text.tokens; i++) text.tags[i] = (uint8_t)(i & 1);
        if (e->tags == 3) text.tags[0] = 3;
    }
    float *video = malloc((e->video ? e->video : 1) * sizeof(float));
    float *audio = malloc((e->audio ? e->audio : 1) * sizeof(float));
    for (size_t i = 0; i < e->video; i++) video[i] = (float)i * 0.25f - 3.0f;
    for (size_t i = 0; i < e->audio; i++) audio[i] = (float)i * -0.5f;
    int stored = h3_cond_disk_store(disk, &text, video, e->video, audio, e->audio,
                                    e->refs, e->ref_count, e->conditioned);
    int result = 0;
    h3_text_embedding back; float *v, *a; size_t ve, ae, rc; h3_layout_ref *refs; int cond;
    if (stored && h3_cond_disk_load(disk, &back, &v, &ve, &a, &ae, &refs, &rc, &cond)) {
        int same = back.tokens == text.tokens && back.width == text.width &&
            !memcmp(back.values, text.values, text.tokens * text.width * 2) &&
            (!e->tags || (back.tags && !memcmp(back.tags, text.tags, text.tokens))) &&
            (e->tags || !back.tags) && ve == e->video && ae == e->audio &&
            (!ve || !memcmp(v, video, ve * sizeof(float))) &&
            (!ae || !memcmp(a, audio, ae * sizeof(float))) && rc == e->ref_count &&
            (!rc || !memcmp(refs, e->refs, rc * sizeof(*refs))) && cond == e->conditioned;
        result = same ? 1 : -1;
        h3_text_embedding_free(&back); free(v); free(a); free(refs);
    }
    h3_text_embedding_free(&text);
    free(video);
    free(audio);
    return result;
}

static int loads(h3_cond_disk *disk) {
    h3_text_embedding back; float *v, *a; size_t ve, ae, rc; h3_layout_ref *refs; int cond;
    int hit = h3_cond_disk_load(disk, &back, &v, &ve, &a, &ae, &refs, &rc, &cond);
    if (hit) { h3_text_embedding_free(&back); free(v); free(a); free(refs); }
    else CHECK(!back.values && !v && !a && !refs && !ve && !ae);
    return hit;
}

static int exists(const char *path) { return access(path, F_OK) == 0; }

static void flip_byte(const char *path, long offset_from_end) {
    FILE *f = fopen(path, "r+b");
    fseek(f, -offset_from_end, SEEK_END); int c = fgetc(f);
    fseek(f, -offset_from_end, SEEK_END); fputc(c ^ 0x40, f); fclose(f);
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    snprintf(root, sizeof root, "%s", argv[1]);
    mkdir(at("te"), 0700); write_file(at("te/model.safetensors"), "weights");
    mkdir(at("video_vae"), 0700); mkdir(at("video_vae/source"), 0700);
    write_file(at("video_vae/source/model.safetensors"), "vae");
    write_file(at("video_vae/config.json"), "{\"latents_mean\": [0.1]}");
    mkdir(at("avae"), 0700); write_file(at("avae/config.json"), "{}");
    write_file(at("tokenizer.json"), "{}");
    write_file(at("h3_shaders.metal"), "kernel");
    write_file(at("first.png"), "image-a");
    write_file(at("ref.png"), "ref-image");
    setenv("H3_COND_CACHE_DIR", at("cache"), 1);
    h3_params t2v = H3_PARAMS_DEFAULT;
    h3_cond_disk disk;
    h3_cond_disk_inputs in;

    /* ---- T2V round trip, key sensitivity */
    in = inputs("a cat", &t2v, 0);
    CHECK(h3_cond_disk_open(&disk, &in));
    CHECK(!loads(&disk));                                   /* cold miss */
    entry e_t2v = {.tokens = 7};
    CHECK(roundtrip(&disk, &e_t2v) == 1);                   /* exact restore */
    char t2v_path[1024]; snprintf(t2v_path, sizeof t2v_path, "%s", disk.path);
    h3_cond_disk_close(&disk);
    in = inputs("a dog", &t2v, 0);
    CHECK(h3_cond_disk_open(&disk, &in));
    CHECK(strcmp(disk.path, t2v_path) && !loads(&disk));    /* prompt -> other key */
    h3_cond_disk_close(&disk);
    setenv("H3_SPARSE", "t0=1", 1); setenv("H3_ATTN", "nax", 1);   /* DiT-only env */
    in = inputs("a cat", &t2v, 0);
    CHECK(h3_cond_disk_open(&disk, &in));
    CHECK(!strcmp(disk.path, t2v_path) && loads(&disk));
    h3_cond_disk_close(&disk);
    setenv("H3_TE", "int8", 1);                             /* other H3_ env */
    CHECK(h3_cond_disk_open(&disk, &in));
    CHECK(strcmp(disk.path, t2v_path));
    h3_cond_disk_close(&disk);
    unsetenv("H3_TE");

    /* ---- FL2VA: content-keyed media, video VAE config in the key (finding 1) */
    h3_params fl2va = H3_PARAMS_DEFAULT;
    fl2va.first_frame = at("first.png");
    in = inputs("a cat", &fl2va, 0);
    CHECK(h3_cond_disk_open(&disk, &in));
    entry e_fl = {.tokens = 5, .tags = 1, .video = FL2VA_ROWS, .conditioned = 1};
    CHECK(roundtrip(&disk, &e_fl) == 1);
    char fl_path[1024]; snprintf(fl_path, sizeof fl_path, "%s", disk.path);
    h3_cond_disk_close(&disk);
    write_file(at("copy.png"), "image-a");                  /* same bytes, other path */
    fl2va.first_frame = at("copy.png");
    in = inputs("a cat", &fl2va, 0);
    CHECK(h3_cond_disk_open(&disk, &in));
    CHECK(!strcmp(disk.path, fl_path) && loads(&disk));
    h3_cond_disk_close(&disk);
    write_file(at("copy.png"), "image-b");                  /* other bytes */
    CHECK(h3_cond_disk_open(&disk, &in));
    CHECK(strcmp(disk.path, fl_path) && !loads(&disk));
    h3_cond_disk_close(&disk);
    fl2va.first_frame = at("first.png");
    write_file(at("video_vae/config.json"), "{\"latents_mean\": [0.2]}");
    in = inputs("a cat", &fl2va, 0);
    CHECK(h3_cond_disk_open(&disk, &in));
    CHECK(strcmp(disk.path, fl_path));                      /* latents_mean changed */
    h3_cond_disk_close(&disk);
    in = inputs("a cat", &t2v, 0);
    CHECK(h3_cond_disk_open(&disk, &in));
    CHECK(!strcmp(disk.path, t2v_path));                    /* T2V does not read it */
    h3_cond_disk_close(&disk);
    write_file(at("video_vae/config.json"), "{\"latents_mean\": [0.1]}");
    fl2va.first_frame = at("missing.png");                  /* unidentifiable input */
    in = inputs("a cat", &fl2va, 0);
    CHECK(!h3_cond_disk_open(&disk, &in));
    fl2va.first_frame = at("first.png");

    /* ---- entries inconsistent with the request: miss, dropped (finding 2) */
    struct { const char *what; int req; entry e; } bad[] = {   /* req: 0 T2V, 1 FL2VA, 2 Ref2VA */
        {"T2V, wrong text width", 0, {.tokens = 7, .width = 16}},
        {"T2V, tags present", 0, {.tokens = 7, .tags = 1}},
        {"T2V, conditioned", 0, {.tokens = 7, .conditioned = 1}},
        {"T2V, keyframe rows", 0, {.tokens = 7, .video = 96}},
        {"FL2VA, tag value 3", 1, {.tokens = 5, .tags = 3, .video = FL2VA_ROWS, .conditioned = 1}},
        {"FL2VA, wrong row count", 1, {.tokens = 5, .tags = 1, .video = FL2VA_ROWS - 96, .conditioned = 1}},
        {"FL2VA, not conditioned", 1, {.tokens = 5, .tags = 1, .video = FL2VA_ROWS}},
        {"FL2VA, audio rows", 1, {.tokens = 5, .tags = 1, .video = FL2VA_ROWS, .audio = 64, .conditioned = 1}},
        {"Ref2VA, no references", 2, {.tokens = 5, .tags = 1, .video = 96, .conditioned = 1}},
        {"Ref2VA, negative latent", 2, {.tokens = 5, .tags = 1, .video = 0, .conditioned = 1, .ref_count = 1,
                                         .refs = {{H3_LAYOUT_REF_IMAGE, 1, -4, 4, 0}}}},
        {"Ref2VA, kind audio for an image", 2, {.tokens = 5, .tags = 1, .audio = 64, .conditioned = 1, .ref_count = 1,
                                                 .refs = {{H3_LAYOUT_REF_AUDIO, 0, 0, 0, 1}}}},
        {"Ref2VA, rows != geometry", 2, {.tokens = 5, .tags = 1, .video = 96, .conditioned = 1, .ref_count = 1,
                                          .refs = {{H3_LAYOUT_REF_IMAGE, 1, 4, 4, 0}}}},
    };
    h3_params ref = H3_PARAMS_DEFAULT;
    h3_reference image_ref = {H3_REFERENCE_IMAGE, at("ref.png"), NULL, 0};
    ref.references = &image_ref;
    ref.reference_count = 1;
    for (size_t i = 0; i < sizeof bad / sizeof *bad; i++) {
        in = bad[i].req == 2 ? inputs("ref", &ref, 1) :
             bad[i].req == 1 ? inputs("a cat", &fl2va, 0) : inputs("a cat", &t2v, 0);
        CHECK(h3_cond_disk_open(&disk, &in));
        int r = roundtrip(&disk, &bad[i].e);
        if (r != 0) fprintf(stderr, "  case: %s\n", bad[i].what);
        CHECK(r == 0);
        CHECK(!exists(disk.path));
        h3_cond_disk_close(&disk);
    }
    in = inputs("ref", &ref, 1);                            /* the valid Ref2VA entry hits */
    CHECK(h3_cond_disk_open(&disk, &in));
    entry e_ref = {.tokens = 5, .tags = 1, .video = 4 * 4 / 4 * 96, .conditioned = 1, .ref_count = 1,
                   .refs = {{H3_LAYOUT_REF_IMAGE, 1, 4, 4, 0}}};
    CHECK(roundtrip(&disk, &e_ref) == 1);
    h3_cond_disk_close(&disk);

    /* ---- crafted header with wrapping sizes: miss, dropped, no crash (finding 3) */
    in = inputs("a cat", &t2v, 0);
    CHECK(h3_cond_disk_open(&disk, &in));
    struct {
        char magic[8]; uint32_t version, header_bytes;
        uint64_t key_bytes, payload_bytes, tokens, width, video, audio, refs;
        uint32_t conditioned, has_tags; unsigned char sha[64];
    } h = {{'H','3','C','O','N','D','C','1'}, 2, 0, 0, 0, 1ull << 62, W, UINT64_MAX / 2, 0, 1ull << 62, 0, 0, {0}};
    h.header_bytes = sizeof h;
    h.key_bytes = strlen(disk.key);
    FILE *f = fopen(disk.path, "wb"); fwrite(&h, sizeof h, 1, f); fputs(disk.key, f); fclose(f);
    CHECK(!loads(&disk) && !exists(disk.path));
    h.tokens = 7; h.video = 0; h.refs = (UINT64_MAX / 20) + 2;   /* refs * 20 wraps */
    f = fopen(disk.path, "wb"); fwrite(&h, sizeof h, 1, f); fputs(disk.key, f); fclose(f);
    CHECK(!loads(&disk) && !exists(disk.path));

    /* ---- corruption: payload byte, header field, truncation, garbage */
    CHECK(roundtrip(&disk, &e_t2v) == 1);
    flip_byte(disk.path, 3);
    CHECK(!loads(&disk) && !exists(disk.path));
    CHECK(roundtrip(&disk, &e_t2v) == 1);
    f = fopen(disk.path, "r+b"); fseek(f, 72, SEEK_SET); fputc(1, f); fclose(f);   /* conditioned field: covered by the entry hash */
    CHECK(!loads(&disk) && !exists(disk.path));
    CHECK(roundtrip(&disk, &e_t2v) == 1);
    truncate(disk.path, 100);
    CHECK(!loads(&disk) && !exists(disk.path));
    write_file(disk.path, "garbage that is not an entry at all ..........................................................................................................................................................................");
    CHECK(!loads(&disk) && !exists(disk.path));

    /* ---- transient: a smaller limit (another process) is a miss that keeps the file (finding 4) */
    entry big = {.tokens = 200};                             /* ~2 MB */
    CHECK(roundtrip(&disk, &big) == 1);
    h3_cond_disk_close(&disk);
    setenv("H3_COND_CACHE_MAX_MB", "1", 1);
    CHECK(h3_cond_disk_open(&disk, &in));
    CHECK(!loads(&disk) && exists(disk.path));
    h3_cond_disk_close(&disk);
    unsetenv("H3_COND_CACHE_MAX_MB");
    CHECK(h3_cond_disk_open(&disk, &in));
    CHECK(loads(&disk));                                    /* still valid */
    /* a symlink at the entry path is not followed, and nothing is deleted */
    rename(disk.path, at("real.h3cc"));
    symlink(at("real.h3cc"), disk.path);
    CHECK(!loads(&disk));
    CHECK(exists(at("real.h3cc")));
    struct stat ls;
    CHECK(lstat(disk.path, &ls) == 0 && S_ISLNK(ls.st_mode));
    unlink(disk.path);
    h3_cond_disk_close(&disk);

    /* ---- H3_COND_CACHE_MAX_MB parsing (finding 5) */
    struct { const char *value; uint64_t bytes; } limits[] = {
        {"99999999999999999999", (1ull << 20) << 20}, {"18446744073709551615", (1ull << 20) << 20},
        {"abc", 2048ull << 20}, {"-5", 2048ull << 20}, {"0", 1ull << 20}, {"300", 300ull << 20}};
    for (size_t i = 0; i < sizeof limits / sizeof *limits; i++) {
        setenv("H3_COND_CACHE_MAX_MB", limits[i].value, 1);
        CHECK(h3_cond_disk_open(&disk, &in) && disk.max_bytes == limits[i].bytes);
        h3_cond_disk_close(&disk);
    }
    unsetenv("H3_COND_CACHE_MAX_MB");

    /* ---- store failure: read-only directory, nothing of anyone else touched */
    write_file(at("cache/.tmp-foreign"), "someone else's temporary file");
    CHECK(h3_cond_disk_open(&disk, &in));
    unlink(disk.path);
    chmod(at("cache"), 0500);
    CHECK(roundtrip(&disk, &e_t2v) == 0);
    chmod(at("cache"), 0700);
    CHECK(exists(at("cache/.tmp-foreign")));
    h3_cond_disk_close(&disk);

    /* ---- LRU: 1 MiB, ~307 KiB entries */
    setenv("H3_COND_CACHE_MAX_MB", "1", 1);
    char prompts[8][16];
    entry lru = {.tokens = 30};
    for (int i = 0; i < 8; i++) {
        snprintf(prompts[i], sizeof prompts[i], "prompt %d", i);
        in = inputs(prompts[i], &t2v, 0);
        CHECK(h3_cond_disk_open(&disk, &in));
        CHECK(roundtrip(&disk, &lru) == 1);
        h3_cond_disk_close(&disk);
        if (i == 2) {   /* touch 0: now 1 is the least recently used */
            sleep(1);
            in = inputs(prompts[0], &t2v, 0);
            CHECK(h3_cond_disk_open(&disk, &in) && loads(&disk));
            h3_cond_disk_close(&disk);
            sleep(1);
        }
        if (i == 3) {   /* storing 3 evicted 1 and kept the touched 0 */
            in = inputs(prompts[1], &t2v, 0);
            h3_cond_disk_open(&disk, &in);
            CHECK(!exists(disk.path));
            h3_cond_disk_close(&disk);
            in = inputs(prompts[0], &t2v, 0);
            h3_cond_disk_open(&disk, &in);
            CHECK(exists(disk.path));
            h3_cond_disk_close(&disk);
        }
    }
    int alive = 0;
    for (int i = 0; i < 8; i++) {
        in = inputs(prompts[i], &t2v, 0);
        h3_cond_disk_open(&disk, &in);
        alive += exists(disk.path);
        h3_cond_disk_close(&disk);
    }
    CHECK(alive == 3);
    unsetenv("H3_COND_CACHE_DIR");
    CHECK(!h3_cond_disk_open(&disk, &in));                  /* disabled */
    printf(failures ? "cond cache test: %d FAILED\n" : "cond cache test: OK\n", failures);
    return failures != 0;
}
