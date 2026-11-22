/* CPU test of the rotated-pack identity checks (h3_rot.h): the strict marker
 * reader, the per-file __metadata__ cross-check and h3_study_fopen.
 * Usage: h3_rot_pack_test SCRATCH_DIR [PACK_DIR...]  (SCRATCH_DIR: empty;
 * each PACK_DIR must pass and its mask is printed). */
#include "h3_rot.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures;

static void expect(int condition, const char *what) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", what);
        failures++;
    }
}

static void write_text(const char *path, const char *text) {
    FILE *file = fopen(path, "wb");
    fputs(text, file);
    fclose(file);
}

static int marker(const char *dir, const char *text, unsigned *mask) {
    char path[1024], error[512];
    snprintf(path, sizeof(path), "%s/h3-int8-pack.json", dir);
    write_text(path, text);
    return h3_rot_read_marker(path, mask, error, sizeof(error));
}

/* A one-tensor safetensors file with this __metadata__ (or none). */
static void safetensors(const char *dir, const char *name, const char *meta) {
    char path[1024], header[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    int length = snprintf(header, sizeof(header),
        "{%s%s\"x\":{\"dtype\":\"I8\",\"shape\":[4],\"data_offsets\":[0,4]}}"
        "        ", meta ? "\"__metadata__\":" : "", meta ? meta : "");
    if (meta) {
        /* insert the comma after the metadata object */
        char fixed[512];
        snprintf(fixed, sizeof(fixed),
                 "{\"__metadata__\":%s,\"x\":{\"dtype\":\"I8\",\"shape\":[4],"
                 "\"data_offsets\":[0,4]}}        ", meta);
        length = (int)strlen(fixed);
        memcpy(header, fixed, (size_t)length + 1);
    }
    uint64_t size = (uint64_t)length;
    unsigned char prefix[8];
    for (int index = 0; index < 8; index++) prefix[index] = (unsigned char)(size >> (8 * index));
    FILE *file = fopen(path, "wb");
    fwrite(prefix, 1, 8, file);
    fwrite(header, 1, (size_t)length, file);
    fwrite("abcd", 1, 4, file);
    fclose(file);
}

static int check(const char *dir, unsigned mask) {
    char error[1024];
    int ok = h3_rot_check_pack(dir, mask, error, sizeof(error));
    if (!ok && getenv("VERBOSE")) fprintf(stderr, "  (%s)\n", error);
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    const char *scratch = argv[1];
    unsigned mask = 99;
    unsigned qkv_fc1_fc2 = H3_ROT_QKV | H3_ROT_FC1 | H3_ROT_FC2;

    /* marker */
    expect(marker(scratch, "{\"format\": \"h3-int8-pack-v1\", \"source\": \"/a\", "
                  "\"max_steps\": 20, \"verified\": true, \"packed_at\": 1}\n",
                  &mask) && mask == 0, "v1 marker");
    expect(marker(scratch, "{\"format\": \"h3-int8-pack-v2\", \"source\": \"/a\\\"b\\\\c\\u001f\", "
                  "\"rot\": \"qkv,fc1,fc2\", \"rotation\": \"H256\", \"packed_at\": 1}",
                  &mask) && mask == qkv_fc1_fc2, "v2 marker with escapes");
    expect(!marker(scratch, "{\"format\": \"h3-int8-pack-v2\", \"source\": \"/a\"}", &mask),
           "v2 without rot refused");
    expect(!marker(scratch, "{\"format\": \"h3-int8-pack-v2\", \"rot\": \"none\"}", &mask),
           "v2 with empty rotation refused");
    expect(!marker(scratch, "{\"format\": \"h3-int8-pack-v2\", \"rot\": null, "
                  "\"rotation\": \"qkv\"}", &mask), "rot null refused");
    expect(!marker(scratch, "{\"format\": \"h3-int8-pack-v3\", \"rot\": \"qkv\"}", &mask),
           "unknown format refused");
    expect(!marker(scratch, "{\"source\": \"/a\"}", &mask), "missing format refused");
    expect(!marker(scratch, "{\"format\": \"h3-int8-pack-v1\", \"rot\": \"qkv\"}", &mask),
           "v1 naming a rotation refused");
    expect(!marker(scratch, "{\"format\": \"h3-int8-pack-v2\", \"rot\": \"qkv,\"}", &mask),
           "trailing comma in list refused");
    expect(!marker(scratch, "{\"format\": \"h3-int8-pack-v2\", \"rot\": \"qkv\",}", &mask),
           "trailing comma in object refused");
    expect(!marker(scratch, "{\"format\": \"h3-int8-pack-v2\", \"rot\": \"qkv\"} x", &mask),
           "trailing data refused");
    expect(!marker(scratch, "{\"format\": \"h3-int8-pack-v2\", \"rot\": \"qkv\", "
                  "\"rot\": \"fc1\"}", &mask), "duplicate key refused");
    {
        size_t size = 70000;
        char *big = malloc(size + 64);
        int length = snprintf(big, 64, "{\"format\": \"h3-int8-pack-v1\", \"source\": \"");
        memset(big + length, 'a', size);
        strcpy(big + length + size, "\"}");
        expect(!marker(scratch, big, &mask), "oversized marker refused");
        free(big);
    }

    /* per-file identity */
    char pack[1024];
    snprintf(pack, sizeof(pack), "%s/pack", scratch);
    mkdir(pack, 0755);
    expect(!check(pack, 0), "empty directory refused");
    safetensors(pack, "base.safetensors", "{\"format\":\"pt\",\"h3\":\"int8-pack-v1\"}");
    safetensors(pack, "blocks-00.safetensors", "{\"format\":\"pt\",\"h3\":\"int8-pack-v1\"}");
    expect(check(pack, 0), "v1 files match an unrotated marker");
    expect(!check(pack, qkv_fc1_fc2), "v1 files refused under a rotated marker");
    safetensors(pack, "blocks-00.safetensors",
                "{\"format\":\"pt\",\"h3\":\"int8-pack-v2\",\"h3_rot\":\"qkv,fc1,fc2\"}");
    expect(!check(pack, qkv_fc1_fc2), "mixed pack refused");
    expect(!check(pack, 0), "mixed pack refused (unrotated marker)");
    safetensors(pack, "base.safetensors",
                "{\"format\":\"pt\",\"h3\":\"int8-pack-v2\",\"h3_rot\":\"fc2,qkv,fc1\"}");
    expect(check(pack, qkv_fc1_fc2), "v2 files match the rotated marker");
    expect(!check(pack, 0), "rotated files refused under an unrotated marker");
    expect(!check(pack, H3_ROT_QKV | H3_ROT_FC1), "different rotation refused");
    safetensors(pack, "blocks-01.safetensors", NULL);
    expect(!check(pack, qkv_fc1_fc2), "file without metadata refused");
    safetensors(pack, "blocks-01.safetensors", "{\"format\":\"pt\",\"h3\":\"int8-pack-v2\"}");
    expect(!check(pack, qkv_fc1_fc2), "v2 file without h3_rot refused");

    /* study output files */
    char target[1024], link[1024];
    snprintf(target, sizeof(target), "%s/target", scratch);
    snprintf(link, sizeof(link), "%s/link", scratch);
    write_text(target, "keep");
    symlink(target, link);
    FILE *file = h3_study_fopen(link, 0);
    expect(file == NULL, "study output through a symlink refused");
    if (file) fclose(file);
    file = h3_study_fopen(scratch, 0);
    expect(file == NULL, "study output onto a directory refused");
    if (file) fclose(file);
    file = h3_study_fopen(target, 1);
    expect(file != NULL, "study output append to a regular file");
    if (file) fclose(file);

    /* real packs */
    for (int index = 2; index < argc; index++) {
        char path[1024], error[1024], list[96];
        unsigned real = 0;
        snprintf(path, sizeof(path), "%s/h3-int8-pack.json", argv[index]);
        int ok = h3_rot_read_marker(path, &real, error, sizeof(error)) &&
                 h3_rot_check_pack(argv[index], real, error, sizeof(error));
        h3_rot_format(real, list, sizeof(list));
        printf("%s: %s %s\n", argv[index], ok ? "ok" : "REFUSED", ok ? list : error);
        if (!ok) failures++;
    }
    printf(failures ? "rot pack test: %d FAILED\n" : "rot pack test: OK\n", failures);
    return failures ? 1 : 0;
}
