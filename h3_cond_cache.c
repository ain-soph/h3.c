#include "h3_cond_cache.h"

#include <CommonCrypto/CommonDigest.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

#define H3CC_MAGIC "H3CONDC1"
#define H3CC_VERSION 2u                   /* 2: entry hash covers the header */
#define H3CC_SMALL_FILE (64ull << 20)     /* content-hash files up to this size */
#define H3CC_MAX_TOKENS (1u << 20)
#define H3CC_MAX_LATENT 4096              /* per latent axis */
#define H3CC_MAX_AUDIO_T (1 << 20)
#define H3CC_MAX_MB (1ull << 20)          /* H3_COND_CACHE_MAX_MB clamp (1 TiB) */

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint64_t key_bytes;
    uint64_t payload_bytes;
    uint64_t tokens;
    uint64_t width;
    uint64_t video_elements;
    uint64_t audio_elements;
    uint64_t reference_count;
    uint32_t conditioned;
    uint32_t has_tags;
    unsigned char key_sha[CC_SHA256_DIGEST_LENGTH];
    /* SHA-256 of this header (with entry_sha zeroed) + key + payload */
    unsigned char entry_sha[CC_SHA256_DIGEST_LENGTH];
} h3cc_header;

typedef struct {
    char *text;
    size_t length;
    size_t capacity;
    int failed;
} h3cc_text;

static void h3cc_append(h3cc_text *key, const char *format, ...)
    __attribute__((format(printf, 2, 3)));

static void h3cc_append(h3cc_text *key, const char *format, ...) {
    if (key->failed) return;
    va_list arguments;
    va_start(arguments, format);
    va_list copy;
    va_copy(copy, arguments);
    int needed = vsnprintf(NULL, 0, format, copy);
    va_end(copy);
    if (needed < 0) {
        key->failed = 1;
        va_end(arguments);
        return;
    }
    size_t wanted = key->length + (size_t)needed + 1;
    if (wanted > key->capacity) {
        size_t capacity = key->capacity ? key->capacity : 1024;
        while (capacity < wanted) capacity *= 2;
        char *grown = realloc(key->text, capacity);
        if (!grown) {
            key->failed = 1;
            va_end(arguments);
            return;
        }
        key->text = grown;
        key->capacity = capacity;
    }
    vsnprintf(key->text + key->length, key->capacity - key->length, format,
              arguments);
    va_end(arguments);
    key->length += (size_t)needed;
}

/* snprintf into a fixed buffer; 0 when it did not fit. */
static int h3cc_format(char *out, size_t size, const char *format, ...)
    __attribute__((format(printf, 3, 4)));

static int h3cc_format(char *out, size_t size, const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    int written = vsnprintf(out, size, format, arguments);
    va_end(arguments);
    return written >= 0 && (size_t)written < size;
}

static void h3cc_hex(const unsigned char *digest, char *out) {
    static const char digits[] = "0123456789abcdef";
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
        out[2 * i] = digits[digest[i] >> 4];
        out[2 * i + 1] = digits[digest[i] & 15];
    }
    out[2 * CC_SHA256_DIGEST_LENGTH] = '\0';
}

static void h3cc_sha_update(CC_SHA256_CTX *context, const void *data,
                            size_t bytes) {
    const unsigned char *p = data;
    while (bytes) {
        CC_LONG chunk = bytes > (1u << 30) ? (1u << 30) : (CC_LONG)bytes;
        CC_SHA256_Update(context, p, chunk);
        p += chunk;
        bytes -= chunk;
    }
}

static void h3cc_sha(const void *data, size_t bytes, unsigned char *digest) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    h3cc_sha_update(&context, data, bytes);
    CC_SHA256_Final(digest, &context);
}

static void h3cc_entry_sha(const h3cc_header *header, const void *key,
                           const void *payload, unsigned char *digest) {
    h3cc_header copy = *header;
    memset(copy.entry_sha, 0, sizeof(copy.entry_sha));
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    h3cc_sha_update(&context, &copy, sizeof(copy));
    h3cc_sha_update(&context, key, (size_t)header->key_bytes);
    h3cc_sha_update(&context, payload, (size_t)header->payload_bytes);
    CC_SHA256_Final(digest, &context);
}

/* SHA-256 of a file's content into hex; 0 on a read error. */
static int h3cc_file_sha(const char *path, char *hex) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    unsigned char *buffer = malloc(1 << 20);
    if (!buffer) {
        fclose(file);
        return 0;
    }
    size_t got;
    while ((got = fread(buffer, 1, 1 << 20, file)) > 0)
        CC_SHA256_Update(&context, buffer, (CC_LONG)got);
    int ok = !ferror(file);
    fclose(file);
    free(buffer);
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    h3cc_hex(digest, hex);
    return ok;
}

/* One file: its content hash (media and small files) or, for large weight
 * files, the resolved path, size, mtime, device and inode. A file that cannot
 * be identified disables the cache for this request. */
static void h3cc_file(h3cc_text *key, const char *role, const char *path,
                      int content) {
    if (!path) {
        h3cc_append(key, "\n%s=none", role);
        return;
    }
    struct stat status;
    char resolved[PATH_MAX];
    if (stat(path, &status) != 0 || !S_ISREG(status.st_mode) ||
        !realpath(path, resolved)) {
        key->failed = 1;
        return;
    }
    if (content || (uint64_t)status.st_size <= H3CC_SMALL_FILE) {
        char hex[2 * CC_SHA256_DIGEST_LENGTH + 1];
        if (!h3cc_file_sha(path, hex)) {
            key->failed = 1;
            return;
        }
        h3cc_append(key, "\n%s=sha256:%s:%lld", role, hex,
                    (long long)status.st_size);
        if (!content) h3cc_append(key, ":%s", resolved);
        return;
    }
    h3cc_append(key, "\n%s=stat:%s:%lld:%lld.%09ld:%lld:%llu", role, resolved,
                (long long)status.st_size,
                (long long)status.st_mtimespec.tv_sec,
                status.st_mtimespec.tv_nsec, (long long)status.st_dev,
                (unsigned long long)status.st_ino);
}

static int h3cc_compare_names(const void *left, const void *right) {
    return strcmp(*(char *const *)left, *(char *const *)right);
}

/* Every regular file of a weight directory (symlinks followed), sorted. */
static void h3cc_directory(h3cc_text *key, const char *role,
                           const char *directory) {
    DIR *dir = opendir(directory);
    if (!dir) {
        key->failed = 1;
        return;
    }
    char **names = NULL;
    size_t count = 0, capacity = 0;
    struct dirent *entry;
    while ((entry = readdir(dir))) {
        if (entry->d_name[0] == '.') continue;
        if (count == capacity) {
            capacity = capacity ? capacity * 2 : 16;
            char **grown = realloc(names, capacity * sizeof(*names));
            if (!grown) {
                key->failed = 1;
                break;
            }
            names = grown;
        }
        names[count] = strdup(entry->d_name);
        if (!names[count]) {
            key->failed = 1;
            break;
        }
        count++;
    }
    closedir(dir);
    if (count) qsort(names, count, sizeof(*names), h3cc_compare_names);
    for (size_t index = 0; index < count && !key->failed; index++) {
        char path[PATH_MAX], label[PATH_MAX];
        if (!h3cc_format(path, sizeof(path), "%s/%s", directory, names[index]) ||
            !h3cc_format(label, sizeof(label), "%s/%s", role, names[index])) {
            key->failed = 1;
            break;
        }
        struct stat status;
        if (stat(path, &status) == 0 && S_ISREG(status.st_mode))
            h3cc_file(key, label, path, 0);
    }
    for (size_t index = 0; index < count; index++) free(names[index]);
    free(names);
}

static int h3cc_env_ignored(const char *entry) {
    /* Read only by the DiT, the sampler, the wrapper or for diagnostics:
     * they cannot change the conditioning, so presets share entries. */
    static const char *const prefixes[] = {
        "H3_COND_CACHE", "H3_SPARSE=", "H3_ATTN=", "H3_ATTN_VARIANT=",
        "H3_ATTN_STAGE=", "H3_TOKEN_REDUCTION", "H3_DISABLE_TOKEN_REDUCTION=",
        "H3_REUSE_STEPS=", "H3_CPU_SAMPLER=", "H3_GPU_SAMPLER", "H3_DIT_",
        "H3_PROFILE=", "H3_DEBUG_HASH=", "H3_VIDEO_", "H3_MEDIA_",
        /* Track A study outputs / DiT-only inputs (H3_TEXT_EMB_IN/OUT turn
         * the cache off instead). */
        "H3_DUMP_LATENTS=", "H3_DUMP_FRAMES=", "H3_TEACHER_FORCE_"};
    for (size_t i = 0; i < sizeof(prefixes) / sizeof(*prefixes); i++)
        if (!strncmp(entry, prefixes[i], strlen(prefixes[i]))) return 1;
    return 0;
}

static void h3cc_environment(h3cc_text *key) {
    size_t count = 0;
    for (char **e = environ; *e; e++) count++;
    char **list = malloc((count ? count : 1) * sizeof(*list));
    if (!list) {
        key->failed = 1;
        return;
    }
    size_t kept = 0;
    for (char **e = environ; *e; e++)
        if ((!strncmp(*e, "H3_", 3) || !strncmp(*e, "MTL_", 4) ||
             !strncmp(*e, "METAL_", 6)) && !h3cc_env_ignored(*e))
            list[kept++] = *e;
    if (kept) qsort(list, kept, sizeof(*list), h3cc_compare_names);
    for (size_t i = 0; i < kept; i++)
        h3cc_append(key, "\nenv=%zu:%s", strlen(list[i]), list[i]);
    free(list);
}

/* The program on PATH that posix_spawnp would run (or the override). */
static void h3cc_program(h3cc_text *key, const char *role, const char *env,
                         const char *name) {
    const char *override = getenv(env);
    if (override && *override) name = override;
    if (strchr(name, '/')) {
        h3cc_file(key, role, name, 1);
        return;
    }
    const char *path = getenv("PATH");
    char *copy = strdup(path && *path ? path : "/usr/bin:/bin");
    if (!copy) {
        key->failed = 1;
        return;
    }
    char *save = NULL;
    for (char *dir = strtok_r(copy, ":", &save); dir;
         dir = strtok_r(NULL, ":", &save)) {
        char candidate[PATH_MAX];
        if (!h3cc_format(candidate, sizeof(candidate), "%s/%s",
                         *dir ? dir : ".", name)) continue;
        struct stat status;
        if (stat(candidate, &status) == 0 && S_ISREG(status.st_mode) &&
            access(candidate, X_OK) == 0) {
            char resolved[PATH_MAX];
            h3cc_append(key, "\n%s-path=%s", role,
                        realpath(candidate, resolved) ? resolved : candidate);
            h3cc_file(key, role, candidate, 1);
            free(copy);
            return;
        }
    }
    free(copy);
    h3cc_append(key, "\n%s=absent", role);
}

static void h3cc_executable(h3cc_text *key) {
    char path[PATH_MAX];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0) {
        key->failed = 1;
        return;
    }
    h3cc_file(key, "executable", path, 1);
}

static void h3cc_sysctl(h3cc_text *key, const char *name) {
    char value[256];
    size_t size = sizeof(value);
    if (sysctlbyname(name, value, &size, NULL, 0) != 0) {
        h3cc_append(key, "\n%s=?", name);
        return;
    }
    value[sizeof(value) - 1] = '\0';
    h3cc_append(key, "\n%s=%s", name, value);
}

static char *h3cc_sibling(const char *path, const char *name) {
    const char *slash = strrchr(path, '/');
    size_t prefix = slash ? (size_t)(slash - path) + 1 : 0;
    char *result = malloc(prefix + strlen(name) + 1);
    if (!result) return NULL;
    memcpy(result, path, prefix);
    strcpy(result + prefix, name);
    return result;
}

static int h3cc_layout_kind(h3_reference_kind kind) {
    switch (kind) {
    case H3_REFERENCE_IMAGE: return H3_LAYOUT_REF_IMAGE;
    case H3_REFERENCE_AUDIO: return H3_LAYOUT_REF_AUDIO;
    case H3_REFERENCE_VIDEO:
    case H3_REFERENCE_VIDEO_AUDIO: return H3_LAYOUT_REF_VIDEO;
    }
    return -1;
}

/* H3_COND_CACHE_MAX_MB as bytes: default 2048 MiB, clamped to [1, 2^20]. */
static uint64_t h3cc_max_bytes(void) {
    const char *limit = getenv("H3_COND_CACHE_MAX_MB");
    unsigned long long megabytes = 2048;
    if (limit && *limit) {
        char *end = NULL;
        errno = 0;
        unsigned long long parsed = strtoull(limit, &end, 10);
        if (*limit != '-' && end && end != limit && !*end) {
            if (errno == ERANGE || parsed > H3CC_MAX_MB) parsed = H3CC_MAX_MB;
            megabytes = parsed ? parsed : 1;
        }
    }
    return (uint64_t)megabytes << 20;
}

int h3_cond_disk_open(h3_cond_disk *disk, const h3_cond_disk_inputs *in) {
    memset(disk, 0, sizeof(*disk));
    const char *directory = getenv("H3_COND_CACHE_DIR");
    if (!directory || !*directory) return 0;
    const h3_params *params = in->params;
    if (in->ref2va && params->reference_count > H3_COND_MAX_REFERENCES)
        return 0;
    h3cc_text key = {0};
    h3cc_append(&key, "h3 conditioning cache v%u", H3CC_VERSION);
    h3cc_append(&key, "\nmode=%d\nprompt=%zu:%s", in->ref2va,
                strlen(in->prompt), in->prompt);
    int media = in->ref2va || params->first_frame || params->last_frame;
    size_t visual = 0;
    if (media) {
        h3cc_append(&key, "\nrender=%dx%d\nframes=%d\nimage-size=%d",
                    in->render_width, in->render_height, params->frames,
                    params->reference_image_size);
        h3cc_file(&key, "first", params->first_frame, 1);
        h3cc_file(&key, "last", params->last_frame, 1);
        for (size_t index = 0; in->ref2va && index < params->reference_count;
             index++) {
            const h3_reference *reference = &params->references[index];
            int kind = h3cc_layout_kind(reference->kind);
            if (kind < 0) key.failed = 1;
            disk->expect_kinds[index] = kind;
            visual += kind != H3_LAYOUT_REF_AUDIO;
            h3cc_append(&key, "\nref=%d:%d", (int)reference->kind,
                        reference->include_embedded_audio);
            h3cc_file(&key, "ref-media", reference->path, 1);
            h3cc_file(&key, "ref-audio", reference->audio_path, 1);
        }
        if (!in->ref2va)
            visual = (size_t)(params->first_frame != NULL) +
                     (size_t)(params->last_frame != NULL);
        h3cc_program(&key, "ffmpeg", "H3_FFMPEG", "ffmpeg");
        h3cc_program(&key, "ffprobe", "H3_FFPROBE", "ffprobe");
        h3cc_directory(&key, "video-vae", in->video_vae_path);
        /* The video encoder reads latents_mean / latents_std from the config
         * one level above its weights (h3_video_encoder.c). */
        size_t config_size = strlen(in->video_vae_path) + sizeof("/../config.json");
        char *config = malloc(config_size);
        if (config) {
            snprintf(config, config_size, "%s/../config.json", in->video_vae_path);
            h3cc_file(&key, "video-vae-config", config, 1);
            free(config);
        } else {
            key.failed = 1;
        }
        h3cc_directory(&key, "audio-vae", in->audio_vae_path);
    }
    h3cc_file(&key, "tokenizer", in->tokenizer_path, 1);
    h3cc_directory(&key, "text-encoder", in->text_path);
    h3cc_file(&key, "shaders", in->shader_path, 1);
    char *attention = h3cc_sibling(in->shader_path, "h3_attention.metal");
    if (!attention) key.failed = 1;
    else if (access(attention, F_OK) == 0)
        h3cc_file(&key, "attention-shaders", attention, 1);
    else
        h3cc_append(&key, "\nattention-shaders=absent");
    free(attention);
    h3cc_executable(&key);
    h3cc_sysctl(&key, "kern.osversion");
    h3cc_sysctl(&key, "kern.osrelease");
    h3cc_append(&key, "\ngpu=%s", in->device_name ? in->device_name : "?");
    h3cc_environment(&key);
    if (key.failed || !key.text) {
        fprintf(stderr, "h3: conditioning disk cache: cannot identify every "
                        "input; not cached\n");
        free(key.text);
        memset(disk, 0, sizeof(*disk));
        return 0;
    }
    if (mkdir(directory, 0700) != 0 && errno != EEXIST) {
        fprintf(stderr, "h3: conditioning disk cache: cannot create %s: %s\n",
                directory, strerror(errno));
        free(key.text);
        memset(disk, 0, sizeof(*disk));
        return 0;
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    h3cc_sha(key.text, key.length, digest);
    char hex[2 * CC_SHA256_DIGEST_LENGTH + 1];
    h3cc_hex(digest, hex);
    size_t path_size = strlen(directory) + sizeof(hex) + 8;
    disk->path = malloc(path_size);
    disk->directory = strdup(directory);
    if (!disk->path || !disk->directory ||
        !h3cc_format(disk->path, path_size, "%s/%s.h3cc", directory, hex)) {
        free(key.text);
        h3_cond_disk_close(disk);
        return 0;
    }
    disk->key = key.text;
    disk->max_bytes = h3cc_max_bytes();
    disk->ref2va = in->ref2va;
    disk->expect_conditioned = media;
    disk->expect_tags = visual != 0;
    disk->expect_references = in->ref2va ? params->reference_count : 0;
    disk->expect_video_elements = in->ref2va ? 0 : in->fl2va_video_elements;
    return 1;
}

void h3_cond_disk_close(h3_cond_disk *disk) {
    if (!disk) return;
    free(disk->directory);
    free(disk->key);
    free(disk->path);
    memset(disk, 0, sizeof(*disk));
}

/* Payload bytes for the header's counts; 0 on overflow or out-of-range
 * counts (never wraps). */
static int h3cc_sizes(const h3cc_header *header, uint64_t *payload) {
    uint64_t values, video, audio, refs, total;
    if (header->tokens > H3CC_MAX_TOKENS || header->width > (1u << 16) ||
        header->reference_count > H3_COND_MAX_REFERENCES ||
        __builtin_mul_overflow(header->tokens, header->width, &values) ||
        __builtin_mul_overflow(values, (uint64_t)2, &values) ||
        __builtin_mul_overflow(header->video_elements, (uint64_t)4, &video) ||
        __builtin_mul_overflow(header->audio_elements, (uint64_t)4, &audio) ||
        __builtin_mul_overflow(header->reference_count, (uint64_t)20, &refs) ||
        __builtin_add_overflow(values, header->has_tags ? header->tokens : 0,
                               &total) ||
        __builtin_add_overflow(total, video, &total) ||
        __builtin_add_overflow(total, audio, &total) ||
        __builtin_add_overflow(total, refs, &total)) return 0;
    *payload = total;
    return 1;
}

/* Is the restored conditioning what this request's encoders would produce?
 * NULL when it is, else the reason. */
static const char *h3cc_validate(const h3_cond_disk *disk,
                                 const h3cc_header *header,
                                 const uint8_t *tags,
                                 const h3_layout_ref *references) {
    if (header->width != H3_COND_TEXT_WIDTH || !header->tokens)
        return "text shape";
    if ((header->has_tags != 0) != (disk->expect_tags != 0))
        return "modality tags present / absent";
    for (uint64_t i = 0; tags && i < header->tokens; i++)
        if (tags[i] > 2) return "modality tag value";
    if ((header->conditioned != 0) != (disk->expect_conditioned != 0))
        return "conditioned flag";
    if (header->reference_count != disk->expect_references)
        return "reference count";
    if (!disk->ref2va) {
        if (header->video_elements != disk->expect_video_elements ||
            header->audio_elements) return "keyframe rows";
        return NULL;
    }
    uint64_t video = 0, audio = 0;
    for (uint64_t i = 0; i < header->reference_count; i++) {
        const h3_layout_ref *r = &references[i];
        if ((int)r->kind != disk->expect_kinds[i]) return "reference kind";
        if (r->audio_t < 0 || r->audio_t > H3CC_MAX_AUDIO_T)
            return "reference audio length";
        if (r->kind == H3_LAYOUT_REF_AUDIO) {
            if (r->latent_t || r->latent_h || r->latent_w || r->audio_t < 1)
                return "audio reference geometry";
        } else {
            if (r->latent_t < 1 || r->latent_h < 1 || r->latent_w < 1 ||
                r->latent_t > H3CC_MAX_LATENT || r->latent_h > H3CC_MAX_LATENT ||
                r->latent_w > H3CC_MAX_LATENT ||
                (r->kind == H3_LAYOUT_REF_IMAGE && r->audio_t))
                return "visual reference geometry";
            uint64_t span = (uint64_t)r->latent_t * (uint64_t)r->latent_h *
                            (uint64_t)r->latent_w / 4 * 96;
            if (__builtin_add_overflow(video, span, &video))
                return "reference rows";
        }
        if (__builtin_add_overflow(audio, (uint64_t)r->audio_t * 64, &audio))
            return "reference audio rows";
    }
    if (video != header->video_elements || audio != header->audio_elements)
        return "reference rows";
    return NULL;
}

/* Remove a corrupt / inconsistent entry, but only if the path still names
 * the file that was read (another writer may have renamed a new one in). */
static void h3cc_drop(const h3_cond_disk *disk, const struct stat *read,
                      const char *why) {
    struct stat now;
    int same = lstat(disk->path, &now) == 0 && now.st_dev == read->st_dev &&
               now.st_ino == read->st_ino;
    fprintf(stderr, "h3: conditioning disk cache: %s %s (%s); recomputing\n",
            same ? "discarding" : "ignoring", disk->path, why);
    if (same) unlink(disk->path);
}

static void h3cc_miss(const char *path, const char *why) {
    fprintf(stderr, "h3: conditioning disk cache: not using %s (%s); "
                    "recomputing\n", path, why);
}

int h3_cond_disk_load(const h3_cond_disk *disk, h3_text_embedding *text,
                      float **video, size_t *video_elements,
                      float **audio, size_t *audio_elements,
                      h3_layout_ref **references, size_t *reference_count,
                      int *conditioned) {
    memset(text, 0, sizeof(*text));
    *video = NULL;
    *audio = NULL;
    *references = NULL;
    *video_elements = *audio_elements = *reference_count = 0;
    *conditioned = 0;
    int fd = open(disk->path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        if (errno != ENOENT) h3cc_miss(disk->path, strerror(errno));
        return 0;
    }
    struct stat status;
    unsigned char *data = NULL;
    h3_layout_ref *refs = NULL;
    const char *corrupt = NULL;     /* content problem: drop the entry */
    const char *transient = NULL;   /* leave the file alone */
    if (fstat(fd, &status) != 0 || !S_ISREG(status.st_mode)) {
        transient = "not a regular file";
        goto done;
    }
    if (status.st_size < (off_t)sizeof(h3cc_header)) {
        corrupt = "too small";
        goto done;
    }
    if ((uint64_t)status.st_size > disk->max_bytes) {
        transient = "larger than the cache limit";
        goto done;
    }
    size_t size = (size_t)status.st_size;
    data = malloc(size);
    if (!data) {
        transient = "out of memory";
        goto done;
    }
    size_t done_bytes = 0;
    while (done_bytes < size) {
        ssize_t got = read(fd, data + done_bytes, size - done_bytes);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) {
            transient = got < 0 ? "read error" : "file changed while reading";
            goto done;
        }
        done_bytes += (size_t)got;
    }
    h3cc_header header;
    memcpy(&header, data, sizeof(header));
    uint64_t payload_bytes = 0, total = 0;
    size_t key_length = strlen(disk->key);
    if (memcmp(header.magic, H3CC_MAGIC, 8) || header.version != H3CC_VERSION ||
        header.header_bytes != sizeof(header)) {
        corrupt = "bad magic or version";
        goto done;
    }
    if (!h3cc_sizes(&header, &payload_bytes) ||
        payload_bytes != header.payload_bytes ||
        header.key_bytes != key_length ||
        __builtin_add_overflow((uint64_t)sizeof(header), header.key_bytes, &total) ||
        __builtin_add_overflow(total, header.payload_bytes, &total) ||
        total != size) {
        corrupt = "inconsistent sizes";
        goto done;
    }
    const unsigned char *key_bytes = data + sizeof(header);
    const unsigned char *payload = key_bytes + header.key_bytes;
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    if (memcmp(key_bytes, disk->key, key_length)) {
        corrupt = "different key";
        goto done;
    }
    h3cc_sha(key_bytes, key_length, digest);
    if (memcmp(digest, header.key_sha, sizeof(digest))) {
        corrupt = "key hash mismatch";
        goto done;
    }
    h3cc_entry_sha(&header, key_bytes, payload, digest);
    if (memcmp(digest, header.entry_sha, sizeof(digest))) {
        corrupt = "entry hash mismatch";
        goto done;
    }
    size_t values_bytes = (size_t)(header.tokens * header.width * 2);
    const unsigned char *tags = header.has_tags ? payload + values_bytes : NULL;
    const unsigned char *cursor = payload + values_bytes +
        (header.has_tags ? header.tokens : 0) +
        header.video_elements * sizeof(float) +
        header.audio_elements * sizeof(float);
    if (header.reference_count) {
        refs = calloc((size_t)header.reference_count, sizeof(*refs));
        if (!refs) {
            transient = "out of memory";
            goto done;
        }
        for (uint64_t i = 0; i < header.reference_count; i++) {
            int32_t fields[5];
            memcpy(fields, cursor, sizeof(fields));
            cursor += sizeof(fields);
            refs[i] = (h3_layout_ref){(h3_layout_ref_kind)fields[0], fields[1],
                                      fields[2], fields[3], fields[4]};
        }
    }
    corrupt = h3cc_validate(disk, &header, tags, refs);
    if (corrupt) goto done;
    text->values = malloc(values_bytes);
    if (tags) text->tags = malloc((size_t)header.tokens);
    if (header.video_elements)
        *video = malloc((size_t)header.video_elements * sizeof(float));
    if (header.audio_elements)
        *audio = malloc((size_t)header.audio_elements * sizeof(float));
    if (!text->values || (tags && !text->tags) ||
        (header.video_elements && !*video) ||
        (header.audio_elements && !*audio)) {
        transient = "out of memory";
        goto done;
    }
    cursor = payload;
    memcpy(text->values, cursor, values_bytes);
    cursor += values_bytes;
    if (tags) {
        memcpy(text->tags, cursor, (size_t)header.tokens);
        cursor += header.tokens;
    }
    if (header.video_elements)
        memcpy(*video, cursor, (size_t)header.video_elements * sizeof(float));
    cursor += header.video_elements * sizeof(float);
    if (header.audio_elements)
        memcpy(*audio, cursor, (size_t)header.audio_elements * sizeof(float));
    text->tokens = (size_t)header.tokens;
    text->width = (size_t)header.width;
    *video_elements = (size_t)header.video_elements;
    *audio_elements = (size_t)header.audio_elements;
    *reference_count = (size_t)header.reference_count;
    *references = refs;
    *conditioned = (int)header.conditioned;
    free(data);
    futimes(fd, NULL);   /* LRU: a hit makes the entry the newest */
    close(fd);
    return 1;
done:
    h3_text_embedding_free(text);
    memset(text, 0, sizeof(*text));
    free(*video);
    free(*audio);
    *video = NULL;
    *audio = NULL;
    *video_elements = *audio_elements = 0;
    free(refs);
    free(data);
    if (corrupt) h3cc_drop(disk, &status, corrupt);
    else if (transient) h3cc_miss(disk->path, transient);
    close(fd);
    return 0;
}

typedef struct {
    char *path;
    off_t size;
    struct timespec mtime;
} h3cc_entry;

static int h3cc_older(const void *left, const void *right) {
    const h3cc_entry *a = left, *b = right;
    if (a->mtime.tv_sec != b->mtime.tv_sec)
        return a->mtime.tv_sec < b->mtime.tv_sec ? -1 : 1;
    if (a->mtime.tv_nsec != b->mtime.tv_nsec)
        return a->mtime.tv_nsec < b->mtime.tv_nsec ? -1 : 1;
    return strcmp(a->path, b->path);
}

/* Remove the least recently used entries until the total fits, and temporary
 * files older than an hour (a crashed writer). Never removes keep_path. */
static void h3cc_evict(const h3_cond_disk *disk, const char *keep_path) {
    DIR *dir = opendir(disk->directory);
    if (!dir) return;
    h3cc_entry *entries = NULL;
    size_t count = 0, capacity = 0;
    uint64_t total = 0;
    time_t now = time(NULL);
    struct dirent *item;
    while ((item = readdir(dir))) {
        size_t length = strlen(item->d_name);
        int entry = length > 5 && !strcmp(item->d_name + length - 5, ".h3cc");
        int temporary = !strncmp(item->d_name, ".tmp-", 5);
        if (!entry && !temporary) continue;
        char path[PATH_MAX];
        if (!h3cc_format(path, sizeof(path), "%s/%s", disk->directory,
                         item->d_name)) continue;
        struct stat status;
        if (lstat(path, &status) != 0 || !S_ISREG(status.st_mode)) continue;
        if (temporary) {
            if (now - status.st_mtimespec.tv_sec > 3600) unlink(path);
            continue;
        }
        if (count == capacity) {
            capacity = capacity ? capacity * 2 : 64;
            h3cc_entry *grown = realloc(entries, capacity * sizeof(*entries));
            if (!grown) break;
            entries = grown;
        }
        entries[count].path = strdup(path);
        if (!entries[count].path) break;
        entries[count].size = status.st_size;
        entries[count].mtime = status.st_mtimespec;
        total += (uint64_t)status.st_size;
        count++;
    }
    closedir(dir);
    if (count) qsort(entries, count, sizeof(*entries), h3cc_older);
    for (size_t i = 0; i < count && total > disk->max_bytes; i++) {
        if (keep_path && !strcmp(entries[i].path, keep_path)) continue;
        if (unlink(entries[i].path) == 0) total -= (uint64_t)entries[i].size;
    }
    for (size_t i = 0; i < count; i++) free(entries[i].path);
    free(entries);
}

static int h3cc_write_all(int fd, const void *data, size_t bytes) {
    const unsigned char *p = data;
    while (bytes) {
        ssize_t wrote = write(fd, p, bytes > (1u << 30) ? (1u << 30) : bytes);
        if (wrote <= 0) {
            if (wrote < 0 && errno == EINTR) continue;
            return 0;
        }
        p += wrote;
        bytes -= (size_t)wrote;
    }
    return 1;
}

int h3_cond_disk_store(const h3_cond_disk *disk,
                       const h3_text_embedding *text,
                       const float *video, size_t video_elements,
                       const float *audio, size_t audio_elements,
                       const h3_layout_ref *references, size_t reference_count,
                       int conditioned) {
    if (!text || !text->values || !text->tokens || !text->width) return 0;
    h3cc_header header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, H3CC_MAGIC, 8);
    header.version = H3CC_VERSION;
    header.header_bytes = sizeof(header);
    header.key_bytes = strlen(disk->key);
    header.tokens = text->tokens;
    header.width = text->width;
    header.video_elements = video_elements;
    header.audio_elements = audio_elements;
    header.reference_count = reference_count;
    header.conditioned = (uint32_t)(conditioned != 0);
    header.has_tags = text->tags != NULL;
    uint64_t total;
    if (!h3cc_sizes(&header, &header.payload_bytes) ||
        __builtin_add_overflow((uint64_t)sizeof(header), header.key_bytes, &total) ||
        __builtin_add_overflow(total, header.payload_bytes, &total)) return 0;
    if (total > disk->max_bytes) {
        fprintf(stderr, "h3: conditioning disk cache: entry of %llu bytes "
                        "exceeds the cache size; not stored\n",
                (unsigned long long)total);
        return 0;
    }
    unsigned char *payload = malloc((size_t)header.payload_bytes);
    if (!payload) return 0;
    unsigned char *cursor = payload;
    size_t values_bytes = text->tokens * text->width * 2;
    memcpy(cursor, text->values, values_bytes);
    cursor += values_bytes;
    if (text->tags) {
        memcpy(cursor, text->tags, text->tokens);
        cursor += text->tokens;
    }
    if (video_elements) memcpy(cursor, video, video_elements * sizeof(float));
    cursor += video_elements * sizeof(float);
    if (audio_elements) memcpy(cursor, audio, audio_elements * sizeof(float));
    cursor += audio_elements * sizeof(float);
    for (size_t i = 0; i < reference_count; i++) {
        int32_t fields[5] = {(int32_t)references[i].kind,
                             references[i].latent_t, references[i].latent_h,
                             references[i].latent_w, references[i].audio_t};
        memcpy(cursor, fields, sizeof(fields));
        cursor += sizeof(fields);
    }
    h3cc_sha(disk->key, (size_t)header.key_bytes, header.key_sha);
    h3cc_entry_sha(&header, disk->key, payload, header.entry_sha);

    size_t temporary_size = strlen(disk->directory) + 64;
    char *temporary = malloc(temporary_size);
    if (!temporary || !h3cc_format(temporary, temporary_size, "%s/.tmp-%d-%08x",
                                   disk->directory, (int)getpid(),
                                   arc4random())) {
        free(temporary);
        free(payload);
        return 0;
    }
    int fd = open(temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                  0600);
    int ok = 0, error = 0;
    if (fd < 0) {
        error = errno;       /* the name may be someone else's: never unlink it */
    } else {
        ok = h3cc_write_all(fd, &header, sizeof(header)) &&
             h3cc_write_all(fd, disk->key, (size_t)header.key_bytes) &&
             h3cc_write_all(fd, payload, (size_t)header.payload_bytes) &&
             fsync(fd) == 0;
        if (!ok) error = errno;
        if (close(fd) != 0 && ok) {
            ok = 0;
            error = errno;
        }
        if (ok && rename(temporary, disk->path) != 0) {
            ok = 0;
            error = errno;
        }
        if (!ok) unlink(temporary);   /* ours: created with O_EXCL above */
    }
    if (!ok)
        fprintf(stderr, "h3: conditioning disk cache: cannot write %s: %s\n",
                disk->path, strerror(error));
    free(temporary);
    free(payload);
    if (ok) h3cc_evict(disk, disk->path);
    return ok;
}
