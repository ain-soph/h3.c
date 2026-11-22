#ifndef H3_COND_CACHE_H
#define H3_COND_CACHE_H

/* Persistent on-disk cache of the prompt conditioning (H3_COND_CACHE_DIR).
 *
 * Every generation is usually a fresh process, so the in-process conditioning
 * cache never hits there. This cache stores exactly what that one keeps: the
 * text encoder output (BF16 rows + modality tags), the patchified VAE latent
 * rows of keyframes / references, the reference audio latent rows and the
 * reference layout, i.e. everything computed before the seeded augmentation.
 * A hit restores the same bytes, so the render is bit-identical.
 *
 * The key is a SHA-256 over: the prompt and conditioning parameters, the
 * CONTENT of every keyframe / reference file, the identity of the text
 * encoder, tokenizer and VAE files (content hash up to 64 MiB, else path,
 * size, mtime, inode; plus the video VAE config.json above the weights, which
 * holds the latent mean / std), the shader sources, the running executable's
 * content, the OS build and GPU name, the ffmpeg/ffprobe binaries (for media
 * requests) and every H3_ / MTL_ / METAL_ environment variable except those
 * read only by the DiT, the sampler or for diagnostics. Anything that cannot
 * be identified disables the cache for that request.
 *
 * Entries carry a magic, version, the full key text, sizes and a SHA-256 over
 * header + key + payload. A load also checks the payload against the request
 * (text width, tags, reference kinds / geometry, element counts). Corrupt or
 * inconsistent entries are a miss and are removed (only if the file is still
 * the one that was read); transient problems (I/O, memory, a smaller size
 * limit) are a miss that leaves the file alone. The cache never fails a
 * render. Writes go to a temporary file renamed into place; the directory is
 * bounded by H3_COND_CACHE_MAX_MB (default 2048) with LRU eviction (a hit
 * refreshes the mtime). H3_COND_CACHE_DIR unset or empty disables it. */

#include "h3.h"
#include "h3_host.h"
#include "h3_text_encoder.h"

#include <stddef.h>
#include <stdint.h>

#define H3_COND_TEXT_WIDTH 5120u        /* Qwen3-VL hidden size (TEXT_HIDDEN) */
#define H3_COND_MAX_REFERENCES 12u      /* h3_valid_params limit */

typedef struct {
    char *directory;
    char *key;          /* full key text, stored in the entry and compared */
    char *path;         /* <directory>/<sha256(key)>.h3cc */
    uint64_t max_bytes;
    /* What a valid entry for this request must look like. */
    int ref2va;
    int expect_conditioned;
    int expect_tags;
    size_t expect_references;
    uint64_t expect_video_elements;   /* exact for T2V / FL2VA; Ref2VA: from the layout */
    int expect_kinds[H3_COND_MAX_REFERENCES];   /* h3_layout_ref_kind per reference */
} h3_cond_disk;

typedef struct {
    const char *prompt;
    const h3_params *params;
    int render_width;
    int render_height;
    int ref2va;
    const char *tokenizer_path;
    const char *text_path;
    const char *video_vae_path;
    const char *audio_vae_path;
    const char *shader_path;
    const char *device_name;
    /* FL2VA: keyframe count x patchified rows of one keyframe x 96 (what the
     * VAE encoder produces for this render size); 0 for T2V and Ref2VA. */
    uint64_t fl2va_video_elements;
} h3_cond_disk_inputs;

/* Returns 1 when the cache is enabled and the key was built; 0 when disabled
 * or the key could not be built (a warning is printed; generation goes on
 * uncached). */
int h3_cond_disk_open(h3_cond_disk *disk, const h3_cond_disk_inputs *inputs);
void h3_cond_disk_close(h3_cond_disk *disk);

/* 1 on a verified hit (outputs owned by the caller), 0 on a miss (outputs
 * NULL / 0). Never fails otherwise. */
int h3_cond_disk_load(const h3_cond_disk *disk, h3_text_embedding *text,
                      float **video, size_t *video_elements,
                      float **audio, size_t *audio_elements,
                      h3_layout_ref **references, size_t *reference_count,
                      int *conditioned);

/* Best effort: 1 when stored, 0 otherwise (a warning is printed). */
int h3_cond_disk_store(const h3_cond_disk *disk,
                       const h3_text_embedding *text,
                       const float *video, size_t video_elements,
                       const float *audio, size_t audio_elements,
                       const h3_layout_ref *references, size_t reference_count,
                       int conditioned);

#endif
