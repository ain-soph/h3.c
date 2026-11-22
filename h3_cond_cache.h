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
 * size, mtime, inode), the shader sources, the running executable's content,
 * the OS build and GPU name, the ffmpeg/ffprobe binaries (for media
 * requests) and every H3_ / MTL_ environment variable except those read only
 * by the DiT, the sampler or for diagnostics. Entries carry a magic, version,
 * the full key text, sizes and a payload SHA-256; any mismatch is a miss (the
 * file is removed and recomputed). Writes go to a temporary file renamed into
 * place. Total size is bounded (H3_COND_CACHE_MAX_MB, default 2048) by
 * evicting the least recently used entries (mtime is refreshed on a hit).
 * H3_COND_CACHE_DIR unset or empty disables it. */

#include "h3.h"
#include "h3_host.h"
#include "h3_text_encoder.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    char *directory;
    char *key;          /* full key text, stored in the entry and compared */
    char *path;         /* <directory>/<sha256(key)>.h3cc */
    uint64_t max_bytes;
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
} h3_cond_disk_inputs;

/* Returns 1 when the cache is enabled and the key was built; 0 when disabled
 * or the key could not be built (a warning is printed; generation goes on
 * uncached). */
int h3_cond_disk_open(h3_cond_disk *disk, const h3_cond_disk_inputs *inputs);
void h3_cond_disk_close(h3_cond_disk *disk);

/* 1 on a verified hit (outputs owned by the caller), 0 on a miss. */
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
