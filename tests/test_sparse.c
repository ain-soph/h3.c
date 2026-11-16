/* CPU check of block-sparse plans on H3's real target layout: permutation
 * validity and kept-block density per parameter set.
 * Usage: h3_sparse_test WIDTH HEIGHT FRAMES "H3_SPARSE value"... */
#include "h3_sparse.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 5) return 2;
    int width = atoi(argv[1]), height = atoi(argv[2]), frames = atoi(argv[3]);
    int lw = 0, lh = 0;
    h3_latent_canvas(width, height, &lw, &lh);
    h3_temporal_shape shape = h3_temporal(frames);
    h3_layout_spec spec = {100, shape.video_t, lh, lw, shape.audio_t,
                           shape.frame_count, NULL, 0, NULL, 0};
    h3_layout layout;
    char error[256];
    if (!h3_layout_build(&spec, &layout, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    printf("layout %dx%d %d frames: latent %dx%dx%d, %zu rows\n", width, height,
           frames, shape.video_t, lh, lw, layout.seq_len);
    for (int arg = 4; arg < argc; arg++) {
        setenv("H3_SPARSE", argv[arg], 1);
        h3_sparse_params params;
        h3_sparse_plan plan;
        if (!h3_sparse_params_from_env(&params) ||
            !h3_sparse_plan_build(&layout, &params, 64, 32, &plan, error,
                                  sizeof(error))) {
            fprintf(stderr, "%s\n", error);
            return 1;
        }
        char *seen = calloc(plan.rows, 1);
        int valid = 1;
        for (uint32_t i = 0; i < plan.rows; i++) {
            if (plan.perm[i] >= plan.rows || seen[plan.perm[i]]) valid = 0;
            else seen[plan.perm[i]] = 1;
        }
        free(seen);
        printf("%-48s density %.3f  kept %zu  perm %s\n", argv[arg],
               h3_sparse_plan_density(&plan), plan.kept, valid ? "ok" : "BAD");
        h3_sparse_plan_free(&plan);
    }
    h3_layout_free(&layout);
    return 0;
}
