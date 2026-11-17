// Time and bit-compare tile variants of the video-VAE linear against h3.c's h3_linear_bf16_f32out_nax.
// Build: clang -O2 -fobjc-arc tests/bench_vae_gemm.m -framework Foundation -framework Metal -o h3_bench_vae_gemm
// Usage: ./h3_bench_vae_gemm h3_attention.metal tests/vae_gemm_variants.metal [ROWS]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t rows, in, out, swap; } dims_t;

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static float uniform(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
}
static uint16_t to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

static id<MTLLibrary> compile(id<MTLDevice> dev, const char *path) {
    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:@(path) encoding:NSUTF8StringEncoding error:&err];
    if (!src) { fprintf(stderr, "read %s failed\n", path); exit(1); }
    MTLCompileOptions *opt = [[MTLCompileOptions alloc] init];
    opt.mathMode = MTLMathModeSafe;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
    if (!lib) { fprintf(stderr, "compile %s: %s\n", path, err.localizedDescription.UTF8String); exit(1); }
    return lib;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 3) { fprintf(stderr, "usage: bench H3_ATTENTION_METAL VARIANTS_METAL [ROWS]\n"); return 1; }
        uint32_t rows = argc > 3 ? (uint32_t)atoi(argv[3]) : 2805;
        uint32_t padded = (rows + 127) & ~127u;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        id<MTLLibrary> h3lib = compile(dev, argv[1]), varlib = compile(dev, argv[2]);
        struct { const char *name; uint32_t in, out; } shapes[] = {
            {"qkv", 2048, 6144}, {"out", 2048, 2048}, {"ff1", 2048, 16384}, {"ff2", 8192, 2048}};
        struct { const char *fn; int bm, bn, threads; } vars[] = {
            {"loopk_128_128_128_4", 128, 128, 128}, {"loopk_128_128_64_4", 128, 128, 128},
            {"loopk_128_128_32_4", 128, 128, 128}, {"loopk_128_128_256_4", 128, 128, 128},
            {"loopk_128_64_128_4", 128, 64, 128}, {"loopk_64_128_128_4", 64, 128, 128},
            {"loopk_64_64_128_4", 64, 64, 128}, {"loopk_128_128_128_8", 128, 128, 256},
            {"loopk_128_64_64_4", 128, 64, 128}, {"loopk_64_128_64_4", 64, 128, 128},
            {"loopk_128_256_64_8", 128, 256, 256}, {"loopk_256_128_64_8", 256, 128, 256}};
        int nvars = sizeof(vars) / sizeof(vars[0]);
        NSError *err = nil;
        id<MTLComputePipelineState> ref = [dev newComputePipelineStateWithFunction:
            [h3lib newFunctionWithName:@"h3_linear_bf16_f32out_nax"] error:&err];
        if (!ref) { fprintf(stderr, "ref pipeline: %s\n", err.localizedDescription.UTF8String); return 1; }
        printf("rows %u (padded %u)\n", rows, padded);
        for (int s = 0; s < 4; s++) {
            uint32_t in = shapes[s].in, out = shapes[s].out;
            id<MTLBuffer> x = [dev newBufferWithLength:(size_t)padded * in * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> w = [dev newBufferWithLength:(size_t)out * in * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> b = [dev newBufferWithLength:(size_t)out * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> y0 = [dev newBufferWithLength:(size_t)rows * out * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> y = [dev newBufferWithLength:(size_t)rows * out * 4 options:MTLResourceStorageModeShared];
            uint16_t *xp = x.contents, *wp = w.contents; float *bp = b.contents;
            for (size_t i = 0; i < (size_t)padded * in; i++) xp[i] = to_bf16(i < (size_t)rows * in ? uniform() * 3.f : 0.f);
            for (size_t i = 0; i < (size_t)out * in; i++) wp[i] = to_bf16(uniform() * 0.03f);
            for (uint32_t i = 0; i < out; i++) bp[i] = uniform() * 0.01f;
            double flops = 2.0 * rows * in * out;
            // reference: the deployed kernel, dims {rows,in,out}
            uint32_t rdims[3] = {rows, in, out};
            double best = 1e9;
            for (int it = 0; it < 8; it++) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:ref];
                [enc setBuffer:x offset:0 atIndex:0]; [enc setBuffer:w offset:0 atIndex:1];
                [enc setBuffer:b offset:0 atIndex:2]; [enc setBuffer:y0 offset:0 atIndex:3];
                [enc setBytes:rdims length:sizeof(rdims) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(out / 128, padded / 128, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                double t = cb.GPUEndTime - cb.GPUStartTime;
                if (it && t < best) best = t;
            }
            printf("%-4s %5ux%-5u  %-24s %7.3f ms %6.1f TFLOPS  ref\n", shapes[s].name, in, out,
                   "h3_linear_bf16_f32out_nax", best * 1e3, flops / best / 1e12);
            for (int v = 0; v < nvars; v++) {
                id<MTLFunction> fn = [varlib newFunctionWithName:@(vars[v].fn)];
                id<MTLComputePipelineState> ps = fn ? [dev newComputePipelineStateWithFunction:fn error:&err] : nil;
                if (!ps) { printf("     %-24s unavailable\n", vars[v].fn); continue; }
                if (out % vars[v].bn || padded % vars[v].bm) { printf("     %-24s shape mismatch\n", vars[v].fn); continue; }
                for (uint32_t swap = 0; swap < 2; swap++) {
                    dims_t dims = {rows, in, out, swap};
                    MTLSize grid = swap ? MTLSizeMake(padded / vars[v].bm, out / vars[v].bn, 1)
                                        : MTLSizeMake(out / vars[v].bn, padded / vars[v].bm, 1);
                    best = 1e9;
                    memset(y.contents, 0xff, y.length);
                    for (int it = 0; it < 8; it++) {
                        id<MTLCommandBuffer> cb = [queue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:ps];
                        [enc setBuffer:x offset:0 atIndex:0]; [enc setBuffer:w offset:0 atIndex:1];
                        [enc setBuffer:b offset:0 atIndex:2]; [enc setBuffer:y offset:0 atIndex:3];
                        [enc setBytes:&dims length:sizeof(dims) atIndex:4];
                        [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(vars[v].threads, 1, 1)];
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                        double t = cb.GPUEndTime - cb.GPUStartTime;
                        if (it && t < best) best = t;
                    }
                    const float *a = y0.contents, *c = y.contents;
                    size_t n = (size_t)rows * out, diff = 0; double maxrel = 0, num = 0, den = 0;
                    for (size_t i = 0; i < n; i++) {
                        if (memcmp(&a[i], &c[i], 4)) diff++;
                        double e = (double)c[i] - a[i];
                        num += e * e; den += (double)a[i] * a[i];
                        double r = fabs(e) / (fabs(a[i]) + 1e-3);
                        if (r > maxrel) maxrel = r;
                    }
                    printf("%-4s %5ux%-5u  %-22s%s %7.3f ms %6.1f TFLOPS  %s (diff %zu, rel-rms %.2e)\n",
                           shapes[s].name, in, out, vars[v].fn, swap ? "/s" : "  ", best * 1e3, flops / best / 1e12,
                           diff ? "DIFFERENT" : "bit-identical", diff, sqrt(num / den));
                }
            }
        }
    }
    return 0;
}
