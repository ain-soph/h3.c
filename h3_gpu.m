#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "h3_gpu.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

@class H3GPU;

/* A read-only file mapping behind zero-copy weight buffers. Every MTLBuffer made over it keeps it alive through its
   deallocator, so a buffer that a command buffer still holds outlives the tensor that dropped it. */
@interface H3Mapping : NSObject
@property(nonatomic) void *mapping;
@property(nonatomic) size_t mapBytes;
@property(nonatomic) void *values;
@end
@implementation H3Mapping
- (void)dealloc {
    if (_mapping) munmap(_mapping, _mapBytes);
}
@end
@interface H3Tensor : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic) size_t elements;
@property(nonatomic) size_t bytes;
@property(nonatomic) h3_gpu_dtype dtype;
@property(nonatomic, weak) H3GPU *owner;
@property(nonatomic, strong) MPSGraphTensorData *graphData;
@property(nonatomic, strong) NSArray<NSNumber *> *graphDataShape;
@property(nonatomic) MPSDataType graphDataType;
@property(nonatomic, strong) H3Mapping *mapping;
@end
@implementation H3Tensor
@end

@interface H3SDPA : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *query;
@property(nonatomic, strong) MPSGraphTensor *key;
@property(nonatomic, strong) MPSGraphTensor *value;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *inputShape;
@property(nonatomic, strong) NSArray<NSNumber *> *outputShape;
@end
@implementation H3SDPA
@end

@interface H3GQA : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *query;
@property(nonatomic, strong) MPSGraphTensor *key;
@property(nonatomic, strong) MPSGraphTensor *value;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *queryShape;
@property(nonatomic, strong) NSArray<NSNumber *> *kvShape;
@end
@implementation H3GQA
@end

@interface H3Linear : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input;
@property(nonatomic, strong) MPSGraphTensor *weight;
@property(nonatomic, strong) MPSGraphTensor *bias;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *inputShape;
@property(nonatomic, strong) NSArray<NSNumber *> *weightShape;
@property(nonatomic, strong) NSArray<NSNumber *> *biasShape;
@property(nonatomic, strong) NSArray<NSNumber *> *outputShape;
@end
@implementation H3Linear
@end

@interface H3MLP : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input;
@property(nonatomic, strong) MPSGraphTensor *fc1Weight;
@property(nonatomic, strong) MPSGraphTensor *fc2Weight;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *inputShape;
@property(nonatomic, strong) NSArray<NSNumber *> *fc1Shape;
@property(nonatomic, strong) NSArray<NSNumber *> *fc2Shape;
@property(nonatomic, strong) NSArray<NSNumber *> *outputShape;
@end
@implementation H3MLP
@end

@interface H3Conv : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input;
@property(nonatomic, strong) MPSGraphTensor *weight;
@property(nonatomic, strong) MPSGraphTensor *bias;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *inputShape;
@property(nonatomic, strong) NSArray<NSNumber *> *weightShape;
@property(nonatomic, strong) NSArray<NSNumber *> *biasShape;
@property(nonatomic, strong) NSArray<NSNumber *> *outputShape;
@end
@implementation H3Conv
@end

/* Block-sparse DiT attention plan (see h3_sparse.h), resident on the GPU. */
@interface H3AttentionPlan : NSObject
@property(nonatomic) uint32_t rows;
@property(nonatomic) uint32_t qBlocks;
@property(nonatomic, strong) id<MTLBuffer> perm;
@property(nonatomic, strong) id<MTLBuffer> inverse;
@property(nonatomic, strong) id<MTLBuffer> offsets;
@property(nonatomic, strong) id<MTLBuffer> indices;
@end
@implementation H3AttentionPlan
@end

@interface H3GPU : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) id<MTLCommandBuffer> command;
@property(nonatomic, strong) NSMutableArray<id<MTLCommandBuffer>> *inflightCommands;
@property(nonatomic, strong) NSDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
@property(nonatomic, strong) NSMutableDictionary<NSString *, H3SDPA *> *sdpaCache;
@property(nonatomic, strong) NSMutableDictionary<NSString *, H3GQA *> *gqaCache;
@property(nonatomic, strong) NSMutableDictionary<NSString *, H3Linear *> *linearCache;
@property(nonatomic, strong) NSMutableDictionary<NSString *, H3MLP *> *mlpCache;
@property(nonatomic, strong) NSMutableDictionary<NSString *, H3Conv *> *convCache;
@property(nonatomic, strong) MPSCommandBuffer *mpsCommand;
@property(nonatomic) BOOL reuseMPSCommandDefault;
@property(nonatomic, copy) NSString *lastError;
@property(nonatomic) h3_gpu_stats stats;
@property(nonatomic, copy) NSString *profileLabel;
@property(nonatomic) BOOL tensorOpsEnabled;
@property(nonatomic) NSUInteger tensorOpsMode;
@property(nonatomic) BOOL headMajorSDPAInputs;
/* h3_attention.metal: the NAX DiT attention kernels (optional; M5 only). */
@property(nonatomic, strong) id<MTLLibrary> attentionLibrary;
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *attentionPipelines;
/* Scratch for H3_ATTN=int8: int8 Q/K, their per-(head,row) scales, K channel means. */
@property(nonatomic, strong) id<MTLBuffer> attentionQ8;
@property(nonatomic, strong) id<MTLBuffer> attentionK8;
@property(nonatomic, strong) id<MTLBuffer> attentionScales;
@property(nonatomic, strong) id<MTLBuffer> attentionMean;
@property(nonatomic) size_t attentionScratchElements;
/* Sparse plan in effect for the next head-major attention calls (nil = dense)
 * and its reordered BF16 Q/K/V/O scratch. */
@property(nonatomic, strong) H3AttentionPlan *attentionPlan;
@property(nonatomic, strong) id<MTLBuffer> attentionQP;
@property(nonatomic, strong) id<MTLBuffer> attentionKP;
@property(nonatomic, strong) id<MTLBuffer> attentionVP;
@property(nonatomic, strong) id<MTLBuffer> attentionOP;
@property(nonatomic) size_t attentionPermutedElements;
@property(nonatomic) h3_gpu_stats profileStartStats;
@property(nonatomic) h3_gpu_stats profileMarkStats;
@property(nonatomic) double profileStartWall;
@property(nonatomic) double profileMarkWall;
@property(nonatomic) double commandStartWall;
@end
@implementation H3GPU
@end

static H3GPU *GPU(h3_gpu *gpu) {
    return (__bridge H3GPU *)gpu;
}

static H3Tensor *TENSOR(const h3_gpu_tensor *tensor) {
    return (__bridge H3Tensor *)(void *)tensor;
}

static MPSGraphTensorData *h3_gpu_graph_data(const h3_gpu_tensor *tensor,
                                             NSArray<NSNumber *> *shape,
                                             MPSDataType data_type,
                                             int stable) {
    H3Tensor *object = TENSOR(tensor);
    if (!stable || getenv("H3_DISABLE_GRAPH_DATA_CACHE"))
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:object.buffer
            shape:shape dataType:data_type];
    if (object.graphData && object.graphDataType == data_type &&
        (object.graphDataShape == shape ||
         [object.graphDataShape isEqualToArray:shape])) return object.graphData;
    MPSGraphTensorData *data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:object.buffer shape:shape dataType:data_type];
    if (!object.graphData) {
        object.graphData = data;
        object.graphDataShape = shape;
        object.graphDataType = data_type;
    }
    return data;
}

static MPSCommandBuffer *h3_gpu_mps_command(H3GPU *gpu) {
    const char *override = getenv("H3_REUSE_MPS_COMMAND");
    BOOL reuse = override ? (*override && strcmp(override, "0")) :
                            gpu.reuseMPSCommandDefault;
    if (!reuse)
        return [MPSCommandBuffer commandBufferWithCommandBuffer:gpu.command];
    if (!gpu.mpsCommand || gpu.mpsCommand.rootCommandBuffer != gpu.command)
        gpu.mpsCommand =
            [MPSCommandBuffer commandBufferWithCommandBuffer:gpu.command];
    return gpu.mpsCommand;
}

static double h3_gpu_now(void) {
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0) return 0.0;
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

static int h3_gpu_profile_enabled(void) {
    const char *value = getenv("H3_PROFILE");
    return value && *value && strcmp(value, "0");
}

static uint64_t h3_gpu_counter_delta(uint64_t value, uint64_t start) {
    return value >= start ? value - start : 0;
}

static void h3_gpu_profile_emit(H3GPU *gpu, NSString *phase,
                                h3_gpu_stats start, double wall_start) {
    if (!h3_gpu_profile_enabled()) return;
    h3_gpu_stats value = gpu.stats;
    double wall = h3_gpu_now() - wall_start;
    NSString *label = gpu.profileLabel ? gpu.profileLabel : @"Metal context";
    fprintf(stderr,
        "h3 profile: %-24s %-14s wall=%8.3fs encode=%7.3fs "
        "wait=%8.3fs root-gpu=%7.3fs "
        "peak=%7.3fGiB alloc=%7.3fGiB submissions=%llu "
        "direct=%llu linear=%llu conv=%llu attention=%llu\n",
        label.UTF8String, phase.UTF8String, wall,
        value.command_encode_seconds - start.command_encode_seconds,
        value.command_wait_seconds - start.command_wait_seconds,
        value.gpu_seconds - start.gpu_seconds,
        (double)value.peak_live_bytes / (1024.0 * 1024.0 * 1024.0),
        (double)h3_gpu_counter_delta(value.allocated_bytes,
                                     start.allocated_bytes) /
            (1024.0 * 1024.0 * 1024.0),
        (unsigned long long)h3_gpu_counter_delta(value.submissions,
                                                 start.submissions),
        (unsigned long long)h3_gpu_counter_delta(value.direct_dispatches,
                                                 start.direct_dispatches),
        (unsigned long long)h3_gpu_counter_delta(value.mps_linear_dispatches,
                                                 start.mps_linear_dispatches),
        (unsigned long long)h3_gpu_counter_delta(value.mps_conv_dispatches,
                                                 start.mps_conv_dispatches),
        (unsigned long long)h3_gpu_counter_delta(value.mps_sdpa_dispatches,
                                                 start.mps_sdpa_dispatches));
}

static void h3_gpu_set_error(H3GPU *gpu, NSString *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    gpu.lastError = [[NSString alloc] initWithFormat:format arguments:arguments];
    va_end(arguments);
}

static int h3_gpu_require_command(H3GPU *gpu) {
    if (!gpu || !gpu.command) {
        if (gpu) h3_gpu_set_error(gpu, @"h3_gpu_begin() was not called");
        return 0;
    }
    return 1;
}

static int h3_gpu_require_elements(H3GPU *gpu, const h3_gpu_tensor *tensor,
                                   size_t elements, NSString *label) {
    if (!tensor || TENSOR(tensor).elements < elements) {
        h3_gpu_set_error(gpu, @"%@ tensor is absent or too small", label);
        return 0;
    }
    return 1;
}

static id<MTLComputePipelineState> h3_gpu_pipeline(H3GPU *gpu, NSString *name) {
    id<MTLComputePipelineState> pipeline = gpu.pipelines[name];
    if (!pipeline) h3_gpu_set_error(gpu, @"missing Metal pipeline %@", name);
    return pipeline;
}

static int h3_gpu_dispatch_1d(H3GPU *gpu, NSString *name, uint32_t count,
                              void (^bindings)(id<MTLComputeCommandEncoder>)) {
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, name);
    if (!pipeline) return 0;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        bindings(encoder);
        NSUInteger width = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_dispatch_2d(H3GPU *gpu, NSString *name, uint32_t width,
                              uint32_t height,
                              void (^bindings)(id<MTLComputeCommandEncoder>)) {
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, name);
    if (!pipeline) return 0;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        bindings(encoder);
        NSUInteger x = 16;
        NSUInteger y = MIN((NSUInteger)16, pipeline.maxTotalThreadsPerThreadgroup / x);
        [encoder dispatchThreads:MTLSizeMake(width, height, 1)
           threadsPerThreadgroup:MTLSizeMake(x, y, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_dispatch_3d(H3GPU *gpu, NSString *name, MTLSize grid,
                              void (^bindings)(id<MTLComputeCommandEncoder>)) {
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, name);
    if (!pipeline) return 0;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        bindings(encoder);
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_dispatch_rows(H3GPU *gpu, NSString *name, uint32_t rows,
                                void (^bindings)(id<MTLComputeCommandEncoder>)) {
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, name);
    if (!pipeline) return 0;
    NSUInteger maximum = MIN((NSUInteger)256,
                             pipeline.maxTotalThreadsPerThreadgroup);
    NSUInteger threads = 1;
    while (threads * 2 <= maximum) threads *= 2;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        bindings(encoder);
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

h3_gpu *h3_gpu_create(const char *shader_source_path,
                      char *error, size_t error_size) {
    @autoreleasepool {
        H3GPU *gpu = [[H3GPU alloc] init];
        gpu.profileLabel = @"Metal context";
        gpu.profileStartWall = h3_gpu_now();
        gpu.profileMarkWall = gpu.profileStartWall;
        gpu.device = MTLCreateSystemDefaultDevice();
        gpu.queue = [gpu.device newCommandQueue];
        gpu.reuseMPSCommandDefault = YES;
        gpu.inflightCommands = [NSMutableArray array];
        gpu.sdpaCache = [NSMutableDictionary dictionary];
        gpu.gqaCache = [NSMutableDictionary dictionary];
        gpu.linearCache = [NSMutableDictionary dictionary];
        gpu.mlpCache = [NSMutableDictionary dictionary];
        gpu.convCache = [NSMutableDictionary dictionary];
        if (!gpu.device || !gpu.queue) {
            if (error && error_size) snprintf(error, error_size, "cannot initialize Metal");
            return NULL;
        }
        if (getenv("H3_DEBUG_GPU_MEMORY")) {
            fprintf(stderr, "h3: Metal live allocation at GPU startup: "
                    "%.3f GiB\n", (double)gpu.device.currentAllocatedSize /
                    (1024.0 * 1024.0 * 1024.0));
        }
        const char *source_path = shader_source_path ? shader_source_path :
                                                       "h3_shaders.metal";
        NSString *path = [NSString stringWithUTF8String:source_path];
        NSError *libraryError = nil;
        NSString *source = [NSString stringWithContentsOfFile:path
                                                     encoding:NSUTF8StringEncoding
                                                        error:&libraryError];
        if (source) {
            MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
            options.mathMode = MTLMathModeSafe;
            const char *nax = getenv("H3_NAX");
            BOOL m5 = [gpu.device.name rangeOfString:@"M5"].location !=
                      NSNotFound;
            BOOL wantsTensorOps =
                m5 && (!nax || !*nax || strcmp(nax, "0") != 0);
            if (wantsTensorOps)
                options.preprocessorMacros = @{ @"H3_METAL_HAS_TENSOR": @"1" };
            gpu.library = [gpu.device newLibraryWithSource:source
                                                   options:options
                                                     error:&libraryError];
            gpu.tensorOpsEnabled = gpu.library && wantsTensorOps;
            if (gpu.tensorOpsEnabled) {
                const char *mode = nax && *nax ? nax : "qkv-attn";
                gpu.tensorOpsMode = !strcmp(mode, "attn") ? 2u :
                                    !strcmp(mode, "qkv-attn") ? 3u :
                                    !strcmp(mode, "qkv") ? 4u :
                                    !strcmp(mode, "mlp") ? 5u : 1u;
            }
            if (!gpu.library && wantsTensorOps) {
                /* TensorOps is optional: an older runtime must retain the
                 * ordinary MPSGraph/direct Metal implementation. */
                if (getenv("H3_NAX_DIAGNOSTIC"))
                    fprintf(stderr, "h3: TensorOps compile failed: %s\n",
                            libraryError.localizedDescription.UTF8String);
                options.preprocessorMacros = @{};
                libraryError = nil;
                gpu.library = [gpu.device newLibraryWithSource:source
                                                       options:options
                                                         error:&libraryError];
                gpu.tensorOpsEnabled = NO;
                gpu.tensorOpsMode = 0;
            }
        }
        if (gpu.library && gpu.tensorOpsEnabled) {
            NSString *attentionPath = [[path stringByDeletingLastPathComponent]
                stringByAppendingPathComponent:@"h3_attention.metal"];
            NSString *attentionSource = [NSString
                stringWithContentsOfFile:attentionPath
                                encoding:NSUTF8StringEncoding error:nil];
            if (attentionSource) {
                NSError *attentionError = nil;
                MTLCompileOptions *attentionOptions = [[MTLCompileOptions alloc] init];
                attentionOptions.mathMode = MTLMathModeSafe;
                gpu.attentionLibrary = [gpu.device
                    newLibraryWithSource:attentionSource
                                 options:attentionOptions
                                   error:&attentionError];
                if (!gpu.attentionLibrary && getenv("H3_ATTN"))
                    fprintf(stderr, "h3: NAX attention compile failed: %s\n",
                            attentionError.localizedDescription.UTF8String);
                gpu.attentionPipelines = [NSMutableDictionary dictionary];
            }
        }
        if (!gpu.library) {
            if (error && error_size) {
                const char *description = libraryError.localizedDescription.UTF8String;
                snprintf(error, error_size, "cannot compile %s: %s",
                         source_path, description ? description : "unknown error");
            }
            return NULL;
        }
        NSMutableArray<NSString *> *names = [@[
            @"h3_linear_f32", @"h3_linear_f32_tiled",
            @"h3_linear_f32_tiled_bf16", @"h3_silu_f32",
            @"h3_linear_f32_tiled_bf16_map",
            @"h3_cast_f32_to_bf16",
            @"h3_cast_bf16_to_f32",
            @"h3_rms_norm_f32", @"h3_rms_norm_f32_to_bf16",
            @"h3_scale_add_f32", @"h3_layer_norm_f32",
            @"h3_video_qkv_rope_f32", @"h3_video_qkv_rope_bf16",
            @"h3_swiglu_f32_to_bf16",
            @"h3_adaln_f32", @"h3_gate_f32", @"h3_qkv_rope_f32",
            @"h3_swiglu_f32", @"h3_linear_bf16", @"h3_silu_bf16",
            @"h3_rms_norm_bf16", @"h3_adaln_bf16", @"h3_gate_bf16",
            @"h3_rms_inverse_bf16", @"h3_adaln_linear_bf16",
            @"h3_gate_adaln_bf16", @"h3_gate_adaln_bf16_exact_simd",
            @"h3_qkv_rope_bf16", @"h3_qkv_rope_bf16_coop",
            @"h3_qkv_rope_bf16_coop_uncached",
            @"h3_swiglu_bf16",
            @"h3_layer_norm_bf16", @"h3_gelu_bf16",
            @"h3_vision_qkv_rope_bf16",
            @"h3_embedding_bf16", @"h3_text_qk_rope_bf16",
            @"h3_rotate256_bf16", @"h3_dequantize_int8_rows_bf16",
            @"h3_head_rms_norm_bf16", @"h3_rope_text_bf16",
            @"h3_gqa_causal_bf16", @"h3_add_bf16", @"h3_sub_bf16",
            @"h3_token_pool_bf16", @"h3_token_pool_adaln_bf16",
            @"h3_token_expand_delta_bf16",
            @"h3_token_expand_adaln_bf16",
            @"h3_euler_bf16", @"h3_silu_mul_bf16",
            @"h3_weight_norm_f32", @"h3_add_scaled_f32",
            @"h3_alias_free_snake_f32", @"h3_snake1d_f32",
            @"h3_audio_qkv_split_f32", @"h3_audio_attention_pool_f32",
            @"h3_geglu_f32", @"h3_clip_f32",
            @"h3_vae_encoder_pad_f32",
            @"h3_vae_encoder_group_norm_silu_f32",
            @"h3_vae_encoder_group_norm_silu_pad_f32"
        ] mutableCopy];
        if (gpu.tensorOpsEnabled) {
            [names addObject:@"h3_linear_bf16_nax_r128"];
            [names addObject:@"h3_linear_bf16_nax_r128_morton"];
            [names addObject:@"h3_linear_bf16_nax_r128_morton4"];
            [names addObject:
                @"h3_qkv_project_split_bf16_nax_r128_morton4"];
            [names addObject:@"h3_qk_rope_bf16_nax_inplace"];
            [names addObject:@"h3_fc1_swiglu_bf16_nax_r128"];
            [names addObject:@"h3_fc1_swiglu_bf16_nax_r128_morton"];
            [names addObject:@"h3_fc1_swiglu_bf16_nax_r128_morton4"];
            [names addObject:@"h3_quantize_bf16_int8_rows"];
            [names addObject:@"h3_quantize_bf16_int8_rows_scalar"];
            [names addObject:
                @"h3_quantize_bf16_int8_head_major_to_rows_cached"];
            [names addObject:@"h3_quantize_bf16_int8_groups"];
            [names addObject:@"h3_quantize_bf16_int8_groups_scalar"];
            [names addObject:@"h3_quantize_bf16_int8_groups_scalar128"];
            [names addObject:
                @"h3_quantize_bf16_int8_groups_scalar128_cached"];
            [names addObject:
                @"h3_qkv_project_split_int8_nax_r128_morton4"];
            [names addObject:
                @"h3_qkv_project_split_int8_rope_nax_r128_morton4"];
            [names addObject:
                @"h3_qkv_project_split_int8_rope_nax_r128_k5376_morton4"];
            [names addObject:
                @"h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4"];
            [names addObject:
                @"h3_qkv_project_split_int8_rope_local_scales_nax_r128_k5376_morton4"];
            [names addObject:@"h3_fc1_swiglu_int8_nax_r128"];
            [names addObject:@"h3_fc1_swiglu_int8_nax_r128_k5376"];
            [names addObject:@"h3_fc1_swiglu_int8_nax_r128_full_k5376"];
            [names addObject:@"h3_fc1_swiglu_int8_local_nax_r128"];
            [names addObject:@"h3_linear_int8_nax_r128"];
            [names addObject:
                @"h3_linear_int8_nax_r128_full_k14336"];
            [names addObject:
                @"h3_linear_int8_nax_r128x256_full_k14336"];
            [names addObject:@"h3_linear_int8_local_scales_nax_r128"];
            [names addObject:@"h3_linear_int8_local_scales_nax_r128_k7168"];
            [names addObject:@"h3_gate_adaln_quantize_int8"];
            [names addObject:@"h3_gate_adaln_quantize_int8_scalar"];
            [names addObject:@"h3_gate_adaln_quantize_int8_rot256"];
            [names addObject:@"h3_quantize_bf16_int8_rows_rot256"];
            [names addObject:
                @"h3_quantize_bf16_int8_head_major_to_rows_rot256"];
            [names addObject:@"h3_quantize_bf16_int8_groups1024_rot256"];
            [names addObject:@"h3_linear_int8_grouped_nax_r128x64"];
            [names addObject:
                @"h3_linear_int8_grouped_local_nax_r128x64"];
            [names addObject:
                @"h3_linear_int8_grouped_local_nax_r128x128"];
        }
        NSMutableDictionary *pipelines = [NSMutableDictionary dictionary];
        for (NSString *name in names) {
            id<MTLFunction> function = [gpu.library newFunctionWithName:name];
            NSError *pipelineError = nil;
            id<MTLComputePipelineState> pipeline =
                function ? [gpu.device newComputePipelineStateWithFunction:function
                                                                       error:&pipelineError] : nil;
            if (!pipeline) {
                if (error && error_size) {
                    const char *description = pipelineError.localizedDescription.UTF8String;
                    snprintf(error, error_size, "cannot build %s: %s", name.UTF8String,
                             description ? description : "function missing");
                }
                return NULL;
            }
            pipelines[name] = pipeline;
        }
        gpu.pipelines = pipelines;
        return (__bridge_retained h3_gpu *)gpu;
    }
}

void h3_gpu_free(h3_gpu *gpu) {
    if (!gpu) return;
    @autoreleasepool {
        H3GPU *object = CFBridgingRelease(gpu);
        h3_gpu_profile_emit(object, @"total", object.profileStartStats,
                            object.profileStartWall);
        id<MTLDevice> device = object.device;
        NSUInteger before = device.currentAllocatedSize;
        object.command = nil;
        object.mpsCommand = nil;
        for (id<MTLCommandBuffer> pending in object.inflightCommands)
            [pending waitUntilCompleted];
        object.inflightCommands = nil;
        object.sdpaCache = nil;
        object.linearCache = nil;
        object.pipelines = nil;
        object.library = nil;
        object.queue = nil;
        object.device = nil;
        if (getenv("H3_DEBUG_GPU_MEMORY")) {
            fprintf(stderr, "h3: Metal live allocation at GPU teardown: "
                    "%.3f GiB\n", (double)before /
                    (1024.0 * 1024.0 * 1024.0));
        }
    }
}

int h3_gpu_is_m5(const h3_gpu *opaque) {
    if (!opaque) return 0;
    H3GPU *gpu = GPU((h3_gpu *)(void *)opaque);
    return [gpu.device.name rangeOfString:@"M5"].location != NSNotFound;
}

int h3_gpu_has_nax_mlp(const h3_gpu *opaque) {
    if (!opaque) return 0;
    H3GPU *gpu = GPU((h3_gpu *)(void *)opaque);
    return gpu.tensorOpsEnabled && gpu.tensorOpsMode == 5;
}

int h3_gpu_has_int8_mlp(const h3_gpu *opaque) {
    if (!opaque) return 0;
    H3GPU *gpu = GPU((h3_gpu *)(void *)opaque);
    return gpu.tensorOpsEnabled;
}

static h3_gpu_tensor *h3_gpu_tensor_new(h3_gpu *opaque, const void *values,
                                        size_t elements, size_t item_size,
                                        h3_gpu_dtype dtype) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || elements > SIZE_MAX / item_size) return NULL;
    size_t bytes = elements * item_size;
    H3Tensor *tensor = [[H3Tensor alloc] init];
    tensor.elements = elements;
    tensor.bytes = bytes;
    tensor.dtype = dtype;
    tensor.buffer = [gpu.device newBufferWithLength:MAX(bytes, (size_t)1)
                                            options:MTLResourceStorageModeShared];
    if (!tensor.buffer) {
        h3_gpu_set_error(gpu, @"cannot allocate %zu-byte Metal buffer", bytes);
        return NULL;
    }
    tensor.owner = gpu;
    if (values && bytes) memcpy(tensor.buffer.contents, values, bytes);
    h3_gpu_stats stats = gpu.stats;
    stats.allocated_bytes += bytes;
    stats.live_bytes += bytes;
    if (stats.live_bytes > stats.peak_live_bytes)
        stats.peak_live_bytes = stats.live_bytes;
    stats.tensor_allocations++;
    gpu.stats = stats;
    return (__bridge_retained h3_gpu_tensor *)tensor;
}

h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements) {
    return h3_gpu_tensor_new(gpu, NULL, elements, sizeof(float), H3_GPU_F32);
}

h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements) {
    return h3_gpu_tensor_new(gpu, NULL, elements, sizeof(uint16_t), H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements) {
    return h3_gpu_tensor_new(gpu, NULL, elements, sizeof(int8_t), H3_GPU_I8);
}

h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values,
                                      size_t elements) {
    return h3_gpu_tensor_new(gpu, values, elements, sizeof(float), H3_GPU_F32);
}

h3_gpu_tensor *h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values,
                                       size_t elements) {
    return h3_gpu_tensor_new(gpu, values, elements, sizeof(uint16_t), H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values,
                                      size_t elements) {
    return h3_gpu_tensor_new(gpu, values, elements, sizeof(uint32_t), H3_GPU_U32);
}

static id<MTLBuffer> h3_gpu_mapped_buffer(H3GPU *gpu, H3Mapping *file_map,
                                          size_t bytes) {
    return [gpu.device newBufferWithBytesNoCopy:file_map.values
                                         length:MAX(bytes, (size_t)1)
                                        options:MTLResourceStorageModeShared
                                    deallocator:^(void *pointer, NSUInteger length) {
        (void)pointer;
        (void)length;
        (void)file_map;  /* keeps the mapping until Metal releases the buffer */
    }];
}

int h3_gpu_tensor_is_mapped(const h3_gpu_tensor *tensor) {
    return tensor && TENSOR(tensor).mapping != nil;
}

int h3_gpu_tensor_detach(h3_gpu_tensor *opaque) {
    if (!opaque) return 0;
    H3Tensor *tensor = TENSOR(opaque);
    if (!tensor.mapping) return 0;
    @autoreleasepool {
        tensor.graphData = nil;
        tensor.graphDataShape = nil;
        tensor.buffer = nil;
    }
    return 1;
}

int h3_gpu_tensor_attach(h3_gpu *opaque, h3_gpu_tensor *tensor_opaque) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || !tensor_opaque) return 0;
    H3Tensor *tensor = TENSOR(tensor_opaque);
    if (tensor.buffer) return 1;
    if (!tensor.mapping) return 0;
    @autoreleasepool {
        tensor.buffer = h3_gpu_mapped_buffer(gpu, tensor.mapping, tensor.bytes);
    }
    if (!tensor.buffer) {
        h3_gpu_set_error(gpu, @"cannot map %zu-byte Metal buffer", tensor.bytes);
        return 0;
    }
    return 1;
}

static h3_gpu_tensor *h3_gpu_tensor_load_file(h3_gpu *opaque, const char *path,
                                              uint64_t file_offset,
                                              size_t elements,
                                              size_t item_size,
                                              h3_gpu_dtype dtype,
                                              const char *label) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || !path || !*path || file_offset > INT64_MAX ||
        elements > SIZE_MAX / item_size) return NULL;
    size_t bytes = elements * item_size;
    if ((uint64_t)bytes > (uint64_t)INT64_MAX - file_offset) return NULL;
    const char *zero_copy = getenv("H3_ZERO_COPY_WEIGHTS");
    /* The BF16 DiT and the packed INT8 DiT (tools/align_int8_pack.py puts every whole-page tensor on a page): files
       directly in a directory named transformer, transformer-int8 or transformer-int8-<variant> (e.g. rot256). */
    int transformer_weight = 0;
    {
        const char *slash = strrchr(path, '/');
        if (slash) {
            const char *start = slash;
            while (start > path && start[-1] != '/') start--;
            size_t length = (size_t)(slash - start);
            transformer_weight =
                (length == 11 && !strncmp(start, "transformer", 11)) ||
                (length == 16 && !strncmp(start, "transformer-int8", 16)) ||
                (length > 17 && !strncmp(start, "transformer-int8-", 17));
        }
    }
    int m5 = [gpu.device.name rangeOfString:@"M5"].location != NSNotFound;
    size_t page = (size_t)getpagesize();
    /* newBufferWithBytesNoCopy needs a page-aligned pointer and a whole number of pages; anything else (unaligned
       files, small scale/norm tensors) is copied as before. */
    int mappable = bytes && file_offset % page == 0 && bytes % page == 0;
    int map_weight = mappable &&
        ((zero_copy && !strcmp(zero_copy, "1")) ||
         (transformer_weight &&
          ((zero_copy && !strcmp(zero_copy, "transformer")) ||
           (!zero_copy && m5))));
    if (map_weight) {
        int descriptor = open(path, O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) {
            h3_gpu_set_error(gpu, @"cannot open %s: %s", path,
                             strerror(errno));
            return NULL;
        }
        uint64_t aligned_offset = file_offset - file_offset % page;
        size_t delta = (size_t)(file_offset - aligned_offset);
        if (bytes > SIZE_MAX - delta) {
            close(descriptor);
            return NULL;
        }
        size_t map_bytes = bytes + delta;
        void *mapping = mmap(NULL, map_bytes, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE, descriptor, (off_t)aligned_offset);
        int map_error = errno;
        close(descriptor);
        if (mapping == MAP_FAILED) {
            h3_gpu_set_error(gpu, @"cannot map %s payload from %s: %s", label,
                             path, strerror(map_error));
            return NULL;
        }
        H3Mapping *file_map = [[H3Mapping alloc] init];
        file_map.mapping = mapping;
        file_map.mapBytes = map_bytes;
        file_map.values = (unsigned char *)mapping + delta;
        id<MTLBuffer> buffer = h3_gpu_mapped_buffer(gpu, file_map, bytes);
        if (!buffer) {
            h3_gpu_set_error(gpu, @"cannot map %zu-byte Metal buffer", bytes);
            return NULL;
        }
        H3Tensor *tensor = [[H3Tensor alloc] init];
        tensor.elements = elements;
        tensor.bytes = bytes;
        tensor.dtype = dtype;
        tensor.buffer = buffer;
        tensor.mapping = file_map;
        tensor.owner = gpu;
        h3_gpu_stats stats = gpu.stats;
        stats.allocated_bytes += bytes;
        stats.live_bytes += bytes;
        if (stats.live_bytes > stats.peak_live_bytes)
            stats.peak_live_bytes = stats.live_bytes;
        stats.tensor_allocations++;
        gpu.stats = stats;
        return (__bridge_retained h3_gpu_tensor *)tensor;
    }
    h3_gpu_tensor *opaque_tensor = h3_gpu_tensor_new(
        opaque, NULL, elements, item_size, dtype);
    if (!opaque_tensor) return NULL;
    H3Tensor *tensor = TENSOR(opaque_tensor);
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        h3_gpu_set_error(gpu, @"cannot open %s: %s", path, strerror(errno));
        h3_gpu_tensor_free(opaque_tensor);
        return NULL;
    }
    size_t remaining = bytes;
    size_t completed = 0;
    while (remaining) {
        size_t request = MIN(remaining, (size_t)SSIZE_MAX);
        ssize_t count = pread(descriptor,
                              (unsigned char *)tensor.buffer.contents + completed,
                              request, (off_t)(file_offset + completed));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            int detail = count < 0 ? errno : 0;
            h3_gpu_set_error(gpu, @"cannot read %s payload from %s: %s", label,
                             path, detail ? strerror(detail) :
                                            "unexpected end of file");
            close(descriptor);
            h3_gpu_tensor_free(opaque_tensor);
            return NULL;
        }
        completed += (size_t)count;
        remaining -= (size_t)count;
    }
    close(descriptor);
    return opaque_tensor;
}

h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *opaque, const char *path,
                                       uint64_t file_offset, size_t elements) {
    return h3_gpu_tensor_load_file(opaque, path, file_offset, elements,
                                   sizeof(uint16_t), H3_GPU_BF16, "BF16");
}

h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *opaque, const char *path,
                                      uint64_t file_offset, size_t elements) {
    return h3_gpu_tensor_load_file(opaque, path, file_offset, elements,
                                   sizeof(float), H3_GPU_F32, "F32");
}

h3_gpu_tensor *h3_gpu_tensor_load_i8(h3_gpu *opaque, const char *path,
                                     uint64_t file_offset, size_t elements) {
    return h3_gpu_tensor_load_file(opaque, path, file_offset, elements,
                                   sizeof(int8_t), H3_GPU_I8, "I8");
}

static int h3_gpu_tensor_read_file_mode(
                                 h3_gpu_tensor *opaque, const char *path,
                                 uint64_t file_offset, size_t bytes,
                                 int uncached,
                                 char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!opaque || !path || !*path || bytes != TENSOR(opaque).bytes ||
        file_offset > INT64_MAX) {
        if (error && error_size)
            snprintf(error, error_size, "invalid tensor file read request");
        return 0;
    }
    if ((uint64_t)bytes > (uint64_t)INT64_MAX - file_offset) {
        if (error && error_size)
            snprintf(error, error_size, "tensor file read range overflows");
        return 0;
    }
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        if (error && error_size)
            snprintf(error, error_size, "cannot open %s: %s", path,
                     strerror(errno));
        return 0;
    }
#ifdef F_NOCACHE
    if (uncached) (void)fcntl(descriptor, F_NOCACHE, 1);
#else
    (void)uncached;
#endif
    unsigned char *destination = TENSOR(opaque).buffer.contents;
    size_t completed = 0;
    while (completed < bytes) {
        size_t request = MIN(bytes - completed, (size_t)SSIZE_MAX);
        ssize_t count = pread(descriptor, destination + completed, request,
                              (off_t)(file_offset + completed));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            int detail = count < 0 ? errno : 0;
            if (error && error_size) {
                snprintf(error, error_size, "cannot read tensor payload from %s: %s",
                         path, detail ? strerror(detail) :
                                        "unexpected end of file");
            }
            close(descriptor);
            return 0;
        }
        completed += (size_t)count;
    }
    close(descriptor);
    return 1;
}

static int h3_gpu_tensor_read_file_bf16_mode(
                                 h3_gpu_tensor *opaque, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 int uncached,
                                 char *error, size_t error_size) {
    if (!opaque || TENSOR(opaque).dtype != H3_GPU_BF16 ||
        elements != TENSOR(opaque).elements ||
        elements > SIZE_MAX / sizeof(uint16_t)) {
        if (error && error_size)
            snprintf(error, error_size, "invalid BF16 file read request");
        return 0;
    }
    return h3_gpu_tensor_read_file_mode(opaque, path, file_offset,
                                        elements * sizeof(uint16_t), uncached,
                                        error, error_size);
}

int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *opaque, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size) {
    return h3_gpu_tensor_read_file_bf16_mode(
        opaque, path, file_offset, elements, 0, error, error_size);
}

int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *opaque, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size) {
    return h3_gpu_tensor_read_file_bf16_mode(
        opaque, path, file_offset, elements, 1, error, error_size);
}

int h3_gpu_tensor_stream_file(h3_gpu_tensor *opaque, const char *path,
                              uint64_t file_offset,
                              char *error, size_t error_size) {
    if (!opaque) {
        if (error && error_size)
            snprintf(error, error_size, "invalid tensor file read request");
        return 0;
    }
    return h3_gpu_tensor_read_file_mode(opaque, path, file_offset,
                                        TENSOR(opaque).bytes, 1,
                                        error, error_size);
}

void h3_gpu_tensor_free(h3_gpu_tensor *tensor) {
    if (!tensor) return;
    @autoreleasepool {
        H3Tensor *object = CFBridgingRelease(tensor);
        H3GPU *owner = object.owner;
        if (owner) {
            h3_gpu_stats stats = owner.stats;
            stats.live_bytes = stats.live_bytes >= object.bytes ?
                stats.live_bytes - object.bytes : 0;
            owner.stats = stats;
        }
        [object.buffer setPurgeableState:MTLPurgeableStateEmpty];
        object.buffer = nil;
    }
}

size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor) {
    return tensor ? TENSOR(tensor).elements : 0;
}

h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor) {
    return tensor ? TENSOR(tensor).dtype : H3_GPU_F32;
}

int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values,
                           size_t elements) {
    return h3_gpu_tensor_read_f32_range(tensor, 0, values, elements);
}

const float *h3_gpu_tensor_contents_f32(const h3_gpu_tensor *tensor) {
    if (!tensor || TENSOR(tensor).dtype != H3_GPU_F32) return NULL;
    return TENSOR(tensor).buffer.contents;
}

int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements) {
    if (!tensor || !values || TENSOR(tensor).dtype != H3_GPU_F32 ||
        source_offset > TENSOR(tensor).elements ||
        elements > TENSOR(tensor).elements - source_offset) return 0;
    const unsigned char *source = TENSOR(tensor).buffer.contents;
    memcpy(values, source + source_offset * sizeof(float),
           elements * sizeof(float));
    return 1;
}

int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values,
                            size_t elements) {
    if (!tensor || !values || TENSOR(tensor).dtype != H3_GPU_BF16 ||
        elements > TENSOR(tensor).elements) return 0;
    memcpy(values, TENSOR(tensor).buffer.contents, elements * sizeof(uint16_t));
    return 1;
}

int h3_gpu_tensor_read_i8(const h3_gpu_tensor *tensor, int8_t *values,
                          size_t elements) {
    if (!tensor || !values || TENSOR(tensor).dtype != H3_GPU_I8 ||
        elements > TENSOR(tensor).elements) return 0;
    memcpy(values, TENSOR(tensor).buffer.contents, elements);
    return 1;
}

int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values,
                            size_t elements) {
    return h3_gpu_tensor_write_f32_range(tensor, 0, values, elements);
}

int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements) {
    if (!tensor || !values || TENSOR(tensor).dtype != H3_GPU_F32 ||
        destination_offset > TENSOR(tensor).elements ||
        elements > TENSOR(tensor).elements - destination_offset) return 0;
    unsigned char *destination = TENSOR(tensor).buffer.contents;
    memcpy(destination + destination_offset * sizeof(float), values,
           elements * sizeof(float));
    return 1;
}

int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements) {
    return h3_gpu_tensor_write_bf16_range(tensor, 0, values, elements);
}

int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements) {
    if (!tensor || !values || TENSOR(tensor).dtype != H3_GPU_BF16 ||
        destination_offset > TENSOR(tensor).elements ||
        elements > TENSOR(tensor).elements - destination_offset) return 0;
    unsigned char *destination = TENSOR(tensor).buffer.contents;
    memcpy(destination + destination_offset * sizeof(uint16_t), values,
           elements * sizeof(uint16_t));
    return 1;
}

int h3_gpu_tensor_write_i8(h3_gpu_tensor *tensor, const int8_t *values,
                           size_t elements) {
    if (!tensor || !values || TENSOR(tensor).dtype != H3_GPU_I8 ||
        elements > TENSOR(tensor).elements || !TENSOR(tensor).buffer)
        return 0;
    memcpy(TENSOR(tensor).buffer.contents, values, elements);
    return 1;
}

int h3_gpu_begin(h3_gpu *opaque) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || gpu.command || gpu.inflightCommands.count) return 0;
    gpu.lastError = nil;
    @autoreleasepool {
        gpu.command = [gpu.queue commandBuffer];
    }
    if (!gpu.command) {
        h3_gpu_set_error(gpu, @"cannot create Metal command buffer");
        return 0;
    }
    gpu.commandStartWall = h3_gpu_now();
    return 1;
}

int h3_gpu_continue(h3_gpu *opaque) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || !gpu.command) return 0;
    @autoreleasepool {
        id<MTLCommandBuffer> command = gpu.command;
        gpu.command = nil;
        gpu.mpsCommand = nil;
        double commit_time = h3_gpu_now();
        [command commit];
        [gpu.inflightCommands addObject:command];
        h3_gpu_stats stats = gpu.stats;
        stats.submissions++;
        stats.command_encode_seconds += commit_time - gpu.commandStartWall;
        gpu.stats = stats;
        gpu.command = [gpu.queue commandBuffer];
    }
    if (!gpu.command) {
        h3_gpu_set_error(gpu, @"cannot continue Metal command chain");
        return 0;
    }
    gpu.commandStartWall = h3_gpu_now();
    return 1;
}

int h3_gpu_retire(h3_gpu *opaque, unsigned keep) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu) return 0;
    @autoreleasepool {
        while (gpu.inflightCommands.count > keep) {
            id<MTLCommandBuffer> oldest = gpu.inflightCommands.firstObject;
            double wait_started = h3_gpu_now();
            [oldest waitUntilCompleted];
            h3_gpu_stats stats = gpu.stats;
            stats.command_wait_seconds += h3_gpu_now() - wait_started;
            if (oldest.GPUEndTime >= oldest.GPUStartTime)
                stats.gpu_seconds += oldest.GPUEndTime - oldest.GPUStartTime;
            gpu.stats = stats;
            if (oldest.status == MTLCommandBufferStatusError) {
                h3_gpu_set_error(gpu, @"Metal command failed: %@",
                                 oldest.error.localizedDescription);
                return 0;
            }
            [gpu.inflightCommands removeObjectAtIndex:0];
        }
    }
    return 1;
}

int h3_gpu_submit(h3_gpu *opaque) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || !gpu.command) return 0;
    @autoreleasepool {
        id<MTLCommandBuffer> command = gpu.command;
        gpu.command = nil;
        gpu.mpsCommand = nil;
        double commit_time = h3_gpu_now();
        [command commit];
        [gpu.inflightCommands addObject:command];
        for (id<MTLCommandBuffer> pending in gpu.inflightCommands)
            [pending waitUntilCompleted];
        double complete_time = h3_gpu_now();
        for (id<MTLCommandBuffer> pending in gpu.inflightCommands) {
            if (pending.status == MTLCommandBufferStatusError) {
                h3_gpu_set_error(gpu, @"Metal command failed: %@",
                                 pending.error.localizedDescription);
                [gpu.inflightCommands removeAllObjects];
                return 0;
            }
        }
        h3_gpu_stats stats = gpu.stats;
        stats.submissions++;
        stats.command_encode_seconds += commit_time - gpu.commandStartWall;
        stats.command_wait_seconds += complete_time - commit_time;
        for (id<MTLCommandBuffer> pending in gpu.inflightCommands) {
            if (pending.GPUEndTime >= pending.GPUStartTime)
                stats.gpu_seconds += pending.GPUEndTime - pending.GPUStartTime;
        }
        gpu.stats = stats;
        [gpu.inflightCommands removeAllObjects];
    }
    return 1;
}

void h3_gpu_abort(h3_gpu *opaque) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu) return;
    @autoreleasepool {
        /* the open buffer is dropped uncommitted; everything already committed
         * (ours, and roots MPS committed itself) finishes before the marker */
        gpu.command = nil;
        gpu.mpsCommand = nil;
        id<MTLCommandBuffer> marker = [gpu.queue commandBuffer];
        [marker commit];
        for (id<MTLCommandBuffer> pending in gpu.inflightCommands)
            [pending waitUntilCompleted];
        [marker waitUntilCompleted];
        [gpu.inflightCommands removeAllObjects];
    }
}

const char *h3_gpu_error(const h3_gpu *opaque) {
    H3GPU *gpu = GPU((h3_gpu *)(void *)opaque);
    const char *message = gpu.lastError.UTF8String;
    return message ? message : "unknown Metal error";
}

int h3_gpu_get_stats(const h3_gpu *opaque, h3_gpu_stats *stats) {
    if (!opaque || !stats) return 0;
    *stats = GPU((h3_gpu *)(void *)opaque).stats;
    return 1;
}

void h3_gpu_profile_set_label(h3_gpu *opaque, const char *label) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || !label || !*label) return;
    gpu.profileLabel = [NSString stringWithUTF8String:label];
}

void h3_gpu_profile_mark(h3_gpu *opaque, const char *phase) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || !phase || !*phase || !h3_gpu_profile_enabled()) return;
    h3_gpu_profile_emit(gpu, [NSString stringWithUTF8String:phase],
                        gpu.profileMarkStats, gpu.profileMarkWall);
    gpu.profileMarkStats = gpu.stats;
    gpu.profileMarkWall = h3_gpu_now();
}

typedef struct { uint32_t rows, input_dim, output_dim, has_bias; } linear_args;
typedef struct { uint32_t rows, columns; float clip; } int8_quant_args;
typedef struct {
    uint32_t rows, padded_rows, heads, head_dim;
    float clip;
} int8_head_major_quant_args;
typedef struct { uint32_t rows, columns, group_size, groups; }
    int8_group_quant_args;
typedef struct { uint32_t rows, width; float epsilon; } norm_args;
typedef struct {
    uint32_t rows, width, slots, shift_slot, scale_slot;
    float epsilon;
} adaln_args;
typedef struct { uint32_t rows, width, slots, gate_slot; } gate_args;
typedef struct {
    uint32_t sequence, heads, head_dim, rope_half, grouped;
    float epsilon;
} qkv_args;
typedef struct { uint32_t outer, inner; } weight_norm_args;
typedef struct { uint32_t elements; float left_scale, right_scale; }
    add_scaled_args;
typedef struct { uint32_t batch, length, channels; } audio_activation_args;
typedef struct { uint32_t batch, length, heads, head_dim; } audio_qkv_args;
typedef struct {
    uint32_t batch, length, heads, head_dim, output_dim;
} audio_pool_args;
typedef struct { uint32_t elements; float minimum, maximum; } clip_args;
typedef struct {
    uint32_t batch, depth, height, width, channels, depth_front;
    uint32_t height_before, height_after, width_before, width_after;
} vae_encoder_pad_args;
typedef struct {
    uint32_t batch, depth, height, width, channels, groups;
    float epsilon;
} vae_encoder_norm_args;
typedef struct { uint32_t rows, width; } swiglu_args;
typedef struct { uint32_t elements, approximate; } gelu_bf16_args;
typedef struct { uint32_t tokens, vocab_size, width; } embedding_args;
typedef struct {
    uint32_t sequence, query_heads, kv_heads, head_dim;
    float epsilon;
} text_rope_args;
typedef struct { uint32_t sequence, heads, head_dim; float epsilon; } head_norm_args;
typedef struct { uint32_t sequence, query_heads, kv_heads, head_dim; } text_rope_inplace_args;
typedef struct {
    uint32_t sequence, query_heads, kv_heads, head_dim;
    float scale;
} gqa_args;
typedef struct { uint32_t sample_offset, elements; float delta, ratio; }
    euler_args;

static int h3_gpu_linear_mps(H3GPU *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim,
                             MPSDataType dataType);

int h3_gpu_linear_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!h3_gpu_require_elements(gpu, input, input_count, @"linear input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, weight_count, @"linear weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count, @"linear output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_dim, @"linear bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32))) return 0;
    if (!getenv("H3_SCALAR_PATCH") && rows >= 16 && output_dim == 5376 &&
        (input_dim == 32 || input_dim == 96)) {
        linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
        const h3_gpu_tensor *bias_buffer = bias ? bias : input;
        if (!h3_gpu_require_command(gpu)) return 0;
        id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
            gpu, @"h3_linear_f32_tiled");
        if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
            h3_gpu_set_error(gpu,
                             @"device cannot dispatch the F32 16x16 tile");
            return 0;
        }
        @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder =
                [gpu.command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake((output_dim + 15) / 16,
                                                      (rows + 15) / 16, 1)
                     threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [encoder endEncoding];
        }
        h3_gpu_stats stats = gpu.stats;
        stats.direct_dispatches++;
        gpu.stats = stats;
        return 1;
    }
    if (rows >= 32 && input_dim >= 256 && output_dim >= 256 &&
        h3_gpu_linear_mps(gpu, output, input, weight, bias, rows,
                          input_dim, output_dim, MPSDataTypeFloat32)) return 1;
    linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    return h3_gpu_dispatch_2d(gpu, @"h3_linear_f32", output_dim, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

int h3_gpu_patch_linear_bf16_offset(
                             h3_gpu *opaque, h3_gpu_tensor *output,
                             size_t output_offset,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (output_dim != 5376 || (input_dim != 32 && input_dim != 96)) {
        h3_gpu_set_error(gpu, @"unsupported fused patch projection shape");
        return 0;
    }
    if (input_offset > SIZE_MAX - input_count ||
        output_offset > SIZE_MAX - output_count ||
        input_offset > SIZE_MAX / sizeof(float) ||
        output_offset > SIZE_MAX / sizeof(uint16_t)) {
        h3_gpu_set_error(gpu, @"fused patch projection offset is out of range");
        return 0;
    }
    if (!h3_gpu_require_elements(gpu, input, input_offset + input_count,
                                 @"patch projection input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, weight_count,
                                 @"patch projection weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_offset + output_count,
                                 @"patch projection output") ||
        TENSOR(output).dtype != H3_GPU_BF16 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_dim,
                                           @"patch projection bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32)) ||
        !h3_gpu_require_command(gpu)) return 0;
    linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, @"h3_linear_f32_tiled_bf16");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu,
                         @"device cannot dispatch the fused F32/BF16 tile");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer
                       offset:input_offset * sizeof(float) atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(output).buffer
                       offset:output_offset * sizeof(uint16_t) atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake((output_dim + 15) / 16,
                                                  (rows + 15) / 16, 1)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_patch_linear_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    return h3_gpu_patch_linear_bf16_offset(
        opaque, output, 0, input, 0, weight, bias, rows, input_dim,
        output_dim);
}

int h3_gpu_patch_linear_bf16_map(
                             h3_gpu *opaque, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias,
                             const h3_gpu_tensor *row_map,
                             uint32_t output_rows, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)output_rows * output_dim;
    if (output_dim != 5376 || (input_dim != 32 && input_dim != 96)) {
        h3_gpu_set_error(gpu, @"unsupported mapped patch projection shape");
        return 0;
    }
    if (!h3_gpu_require_elements(gpu, input, input_count,
                                 @"mapped patch input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, weight_count,
                                 @"mapped patch weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count,
                                 @"mapped patch output") ||
        TENSOR(output).dtype != H3_GPU_BF16 ||
        !h3_gpu_require_elements(gpu, row_map, rows,
                                 @"mapped patch row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_dim,
                                           @"mapped patch bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32)) ||
        !h3_gpu_require_command(gpu)) return 0;
    linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, @"h3_linear_f32_tiled_bf16_map");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu,
                         @"device cannot dispatch the mapped patch tile");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake((output_dim + 15) / 16,
                                                  (rows + 15) / 16, 1)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_silu_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, input, elements, @"SiLU input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, @"SiLU output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_silu_f32", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        });
}

int h3_gpu_cast_f32_to_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, input, elements, @"cast input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, @"cast output") ||
        TENSOR(output).dtype != H3_GPU_BF16) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_cast_f32_to_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        });
}

int h3_gpu_cast_bf16_to_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, input, elements, @"cast input") ||
        TENSOR(input).dtype != H3_GPU_BF16 ||
        !h3_gpu_require_elements(gpu, output, elements, @"cast output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_cast_bf16_to_f32", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        });
}

int h3_gpu_copy_bf16(h3_gpu *opaque, h3_gpu_tensor *destination,
                     size_t destination_offset,
                     const h3_gpu_tensor *source, size_t source_offset,
                     size_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_command(gpu) || !destination || !source ||
        TENSOR(destination).dtype != H3_GPU_BF16 ||
        TENSOR(source).dtype != H3_GPU_BF16 ||
        source_offset > TENSOR(source).elements ||
        elements > TENSOR(source).elements - source_offset ||
        destination_offset > TENSOR(destination).elements ||
        elements > TENSOR(destination).elements - destination_offset ||
        elements > SIZE_MAX / sizeof(uint16_t)) {
        h3_gpu_set_error(gpu, @"invalid BF16 blit range");
        return 0;
    }
    @autoreleasepool {
        id<MTLBlitCommandEncoder> encoder = [gpu.command blitCommandEncoder];
        [encoder copyFromBuffer:TENSOR(source).buffer
                   sourceOffset:source_offset * sizeof(uint16_t)
                       toBuffer:TENSOR(destination).buffer
              destinationOffset:destination_offset * sizeof(uint16_t)
                           size:elements * sizeof(uint16_t)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.blit_copies++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_copy_f32(h3_gpu *opaque, h3_gpu_tensor *destination,
                    size_t destination_offset,
                    const h3_gpu_tensor *source, size_t source_offset,
                    size_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_command(gpu) || !destination || !source ||
        TENSOR(destination).dtype != H3_GPU_F32 ||
        TENSOR(source).dtype != H3_GPU_F32 ||
        source_offset > TENSOR(source).elements ||
        elements > TENSOR(source).elements - source_offset ||
        destination_offset > TENSOR(destination).elements ||
        elements > TENSOR(destination).elements - destination_offset ||
        elements > SIZE_MAX / sizeof(float)) {
        h3_gpu_set_error(gpu, @"invalid F32 blit range");
        return 0;
    }
    @autoreleasepool {
        id<MTLBlitCommandEncoder> encoder = [gpu.command blitCommandEncoder];
        [encoder copyFromBuffer:TENSOR(source).buffer
                   sourceOffset:source_offset * sizeof(float)
                       toBuffer:TENSOR(destination).buffer
              destinationOffset:destination_offset * sizeof(float)
                           size:elements * sizeof(float)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.blit_copies++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_rms_norm_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, @"RMSNorm input") ||
        !h3_gpu_require_elements(gpu, weight, width, @"RMSNorm weight") ||
        !h3_gpu_require_elements(gpu, output, count, @"RMSNorm output")) return 0;
    norm_args args = {rows, width, epsilon};
    return h3_gpu_dispatch_rows(gpu, @"h3_rms_norm_f32", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_adaln_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, @"AdaLN input") ||
        !h3_gpu_require_elements(gpu, norm_weight, width, @"AdaLN norm") ||
        !h3_gpu_require_elements(gpu, row_map, rows, @"AdaLN row map") ||
        !h3_gpu_require_elements(gpu, output, count, @"AdaLN output") ||
        !modulation || shift_slot >= slots || scale_slot >= slots) return 0;
    adaln_args args = {rows, width, slots, shift_slot, scale_slot, epsilon};
    return h3_gpu_dispatch_2d(gpu, @"h3_adaln_f32", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
            [encoder setBytes:&args length:sizeof(args) atIndex:5];
        });
}

int h3_gpu_gate_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual,
                    const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, residual, count, @"gate residual") ||
        !h3_gpu_require_elements(gpu, branch, count, @"gate branch") ||
        !h3_gpu_require_elements(gpu, row_map, rows, @"gate row map") ||
        !h3_gpu_require_elements(gpu, output, count, @"gate output") ||
        !modulation || gate_slot >= slots) return 0;
    gate_args args = {rows, width, slots, gate_slot};
    return h3_gpu_dispatch_2d(gpu, @"h3_gate_f32", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(branch).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
            [encoder setBytes:&args length:sizeof(args) atIndex:5];
        });
}

int h3_gpu_qkv_rope_f32(h3_gpu *opaque, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_gpu_require_elements(gpu, qkv, count * 3, @"QKV input") ||
        !h3_gpu_require_elements(gpu, q_norm, head_dim, @"Q norm") ||
        !h3_gpu_require_elements(gpu, k_norm, head_dim, @"K norm") ||
        !h3_gpu_require_elements(gpu, rope_cos, rope_count, @"RoPE cosine") ||
        !h3_gpu_require_elements(gpu, rope_sin, rope_count, @"RoPE sine") ||
        !h3_gpu_require_elements(gpu, query, count, @"query") ||
        !h3_gpu_require_elements(gpu, key, count, @"key") ||
        !h3_gpu_require_elements(gpu, value, count, @"value") ||
        rope_half * 2 > head_dim) return 0;
    qkv_args args = {sequence, heads, head_dim, rope_half, 0, epsilon};
    return h3_gpu_dispatch_3d(gpu, @"h3_qkv_rope_f32",
        MTLSizeMake(head_dim, heads, sequence),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:5];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:6];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:7];
            [encoder setBytes:&args length:sizeof(args) atIndex:8];
        });
}

static H3SDPA *h3_gpu_sdpa_graph(H3GPU *gpu, uint32_t batch,
                                 uint32_t sequence,
                                 uint32_t heads, uint32_t head_dim, float scale,
                                 MPSDataType dataType, int causal,
                                 int headMajor, int outputHeadMajor) {
    @autoreleasepool {
        NSString *cacheKey = [NSString stringWithFormat:
                              @"%u:%u:%u:%u:%u:%.9g:%d:%d:%d",
                              (unsigned)dataType, batch, sequence, heads,
                              head_dim, scale, causal, headMajor,
                              outputHeadMajor];
        H3SDPA *cached = gpu.sdpaCache[cacheKey];
        if (cached) return cached;
        MPSGraph *graph = [[MPSGraph alloc] init];
        NSArray<NSNumber *> *rowMajorShape =
            @[@(batch), @(sequence), @(heads), @(head_dim)];
        NSArray<NSNumber *> *headMajorShape =
            @[@(batch), @(heads), @(sequence), @(head_dim)];
        NSArray<NSNumber *> *outputShape = outputHeadMajor ?
            headMajorShape : rowMajorShape;
        NSArray<NSNumber *> *inputShape = headMajor ?
            headMajorShape : rowMajorShape;
        MPSGraphTensor *q = [graph placeholderWithShape:inputShape
                                               dataType:dataType name:nil];
        MPSGraphTensor *k = [graph placeholderWithShape:inputShape
                                               dataType:dataType name:nil];
        MPSGraphTensor *v = [graph placeholderWithShape:inputShape
                                               dataType:dataType name:nil];
        MPSGraphTensor *qt = headMajor ? q :
            [graph transposeTensor:q dimension:1 withDimension:2 name:nil];
        MPSGraphTensor *kt = headMajor ? k :
            [graph transposeTensor:k dimension:1 withDimension:2 name:nil];
        MPSGraphTensor *vt = headMajor ? v :
            [graph transposeTensor:v dimension:1 withDimension:2 name:nil];
        if (![graph respondsToSelector:@selector(scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:scale:name:)]) {
            h3_gpu_set_error(gpu, @"native MPSGraph SDPA is unavailable");
            return nil;
        }
        MPSGraphTensor *attention;
        if (causal) {
            size_t mask_count = (size_t)sequence * sequence;
            float *mask_values = malloc(mask_count * sizeof(*mask_values));
            if (!mask_values) return nil;
            for (uint32_t row = 0; row < sequence; row++)
                for (uint32_t column = 0; column < sequence; column++)
                    mask_values[(size_t)row * sequence + column] =
                        column <= row ? 0.0f : -INFINITY;
            NSData *mask_data = [NSData dataWithBytesNoCopy:mask_values
                length:mask_count * sizeof(*mask_values) freeWhenDone:YES];
            MPSGraphTensor *mask = [graph constantWithData:mask_data
                shape:@[@1, @1, @(sequence), @(sequence)]
                dataType:MPSDataTypeFloat32];
            attention = [graph
                scaledDotProductAttentionWithQueryTensor:qt keyTensor:kt
                valueTensor:vt maskTensor:mask scale:scale name:nil];
        } else {
            attention = [graph scaledDotProductAttentionWithQueryTensor:qt
                keyTensor:kt valueTensor:vt scale:scale name:nil];
        }
        H3SDPA *result = [[H3SDPA alloc] init];
        result.graph = graph;
        result.query = q;
        result.key = k;
        result.value = v;
        result.output = outputHeadMajor ? attention :
            [graph transposeTensor:attention dimension:1 withDimension:2
             name:nil];
        result.inputShape = inputShape;
        result.outputShape = outputShape;
        gpu.sdpaCache[cacheKey] = result;
        return result;
    }
}

static int h3_gpu_sdpa(h3_gpu *opaque, h3_gpu_tensor *output,
                       const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                       const h3_gpu_tensor *value, uint32_t batch,
                       uint32_t sequence,
                       uint32_t heads, uint32_t head_dim, float scale,
                       h3_gpu_dtype tensor_dtype, MPSDataType mps_dtype,
                       int causal, int outputHeadMajor) {
    H3GPU *gpu = GPU(opaque);
    int headMajor = gpu.headMajorSDPAInputs &&
        tensor_dtype == H3_GPU_BF16 && batch == 1 && !causal;
    gpu.headMajorSDPAInputs = NO;
    size_t count = (size_t)batch * sequence * heads * head_dim;
    if (!batch || !sequence || !heads || !head_dim ||
        !h3_gpu_require_command(gpu) ||
        !h3_gpu_require_elements(gpu, query, count, @"SDPA query") ||
        !h3_gpu_require_elements(gpu, key, count, @"SDPA key") ||
        !h3_gpu_require_elements(gpu, value, count, @"SDPA value") ||
        !h3_gpu_require_elements(gpu, output, count, @"SDPA output")) return 0;
    if (TENSOR(query).dtype != tensor_dtype || TENSOR(key).dtype != tensor_dtype ||
        TENSOR(value).dtype != tensor_dtype || TENSOR(output).dtype != tensor_dtype) {
        h3_gpu_set_error(gpu, @"SDPA tensor dtype mismatch");
        return 0;
    }
    H3SDPA *cache = h3_gpu_sdpa_graph(gpu, batch, sequence, heads, head_dim,
                                      scale, mps_dtype, causal, headMajor,
                                      outputHeadMajor);
    if (!cache) return 0;
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        MPSGraphTensorData *(^data)(const h3_gpu_tensor *) =
            ^MPSGraphTensorData *(const h3_gpu_tensor *tensor) {
                return h3_gpu_graph_data(tensor, cache.inputShape,
                                         mps_dtype, 0);
            };
        NSDictionary *feeds = @{
            cache.query: data(query), cache.key: data(key), cache.value: data(value)
        };
        MPSGraphTensorData *outputData = h3_gpu_graph_data(
            output, cache.outputShape, mps_dtype, 0);
        NSDictionary *results = @{cache.output: outputData};
        @try {
            [cache.graph encodeToCommandBuffer:command feeds:feeds targetOperations:nil
                             resultsDictionary:results executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph SDPA failed: %@", exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_sdpa_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_sdpa_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale) {
    return h3_gpu_sdpa(opaque, output, query, key, value, 1, sequence, heads,
                       head_dim, scale, H3_GPU_F32, MPSDataTypeFloat32, 0, 0);
}

int h3_gpu_sdpa_causal_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t batch,
                    uint32_t sequence, uint32_t heads, uint32_t head_dim,
                    float scale) {
    return h3_gpu_sdpa(opaque, output, query, key, value, batch, sequence,
                       heads, head_dim, scale, H3_GPU_F32,
                       MPSDataTypeFloat32, 1, 0);
}

int h3_gpu_sdpa_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    return h3_gpu_sdpa(opaque, output, query, key, value, 1, sequence, heads,
                       head_dim, scale, H3_GPU_BF16, MPSDataTypeBFloat16, 0,
                       0);
}

static int h3_gpu_require_bf16(H3GPU *gpu, const h3_gpu_tensor *tensor,
                               size_t elements, NSString *label);

typedef struct {
    int32_t rows;
    int32_t heads;
    float scale;
    int32_t q_blocks;
    int32_t k_full;
    int32_t k_rem;
} h3_attention_args;

static id<MTLComputePipelineState> h3_gpu_attention_pipeline(H3GPU *gpu,
                                                              NSString *name) {
    id<MTLComputePipelineState> pipeline = gpu.attentionPipelines[name];
    if (pipeline || !gpu.attentionLibrary) return pipeline;
    id<MTLFunction> function = [gpu.attentionLibrary newFunctionWithName:name];
    NSError *pipelineError = nil;
    pipeline = function ? [gpu.device newComputePipelineStateWithFunction:function
                                                                   error:&pipelineError] : nil;
    if (pipeline) gpu.attentionPipelines[name] = pipeline;
    else if (getenv("H3_ATTN"))
        fprintf(stderr, "h3: NAX attention pipeline %s unavailable: %s\n",
                name.UTF8String, pipelineError.localizedDescription.UTF8String);
    return pipeline;
}

h3_gpu_attention_plan *h3_gpu_attention_plan_create(
    h3_gpu *opaque, uint32_t rows, const uint32_t *perm,
    const uint32_t *inverse, uint32_t q_blocks, const uint32_t *offsets,
    size_t kept, const uint32_t *indices) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || !rows || !perm || !inverse || !q_blocks || !offsets ||
        !indices) return NULL;
    H3AttentionPlan *plan = [[H3AttentionPlan alloc] init];
    plan.rows = rows;
    plan.qBlocks = q_blocks;
    MTLResourceOptions shared = MTLResourceStorageModeShared;
    plan.perm = [gpu.device newBufferWithBytes:perm length:rows * 4 options:shared];
    plan.inverse = [gpu.device newBufferWithBytes:inverse length:rows * 4 options:shared];
    plan.offsets = [gpu.device newBufferWithBytes:offsets
                                           length:((size_t)q_blocks + 1) * 4
                                          options:shared];
    plan.indices = [gpu.device newBufferWithBytes:indices
                                           length:MAX(kept, (size_t)1) * 4
                                          options:shared];
    if (!plan.perm || !plan.inverse || !plan.offsets || !plan.indices) {
        h3_gpu_set_error(gpu, @"cannot allocate a sparse attention plan");
        return NULL;
    }
    return (__bridge_retained h3_gpu_attention_plan *)plan;
}

void h3_gpu_attention_plan_free(h3_gpu_attention_plan *plan) {
    if (plan) (void)(__bridge_transfer H3AttentionPlan *)plan;
}

void h3_gpu_attention_plan_use(h3_gpu *opaque,
                               h3_gpu_attention_plan *plan) {
    H3GPU *gpu = GPU(opaque);
    if (gpu) gpu.attentionPlan = (__bridge H3AttentionPlan *)plan;
}

/* Which DiT attention to use for head-major BF16 inputs: "mps" (MPSGraph
 * SDPA, the default), "nax" (h3_attention.metal, BF16) or "int8" (NAX with
 * per-row int8 Q/K and mean-smoothed K, BF16 P@V). */
static const char *h3_attention_mode(void) {
    const char *mode = getenv("H3_ATTN");
    return mode && *mode ? mode : "mps";
}

/* Head-major [head][row][128] BF16 Q/K/V -> head-major BF16 output. Returns -1
 * when the kernel is unavailable so the caller can fall back. */
static int h3_gpu_attention_nax(H3GPU *gpu, h3_gpu_tensor *output,
                                const h3_gpu_tensor *query,
                                const h3_gpu_tensor *key,
                                const h3_gpu_tensor *value, uint32_t sequence,
                                uint32_t heads, uint32_t head_dim, float scale,
                                int int8_qk) {
    if (head_dim != 128) return -1;
    const char *variant_env = getenv("H3_ATTN_VARIANT");
    uint32_t BQ = variant_env && !strcmp(variant_env, "b128") ? 128 : 64;
    uint32_t BK = variant_env && !strcmp(variant_env, "k64") ? 64 : 32;
    NSUInteger threads = BQ == 128 ? 256 : 128;
    H3AttentionPlan *plan = gpu.attentionPlan.rows == sequence ?
        gpu.attentionPlan : nil;
    NSString *kernel = int8_qk ? @"h3_attention_nax_qk8_q64_k32" :
                                 @"h3_attention_nax_bf16_q64_k32";
    if (plan) kernel = [kernel stringByAppendingString:@"_sparse"];
    id<MTLComputePipelineState> gather_pipeline = plan ?
        h3_gpu_attention_pipeline(gpu, @"h3_attention_gather_rows") : nil;
    if (plan && !gather_pipeline) return -1;
    const char *stage = getenv("H3_ATTN_STAGE");
    if (stage && *stage) kernel = [kernel stringByAppendingFormat:@"_s%s", stage];
    const char *variant = getenv("H3_ATTN_VARIANT");
    if (variant && *variant) kernel = [kernel stringByAppendingFormat:@"_%s", variant];
    id<MTLComputePipelineState> pipeline = h3_gpu_attention_pipeline(gpu, kernel);
    id<MTLComputePipelineState> mean_pipeline = int8_qk ?
        h3_gpu_attention_pipeline(gpu, @"h3_attention_k_mean") : nil;
    id<MTLComputePipelineState> quantize_pipeline = int8_qk ?
        h3_gpu_attention_pipeline(gpu, @"h3_attention_quantize_qk") : nil;
    if (!pipeline || (int8_qk && (!mean_pipeline || !quantize_pipeline)))
        return -1;
    size_t count = (size_t)sequence * heads * head_dim;
    if (!h3_gpu_require_bf16(gpu, query, count, @"attention query") ||
        !h3_gpu_require_bf16(gpu, key, count, @"attention key") ||
        !h3_gpu_require_bf16(gpu, value, count, @"attention value") ||
        !h3_gpu_require_bf16(gpu, output, count, @"attention output") ||
        !h3_gpu_require_command(gpu)) return 0;
    h3_attention_args args = {(int32_t)sequence, (int32_t)heads, scale,
                              (int32_t)((sequence + BQ - 1) / BQ),
                              (int32_t)(sequence / BK),
                              (int32_t)(sequence % BK)};
    if (int8_qk && gpu.attentionScratchElements < count) {
        gpu.attentionQ8 = [gpu.device newBufferWithLength:count
                                                  options:MTLResourceStorageModePrivate];
        gpu.attentionK8 = [gpu.device newBufferWithLength:count
                                                  options:MTLResourceStorageModePrivate];
        gpu.attentionScales = [gpu.device
            newBufferWithLength:(size_t)sequence * heads * 2 * sizeof(float)
                        options:MTLResourceStorageModePrivate];
        gpu.attentionMean = [gpu.device
            newBufferWithLength:(size_t)heads * 128 * sizeof(float)
                        options:MTLResourceStorageModePrivate];
        if (!gpu.attentionQ8 || !gpu.attentionK8 || !gpu.attentionScales ||
            !gpu.attentionMean) {
            gpu.attentionScratchElements = 0;
            h3_gpu_set_error(gpu, @"cannot allocate int8 attention scratch");
            return 0;
        }
        gpu.attentionScratchElements = count;
    }
    if (plan && gpu.attentionPermutedElements < count) {
        gpu.attentionQP = [gpu.device newBufferWithLength:count * 2
                                                  options:MTLResourceStorageModePrivate];
        gpu.attentionKP = [gpu.device newBufferWithLength:count * 2
                                                  options:MTLResourceStorageModePrivate];
        gpu.attentionVP = [gpu.device newBufferWithLength:count * 2
                                                  options:MTLResourceStorageModePrivate];
        gpu.attentionOP = [gpu.device newBufferWithLength:count * 2
                                                  options:MTLResourceStorageModePrivate];
        if (!gpu.attentionQP || !gpu.attentionKP || !gpu.attentionVP ||
            !gpu.attentionOP) {
            gpu.attentionPermutedElements = 0;
            h3_gpu_set_error(gpu, @"cannot allocate sparse attention scratch");
            return 0;
        }
        gpu.attentionPermutedElements = count;
    }
    size_t k_scale_offset = (size_t)sequence * heads * sizeof(float);
    void (^gather)(id<MTLComputeCommandEncoder>, id<MTLBuffer>, id<MTLBuffer>,
                   id<MTLBuffer>) =
        ^(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> source,
          id<MTLBuffer> destination, id<MTLBuffer> rows) {
            [encoder setComputePipelineState:gather_pipeline];
            [encoder setBuffer:source offset:0 atIndex:0];
            [encoder setBuffer:destination offset:0 atIndex:1];
            [encoder setBuffer:rows offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
            [encoder dispatchThreads:MTLSizeMake((NSUInteger)sequence * 16, heads, 1)
               threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        };
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        if (int8_qk) {
            int32_t no = 0, yes = 1;
            [encoder setComputePipelineState:mean_pipeline];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:0];
            [encoder setBuffer:gpu.attentionMean offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
            [encoder dispatchThreads:MTLSizeMake((NSUInteger)heads * 128, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [encoder setComputePipelineState:quantize_pipeline];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:0];
            [encoder setBuffer:gpu.attentionQ8 offset:0 atIndex:1];
            [encoder setBuffer:gpu.attentionScales offset:0 atIndex:2];
            [encoder setBuffer:gpu.attentionMean offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
            [encoder setBytes:&no length:sizeof(no) atIndex:5];
            int32_t mapped = plan ? 1 : 0;
            [encoder setBuffer:plan ? plan.perm : gpu.attentionMean offset:0 atIndex:6];
            [encoder setBytes:&mapped length:sizeof(mapped) atIndex:7];
            [encoder dispatchThreadgroups:MTLSizeMake(sequence, heads, 1)
                     threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:0];
            [encoder setBuffer:gpu.attentionK8 offset:0 atIndex:1];
            [encoder setBuffer:gpu.attentionScales offset:k_scale_offset atIndex:2];
            [encoder setBytes:&yes length:sizeof(yes) atIndex:5];
            [encoder dispatchThreadgroups:MTLSizeMake(sequence, heads, 1)
                     threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            if (plan) gather(encoder, TENSOR(value).buffer, gpu.attentionVP, plan.perm);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        } else if (plan) {
            gather(encoder, TENSOR(query).buffer, gpu.attentionQP, plan.perm);
            gather(encoder, TENSOR(key).buffer, gpu.attentionKP, plan.perm);
            gather(encoder, TENSOR(value).buffer, gpu.attentionVP, plan.perm);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        [encoder setComputePipelineState:pipeline];
        if (plan) {
            [encoder setBuffer:plan.offsets offset:0 atIndex:7];
            [encoder setBuffer:plan.indices offset:0 atIndex:8];
        }
        if (int8_qk) {
            [encoder setBuffer:gpu.attentionQ8 offset:0 atIndex:0];
            [encoder setBuffer:gpu.attentionK8 offset:0 atIndex:1];
            [encoder setBuffer:gpu.attentionScales offset:0 atIndex:5];
            [encoder setBuffer:gpu.attentionScales offset:k_scale_offset atIndex:6];
        } else {
            [encoder setBuffer:plan ? gpu.attentionQP : TENSOR(query).buffer offset:0 atIndex:0];
            [encoder setBuffer:plan ? gpu.attentionKP : TENSOR(key).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:5];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:6];
        }
        [encoder setBuffer:plan ? gpu.attentionVP : TENSOR(value).buffer offset:0 atIndex:2];
        [encoder setBuffer:plan ? gpu.attentionOP : TENSOR(output).buffer offset:0 atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)args.q_blocks, heads, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        if (plan) {
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            gather(encoder, gpu.attentionOP, TENSOR(output).buffer, plan.inverse);
        }
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches += int8_qk ? 4 : 1;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_has_nax_extras(const h3_gpu *opaque) {
    H3GPU *gpu = GPU(opaque);
    return gpu && gpu.attentionLibrary && gpu.tensorOpsEnabled &&
        h3_gpu_attention_pipeline(gpu, @"h3_linear_bf16_f32out_nax") != nil;
}

/* The mixed-precision NAX linear on a BF16 input that already holds whole
 * 128-row tiles (rows past `rows` are computed and dropped). */
static int h3_gpu_encode_linear_bf16_f32out(H3GPU *gpu, h3_gpu_tensor *output,
                                            const h3_gpu_tensor *input,
                                            const h3_gpu_tensor *weight,
                                            const h3_gpu_tensor *bias,
                                            uint32_t rows, uint32_t input_dim,
                                            uint32_t output_dim) {
    uint32_t padded = (rows + 127u) & ~127u;
    if (input_dim % 128 || output_dim % 128 ||
        !h3_gpu_require_bf16(gpu, input, (size_t)padded * input_dim, @"mixed linear BF16 input") ||
        !h3_gpu_require_bf16(gpu, weight, (size_t)output_dim * input_dim, @"mixed linear weight") ||
        !h3_gpu_require_elements(gpu, output, (size_t)rows * output_dim, @"mixed linear output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_dim, @"mixed linear bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32))) return 0;
    id<MTLComputePipelineState> pipeline =
        h3_gpu_attention_pipeline(gpu, @"h3_linear_bf16_f32out_nax");
    if (!pipeline) {
        h3_gpu_set_error(gpu, @"NAX mixed-precision linear is unavailable");
        return 0;
    }
    if (!h3_gpu_require_command(gpu)) return 0;
    /* 128x64 output tiles; walk the row tiles fastest when the output is wider than the input (QKV). */
    uint32_t row_major = output_dim > input_dim;
    uint32_t dims[4] = {rows, input_dim, output_dim, row_major};
    MTLSize groups = row_major ? MTLSizeMake(padded / 128, output_dim / 64, 1)
                               : MTLSizeMake(output_dim / 64, padded / 128, 1);
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:bias ? TENSOR(bias).buffer : TENSOR(output).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
        [encoder setBytes:dims length:sizeof(dims) atIndex:4];
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_linear_bf16_weight_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *input,
                                  const h3_gpu_tensor *weight,
                                  const h3_gpu_tensor *bias,
                                  h3_gpu_tensor *scratch, uint32_t rows,
                                  uint32_t input_dim, uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu) return 0;
    uint32_t padded = (rows + 127u) & ~127u;
    if (input_dim % 128 || output_dim % 128 ||
        !h3_gpu_require_elements(gpu, input, (size_t)rows * input_dim, @"mixed linear input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, scratch, (size_t)padded * input_dim, @"mixed linear scratch") ||
        TENSOR(scratch).dtype != H3_GPU_BF16) return 0;
    if (!h3_gpu_attention_pipeline(gpu, @"h3_linear_bf16_f32out_nax")) {
        h3_gpu_set_error(gpu, @"NAX mixed-precision linear is unavailable");
        return 0;
    }
    if (!h3_gpu_require_bf16(gpu, weight, (size_t)output_dim * input_dim, @"mixed linear weight") ||
        !h3_gpu_require_elements(gpu, output, (size_t)rows * output_dim, @"mixed linear output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_dim, @"mixed linear bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32))) return 0;
    return h3_gpu_cast_f32_to_bf16(opaque, scratch, input, (uint32_t)((size_t)rows * input_dim)) &&
           h3_gpu_encode_linear_bf16_f32out(gpu, output, scratch, weight, bias,
                                            rows, input_dim, output_dim);
}

int h3_gpu_linear_bf16_input_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *bias, uint32_t rows,
                                 uint32_t input_dim, uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    return gpu && h3_gpu_encode_linear_bf16_f32out(gpu, output, input, weight,
                                                   bias, rows, input_dim,
                                                   output_dim);
}

void h3_gpu_declare_head_major_sdpa_inputs(h3_gpu *opaque) {
    H3GPU *gpu = GPU(opaque);
    if (gpu) gpu.headMajorSDPAInputs = YES;
}

int h3_gpu_sdpa_bf16_head_major_output(
                     h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    H3GPU *gpu = GPU(opaque);
    const char *mode = h3_attention_mode();
    if (gpu && gpu.headMajorSDPAInputs &&
        (!strcmp(mode, "nax") || !strcmp(mode, "int8") ||
         gpu.attentionPlan.rows == sequence)) {
        int done = h3_gpu_attention_nax(gpu, output, query, key, value,
                                        sequence, heads, head_dim, scale,
                                        !strcmp(mode, "int8"));
        if (done >= 0) {
            gpu.headMajorSDPAInputs = NO;
            return done;
        }
    }
    return h3_gpu_sdpa(opaque, output, query, key, value, 1, sequence, heads,
                       head_dim, scale, H3_GPU_BF16, MPSDataTypeBFloat16, 0,
                       1);
}

int h3_gpu_swiglu_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, fused, (size_t)rows * width * 2, @"SwiGLU input") ||
        !h3_gpu_require_elements(gpu, output, (size_t)rows * width, @"SwiGLU output")) return 0;
    swiglu_args args = {rows, width};
    return h3_gpu_dispatch_2d(gpu, @"h3_swiglu_f32", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(fused).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_scale_add_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *scale, uint32_t rows,
                         uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, residual, count, @"scale-add residual") ||
        TENSOR(residual).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, branch, count, @"scale-add branch") ||
        TENSOR(branch).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, scale, width, @"scale-add scale") ||
        TENSOR(scale).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, @"scale-add output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    swiglu_args args = {rows, width};
    return h3_gpu_dispatch_2d(gpu, @"h3_scale_add_f32", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(branch).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(scale).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

int h3_gpu_layer_norm_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, @"LayerNorm input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, width, @"LayerNorm weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, bias, width, @"LayerNorm bias") ||
        TENSOR(bias).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, @"LayerNorm output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    norm_args args = {rows, width, epsilon};
    return h3_gpu_dispatch_rows(gpu, @"h3_layer_norm_f32", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

int h3_gpu_video_qkv_rope_f32(h3_gpu *opaque, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_gpu_require_elements(gpu, qkv, count * 3, @"video QKV") ||
        TENSOR(qkv).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, rope_cos, rope_count, @"video RoPE cosine") ||
        TENSOR(rope_cos).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, rope_sin, rope_count, @"video RoPE sine") ||
        TENSOR(rope_sin).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, query, count, @"video query") ||
        TENSOR(query).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, key, count, @"video key") ||
        TENSOR(key).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, value, count, @"video value") ||
        TENSOR(value).dtype != H3_GPU_F32 || rope_half * 2 > head_dim) return 0;
    qkv_args args = {sequence, heads, head_dim, rope_half, 0, epsilon};
    return h3_gpu_dispatch_3d(gpu, @"h3_video_qkv_rope_f32",
        MTLSizeMake(head_dim, heads, sequence),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
        });
}

/* Track B fused VAE ops (bit-identical to the F32 kernel + cast chains). */
int h3_gpu_rms_norm_f32_to_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                                const h3_gpu_tensor *input,
                                const h3_gpu_tensor *weight, uint32_t rows,
                                uint32_t width, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, @"RMSNorm input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, width, @"RMSNorm weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_bf16(gpu, output, count, @"RMSNorm BF16 output")) return 0;
    /* The reduction tree depends on the threadgroup width: use exactly the
     * width h3_rms_norm_f32 gets, or refuse (the caller then keeps the chain). */
    id<MTLComputePipelineState> reference = h3_gpu_pipeline(gpu, @"h3_rms_norm_f32");
    id<MTLComputePipelineState> fused = h3_gpu_pipeline(gpu, @"h3_rms_norm_f32_to_bf16");
    if (!reference || !fused) return 0;
    NSUInteger reference_threads = 1, fused_threads = 1;
    while (reference_threads * 2 <= MIN((NSUInteger)256, reference.maxTotalThreadsPerThreadgroup))
        reference_threads *= 2;
    while (fused_threads * 2 <= MIN((NSUInteger)256, fused.maxTotalThreadsPerThreadgroup))
        fused_threads *= 2;
    if (reference_threads != fused_threads) {
        h3_gpu_set_error(gpu, @"fused RMSNorm threadgroup width differs");
        return 0;
    }
    norm_args args = {rows, width, epsilon};
    return h3_gpu_dispatch_rows(gpu, @"h3_rms_norm_f32_to_bf16", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_swiglu_f32_to_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                              const h3_gpu_tensor *fused, uint32_t rows,
                              uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, fused, (size_t)rows * width * 2, @"SwiGLU input") ||
        TENSOR(fused).dtype != H3_GPU_F32 ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * width, @"SwiGLU BF16 output")) return 0;
    swiglu_args args = {rows, width};
    return h3_gpu_dispatch_2d(gpu, @"h3_swiglu_f32_to_bf16", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(fused).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_video_qkv_rope_bf16(h3_gpu *opaque, h3_gpu_tensor *query,
                               h3_gpu_tensor *key, h3_gpu_tensor *value,
                               const h3_gpu_tensor *qkv,
                               const h3_gpu_tensor *rope_cos,
                               const h3_gpu_tensor *rope_sin,
                               uint32_t sequence, uint32_t heads,
                               uint32_t head_dim, uint32_t rope_half,
                               float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)sequence * heads * head_dim;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!gpu) return 0;
    if (!sequence || !heads || head_dim != 64 || rope_half * 2 > head_dim) {
        h3_gpu_set_error(gpu, @"video QKV RoPE needs head_dim 64 and a non-empty grid");
        return 0;
    }
    if (!h3_gpu_require_elements(gpu, qkv, count * 3, @"video QKV") ||
        TENSOR(qkv).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, rope_cos, rope_count, @"video RoPE cosine") ||
        TENSOR(rope_cos).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, rope_sin, rope_count, @"video RoPE sine") ||
        TENSOR(rope_sin).dtype != H3_GPU_F32 ||
        !h3_gpu_require_bf16(gpu, query, count, @"video BF16 query") ||
        !h3_gpu_require_bf16(gpu, key, count, @"video BF16 key") ||
        !h3_gpu_require_bf16(gpu, value, count, @"video BF16 value") ||
        !h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline =
        h3_gpu_pipeline(gpu, @"h3_video_qkv_rope_bf16");
    if (!pipeline) return 0;
    if (pipeline.threadExecutionWidth != 32) {
        h3_gpu_set_error(gpu, @"video QKV RoPE needs 32-wide simdgroups");
        return 0;
    }
    qkv_args args = {sequence, heads, head_dim, rope_half, 0, epsilon};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:5];
        [encoder setBytes:&args length:sizeof(args) atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake(heads, sequence, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static H3Conv *h3_gpu_conv_graph(H3GPU *gpu, uint32_t batch,
                                 uint32_t length, uint32_t input_channels,
                                 uint32_t output_channels, uint32_t kernel,
                                 uint32_t stride, uint32_t padding,
                                 uint32_t dilation, uint32_t output_length,
                                 int transpose, int has_bias) {
    @autoreleasepool {
        NSString *key = [NSString stringWithFormat:
            @"%d:%u:%u:%u:%u:%u:%u:%u:%u:%d", transpose, batch, length,
            input_channels, output_channels, kernel, stride, padding,
            dilation, has_bias];
        H3Conv *cached = gpu.convCache[key];
        if (cached) return cached;

        H3Conv *conv = [[H3Conv alloc] init];
        conv.graph = [[MPSGraph alloc] init];
        conv.inputShape = @[@(batch), @1, @(length), @(input_channels)];
        conv.weightShape = transpose ?
            @[@(input_channels), @(output_channels), @1, @(kernel)] :
            @[@(output_channels), @(input_channels), @1, @(kernel)];
        conv.biasShape = @[@1, @1, @1, @(output_channels)];
        conv.outputShape = @[@(batch), @1, @(output_length),
                             @(output_channels)];
        conv.input = [conv.graph placeholderWithShape:conv.inputShape
                                              dataType:MPSDataTypeFloat32
                                                  name:nil];
        conv.weight = [conv.graph placeholderWithShape:conv.weightShape
                                               dataType:MPSDataTypeFloat32
                                                   name:nil];
        MPSGraphConvolution2DOpDescriptor *descriptor =
            [MPSGraphConvolution2DOpDescriptor
                descriptorWithStrideInX:stride strideInY:1
                dilationRateInX:dilation dilationRateInY:1 groups:1
                paddingLeft:padding paddingRight:padding
                paddingTop:0 paddingBottom:0
                paddingStyle:MPSGraphPaddingStyleExplicit
                dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        MPSGraphTensor *result = transpose ?
            [conv.graph convolutionTranspose2DWithSourceTensor:conv.input
                 weightsTensor:conv.weight outputShape:conv.outputShape
                 descriptor:descriptor name:nil] :
            [conv.graph convolution2DWithSourceTensor:conv.input
                 weightsTensor:conv.weight descriptor:descriptor name:nil];
        if (has_bias) {
            conv.bias = [conv.graph placeholderWithShape:conv.biasShape
                                                dataType:MPSDataTypeFloat32
                                                    name:nil];
            result = [conv.graph additionWithPrimaryTensor:result
                                           secondaryTensor:conv.bias name:nil];
        }
        conv.output = result;
        gpu.convCache[key] = conv;
        return conv;
    }
}

static int h3_gpu_conv_mps(H3GPU *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t batch,
                           uint32_t length, uint32_t input_channels,
                           uint32_t output_channels, uint32_t kernel,
                           uint32_t stride, uint32_t padding,
                           uint32_t dilation, uint32_t output_length,
                           int transpose) {
    if (!h3_gpu_require_command(gpu)) return 0;
    H3Conv *conv = h3_gpu_conv_graph(
        gpu, batch, length, input_channels, output_channels, kernel, stride,
        padding, dilation, output_length, transpose, bias != NULL);
    if (!conv) return 0;
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        MPSGraphTensorData *input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(input).buffer shape:conv.inputShape
            dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *weight_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(weight).buffer shape:conv.weightShape
            dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(output).buffer shape:conv.outputShape
            dataType:MPSDataTypeFloat32];
        NSMutableDictionary *feeds = [@{conv.input: input_data,
                                         conv.weight: weight_data} mutableCopy];
        if (bias) {
            MPSGraphTensorData *bias_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:TENSOR(bias).buffer shape:conv.biasShape
                dataType:MPSDataTypeFloat32];
            feeds[conv.bias] = bias_data;
        }
        NSDictionary *results = @{conv.output: output_data};
        @try {
            [conv.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil resultsDictionary:results
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph Conv1d failed: %@",
                             exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_conv_dispatches++;
    gpu.stats = stats;
    return 1;
}

/* mode 0: F32 (the reference); 1: BF16 compute (H3_VAE_ENC_BF16, opt-in:
 * F32 input cast to BF16 in the graph, BF16 weights, result cast back to F32
 * before the F32 bias). */
static H3Conv *h3_gpu_conv3d_graph(
        H3GPU *gpu, uint32_t batch, uint32_t depth, uint32_t height,
        uint32_t width, uint32_t input_channels, uint32_t output_channels,
        uint32_t kernel_depth, uint32_t kernel_height, uint32_t kernel_width,
        uint32_t stride_depth, uint32_t stride_height, uint32_t stride_width,
        uint32_t output_depth, uint32_t output_height, uint32_t output_width,
        int has_bias, int mode) {
    @autoreleasepool {
        NSString *key = [NSString stringWithFormat:
            @"3:%u:%u:%u:%u:%u:%u:%u:%u:%u:%u:%u:%u:%d%@", batch, depth,
            height, width, input_channels, output_channels, kernel_depth,
            kernel_height, kernel_width, stride_depth, stride_height,
            stride_width, has_bias,
            mode ? [NSString stringWithFormat:@":m%d", mode] : @""];
        H3Conv *cached = gpu.convCache[key];
        if (cached) return cached;
        H3Conv *conv = [[H3Conv alloc] init];
        conv.graph = [[MPSGraph alloc] init];
        conv.inputShape = @[@(batch), @(depth), @(height), @(width),
                            @(input_channels)];
        conv.weightShape = @[@(output_channels), @(input_channels),
                             @(kernel_depth), @(kernel_height), @(kernel_width)];
        conv.biasShape = @[@1, @1, @1, @1, @(output_channels)];
        conv.outputShape = @[@(batch), @(output_depth), @(output_height),
                             @(output_width), @(output_channels)];
        conv.input = [conv.graph placeholderWithShape:conv.inputShape
                                              dataType:MPSDataTypeFloat32
                                                  name:nil];
        conv.weight = [conv.graph placeholderWithShape:conv.weightShape
                                               dataType:mode == 1 ?
                                                   MPSDataTypeBFloat16 :
                                                   MPSDataTypeFloat32
                                                   name:nil];
        MPSGraphConvolution3DOpDescriptor *descriptor =
            [MPSGraphConvolution3DOpDescriptor
                descriptorWithStrideInX:stride_width
                strideInY:stride_height strideInZ:stride_depth
                dilationRateInX:1 dilationRateInY:1 dilationRateInZ:1 groups:1
                paddingLeft:0 paddingRight:0 paddingTop:0 paddingBottom:0
                paddingFront:0 paddingBack:0
                paddingStyle:MPSGraphPaddingStyleExplicit
                dataLayout:MPSGraphTensorNamedDataLayoutNDHWC
                weightsLayout:MPSGraphTensorNamedDataLayoutOIDHW];
        MPSGraphTensor *source = mode == 1 ?
            [conv.graph castTensor:conv.input toType:MPSDataTypeBFloat16 name:nil] :
            conv.input;
        MPSGraphTensor *result = [conv.graph
            convolution3DWithSourceTensor:source weightsTensor:conv.weight
            descriptor:descriptor name:nil];
        if (mode == 1)
            result = [conv.graph castTensor:result toType:MPSDataTypeFloat32 name:nil];
        if (has_bias) {
            conv.bias = [conv.graph placeholderWithShape:conv.biasShape
                                                dataType:MPSDataTypeFloat32
                                                    name:nil];
            result = [conv.graph additionWithPrimaryTensor:result
                                           secondaryTensor:conv.bias name:nil];
        }
        conv.output = result;
        gpu.convCache[key] = conv;
        return conv;
    }
}

static int h3_gpu_conv3d(h3_gpu *opaque, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight,
                         const h3_gpu_tensor *bias, uint32_t batch,
                         uint32_t depth, uint32_t height, uint32_t width,
                         uint32_t input_channels, uint32_t output_channels,
                         uint32_t kernel_depth, uint32_t kernel_height,
                         uint32_t kernel_width, uint32_t stride_depth,
                         uint32_t stride_height, uint32_t stride_width,
                         int stable_weights, int mode) {
    H3GPU *gpu = GPU(opaque);
    if (!batch || !depth || !height || !width || !input_channels ||
        !output_channels || !kernel_depth || !kernel_height || !kernel_width ||
        !stride_depth || !stride_height || !stride_width ||
        height < kernel_height || width < kernel_width ||
        depth < kernel_depth || (mode != 0 && mode != 1))
        return 0;
    uint32_t output_depth = (depth - kernel_depth) / stride_depth + 1;
    uint32_t output_height = (height - kernel_height) / stride_height + 1;
    uint32_t output_width = (width - kernel_width) / stride_width + 1;
    size_t input_count = (size_t)batch * depth * height * width * input_channels;
    size_t weight_count = (size_t)output_channels * input_channels *
                          kernel_depth * kernel_height * kernel_width;
    size_t output_count = (size_t)batch * output_depth * output_height *
                          output_width * output_channels;
    if (!h3_gpu_require_elements(gpu, input, input_count, @"Conv3d input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, weight_count, @"Conv3d weight") ||
        TENSOR(weight).dtype != (mode == 1 ? H3_GPU_BF16 : H3_GPU_F32) ||
        !h3_gpu_require_elements(gpu, output, output_count, @"Conv3d output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_channels,
                                           @"Conv3d bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32)) ||
        !h3_gpu_require_command(gpu)) return 0;
    H3Conv *conv = h3_gpu_conv3d_graph(
        gpu, batch, depth, height, width, input_channels, output_channels,
        kernel_depth, kernel_height, kernel_width, stride_depth, stride_height,
        stride_width, output_depth, output_height, output_width, bias != NULL,
        mode);
    if (!conv) return 0;
    MPSDataType weight_type = mode == 1 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32;
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        MPSGraphTensorData *input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(input).buffer shape:conv.inputShape
            dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *weight_data = h3_gpu_graph_data(
            weight, conv.weightShape, weight_type, stable_weights);
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(output).buffer shape:conv.outputShape
            dataType:MPSDataTypeFloat32];
        NSMutableDictionary *feeds = [@{conv.input: input_data,
                                         conv.weight: weight_data} mutableCopy];
        if (bias) {
            MPSGraphTensorData *bias_data = h3_gpu_graph_data(
                bias, conv.biasShape, MPSDataTypeFloat32, stable_weights);
            feeds[conv.bias] = bias_data;
        }
        NSDictionary *results = @{conv.output: output_data};
        @try {
            [conv.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil resultsDictionary:results
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph Conv3d failed: %@",
                             exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_conv_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_conv3d_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t input_channels, uint32_t output_channels,
                      uint32_t kernel_depth, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride_depth,
                      uint32_t stride_height, uint32_t stride_width) {
    return h3_gpu_conv3d(opaque, output, input, weight, bias, batch, depth,
                         height, width, input_channels, output_channels,
                         kernel_depth, kernel_height, kernel_width,
                         stride_depth, stride_height, stride_width, 0, 0);
}

int h3_gpu_conv3d_f32_ex(h3_gpu *opaque, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight,
                         const h3_gpu_tensor *bias, uint32_t batch,
                         uint32_t depth, uint32_t height, uint32_t width,
                         uint32_t input_channels, uint32_t output_channels,
                         uint32_t kernel_depth, uint32_t kernel_height,
                         uint32_t kernel_width, uint32_t stride_depth,
                         uint32_t stride_height, uint32_t stride_width,
                         int mode) {
    return h3_gpu_conv3d(opaque, output, input, weight, bias, batch, depth,
                         height, width, input_channels, output_channels,
                         kernel_depth, kernel_height, kernel_width,
                         stride_depth, stride_height, stride_width, 1, mode);
}

int h3_gpu_conv1d_stride_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding,
                      uint32_t dilation) {
    H3GPU *gpu = GPU(opaque);
    uint64_t effective = (uint64_t)dilation * (kernel - 1) + 1;
    if (!batch || !length || !input_channels || !output_channels || !kernel ||
        !stride || !dilation || (uint64_t)length + 2 * padding < effective)
        return 0;
    uint32_t output_length = (uint32_t)(((uint64_t)length + 2 * padding -
                                         effective) / stride + 1);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)output_channels * input_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (!h3_gpu_require_elements(gpu, input, input_count, @"Conv1d input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, weight_count, @"Conv1d weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count, @"Conv1d output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_channels,
                                           @"Conv1d bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32))) return 0;
    return h3_gpu_conv_mps(gpu, output, input, weight, bias, batch, length,
        input_channels, output_channels, kernel, stride, padding, dilation,
        output_length, 0);
}

int h3_gpu_conv1d_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation) {
    return h3_gpu_conv1d_stride_f32(opaque, output, input, weight, bias,
        batch, length, input_channels, output_channels, kernel, 1, padding,
        dilation);
}

int h3_gpu_conv_transpose1d_f32(
                      h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding) {
    H3GPU *gpu = GPU(opaque);
    if (!batch || !length || !input_channels || !output_channels || !kernel ||
        !stride || (uint64_t)(length - 1) * stride + kernel < 2 * padding)
        return 0;
    uint32_t output_length = (uint32_t)((uint64_t)(length - 1) * stride +
                                        kernel - 2 * padding);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)input_channels * output_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (!h3_gpu_require_elements(gpu, input, input_count,
                                 @"ConvTranspose1d input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, weight_count,
                                 @"ConvTranspose1d weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count,
                                 @"ConvTranspose1d output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_channels,
                                           @"ConvTranspose1d bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32))) return 0;
    return h3_gpu_conv_mps(gpu, output, input, weight, bias, batch, length,
        input_channels, output_channels, kernel, stride, padding, 1,
        output_length, 1);
}

int h3_gpu_weight_norm_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude,
                           uint32_t outer, uint32_t inner) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)outer * inner;
    if (!outer || !inner ||
        !h3_gpu_require_elements(gpu, vector, count, @"weight-norm vector") ||
        TENSOR(vector).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, magnitude, outer,
                                 @"weight-norm magnitude") ||
        TENSOR(magnitude).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, @"weight-norm output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    weight_norm_args args = {outer, inner};
    return h3_gpu_dispatch_1d(gpu, @"h3_weight_norm_f32", outer,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(vector).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(magnitude).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_add_scaled_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left,
                          const h3_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, left, elements, @"scaled-add left") ||
        TENSOR(left).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, right, elements, @"scaled-add right") ||
        TENSOR(right).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, @"scaled-add output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    add_scaled_args args = {elements, left_scale, right_scale};
    return h3_gpu_dispatch_1d(gpu, @"h3_add_scaled_f32", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(left).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(right).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_alias_free_snake_f32(
                          h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *alpha_log,
                          const h3_gpu_tensor *beta_log,
                          const h3_gpu_tensor *upsample_filter,
                          const h3_gpu_tensor *downsample_filter,
                          uint32_t batch, uint32_t length,
                          uint32_t channels) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)batch * length * channels;
    if (!batch || !length || !channels ||
        !h3_gpu_require_elements(gpu, input, count, @"Snake input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, @"Snake output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, alpha_log, channels, @"Snake alpha") ||
        TENSOR(alpha_log).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, beta_log, channels, @"Snake beta") ||
        TENSOR(beta_log).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, upsample_filter, 12,
                                 @"Snake upsample filter") ||
        TENSOR(upsample_filter).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, downsample_filter, 12,
                                 @"Snake downsample filter") ||
        TENSOR(downsample_filter).dtype != H3_GPU_F32) return 0;
    audio_activation_args args = {batch, length, channels};
    return h3_gpu_dispatch_3d(gpu, @"h3_alias_free_snake_f32",
        MTLSizeMake(channels, length, batch),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(alpha_log).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(beta_log).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(upsample_filter).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(downsample_filter).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
        });
}

int h3_gpu_snake1d_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *alpha, uint32_t batch,
                       uint32_t length, uint32_t channels) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)batch * length * channels;
    if (!batch || !length || !channels || count > UINT32_MAX ||
        !h3_gpu_require_elements(gpu, input, count, @"Snake1d input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, alpha, channels, @"Snake1d alpha") ||
        TENSOR(alpha).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, @"Snake1d output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    audio_activation_args args = {batch, length, channels};
    return h3_gpu_dispatch_1d(gpu, @"h3_snake1d_f32", (uint32_t)count,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(alpha).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_audio_qkv_split_f32(h3_gpu *opaque,
                       h3_gpu_tensor *query, h3_gpu_tensor *key,
                       h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                       const h3_gpu_tensor *q_bias,
                       const h3_gpu_tensor *k_bias,
                       const h3_gpu_tensor *v_bias, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t width = (size_t)heads * head_dim;
    size_t count = (size_t)batch * length * width;
    if (!batch || !length || !heads || !head_dim || count > UINT32_MAX ||
        !h3_gpu_require_elements(gpu, qkv, count * 3, @"audio QKV") ||
        TENSOR(qkv).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, q_bias, width, @"audio Q bias") ||
        !h3_gpu_require_elements(gpu, k_bias, width, @"audio K bias") ||
        !h3_gpu_require_elements(gpu, v_bias, width, @"audio V bias") ||
        TENSOR(q_bias).dtype != H3_GPU_F32 ||
        TENSOR(k_bias).dtype != H3_GPU_F32 ||
        TENSOR(v_bias).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, query, count, @"audio query") ||
        !h3_gpu_require_elements(gpu, key, count, @"audio key") ||
        !h3_gpu_require_elements(gpu, value, count, @"audio value") ||
        TENSOR(query).dtype != H3_GPU_F32 || TENSOR(key).dtype != H3_GPU_F32 ||
        TENSOR(value).dtype != H3_GPU_F32) return 0;
    audio_qkv_args args = {batch, length, heads, head_dim};
    return h3_gpu_dispatch_1d(gpu, @"h3_audio_qkv_split_f32",
        (uint32_t)count, ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(q_bias).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(k_bias).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(v_bias).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:5];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:6];
            [encoder setBytes:&args length:sizeof(args) atIndex:7];
        });
}

int h3_gpu_audio_attention_pool_f32(h3_gpu *opaque,
                       h3_gpu_tensor *output,
                       const h3_gpu_tensor *attended, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim, uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t input_count = (size_t)batch * length * heads * head_dim;
    size_t output_count = (size_t)batch * length * output_dim;
    if (!batch || !length || !heads || !head_dim || !output_dim ||
        head_dim % output_dim || output_count > UINT32_MAX ||
        !h3_gpu_require_elements(gpu, attended, input_count,
                                 @"audio attended values") ||
        TENSOR(attended).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count,
                                 @"audio pooled values") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    audio_pool_args args = {batch, length, heads, head_dim, output_dim};
    return h3_gpu_dispatch_1d(gpu, @"h3_audio_attention_pool_f32",
        (uint32_t)output_count, ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(attended).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_geglu_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate,
                     const h3_gpu_tensor *linear, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, gate, elements, @"GeGLU gate") ||
        !h3_gpu_require_elements(gpu, linear, elements, @"GeGLU linear") ||
        !h3_gpu_require_elements(gpu, output, elements, @"GeGLU output") ||
        TENSOR(gate).dtype != H3_GPU_F32 ||
        TENSOR(linear).dtype != H3_GPU_F32 ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_geglu_f32", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(gate).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(linear).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:3];
        });
}

int h3_gpu_clip_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum) {
    H3GPU *gpu = GPU(opaque);
    if (!(minimum <= maximum) ||
        !h3_gpu_require_elements(gpu, input, elements, @"clip input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, @"clip output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    clip_args args = {elements, minimum, maximum};
    return h3_gpu_dispatch_1d(gpu, @"h3_clip_f32", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_vae_encoder_pad_f32(
                    h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t batch,
                    uint32_t depth, uint32_t height, uint32_t width,
                    uint32_t channels, uint32_t depth_front,
                    uint32_t height_before, uint32_t height_after,
                    uint32_t width_before, uint32_t width_after) {
    H3GPU *gpu = GPU(opaque);
    if (!batch || !depth || height < 2 || width < 2 || !channels ||
        height_before >= height || height_after >= height ||
        width_before >= width || width_after >= width) return 0;
    uint32_t output_depth = depth + depth_front;
    uint32_t output_height = height + height_before + height_after;
    uint32_t output_width = width + width_before + width_after;
    size_t input_count = (size_t)batch * depth * height * width * channels;
    size_t output_count = (size_t)batch * output_depth * output_height *
                          output_width * channels;
    if (!h3_gpu_require_elements(gpu, input, input_count,
                                 @"VAE encoder pad input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count,
                                 @"VAE encoder pad output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    vae_encoder_pad_args args = {
        batch, depth, height, width, channels, depth_front,
        height_before, height_after, width_before, width_after
    };
    return h3_gpu_dispatch_3d(gpu, @"h3_vae_encoder_pad_f32",
        MTLSizeMake(channels, output_width,
                    (NSUInteger)batch * output_depth * output_height),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_vae_encoder_group_norm_silu_f32(
                      h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t groups, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)batch * depth * height * width * channels;
    if (!batch || !depth || !height || !width || !channels || !groups ||
        channels % groups || !(epsilon > 0.0f) ||
        !h3_gpu_require_elements(gpu, input, count,
                                 @"VAE encoder norm input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, channels,
                                 @"VAE encoder norm weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, bias, channels,
                                 @"VAE encoder norm bias") ||
        TENSOR(bias).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count,
                                 @"VAE encoder norm output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    vae_encoder_norm_args args = {
        batch, depth, height, width, channels, groups, epsilon
    };
    uint64_t rows = (uint64_t)batch * depth * groups;
    if (rows > UINT32_MAX) return 0;
    return h3_gpu_dispatch_rows(
        gpu, @"h3_vae_encoder_group_norm_silu_f32", (uint32_t)rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

typedef struct {
    uint32_t batch, depth, height, width, channels, groups;
    float epsilon;
    uint32_t depth_front, height_before, height_after, width_before, width_after;
} vae_encoder_norm_pad_args;

int h3_gpu_vae_encoder_group_norm_silu_pad_f32(
                      h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t groups, float epsilon,
                      uint32_t depth_front, uint32_t height_before,
                      uint32_t height_after, uint32_t width_before,
                      uint32_t width_after) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)batch * depth * height * width * channels;
    size_t padded = (size_t)batch * (depth + depth_front) *
        (height + height_before + height_after) *
        (width + width_before + width_after) * channels;
    if (!batch || !depth || height < 2 || width < 2 || !channels || !groups ||
        channels % groups || !(epsilon > 0.0f) || height_before > 1 ||
        height_after > 1 || width_before > 1 || width_after > 1 ||
        !h3_gpu_require_elements(gpu, input, count, @"VAE encoder norm input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, channels, @"VAE encoder norm weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, bias, channels, @"VAE encoder norm bias") ||
        TENSOR(bias).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, padded, @"VAE encoder padded norm output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    /* same threadgroup width as the unfused norm, so the same reduction */
    id<MTLComputePipelineState> reference =
        h3_gpu_pipeline(gpu, @"h3_vae_encoder_group_norm_silu_f32");
    id<MTLComputePipelineState> fused =
        h3_gpu_pipeline(gpu, @"h3_vae_encoder_group_norm_silu_pad_f32");
    if (!reference || !fused) return 0;
    NSUInteger reference_threads = 1, fused_threads = 1;
    while (reference_threads * 2 <= MIN((NSUInteger)256, reference.maxTotalThreadsPerThreadgroup))
        reference_threads *= 2;
    while (fused_threads * 2 <= MIN((NSUInteger)256, fused.maxTotalThreadsPerThreadgroup))
        fused_threads *= 2;
    if (reference_threads != fused_threads) {
        h3_gpu_set_error(gpu, @"fused encoder norm threadgroup width differs");
        return 0;
    }
    vae_encoder_norm_pad_args args = {
        batch, depth, height, width, channels, groups, epsilon,
        depth_front, height_before, height_after, width_before, width_after
    };
    uint64_t rows = (uint64_t)batch * depth * groups;
    if (rows > UINT32_MAX) return 0;
    return h3_gpu_dispatch_rows(
        gpu, @"h3_vae_encoder_group_norm_silu_pad_f32", (uint32_t)rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

static int h3_gpu_require_bf16(H3GPU *gpu, const h3_gpu_tensor *tensor,
                               size_t elements, NSString *label) {
    if (!h3_gpu_require_elements(gpu, tensor, elements, label)) return 0;
    if (TENSOR(tensor).dtype != H3_GPU_BF16) {
        h3_gpu_set_error(gpu, @"%@ tensor is not BF16", label);
        return 0;
    }
    return 1;
}

static int h3_gpu_require_i8(H3GPU *gpu, const h3_gpu_tensor *tensor,
                             size_t elements, NSString *label) {
    if (!h3_gpu_require_elements(gpu, tensor, elements, label)) return 0;
    if (TENSOR(tensor).dtype != H3_GPU_I8) {
        h3_gpu_set_error(gpu, @"%@ tensor is not int8", label);
        return 0;
    }
    return 1;
}

static int h3_gpu_require_f32(H3GPU *gpu, const h3_gpu_tensor *tensor,
                              size_t elements, NSString *label) {
    if (!h3_gpu_require_elements(gpu, tensor, elements, label)) return 0;
    if (TENSOR(tensor).dtype != H3_GPU_F32) {
        h3_gpu_set_error(gpu, @"%@ tensor is not F32", label);
        return 0;
    }
    return 1;
}

static H3Linear *h3_gpu_linear_graph(H3GPU *gpu, uint32_t rows,
                                     uint32_t input_dim, uint32_t output_dim,
                                     int has_bias, MPSDataType dataType) {
    @autoreleasepool {
        NSString *key = [NSString stringWithFormat:@"%u:%u:%u:%d:%u", rows,
                         input_dim, output_dim, has_bias, (unsigned)dataType];
        H3Linear *cached = gpu.linearCache[key];
        if (cached) return cached;

        H3Linear *linear = [[H3Linear alloc] init];
        linear.graph = [[MPSGraph alloc] init];
        linear.inputShape = @[@1, @(rows), @(input_dim)];
        linear.weightShape = @[@1, @(output_dim), @(input_dim)];
        linear.biasShape = @[@1, @1, @(output_dim)];
        linear.outputShape = @[@1, @(rows), @(output_dim)];
        linear.input = [linear.graph placeholderWithShape:linear.inputShape
                                                 dataType:dataType name:nil];
        linear.weight = [linear.graph placeholderWithShape:linear.weightShape
                                                  dataType:dataType name:nil];
        MPSGraphTensor *transposed =
            [linear.graph transposeTensor:linear.weight dimension:1
                            withDimension:2 name:nil];
        MPSGraphTensor *output =
            [linear.graph matrixMultiplicationWithPrimaryTensor:linear.input
                                                secondaryTensor:transposed name:nil];
        if (has_bias) {
            linear.bias = [linear.graph placeholderWithShape:linear.biasShape
                                                    dataType:dataType name:nil];
            output = [linear.graph additionWithPrimaryTensor:output
                                             secondaryTensor:linear.bias name:nil];
        }
        linear.output = [linear.graph castTensor:output toType:dataType name:nil];
        gpu.linearCache[key] = linear;
        return linear;
    }
}

static int h3_gpu_linear_mps(H3GPU *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim,
                             MPSDataType dataType) {
    if (!h3_gpu_require_command(gpu)) return 0;
    H3Linear *linear = h3_gpu_linear_graph(gpu, rows, input_dim, output_dim,
                                           bias != NULL, dataType);
    if (!linear) return 0;
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        MPSGraphTensorData *input_data = h3_gpu_graph_data(
            input, linear.inputShape, dataType, 0);
        MPSGraphTensorData *weight_data = h3_gpu_graph_data(
            weight, linear.weightShape, dataType, 1);
        MPSGraphTensorData *output_data = h3_gpu_graph_data(
            output, linear.outputShape, dataType, 0);
        NSMutableDictionary *feeds = [@{linear.input: input_data,
                                         linear.weight: weight_data} mutableCopy];
        if (bias) {
            MPSGraphTensorData *bias_data = h3_gpu_graph_data(
                bias, linear.biasShape, dataType, 1);
            feeds[linear.bias] = bias_data;
        }
        NSDictionary *results = @{linear.output: output_data};
        @try {
            [linear.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil resultsDictionary:results
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph linear failed: %@", exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_linear_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_linear_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!h3_gpu_require_bf16(gpu, input, input_count, @"linear input") ||
        !h3_gpu_require_bf16(gpu, weight, weight_count, @"linear weight") ||
        !h3_gpu_require_bf16(gpu, output, output_count, @"linear output") ||
        (bias && !h3_gpu_require_bf16(gpu, bias, output_dim, @"linear bias"))) return 0;
    const char *splitRowsValue = getenv("H3_NAX_SPLIT_ROWS");
    BOOL autoSplitRows = gpu.tensorOpsEnabled && rows > 2048 && rows <= 3072 &&
        (gpu.tensorOpsMode == 2 || gpu.tensorOpsMode == 3 ||
         gpu.tensorOpsMode == 4);
    BOOL splitRows = rows > 2048 &&
        (autoSplitRows || (splitRowsValue && *splitRowsValue));
    BOOL specializedRows = rows <= 2048 || splitRows;
    BOOL naxShape = gpu.tensorOpsMode == 1 ||
        (specializedRows && gpu.tensorOpsMode == 2 &&
         input_dim == 7168 && output_dim == 5376) ||
        (specializedRows && gpu.tensorOpsMode == 3 &&
         ((input_dim == 7168 && output_dim == 5376) ||
          (input_dim == 5376 && output_dim == 21504))) ||
        (specializedRows && gpu.tensorOpsMode == 4 &&
         input_dim == 5376 && output_dim == 21504);
    if (gpu.tensorOpsEnabled && naxShape && !bias && rows >= 128 &&
        !getenv("H3_DISABLE_NAX_LINEAR") &&
        (input_dim % 32) == 0 && (output_dim % 64) == 0) {
        if (!h3_gpu_require_command(gpu)) return 0;
        uint32_t column_tiles = (output_dim + 63) / 64;
        BOOL morton4 = (rows <= 2048 || splitRows) &&
                       !(column_tiles % 4) &&
                       getenv("H3_DISABLE_NAX_LINEAR_MORTON4") == NULL;
        id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
            gpu, morton4 ? @"h3_linear_bf16_nax_r128_morton4" :
                           @"h3_linear_bf16_nax_r128");
        if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 128) {
            h3_gpu_set_error(gpu, @"device cannot dispatch M5 BF16 TensorOps");
            return 0;
        }
        uint32_t splitAt = splitRows ? 2048 : rows;
        if (splitRows) {
            unsigned long requested = splitRowsValue ?
                strtoul(splitRowsValue, NULL, 10) : 0;
            if (requested >= 128 && requested <= 2048)
                splitAt = (uint32_t)requested;
        }
        uint32_t dispatches = 0;
        for (uint32_t rowOffset = 0; rowOffset < rows;
             rowOffset += splitAt) @autoreleasepool {
            uint32_t chunkRows = MIN(splitAt, rows - rowOffset);
            linear_args args = {chunkRows, input_dim, output_dim, 0};
            uint32_t row_tiles = (chunkRows + 127) / 128;
            id<MTLComputeCommandEncoder> encoder =
                [gpu.command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:TENSOR(input).buffer
                         offset:(NSUInteger)rowOffset * input_dim *
                                sizeof(uint16_t) atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer
                         offset:(NSUInteger)rowOffset * output_dim *
                                sizeof(uint16_t) atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
            if (morton4) {
                [encoder dispatchThreadgroups:
                    MTLSizeMake((NSUInteger)row_tiles * column_tiles, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                [encoder dispatchThreadgroups:
                    MTLSizeMake(row_tiles, column_tiles, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            [encoder endEncoding];
            dispatches++;
        }
        h3_gpu_stats stats = gpu.stats;
        stats.direct_dispatches += dispatches;
        gpu.stats = stats;
        return 1;
    }
    if (rows >= 32 && input_dim >= 256 && output_dim >= 256 &&
        h3_gpu_linear_mps(gpu, output, input, weight, bias, rows,
                          input_dim, output_dim,
                          MPSDataTypeBFloat16)) return 1;
    linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu,
                                                           @"h3_linear_bf16");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu, @"device cannot dispatch the 16x16 BF16 linear tile");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake((output_dim + 15) / 16,
                                                  (rows + 15) / 16, 1)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static H3MLP *h3_gpu_mlp_graph(H3GPU *gpu, uint32_t rows,
                               uint32_t input_dim, uint32_t hidden_dim,
                               uint32_t output_dim) {
    @autoreleasepool {
        NSString *key = [NSString stringWithFormat:@"%u:%u:%u:%u", rows,
                         input_dim, hidden_dim, output_dim];
        H3MLP *cached = gpu.mlpCache[key];
        if (cached) return cached;

        H3MLP *mlp = [[H3MLP alloc] init];
        mlp.graph = [[MPSGraph alloc] init];
        mlp.inputShape = @[@1, @(rows), @(input_dim)];
        mlp.fc1Shape = @[@1, @(hidden_dim * 2), @(input_dim)];
        mlp.fc2Shape = @[@1, @(output_dim), @(hidden_dim)];
        mlp.outputShape = @[@1, @(rows), @(output_dim)];
        mlp.input = [mlp.graph placeholderWithShape:mlp.inputShape
                                           dataType:MPSDataTypeBFloat16 name:nil];
        mlp.fc1Weight = [mlp.graph placeholderWithShape:mlp.fc1Shape
                                               dataType:MPSDataTypeBFloat16 name:nil];
        mlp.fc2Weight = [mlp.graph placeholderWithShape:mlp.fc2Shape
                                               dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *fc1Transposed =
            [mlp.graph transposeTensor:mlp.fc1Weight dimension:1
                         withDimension:2 name:nil];
        MPSGraphTensor *fused =
            [mlp.graph matrixMultiplicationWithPrimaryTensor:mlp.input
                                             secondaryTensor:fc1Transposed name:nil];
        NSArray<MPSGraphTensor *> *halves =
            [mlp.graph splitTensor:fused numSplits:2 axis:2 name:nil];
        MPSGraphTensor *sigmoid = [mlp.graph sigmoidWithTensor:halves[0]
                                                              name:nil];
        MPSGraphTensor *silu =
            [mlp.graph multiplicationWithPrimaryTensor:halves[0]
                                       secondaryTensor:sigmoid name:nil];
        MPSGraphTensor *activated =
            [mlp.graph multiplicationWithPrimaryTensor:silu
                                       secondaryTensor:halves[1] name:nil];
        MPSGraphTensor *fc2Transposed =
            [mlp.graph transposeTensor:mlp.fc2Weight dimension:1
                         withDimension:2 name:nil];
        MPSGraphTensor *result =
            [mlp.graph matrixMultiplicationWithPrimaryTensor:activated
                                             secondaryTensor:fc2Transposed name:nil];
        mlp.output = [mlp.graph castTensor:result
                                    toType:MPSDataTypeBFloat16 name:nil];
        gpu.mlpCache[key] = mlp;
        return mlp;
    }
}

int h3_gpu_mlp_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input,
                    const h3_gpu_tensor *fc1_weight,
                    const h3_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, input, (size_t)rows * input_dim,
                             @"MLP input") ||
        !h3_gpu_require_bf16(gpu, fc1_weight,
                             (size_t)hidden_dim * 2 * input_dim,
                             @"MLP fc1 weight") ||
        !h3_gpu_require_bf16(gpu, fc2_weight,
                             (size_t)output_dim * hidden_dim,
                             @"MLP fc2 weight") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * output_dim,
                             @"MLP output") ||
        !h3_gpu_require_command(gpu)) return 0;
    H3MLP *mlp = h3_gpu_mlp_graph(gpu, rows, input_dim, hidden_dim,
                                  output_dim);
    if (!mlp) return 0;
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        MPSGraphTensorData *inputData = h3_gpu_graph_data(
            input, mlp.inputShape, MPSDataTypeBFloat16, 0);
        MPSGraphTensorData *fc1Data = h3_gpu_graph_data(
            fc1_weight, mlp.fc1Shape, MPSDataTypeBFloat16, 1);
        MPSGraphTensorData *fc2Data = h3_gpu_graph_data(
            fc2_weight, mlp.fc2Shape, MPSDataTypeBFloat16, 1);
        MPSGraphTensorData *outputData = h3_gpu_graph_data(
            output, mlp.outputShape, MPSDataTypeBFloat16, 0);
        NSDictionary *feeds = @{mlp.input: inputData,
                                mlp.fc1Weight: fc1Data,
                                mlp.fc2Weight: fc2Data};
        NSDictionary *results = @{mlp.output: outputData};
        @try {
            [mlp.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil resultsDictionary:results
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph MLP failed: %@", exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_linear_dispatches += 2;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_fc1_swiglu_nax_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                                      const h3_gpu_tensor *input,
                                      const h3_gpu_tensor *weight,
                                      uint32_t rows, uint32_t input_dim,
                                      uint32_t hidden_dim) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu.tensorOpsEnabled ||
        (gpu.tensorOpsMode != 5 && !h3_gpu_has_int8_mlp(opaque)) ||
        !h3_gpu_require_bf16(gpu, input, (size_t)rows * input_dim,
                             @"NAX FC1 input") ||
        !h3_gpu_require_bf16(gpu, weight,
                             (size_t)hidden_dim * 2 * input_dim,
                             @"NAX FC1 weight") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * hidden_dim,
                             @"NAX FC1 output") ||
        !rows || (input_dim % 32) || (hidden_dim % 64) ||
        !h3_gpu_require_command(gpu)) return 0;
    BOOL morton = getenv("H3_DISABLE_NAX_MORTON") == NULL &&
                  getenv("H3_DISABLE_NAX_MORTON_FC1") == NULL;
    uint32_t row_tiles = (rows + 127) / 128;
    uint32_t column_tiles = (hidden_dim + 63) / 64;
    BOOL morton4 = morton && !(column_tiles % 4) &&
                   getenv("H3_DISABLE_NAX_MORTON4") == NULL &&
                   getenv("H3_DISABLE_NAX_MORTON4_FC1") == NULL;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, morton4 ? @"h3_fc1_swiglu_bf16_nax_r128_morton4" :
        morton ? @"h3_fc1_swiglu_bf16_nax_r128_morton" :
                 @"h3_fc1_swiglu_bf16_nax_r128");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 128) {
        h3_gpu_set_error(gpu, @"device cannot dispatch fused M5 FC1/SwiGLU");
        return 0;
    }
    linear_args args = {rows, input_dim, hidden_dim, 0};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        [encoder setThreadgroupMemoryLength:128 * 64 * 2 * sizeof(uint16_t)
                                   atIndex:0];
        if (morton4) {
            [encoder dispatchThreadgroups:
                MTLSizeMake((NSUInteger)row_tiles * column_tiles, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (morton) {
            uint32_t row_codes = 1, column_codes = 1;
            while (row_codes < row_tiles) row_codes <<= 1;
            while (column_codes < column_tiles) column_codes <<= 1;
            [encoder dispatchThreadgroups:
                MTLSizeMake((NSUInteger)row_codes * column_codes, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [encoder dispatchThreadgroups:
                MTLSizeMake(row_tiles, column_tiles, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_mlp_nax_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                        h3_gpu_tensor *activated,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *fc1_weight,
                        const h3_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu.tensorOpsEnabled || gpu.tensorOpsMode != 5 || rows < 128 ||
        (hidden_dim % 32) || (output_dim % 64) ||
        !h3_gpu_require_bf16(gpu, fc2_weight,
                             (size_t)output_dim * hidden_dim,
                             @"NAX FC2 weight") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * output_dim,
                             @"NAX FC2 output") ||
        !h3_gpu_fc1_swiglu_nax_bf16(opaque, activated, input, fc1_weight,
                                     rows, input_dim, hidden_dim)) return 0;
    BOOL morton = getenv("H3_DISABLE_NAX_MORTON") == NULL &&
                  getenv("H3_DISABLE_NAX_MORTON_FC2") == NULL;
    uint32_t row_tiles = (rows + 127) / 128;
    uint32_t column_tiles = (output_dim + 63) / 64;
    BOOL morton4 = morton && !(column_tiles % 4) &&
                   getenv("H3_DISABLE_NAX_MORTON4") == NULL &&
                   getenv("H3_DISABLE_NAX_MORTON4_FC2") == NULL;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, morton4 ? @"h3_linear_bf16_nax_r128_morton4" :
        morton ? @"h3_linear_bf16_nax_r128_morton" :
                 @"h3_linear_bf16_nax_r128");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 128) {
        h3_gpu_set_error(gpu, @"device cannot dispatch M5 BF16 TensorOps FC2");
        return 0;
    }
    linear_args args = {rows, hidden_dim, output_dim, 0};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(activated).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(fc2_weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        if (morton4) {
            [encoder dispatchThreadgroups:
                MTLSizeMake((NSUInteger)row_tiles * column_tiles, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (morton) {
            uint32_t row_codes = 1, column_codes = 1;
            while (row_codes < row_tiles) row_codes <<= 1;
            while (column_codes < column_tiles) column_codes <<= 1;
            [encoder dispatchThreadgroups:
                MTLSizeMake((NSUInteger)row_codes * column_codes, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [encoder dispatchThreadgroups:
                MTLSizeMake(row_tiles, column_tiles, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_quantize_bf16_int8_rows(
                        h3_gpu *opaque, h3_gpu_tensor *output,
                        h3_gpu_tensor *scales,
                        const h3_gpu_tensor *input, uint32_t rows,
                        uint32_t dispatch_rows, uint32_t columns,
                        float clip, NSString *label, BOOL rotate) {
    H3GPU *gpu = GPU(opaque);
    if (rotate && (columns % 256u || columns > 14336u)) {
        h3_gpu_set_error(gpu, @"rotated int8 quantizer needs K %% 256 == 0 "
                         "and K <= 14336");
        return 0;
    }
    if (!gpu.tensorOpsEnabled || !rows || dispatch_rows < rows || !columns ||
        !h3_gpu_require_bf16(gpu, input, (size_t)rows * columns, label) ||
        !h3_gpu_require_i8(gpu, output,
                           (size_t)dispatch_rows * columns,
                           @"int8 quantized output") ||
        !h3_gpu_require_f32(gpu, scales, dispatch_rows,
                            @"int8 quantization scales") ||
        !h3_gpu_require_command(gpu)) return 0;
    BOOL vector_quantizer =
        [label isEqualToString:@"int8 MLP FC2 input"] ||
        getenv("H3_INT8_VECTOR_QUANT");
    NSString *kernel = rotate ? @"h3_quantize_bf16_int8_rows_rot256" :
        vector_quantizer ?
        @"h3_quantize_bf16_int8_rows" :
        @"h3_quantize_bf16_int8_rows_scalar";
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, kernel);
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu, @"device cannot dispatch M5 int8 quantizer");
        return 0;
    }
    int8_quant_args args = {rows, columns, clip};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(scales).buffer offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(dispatch_rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_quantize_bf16_int8_head_major_rows(
                        h3_gpu *opaque, h3_gpu_tensor *output,
                        h3_gpu_tensor *scales,
                        const h3_gpu_tensor *input, uint32_t rows,
                        uint32_t padded_rows, uint32_t heads,
                        uint32_t head_dim, BOOL rotate) {
    H3GPU *gpu = GPU(opaque);
    size_t columns = (size_t)heads * head_dim;
    if (!gpu.tensorOpsEnabled || !rows || padded_rows < rows ||
        heads != 56 || head_dim != 128 ||
        !h3_gpu_require_bf16(gpu, input, (size_t)rows * columns,
                             @"head-major int8 linear input") ||
        !h3_gpu_require_i8(gpu, output, (size_t)padded_rows * columns,
                           @"head-major int8 quantized output") ||
        !h3_gpu_require_f32(gpu, scales, padded_rows,
                            @"head-major int8 quantization scales") ||
        !h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, rotate ? @"h3_quantize_bf16_int8_head_major_to_rows_rot256" :
                      @"h3_quantize_bf16_int8_head_major_to_rows_cached");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(
            gpu, @"device cannot dispatch M5 head-major int8 quantizer");
        return 0;
    }
    int8_head_major_quant_args args = {
        rows, padded_rows, heads, head_dim, 1.0f
    };
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(scales).buffer offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(padded_rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_quantize_bf16_int8_groups(
                        h3_gpu *opaque, h3_gpu_tensor *output,
                        h3_gpu_tensor *scales,
                        const h3_gpu_tensor *input, uint32_t rows,
                        uint32_t dispatch_rows, uint32_t columns,
                        uint32_t group_size,
                        int use_slower_grouped_quantizer, BOOL rotate) {
    H3GPU *gpu = GPU(opaque);
    if (!group_size || (group_size % 4)) return 0;
    if (rotate && group_size != 1024u) {
        h3_gpu_set_error(gpu, @"rotated grouped int8 quantizer needs "
                         "1024-wide groups");
        return 0;
    }
    uint32_t groups = (columns + group_size - 1) / group_size;
    if (!gpu.tensorOpsEnabled || !rows || dispatch_rows < rows || !columns ||
        (columns % group_size) ||
        !h3_gpu_require_bf16(gpu, input, (size_t)rows * columns,
                             @"grouped int8 input") ||
        !h3_gpu_require_i8(gpu, output,
                           (size_t)dispatch_rows * columns,
                           @"grouped int8 output") ||
        !h3_gpu_require_f32(gpu, scales,
                            (size_t)dispatch_rows * groups,
                            @"grouped int8 scales") ||
        !h3_gpu_require_command(gpu)) return 0;
    BOOL scalar128 = !use_slower_grouped_quantizer && rows <= 2048;
    const char *quantizer_override = getenv("H3_INT8_GROUP_QUANT_128");
    if (quantizer_override)
        scalar128 = *quantizer_override && strcmp(quantizer_override, "0");
    BOOL cached128 = scalar128 && group_size == 1024;
    const char *cache_override = getenv("H3_INT8_GROUP_QUANT_128_CACHE");
    if (cache_override)
        cached128 = cached128 && *cache_override &&
            strcmp(cache_override, "0");
    if (rotate) scalar128 = NO;
    NSString *kernel = rotate ? @"h3_quantize_bf16_int8_groups1024_rot256" :
        getenv("H3_INT8_VECTOR_QUANT") ?
        @"h3_quantize_bf16_int8_groups" : cached128 ?
        @"h3_quantize_bf16_int8_groups_scalar128_cached" : scalar128 ?
        @"h3_quantize_bf16_int8_groups_scalar128" :
        @"h3_quantize_bf16_int8_groups_scalar";
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, kernel);
    NSUInteger threads = scalar128 ? 128u : 256u;
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < threads) {
        h3_gpu_set_error(gpu,
                         @"device cannot dispatch grouped M5 int8 quantizer");
        return 0;
    }
    int8_group_quant_args args = {
        rows, columns, group_size, groups
    };
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(scales).buffer offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(dispatch_rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_quantize_weight_int8(h3_gpu *opaque, h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t columns) {
    return h3_gpu_quantize_bf16_int8_rows(
        opaque, output, scales, input, rows, rows, columns,
        1.0f, @"BF16 weight to quantize", NO);
}

static int h3_gpu_linear_int8_bf16_layout(
                            h3_gpu *opaque, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales,
                            BOOL headMajorInput, uint32_t heads,
                            uint32_t headDim, BOOL rotate) {
    H3GPU *gpu = GPU(opaque);
    uint32_t padded_rows = (rows + 127u) & ~127u;
    /* Fewer than 128 rows (Qwen text prompts) work too: the quantizer
     * zero-fills the padded rows and the GEMM stores only rows < args.rows. */
    if (!gpu.tensorOpsEnabled || !rows || input_dim % 128 ||
        output_dim % 128 ||
        !h3_gpu_require_i8(gpu, weight, (size_t)output_dim * input_dim,
                           @"int8 linear weight") ||
        !h3_gpu_require_f32(gpu, weight_scales, output_dim,
                            @"int8 linear weight scales") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * output_dim,
                             @"int8 linear output") ||
        !h3_gpu_require_command(gpu)) return 0;
    if (headMajorInput) {
        if (use_slower_uncached_int8_scales || heads * headDim != input_dim ||
            !h3_gpu_quantize_bf16_int8_head_major_rows(
                opaque, quantized_input, input_scales, input, rows,
                padded_rows, heads, headDim, rotate)) return 0;
    } else if (!h3_gpu_quantize_bf16_int8_rows(
                   opaque, quantized_input, input_scales, input, rows,
                   padded_rows, input_dim, 1.0f,
                   @"int8 linear input", rotate)) return 0;
    BOOL local_scales = !use_slower_uncached_int8_scales &&
        getenv("H3_DISABLE_INT8_LOCAL_SCALES") == NULL;
    BOOL known_linear = local_scales && rows <= 2048 && input_dim == 7168 &&
        output_dim == 5376 &&
        getenv("H3_DISABLE_INT8_LINEAR_KNOWN") == NULL;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, known_linear ? @"h3_linear_int8_local_scales_nax_r128_k7168" :
             local_scales ? @"h3_linear_int8_local_scales_nax_r128" :
                            @"h3_linear_int8_nax_r128");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu, @"int8 M5 linear projection is unavailable");
        return 0;
    }
    linear_args args = {rows, input_dim, output_dim, 0};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(quantized_input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(input_scales).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(weight_scales).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
        NSUInteger groups = (NSUInteger)(padded_rows / 128u) *
                            (output_dim / 128u);
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_linear_int8_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales) {
    return h3_gpu_linear_int8_bf16_layout(
        opaque, output, quantized_input, input_scales, input, weight,
        weight_scales, rows, input_dim, output_dim,
        use_slower_uncached_int8_scales, NO, 0, 0, NO);
}

int h3_gpu_linear_int8_bf16_rotated(h3_gpu *opaque, h3_gpu_tensor *output,
                                    h3_gpu_tensor *quantized_input,
                                    h3_gpu_tensor *input_scales,
                                    const h3_gpu_tensor *input,
                                    const h3_gpu_tensor *weight,
                                    const h3_gpu_tensor *weight_scales,
                                    uint32_t rows, uint32_t input_dim,
                                    uint32_t output_dim,
                                    int use_slower_uncached_int8_scales,
                                    int rotate_input) {
    return h3_gpu_linear_int8_bf16_layout(
        opaque, output, quantized_input, input_scales, input, weight,
        weight_scales, rows, input_dim, output_dim,
        use_slower_uncached_int8_scales, NO, 0, 0, rotate_input != 0);
}

int h3_gpu_linear_int8_head_major_bf16(
                            h3_gpu *opaque, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim, uint32_t output_dim,
                            int rotate_input) {
    return h3_gpu_linear_int8_bf16_layout(
        opaque, output, quantized_input, input_scales, input, weight,
        weight_scales, rows, heads * head_dim, output_dim, 0, YES, heads,
        head_dim, rotate_input != 0);
}

int h3_gpu_mlp_int8_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                         h3_gpu_tensor *activated,
                         h3_gpu_tensor *quantized_activation,
                         h3_gpu_tensor *activation_scales,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *fc1_weight,
                         const h3_gpu_tensor *fc1_scales,
                         const h3_gpu_tensor *fc2_weight,
                         const h3_gpu_tensor *fc2_scales,
                         const h3_gpu_tensor *fc1_bf16,
                         const h3_gpu_tensor *fc2_bf16, uint32_t rows,
                         uint32_t input_dim, uint32_t hidden_dim,
                         uint32_t output_dim,
                         int use_slower_grouped_quantizer,
                         int use_slower_dynamic_fc1_k,
                         int use_int8_row_fc2,
                         int input_is_quantized,
                         unsigned rotate) {
    H3GPU *gpu = GPU(opaque);
    uint32_t padded_rows = (rows + 127u) & ~127u;
    BOOL rotate_fc1 = (rotate & H3_GPU_ROTATE_FC1_INPUT) != 0;
    BOOL rotate_fc2 = (rotate & H3_GPU_ROTATE_FC2_INPUT) != 0;
    uint32_t fc2_scale_groups = hidden_dim / 1024u;
    size_t activation_capacity = (size_t)padded_rows *
        MAX(input_dim, hidden_dim);
    const char *stage = getenv("H3_INT8_MLP_STAGE");
    BOOL int8_fc1 = !stage || (strcmp(stage, "fc2") &&
                               strcmp(stage, "bf16"));
    BOOL int8_fc2 = !stage || (strcmp(stage, "fc1") &&
                               strcmp(stage, "bf16"));
    if (!gpu.tensorOpsEnabled || rows < 128 || rows > UINT32_MAX - 127u ||
        (input_dim % 128) || (hidden_dim % 128) || (output_dim % 128) ||
        !h3_gpu_require_i8(gpu, quantized_activation, activation_capacity,
                           @"int8 MLP activation") ||
        !h3_gpu_require_f32(gpu, activation_scales,
                            (size_t)padded_rows * MAX(fc2_scale_groups, 1u),
                            @"int8 MLP activation scales") ||
        !h3_gpu_require_i8(gpu, fc1_weight,
                           (size_t)hidden_dim * 2 * input_dim,
                           @"int8 MLP FC1 weight") ||
        !h3_gpu_require_f32(gpu, fc1_scales, (size_t)hidden_dim * 2,
                            @"int8 MLP FC1 scales") ||
        !h3_gpu_require_i8(gpu, fc2_weight,
                           (size_t)output_dim * hidden_dim,
                           @"int8 MLP FC2 weight") ||
        !h3_gpu_require_f32(gpu, fc2_scales, output_dim,
                            @"int8 MLP FC2 scales") ||
        (!int8_fc1 && !h3_gpu_require_bf16(
            gpu, fc1_bf16, (size_t)hidden_dim * 2 * input_dim,
            @"diagnostic BF16 MLP FC1 weight")) ||
        (!int8_fc2 && !h3_gpu_require_bf16(
            gpu, fc2_bf16, (size_t)output_dim * hidden_dim,
            @"diagnostic BF16 MLP FC2 weight")) ||
        (!input_is_quantized &&
         !h3_gpu_require_bf16(gpu, input, (size_t)rows * input_dim,
                              @"int8 MLP input")) ||
        !h3_gpu_require_bf16(gpu, activated,
                             (size_t)rows * hidden_dim,
                             @"int8 MLP activated") ||
        !h3_gpu_require_bf16(gpu, output,
                             (size_t)rows * output_dim,
                             @"int8 MLP output") ||
        !h3_gpu_require_command(gpu)) return 0;

    id<MTLComputePipelineState> fc1 = h3_gpu_pipeline(
        gpu, @"h3_fc1_swiglu_int8_nax_r128");
    id<MTLComputePipelineState> fc1_known = h3_gpu_pipeline(
        gpu, @"h3_fc1_swiglu_int8_nax_r128_k5376");
    id<MTLComputePipelineState> fc1_full = h3_gpu_pipeline(
        gpu, @"h3_fc1_swiglu_int8_nax_r128_full_k5376");
    id<MTLComputePipelineState> fc1_local = h3_gpu_pipeline(
        gpu, @"h3_fc1_swiglu_int8_local_nax_r128");
    id<MTLComputePipelineState> fc2 = h3_gpu_pipeline(
        gpu, @"h3_linear_int8_nax_r128");
    id<MTLComputePipelineState> fc2_full = h3_gpu_pipeline(
        gpu, @"h3_linear_int8_nax_r128_full_k14336");
    id<MTLComputePipelineState> fc2_full_n256 = h3_gpu_pipeline(
        gpu, @"h3_linear_int8_nax_r128x256_full_k14336");
    id<MTLComputePipelineState> fc2_grouped = h3_gpu_pipeline(
        gpu, @"h3_linear_int8_grouped_nax_r128x64");
    id<MTLComputePipelineState> fc2_grouped_local = h3_gpu_pipeline(
        gpu, @"h3_linear_int8_grouped_local_nax_r128x64");
    id<MTLComputePipelineState> fc2_grouped_local128 = h3_gpu_pipeline(
        gpu, @"h3_linear_int8_grouped_local_nax_r128x128");
    if (!fc1 || !fc1_known || !fc1_full || !fc1_local || !fc2 ||
        !fc2_full || !fc2_full_n256 ||
        fc1.maxTotalThreadsPerThreadgroup < 256 ||
        fc1_known.maxTotalThreadsPerThreadgroup < 256 ||
        fc1_full.maxTotalThreadsPerThreadgroup < 256 ||
        fc1_local.maxTotalThreadsPerThreadgroup < 256 ||
        fc2.maxTotalThreadsPerThreadgroup < 256 ||
        fc2_full.maxTotalThreadsPerThreadgroup < 256 ||
        fc2_full_n256.maxTotalThreadsPerThreadgroup < 512 || !fc2_grouped ||
        fc2_grouped.maxTotalThreadsPerThreadgroup < 128 ||
        !fc2_grouped_local ||
        fc2_grouped_local.maxTotalThreadsPerThreadgroup < 256 ||
        !fc2_grouped_local128 ||
        fc2_grouped_local128.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu, @"device cannot dispatch M5 int8 MLP");
        return 0;
    }
    float activation_clip = 1.0f;
    const char *clip_value = getenv("H3_INT8_ACTIVATION_CLIP");
    if (clip_value) {
        float parsed = strtof(clip_value, NULL);
        if (parsed >= 0.1f && parsed <= 1.0f) activation_clip = parsed;
    }
    BOOL grouped_fc2 = int8_fc2 && !use_int8_row_fc2;
    BOOL row_fc2_n256 = use_int8_row_fc2 && rows <= 2048;
    BOOL grouped_fc2_local = grouped_fc2 &&
        getenv("H3_INT8_GROUP_FC2_THREADGROUP") == NULL;
    BOOL grouped_fc2_local128 = grouped_fc2 &&
        getenv("H3_INT8_GROUP_FC2_LOCAL128") != NULL;
    BOOL int8_fc1_local = int8_fc1 &&
        getenv("H3_INT8_FC1_LOCAL") != NULL;
    BOOL int8_fc1_known = int8_fc1 && input_dim == 5376 &&
        !use_slower_dynamic_fc1_k;
    const char *known_override = getenv("H3_INT8_FC1_KNOWN");
    if (known_override)
        int8_fc1_known = int8_fc1 && input_dim == 5376 &&
            strcmp(known_override, "0") != 0;
    BOOL int8_fc1_full = int8_fc1_known &&
        getenv("H3_DISABLE_FC1_FULL_K") == NULL;
    uint32_t row_tiles = padded_rows / 128;
    if (input_is_quantized && !int8_fc1) {
        h3_gpu_set_error(gpu,
            @"prequantized MLP input requires the int8 FC1 path");
        return 0;
    }
    if (int8_fc1 && !input_is_quantized &&
        !h3_gpu_quantize_bf16_int8_rows(
            opaque, quantized_activation, activation_scales, input, rows,
            padded_rows, input_dim, activation_clip,
            @"int8 MLP input", rotate_fc1)) return 0;
    if (int8_fc1) @autoreleasepool {
        linear_args fc1_args = {rows, input_dim, hidden_dim, 0};
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:int8_fc1_local ? fc1_local :
            int8_fc1_full ? fc1_full :
            int8_fc1_known ? fc1_known : fc1];
        [encoder setBuffer:TENSOR(quantized_activation).buffer
                         offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(fc1_weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(activation_scales).buffer
                         offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(fc1_scales).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(activated).buffer offset:0 atIndex:4];
        [encoder setBytes:&fc1_args length:sizeof(fc1_args) atIndex:5];
        [encoder dispatchThreadgroups:
            MTLSizeMake((NSUInteger)row_tiles * (hidden_dim / 128), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    } else if (!h3_gpu_fc1_swiglu_nax_bf16(
                   opaque, activated, input, fc1_bf16,
                   rows, input_dim, hidden_dim)) {
        return 0;
    }
    if (int8_fc1) {
        h3_gpu_stats stats = gpu.stats;
        stats.direct_dispatches++;
        gpu.stats = stats;
    }

    if (int8_fc2 && grouped_fc2 &&
        !h3_gpu_quantize_bf16_int8_groups(
            opaque, quantized_activation, activation_scales, activated, rows,
            padded_rows, hidden_dim, 1024,
            use_slower_grouped_quantizer, rotate_fc2)) return 0;
    if (int8_fc2 && !grouped_fc2 &&
        !h3_gpu_quantize_bf16_int8_rows(
            opaque, quantized_activation, activation_scales, activated, rows,
            padded_rows, hidden_dim, activation_clip,
            @"int8 MLP FC2 input", rotate_fc2)) return 0;
    if (int8_fc2) @autoreleasepool {
        linear_args fc2_args = {rows, hidden_dim, output_dim, 0};
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:grouped_fc2_local128 ?
            fc2_grouped_local128 : grouped_fc2_local ?
            fc2_grouped_local : grouped_fc2 ? fc2_grouped :
            row_fc2_n256 ? fc2_full_n256 :
            hidden_dim == 14336u && output_dim == 5376u ? fc2_full : fc2];
        [encoder setBuffer:TENSOR(quantized_activation).buffer
                         offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(fc2_weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(activation_scales).buffer
                         offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(fc2_scales).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
        [encoder setBytes:&fc2_args length:sizeof(fc2_args) atIndex:5];
        [encoder dispatchThreadgroups:
            MTLSizeMake((NSUInteger)row_tiles *
                        (output_dim /
                         (row_fc2_n256 ? 256u :
                          grouped_fc2 && !grouped_fc2_local128 ?
                          64u : 128u)), 1, 1)
                 threadsPerThreadgroup:
                    MTLSizeMake(row_fc2_n256 ? 512u :
                        grouped_fc2_local ? 256u :
                        grouped_fc2 ? 128u : 256u, 1, 1)];
        [encoder endEncoding];
    } else if (!h3_gpu_linear_bf16(
                   opaque, output, activated, fc2_bf16, NULL,
                   rows, hidden_dim, output_dim)) {
        return 0;
    }
    if (int8_fc2) {
        h3_gpu_stats stats = gpu.stats;
        stats.direct_dispatches++;
        gpu.stats = stats;
    }
    return 1;
}

int h3_gpu_rotate256_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input, uint32_t rows,
                          uint32_t columns) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * columns;
    if (!rows || !columns || (columns % 256u) ||
        !h3_gpu_require_bf16(gpu, input, count, @"rotation input") ||
        !h3_gpu_require_bf16(gpu, output, count, @"rotation output") ||
        !h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline =
        h3_gpu_pipeline(gpu, @"h3_rotate256_bf16");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu, @"device cannot dispatch the 256-group rotation");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
        [encoder setBytes:&columns length:sizeof(columns) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(columns / 256u, rows, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_dequantize_int8_rows_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                                     const h3_gpu_tensor *weight,
                                     const h3_gpu_tensor *scales,
                                     uint32_t rows, uint32_t columns) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * columns;
    if (!rows || (columns % 4u) || count / 4u > UINT32_MAX ||
        !h3_gpu_require_i8(gpu, weight, count, @"int8 weight") ||
        !h3_gpu_require_f32(gpu, scales, rows, @"int8 weight scales") ||
        !h3_gpu_require_bf16(gpu, output, count, @"dequantized weight"))
        return 0;
    uint64_t shape = (uint64_t)rows | ((uint64_t)columns << 32);
    return h3_gpu_dispatch_1d(gpu, @"h3_dequantize_int8_rows_bf16",
                              (uint32_t)(count / 4u),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(scales).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&shape length:sizeof(shape) atIndex:3];
        });
}

int h3_gpu_silu_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, input, elements, @"SiLU input") ||
        !h3_gpu_require_bf16(gpu, output, elements, @"SiLU output")) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_silu_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        });
}

int h3_gpu_rms_norm_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, input, count, @"RMSNorm input") ||
        !h3_gpu_require_bf16(gpu, weight, width, @"RMSNorm weight") ||
        !h3_gpu_require_bf16(gpu, output, count, @"RMSNorm output")) return 0;
    norm_args args = {rows, width, epsilon};
    return h3_gpu_dispatch_rows(gpu, @"h3_rms_norm_bf16", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_layer_norm_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, input, count, @"LayerNorm input") ||
        !h3_gpu_require_bf16(gpu, weight, width, @"LayerNorm weight") ||
        !h3_gpu_require_bf16(gpu, bias, width, @"LayerNorm bias") ||
        !h3_gpu_require_bf16(gpu, output, count, @"LayerNorm output")) return 0;
    norm_args args = {rows, width, epsilon};
    return h3_gpu_dispatch_rows(gpu, @"h3_layer_norm_bf16", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

int h3_gpu_gelu_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, input, elements, @"GELU input") ||
        !h3_gpu_require_bf16(gpu, output, elements, @"GELU output")) return 0;
    gelu_bf16_args args = {elements, approximate ? 1u : 0u};
    return h3_gpu_dispatch_1d(gpu, @"h3_gelu_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_vision_qkv_rope_bf16(
                     h3_gpu *opaque, h3_gpu_tensor *query,
                     h3_gpu_tensor *key, h3_gpu_tensor *value,
                     const h3_gpu_tensor *qkv,
                     const h3_gpu_tensor *rope_cos,
                     const h3_gpu_tensor *rope_sin, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim,
                     uint32_t rope_half) {
    H3GPU *gpu = GPU(opaque);
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_gpu_require_bf16(gpu, qkv, count * 3, @"vision QKV") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count,
                              @"vision RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count,
                              @"vision RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query, count, @"vision query") ||
        !h3_gpu_require_bf16(gpu, key, count, @"vision key") ||
        !h3_gpu_require_bf16(gpu, value, count, @"vision value") ||
        rope_half * 2 != head_dim) return 0;
    qkv_args args = {sequence, heads, head_dim, rope_half, 0, 0.0f};
    return h3_gpu_dispatch_3d(gpu, @"h3_vision_qkv_rope_bf16",
        MTLSizeMake(head_dim, heads, sequence),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
        });
}

int h3_gpu_adaln_bf16_offset(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (input_offset > SIZE_MAX - count ||
        input_offset > SIZE_MAX / sizeof(uint16_t)) {
        h3_gpu_set_error(gpu, @"AdaLN input offset is out of range");
        return 0;
    }
    if (!h3_gpu_require_bf16(gpu, input, input_offset + count,
                             @"AdaLN input") ||
        !h3_gpu_require_bf16(gpu, norm_weight, width, @"AdaLN norm") ||
        !h3_gpu_require_bf16(gpu, modulation, 1, @"AdaLN modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows, @"AdaLN row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, count, @"AdaLN output") ||
        shift_slot >= slots || scale_slot >= slots) return 0;
    adaln_args args = {rows, width, slots, shift_slot, scale_slot, epsilon};
    return h3_gpu_dispatch_rows(gpu, @"h3_adaln_bf16", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer
                         offset:input_offset * sizeof(uint16_t) atIndex:0];
            [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
            [encoder setBytes:&args length:sizeof(args) atIndex:5];
        });
}

int h3_gpu_adaln_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    return h3_gpu_adaln_bf16_offset(
        opaque, output, input, 0, norm_weight, modulation, row_map, rows,
        width, slots, shift_slot, scale_slot, epsilon);
}

int h3_gpu_adaln_linear_bf16(
                      h3_gpu *opaque, h3_gpu_tensor *output,
                      h3_gpu_tensor *inverse,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t width, uint32_t output_dim, uint32_t slots,
                      uint32_t shift_slot, uint32_t scale_slot,
                      float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t input_count = (size_t)rows * width;
    size_t weight_count = (size_t)output_dim * width;
    size_t output_count = (size_t)rows * output_dim;
    if (input_offset > SIZE_MAX - input_count ||
        input_offset > SIZE_MAX / sizeof(uint16_t)) {
        h3_gpu_set_error(gpu, @"fused final head input offset is out of range");
        return 0;
    }
    if (!h3_gpu_require_bf16(gpu, input, input_offset + input_count,
                             @"fused final head input") ||
        !h3_gpu_require_elements(gpu, inverse, rows,
                                 @"fused final head inverse RMS") ||
        TENSOR(inverse).dtype != H3_GPU_F32 ||
        !h3_gpu_require_bf16(gpu, norm_weight, width,
                             @"fused final head norm") ||
        !h3_gpu_require_bf16(gpu, modulation, 1,
                             @"fused final head modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows,
                                 @"fused final head row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, weight, weight_count,
                             @"fused final head weight") ||
        !h3_gpu_require_bf16(gpu, output, output_count,
                             @"fused final head output") ||
        (bias && !h3_gpu_require_bf16(gpu, bias, output_dim,
                                      @"fused final head bias")) ||
        shift_slot >= slots || scale_slot >= slots) return 0;
    norm_args rms_args = {rows, width, epsilon};
    if (!h3_gpu_dispatch_rows(
            gpu, @"h3_rms_inverse_bf16", rows,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:TENSOR(input).buffer
                             offset:input_offset * sizeof(uint16_t) atIndex:0];
                [encoder setBuffer:TENSOR(inverse).buffer offset:0 atIndex:1];
                [encoder setBytes:&rms_args length:sizeof(rms_args) atIndex:2];
            })) return 0;
    typedef struct {
        uint32_t rows, width, output_dim, slots, shift_slot, scale_slot;
        uint32_t has_bias;
    } adaln_linear_args;
    adaln_linear_args args = {
        rows, width, output_dim, slots, shift_slot, scale_slot,
        bias ? 1u : 0u
    };
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, @"h3_adaln_linear_bf16");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu,
                         @"device cannot dispatch fused final BF16 head");
        return 0;
    }
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer
                     offset:input_offset * sizeof(uint16_t) atIndex:0];
        [encoder setBuffer:TENSOR(inverse).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:6];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:7];
        [encoder setBytes:&args length:sizeof(args) atIndex:8];
        [encoder dispatchThreadgroups:
            MTLSizeMake((output_dim + 15) / 16, (rows + 15) / 16, 1)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_gate_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, residual, count, @"gate residual") ||
        !h3_gpu_require_bf16(gpu, branch, count, @"gate branch") ||
        !h3_gpu_require_bf16(gpu, modulation, 1, @"gate modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows, @"gate row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, count, @"gate output") ||
        gate_slot >= slots) return 0;
    gate_args args = {rows, width, slots, gate_slot};
    return h3_gpu_dispatch_2d(gpu, @"h3_gate_bf16", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(branch).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
            [encoder setBytes:&args length:sizeof(args) atIndex:5];
        });
}

int h3_gpu_gate_adaln_bf16(
                     h3_gpu *opaque, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot,
                     uint32_t shift_slot, uint32_t scale_slot,
                     float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t elements = (size_t)rows * width;
    if (!rows || !width || width > 5376 || gate_slot >= slots ||
        shift_slot >= slots || scale_slot >= slots ||
        elements > UINT32_MAX ||
        !h3_gpu_require_bf16(gpu, residual, elements,
                             @"fused gate AdaLN residual") ||
        !h3_gpu_require_bf16(gpu, branch, elements,
                             @"fused gate AdaLN branch") ||
        !h3_gpu_require_bf16(gpu, norm_weight, width,
                             @"fused gate AdaLN norm") ||
        !h3_gpu_require_bf16(gpu, gate_modulation, 1,
                             @"fused gate AdaLN gate modulation") ||
        !h3_gpu_require_bf16(gpu, norm_modulation, 1,
                             @"fused gate AdaLN norm modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows,
                                 @"fused gate AdaLN row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, gated_residual, elements,
                             @"fused gate AdaLN gated residual") ||
        !h3_gpu_require_bf16(gpu, output, elements,
                             @"fused gate AdaLN output") ||
        !h3_gpu_require_command(gpu)) return 0;
    typedef struct {
        uint32_t rows, width, slots, gate_slot, shift_slot, scale_slot;
        float epsilon;
    } gate_adaln_args;
    gate_adaln_args args = {
        rows, width, slots, gate_slot, shift_slot, scale_slot, epsilon
    };
    NSString *pipeline_name = getenv("H3_DISABLE_EXACT_SIMD_GATE_ADALN") ?
        @"h3_gate_adaln_bf16" : @"h3_gate_adaln_bf16_exact_simd";
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, pipeline_name);
    if (!pipeline) return 0;
    const NSUInteger threads = 256;
    if (pipeline.maxTotalThreadsPerThreadgroup < threads) {
        h3_gpu_set_error(gpu,
            @"fused gate AdaLN needs a 256-thread threadgroup");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(branch).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(gate_modulation).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(norm_modulation).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(gated_residual).buffer offset:0 atIndex:6];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:7];
        [encoder setBytes:&args length:sizeof(args) atIndex:8];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_gate_adaln_quantize_int8(
                     h3_gpu *opaque, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *quantized_output,
                     h3_gpu_tensor *quantized_scales,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t padded_rows, uint32_t width, uint32_t slots,
                     uint32_t gate_slot, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon, int rotate) {
    H3GPU *gpu = GPU(opaque);
    if (rotate && width != 5376u) {
        h3_gpu_set_error(gpu, @"rotated fused int8 AdaLN needs width 5376");
        return 0;
    }
    size_t elements = (size_t)rows * width;
    size_t padded_elements = (size_t)padded_rows * width;
    if (!gpu.tensorOpsEnabled || !rows || padded_rows < rows || !width ||
        width > 5376 || gate_slot >= slots || shift_slot >= slots ||
        scale_slot >= slots || elements > UINT32_MAX ||
        !h3_gpu_require_bf16(gpu, residual, elements,
                             @"fused int8 gate AdaLN residual") ||
        !h3_gpu_require_bf16(gpu, branch, elements,
                             @"fused int8 gate AdaLN branch") ||
        !h3_gpu_require_bf16(gpu, norm_weight, width,
                             @"fused int8 gate AdaLN norm") ||
        !h3_gpu_require_bf16(gpu, gate_modulation, 1,
                             @"fused int8 gate modulation") ||
        !h3_gpu_require_bf16(gpu, norm_modulation, 1,
                             @"fused int8 norm modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows,
                                 @"fused int8 gate AdaLN row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, gated_residual, elements,
                             @"fused int8 gated residual") ||
        !h3_gpu_require_i8(gpu, quantized_output, padded_elements,
                           @"fused int8 AdaLN output") ||
        !h3_gpu_require_f32(gpu, quantized_scales, padded_rows,
                            @"fused int8 AdaLN scales") ||
        !h3_gpu_require_command(gpu)) return 0;
    typedef struct {
        uint32_t rows, width, slots, gate_slot, shift_slot, scale_slot;
        float epsilon;
    } gate_adaln_args;
    gate_adaln_args args = {
        rows, width, slots, gate_slot, shift_slot, scale_slot, epsilon
    };
    NSString *pipeline_name = rotate ? @"h3_gate_adaln_quantize_int8_rot256" :
        width == 5376 &&
        getenv("H3_DISABLE_VECTOR_GATE_ADALN") == NULL ?
        @"h3_gate_adaln_quantize_int8" :
        @"h3_gate_adaln_quantize_int8_scalar";
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, pipeline_name);
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu,
            @"fused M5 int8 gate AdaLN needs a 256-thread threadgroup");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(branch).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(gate_modulation).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(norm_modulation).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(gated_residual).buffer offset:0 atIndex:6];
        [encoder setBuffer:TENSOR(quantized_output).buffer offset:0 atIndex:7];
        [encoder setBuffer:TENSOR(quantized_scales).buffer offset:0 atIndex:8];
        [encoder setBytes:&args length:sizeof(args) atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake(padded_rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_qkv_rope_bf16_layout(h3_gpu *opaque, h3_gpu_tensor *query,
                                       h3_gpu_tensor *key,
                                       h3_gpu_tensor *value,
                                       const h3_gpu_tensor *qkv,
                                       const h3_gpu_tensor *q_norm,
                                       const h3_gpu_tensor *k_norm,
                                       const h3_gpu_tensor *rope_cos,
                                       const h3_gpu_tensor *rope_sin,
                                       uint32_t sequence, uint32_t heads,
                                       uint32_t head_dim, uint32_t rope_half,
                                       uint32_t grouped, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_gpu_require_bf16(gpu, qkv, count * 3, @"QKV input") ||
        !h3_gpu_require_bf16(gpu, q_norm, head_dim, @"Q norm") ||
        !h3_gpu_require_bf16(gpu, k_norm, head_dim, @"K norm") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count, @"RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count, @"RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query, count, @"query") ||
        !h3_gpu_require_bf16(gpu, key, count, @"key") ||
        !h3_gpu_require_bf16(gpu, value, count, @"value") ||
        rope_half * 2 > head_dim) return 0;
    qkv_args args = {sequence, heads, head_dim, rope_half, grouped, epsilon};
    if (head_dim == 128 && !(heads % 4) &&
        !getenv("H3_DISABLE_COOP_QKV")) {
        if (!h3_gpu_require_command(gpu)) return 0;
        NSString *pipeline_name = getenv("H3_DISABLE_CACHED_QKV") ?
            @"h3_qkv_rope_bf16_coop_uncached" :
            @"h3_qkv_rope_bf16_coop";
        id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
            gpu, pipeline_name);
        if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < head_dim) {
            h3_gpu_set_error(gpu,
                @"cooperative QKV/RoPE needs a 128-thread threadgroup");
            return 0;
        }
        @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder =
                [gpu.command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:5];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:6];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:7];
            [encoder setBytes:&args length:sizeof(args) atIndex:8];
            [encoder dispatchThreadgroups:MTLSizeMake(heads / 4, sequence, 1)
                     threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
            [encoder endEncoding];
        }
        h3_gpu_stats stats = gpu.stats;
        stats.direct_dispatches++;
        gpu.stats = stats;
        return 1;
    }
    return h3_gpu_dispatch_3d(gpu, @"h3_qkv_rope_bf16",
        MTLSizeMake(head_dim, heads, sequence),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:5];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:6];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:7];
            [encoder setBytes:&args length:sizeof(args) atIndex:8];
        });
}

int h3_gpu_qkv_rope_bf16(h3_gpu *opaque, h3_gpu_tensor *query,
                         h3_gpu_tensor *key, h3_gpu_tensor *value,
                         const h3_gpu_tensor *qkv,
                         const h3_gpu_tensor *q_norm,
                         const h3_gpu_tensor *k_norm,
                         const h3_gpu_tensor *rope_cos,
                         const h3_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim,
                         uint32_t rope_half, float epsilon) {
    return h3_gpu_qkv_rope_bf16_layout(
        opaque, query, key, value, qkv, q_norm, k_norm, rope_cos, rope_sin,
        sequence, heads, head_dim, rope_half, 0, epsilon);
}

int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *opaque, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key, h3_gpu_tensor *value,
                                 const h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon) {
    return h3_gpu_qkv_rope_bf16_layout(
        opaque, query, key, value, qkv, q_norm, k_norm, rope_cos, rope_sin,
        sequence, heads, head_dim, rope_half, 1, epsilon);
}

int h3_gpu_grouped_qkv_linear_rope_bf16(
                                 h3_gpu *opaque,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    gpu.headMajorSDPAInputs = NO;
    uint32_t inner = heads * head_dim;
    BOOL candidate = gpu.tensorOpsEnabled && rows >= 64 && rows <= 2048 &&
        input_dim == 5376 && heads == 56 && head_dim == 128 &&
        rope_half == 48 &&
        (gpu.tensorOpsMode == 1 || gpu.tensorOpsMode == 3 ||
         gpu.tensorOpsMode == 4) &&
        !getenv("H3_DISABLE_NAX_LINEAR") &&
        !getenv("H3_DISABLE_SPLIT_NAX_QKV");
    if (!candidate) {
        return h3_gpu_linear_bf16(opaque, qkv, input, weight, NULL, rows,
                                  input_dim, inner * 3) &&
            h3_gpu_grouped_qkv_rope_bf16(
                opaque, query, key, value, qkv, q_norm, k_norm, rope_cos,
                rope_sin, rows, heads, head_dim, rope_half, epsilon);
    }
    size_t projected = (size_t)rows * inner;
    size_t rope_count = (size_t)rows * rope_half;
    if (!h3_gpu_require_bf16(gpu, input, (size_t)rows * input_dim,
                             @"split QKV projection input") ||
        !h3_gpu_require_bf16(gpu, weight,
                             (size_t)inner * 3 * input_dim,
                             @"split QKV projection weight") ||
        !h3_gpu_require_bf16(gpu, q_norm, head_dim,
                             @"split Q norm") ||
        !h3_gpu_require_bf16(gpu, k_norm, head_dim,
                             @"split K norm") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count,
                             @"split RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count,
                             @"split RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query, projected, @"split query") ||
        !h3_gpu_require_bf16(gpu, key, projected, @"split key") ||
        !h3_gpu_require_bf16(gpu, value, projected, @"split value") ||
        !h3_gpu_require_command(gpu)) return 0;
    uint32_t headMajor = getenv("H3_DISABLE_HEAD_MAJOR_SDPA") == NULL;
    typedef struct {
        uint32_t rows, input_dim, heads, head_dim, rope_half, head_major;
        float epsilon;
    } qkv_project_rope_args;
    qkv_project_rope_args args = {
        rows, input_dim, heads, head_dim, rope_half, headMajor, epsilon
    };
    id<MTLComputePipelineState> projection = h3_gpu_pipeline(
        gpu, @"h3_qkv_project_split_bf16_nax_r128_morton4");
    id<MTLComputePipelineState> rope = h3_gpu_pipeline(
        gpu, @"h3_qk_rope_bf16_nax_inplace");
    const NSUInteger threads = 128;
    if (!projection || !rope ||
        projection.maxTotalThreadsPerThreadgroup < threads ||
        rope.maxTotalThreadsPerThreadgroup < threads) {
        h3_gpu_set_error(gpu,
            @"split M5 QKV projection/RoPE needs 128 threads");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:projection];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
        NSUInteger groups = (NSUInteger)((rows + 127) / 128) * heads * 6;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:rope];
        [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:5];
        [encoder setBytes:&args length:sizeof(args) atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake(heads / 4, rows, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches += 2;
    gpu.stats = stats;
    gpu.headMajorSDPAInputs = headMajor != 0;
    return 1;
}

int h3_gpu_grouped_qkv_linear_rope_int8(
                                 h3_gpu *opaque,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *quantized_input,
                                 h3_gpu_tensor *input_scales,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *weight_scales,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon,
                                 int input_is_quantized,
                                 int use_slower_unfused_qkv_rope,
                                 int use_slower_scalar_qkv_rms,
                                 int use_slower_uncached_int8_scales,
                                 int rotate_input) {
    H3GPU *gpu = GPU(opaque);
    gpu.headMajorSDPAInputs = NO;
    uint32_t inner = heads * head_dim;
    uint32_t padded_rows = (rows + 127u) & ~127u;
    size_t projected = (size_t)rows * inner;
    size_t rope_count = (size_t)rows * rope_half;
    if (!gpu.tensorOpsEnabled || rows < 128 || input_dim % 128 ||
        head_dim != 128 || heads % 4 || rope_half != 48 ||
        !h3_gpu_require_i8(gpu, weight,
                           (size_t)inner * 3 * input_dim,
                           @"int8 QKV projection weight") ||
        !h3_gpu_require_f32(gpu, weight_scales, inner * 3,
                            @"int8 QKV weight scales") ||
        !h3_gpu_require_i8(gpu, quantized_input,
                           (size_t)padded_rows * input_dim,
                           @"int8 QKV input") ||
        !h3_gpu_require_f32(gpu, input_scales, padded_rows,
                            @"int8 QKV input scales") ||
        (!input_is_quantized &&
         !h3_gpu_require_bf16(gpu, input, (size_t)rows * input_dim,
                              @"BF16 QKV input")) ||
        !h3_gpu_require_bf16(gpu, q_norm, head_dim, @"int8 Q norm") ||
        !h3_gpu_require_bf16(gpu, k_norm, head_dim, @"int8 K norm") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count,
                             @"int8 RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count,
                             @"int8 RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query, projected, @"int8 query") ||
        !h3_gpu_require_bf16(gpu, key, projected, @"int8 key") ||
        !h3_gpu_require_bf16(gpu, value, projected, @"int8 value") ||
        !h3_gpu_require_command(gpu)) return 0;
    if (!input_is_quantized && !h3_gpu_quantize_bf16_int8_rows(
            opaque, quantized_input, input_scales, input, rows, padded_rows,
            input_dim, 1.0f, @"int8 QKV input", rotate_input != 0)) return 0;
    BOOL fused_rope = !use_slower_unfused_qkv_rope &&
        getenv("H3_DISABLE_FUSED_INT8_QKV_ROPE") == NULL;
    BOOL local_scales = fused_rope && rows > 2048 &&
        !use_slower_uncached_int8_scales &&
        getenv("H3_DISABLE_QKV_LOCAL_SCALES") == NULL;
    BOOL full_k = fused_rope && input_dim == 5376u &&
        getenv("H3_DISABLE_QKV_FULL_K") == NULL;
    id<MTLComputePipelineState> projection = h3_gpu_pipeline(
        gpu, local_scales && full_k ?
            @"h3_qkv_project_split_int8_rope_local_scales_nax_r128_k5376_morton4" :
            local_scales ?
            @"h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4" :
            full_k ?
            @"h3_qkv_project_split_int8_rope_nax_r128_k5376_morton4" :
            fused_rope ?
            @"h3_qkv_project_split_int8_rope_nax_r128_morton4" :
            @"h3_qkv_project_split_int8_nax_r128_morton4");
    id<MTLComputePipelineState> rope = fused_rope ? nil : h3_gpu_pipeline(
        gpu, @"h3_qk_rope_bf16_nax_inplace");
    if (!projection || projection.maxTotalThreadsPerThreadgroup < 256 ||
        (!fused_rope &&
         (!rope || rope.maxTotalThreadsPerThreadgroup < 128))) {
        h3_gpu_set_error(gpu, @"int8 M5 QKV projection is unavailable");
        return 0;
    }
    typedef struct {
        uint32_t rows, input_dim, heads, head_dim, rope_half, head_major;
        float epsilon;
    } qkv_project_rope_args;
    /* The historical layout field now carries fused-epilogue mode bits:
     * bit 1 selects ordered BF16x4 RMS loads, bit 2 packs norm/RoPE by four. */
    uint32_t rms_mode = 1u;
    if (fused_rope && rows <= 2048 && !use_slower_scalar_qkv_rms &&
        getenv("H3_DISABLE_VECTOR_QKV_RMS") == NULL)
        rms_mode |= 2u;
    if (fused_rope && getenv("H3_DISABLE_VECTOR_QKV_ROPE") == NULL)
        rms_mode |= 4u;
    qkv_project_rope_args args = {
        rows, input_dim, heads, head_dim, rope_half, rms_mode, epsilon
    };
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:projection];
        [encoder setBuffer:TENSOR(quantized_input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(input_scales).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(weight_scales).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:6];
        if (fused_rope) {
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:7];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:8];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:9];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:10];
            [encoder setBytes:&args length:sizeof(args) atIndex:11];
        } else {
            [encoder setBytes:&args length:sizeof(args) atIndex:7];
        }
        NSUInteger groups = (NSUInteger)(padded_rows / 128u) * heads * 3u;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        if (!fused_rope) {
            encoder = [gpu.command computeCommandEncoder];
            [encoder setComputePipelineState:rope];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake(heads / 4, rows, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
        }
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches += fused_rope ? 1 : 2;
    gpu.stats = stats;
    gpu.headMajorSDPAInputs = YES;
    return 1;
}

int h3_gpu_swiglu_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, fused, (size_t)rows * width * 2,
                              @"SwiGLU input") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * width,
                              @"SwiGLU output")) return 0;
    swiglu_args args = {rows, width};
    return h3_gpu_dispatch_2d(gpu, @"h3_swiglu_bf16", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(fused).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_embedding_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, weight, (size_t)vocab_size * width,
                              @"embedding weight") ||
        !h3_gpu_require_elements(gpu, token_ids, tokens, @"token IDs") ||
        TENSOR(token_ids).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, (size_t)tokens * width,
                              @"embedding output")) return 0;
    embedding_args args = {tokens, vocab_size, width};
    return h3_gpu_dispatch_2d(gpu, @"h3_embedding_bf16", width, tokens,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(token_ids).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_text_qk_rope_bf16(h3_gpu *opaque,
                             h3_gpu_tensor *query_output,
                             h3_gpu_tensor *key_output,
                             const h3_gpu_tensor *query_input,
                             const h3_gpu_tensor *key_input,
                             const h3_gpu_tensor *q_norm,
                             const h3_gpu_tensor *k_norm,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin,
                             uint32_t sequence, uint32_t query_heads,
                             uint32_t kv_heads, uint32_t head_dim,
                             float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t key_count = (size_t)sequence * kv_heads * head_dim;
    size_t rope_count = (size_t)sequence * (head_dim / 2);
    if (head_dim % 2 || !kv_heads || query_heads % kv_heads ||
        !h3_gpu_require_bf16(gpu, query_input, query_count, @"text query") ||
        !h3_gpu_require_bf16(gpu, key_input, key_count, @"text key") ||
        !h3_gpu_require_bf16(gpu, q_norm, head_dim, @"text Q norm") ||
        !h3_gpu_require_bf16(gpu, k_norm, head_dim, @"text K norm") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count, @"text RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count, @"text RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query_output, query_count, @"text query output") ||
        !h3_gpu_require_bf16(gpu, key_output, key_count, @"text key output")) return 0;
    text_rope_args args = {sequence, query_heads, kv_heads, head_dim, epsilon};
    return h3_gpu_dispatch_3d(gpu, @"h3_text_qk_rope_bf16",
        MTLSizeMake(head_dim, query_heads, sequence),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(query_input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(key_input).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:5];
            [encoder setBuffer:TENSOR(query_output).buffer offset:0 atIndex:6];
            [encoder setBuffer:TENSOR(key_output).buffer offset:0 atIndex:7];
            [encoder setBytes:&args length:sizeof(args) atIndex:8];
        });
}

int h3_gpu_head_rms_norm_bf16(h3_gpu *opaque, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)sequence * heads * head_dim;
    if (!h3_gpu_require_bf16(gpu, tensor, count, @"head norm tensor") ||
        !h3_gpu_require_bf16(gpu, weight, head_dim, @"head norm weight")) return 0;
    head_norm_args args = {sequence, heads, head_dim, epsilon};
    return h3_gpu_dispatch_2d(gpu, @"h3_head_rms_norm_bf16", sequence, heads,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(tensor).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_rope_text_bf16(h3_gpu *opaque, h3_gpu_tensor *query,
                          h3_gpu_tensor *key,
                          const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32,
                          uint32_t sequence, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t key_count = (size_t)sequence * kv_heads * head_dim;
    size_t rope_count = (size_t)sequence * (head_dim / 2);
    if (head_dim % 2 || !kv_heads || query_heads % kv_heads ||
        !h3_gpu_require_bf16(gpu, query, query_count, @"RoPE query") ||
        !h3_gpu_require_bf16(gpu, key, key_count, @"RoPE key") ||
        !h3_gpu_require_elements(gpu, rope_cos_f32, rope_count, @"RoPE cosine") ||
        TENSOR(rope_cos_f32).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, rope_sin_f32, rope_count, @"RoPE sine") ||
        TENSOR(rope_sin_f32).dtype != H3_GPU_F32) return 0;
    text_rope_inplace_args args = {sequence, query_heads, kv_heads, head_dim};
    uint32_t maximum_heads = query_heads > kv_heads ? query_heads : kv_heads;
    return h3_gpu_dispatch_2d(gpu, @"h3_rope_text_bf16", sequence, maximum_heads,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(rope_cos_f32).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(rope_sin_f32).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

static H3GQA *h3_gpu_gqa_graph(H3GPU *gpu, uint32_t sequence,
                               uint32_t query_heads, uint32_t kv_heads,
                               uint32_t head_dim, float scale) {
    @autoreleasepool {
        NSString *key = [NSString stringWithFormat:@"%u:%u:%u:%u:%.9g",
                         sequence, query_heads, kv_heads, head_dim, scale];
        H3GQA *cached = gpu.gqaCache[key];
        if (cached) return cached;
        if (!kv_heads || query_heads % kv_heads) return nil;
        MPSGraph *graph = [[MPSGraph alloc] init];
        MPSShape *query_shape = @[@1, @(sequence), @(query_heads), @(head_dim)];
        MPSShape *kv_shape = @[@1, @(sequence), @(kv_heads), @(head_dim)];
        MPSGraphTensor *query = [graph placeholderWithShape:query_shape
                                                   dataType:MPSDataTypeBFloat16
                                                       name:nil];
        MPSGraphTensor *key_tensor = [graph placeholderWithShape:kv_shape
                                                         dataType:MPSDataTypeBFloat16
                                                             name:nil];
        MPSGraphTensor *value = [graph placeholderWithShape:kv_shape
                                                   dataType:MPSDataTypeBFloat16
                                                       name:nil];
        MPSGraphTensor *qt = [graph transposeTensor:query dimension:1
                                      withDimension:2 name:nil];
        MPSGraphTensor *kt = [graph transposeTensor:key_tensor dimension:1
                                      withDimension:2 name:nil];
        MPSGraphTensor *vt = [graph transposeTensor:value dimension:1
                                      withDimension:2 name:nil];
        uint32_t groups = query_heads / kv_heads;
        MPSShape *split_shape = @[@1, @(kv_heads), @1, @(sequence), @(head_dim)];
        MPSShape *broadcast_shape = @[@1, @(kv_heads), @(groups),
                                      @(sequence), @(head_dim)];
        MPSShape *expanded_shape = @[@1, @(query_heads), @(sequence),
                                     @(head_dim)];
        kt = [graph reshapeTensor:kt withShape:split_shape name:nil];
        vt = [graph reshapeTensor:vt withShape:split_shape name:nil];
        kt = [graph broadcastTensor:kt toShape:broadcast_shape name:nil];
        vt = [graph broadcastTensor:vt toShape:broadcast_shape name:nil];
        kt = [graph reshapeTensor:kt withShape:expanded_shape name:nil];
        vt = [graph reshapeTensor:vt withShape:expanded_shape name:nil];
        size_t mask_count = (size_t)sequence * sequence;
        uint16_t *mask_values = malloc(mask_count * sizeof(*mask_values));
        if (!mask_values) return nil;
        for (uint32_t row = 0; row < sequence; row++)
            for (uint32_t column = 0; column < sequence; column++)
                mask_values[(size_t)row * sequence + column] =
                    column <= row ? 0u : UINT16_C(0xff80);
        NSData *mask_data = [NSData dataWithBytesNoCopy:mask_values
                                                 length:mask_count * sizeof(*mask_values)
                                           freeWhenDone:YES];
        MPSGraphTensor *mask = [graph constantWithData:mask_data
            shape:@[@1, @1, @(sequence), @(sequence)]
            dataType:MPSDataTypeBFloat16];
        MPSGraphTensor *attention = [graph
            scaledDotProductAttentionWithQueryTensor:qt keyTensor:kt
            valueTensor:vt maskTensor:mask scale:scale name:nil];
        H3GQA *result = [[H3GQA alloc] init];
        result.graph = graph;
        result.query = query;
        result.key = key_tensor;
        result.value = value;
        result.output = [graph transposeTensor:attention dimension:1
                                 withDimension:2 name:nil];
        result.queryShape = query_shape;
        result.kvShape = kv_shape;
        gpu.gqaCache[key] = result;
        return result;
    }
}

static int h3_gpu_gqa_mps(H3GPU *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *query,
                          const h3_gpu_tensor *key,
                          const h3_gpu_tensor *value,
                          uint32_t sequence, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim,
                          float scale) {
    if (!h3_gpu_require_command(gpu)) return 0;
    H3GQA *cache = h3_gpu_gqa_graph(gpu, sequence, query_heads, kv_heads,
                                    head_dim, scale);
    if (!cache) {
        h3_gpu_set_error(gpu, @"cannot build MPSGraph causal GQA");
        return 0;
    }
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        MPSGraphTensorData *query_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(query).buffer shape:cache.queryShape
            dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData *(^kv_data)(const h3_gpu_tensor *) =
            ^MPSGraphTensorData *(const h3_gpu_tensor *tensor) {
                return [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:TENSOR(tensor).buffer shape:cache.kvShape
                    dataType:MPSDataTypeBFloat16];
            };
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(output).buffer shape:cache.queryShape
            dataType:MPSDataTypeBFloat16];
        NSDictionary *feeds = @{cache.query: query_data,
                                cache.key: kv_data(key),
                                cache.value: kv_data(value)};
        NSDictionary *results = @{cache.output: output_data};
        @try {
            [cache.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil resultsDictionary:results
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph causal GQA failed: %@",
                             exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_sdpa_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_gqa_causal_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value,
                           uint32_t sequence, uint32_t query_heads,
                           uint32_t kv_heads, uint32_t head_dim,
                           float scale) {
    H3GPU *gpu = GPU(opaque);
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t kv_count = (size_t)sequence * kv_heads * head_dim;
    if (!sequence || !query_heads || !kv_heads || !head_dim ||
        query_heads % kv_heads || head_dim > 128 ||
        !h3_gpu_require_bf16(gpu, query, query_count, @"GQA query") ||
        !h3_gpu_require_bf16(gpu, key, kv_count, @"GQA key") ||
        !h3_gpu_require_bf16(gpu, value, kv_count, @"GQA value") ||
        !h3_gpu_require_bf16(gpu, output, query_count, @"GQA output") ||
        !h3_gpu_require_command(gpu)) return 0;
    if (getenv("H3_MPS_GQA") && h3_gpu_gqa_mps(
            gpu, output, query, key, value, sequence, query_heads,
            kv_heads, head_dim, scale)) return 1;
    size_t score_bytes = (size_t)sequence * sizeof(float);
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu,
                                                           @"h3_gqa_causal_bf16");
    if (!pipeline) return 0;
    if (score_bytes + pipeline.staticThreadgroupMemoryLength >
        gpu.device.maxThreadgroupMemoryLength) {
        h3_gpu_set_error(gpu, @"causal attention sequence exceeds threadgroup memory");
        return 0;
    }
    NSUInteger maximum_threads = MIN((NSUInteger)128,
                                     pipeline.maxTotalThreadsPerThreadgroup);
    NSUInteger threads = 1;
    while (threads * 2 <= maximum_threads) threads *= 2;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
        gqa_args args = {sequence, query_heads, kv_heads, head_dim, scale};
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        [encoder setThreadgroupMemoryLength:score_bytes atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(sequence, query_heads, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_add_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, left, elements, @"add left") ||
        !h3_gpu_require_bf16(gpu, right, elements, @"add right") ||
        !h3_gpu_require_bf16(gpu, output, elements, @"add output")) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_add_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(left).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(right).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:3];
        });
}

int h3_gpu_sub_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, left, elements, @"subtract left") ||
        !h3_gpu_require_bf16(gpu, right, elements, @"subtract right") ||
        !h3_gpu_require_bf16(gpu, output, elements, @"subtract output"))
        return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_sub_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(left).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(right).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:3];
        });
}

int h3_gpu_token_pool_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           size_t input_offset,
                           h3_gpu_tensor *original,
                           size_t original_offset,
                           h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    size_t elements = (size_t)rows * width;
    size_t input_elements = (size_t)input_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!input_rows || !rows || rows > input_rows ||
        baseline_rows > rows || !width ||
        elements > UINT32_MAX || input_offset > UINT32_MAX ||
        input_elements > UINT32_MAX - input_offset ||
        original_offset > UINT32_MAX ||
        input_elements > UINT32_MAX - original_offset ||
        baseline_offset > UINT32_MAX ||
        baseline_elements > UINT32_MAX - baseline_offset ||
        !input || TENSOR(input).dtype != H3_GPU_BF16 ||
        input_offset > TENSOR(input).elements ||
        input_elements > TENSOR(input).elements - input_offset ||
        !original || TENSOR(original).dtype != H3_GPU_BF16 ||
        original_offset > TENSOR(original).elements ||
        input_elements > TENSOR(original).elements - original_offset ||
        !h3_gpu_require_bf16(gpu, output, elements, @"token pool output") ||
        !baseline || TENSOR(baseline).dtype != H3_GPU_BF16 ||
        baseline_offset > TENSOR(baseline).elements ||
        baseline_elements > TENSOR(baseline).elements - baseline_offset ||
        !h3_gpu_require_elements(gpu, baseline_indices, rows,
                                 @"token pool baseline indices") ||
        TENSOR(baseline_indices).dtype != H3_GPU_U32 ||
        !h3_gpu_require_elements(gpu, pairs, (size_t)rows * 2,
                                 @"token pool pairs") ||
        TENSOR(pairs).dtype != H3_GPU_U32) return 0;
    typedef struct {
        uint32_t input_offset, original_offset, baseline_offset;
        uint32_t rows, width;
    } token_pool_args;
    token_pool_args args = {
        (uint32_t)input_offset, (uint32_t)original_offset,
        (uint32_t)baseline_offset, rows, width
    };
    return h3_gpu_dispatch_2d(gpu, @"h3_token_pool_bf16", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(pairs).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(baseline).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(baseline_indices).buffer
                          offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(original).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
        });
}

int h3_gpu_token_pool_adaln_bf16(
                           h3_gpu *opaque, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, size_t input_offset,
                           h3_gpu_tensor *original, size_t original_offset,
                           h3_gpu_tensor *baseline, size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t input_rows, uint32_t rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t elements = (size_t)rows * width;
    size_t input_elements = (size_t)input_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!input_rows || !rows || rows > input_rows || !width || width > 5376 ||
        baseline_rows > rows || shift_slot >= slots || scale_slot >= slots ||
        elements > UINT32_MAX || input_offset > UINT32_MAX ||
        input_elements > UINT32_MAX - input_offset ||
        original_offset > UINT32_MAX ||
        input_elements > UINT32_MAX - original_offset ||
        baseline_offset > UINT32_MAX ||
        baseline_elements > UINT32_MAX - baseline_offset ||
        !input || TENSOR(input).dtype != H3_GPU_BF16 ||
        input_offset > TENSOR(input).elements ||
        input_elements > TENSOR(input).elements - input_offset ||
        !original || TENSOR(original).dtype != H3_GPU_BF16 ||
        original_offset > TENSOR(original).elements ||
        input_elements > TENSOR(original).elements - original_offset ||
        !h3_gpu_require_bf16(gpu, residual, elements,
                             @"fused token pool residual") ||
        !baseline || TENSOR(baseline).dtype != H3_GPU_BF16 ||
        baseline_offset > TENSOR(baseline).elements ||
        baseline_elements > TENSOR(baseline).elements - baseline_offset ||
        !h3_gpu_require_elements(gpu, baseline_indices, rows,
                                 @"fused token pool baseline indices") ||
        TENSOR(baseline_indices).dtype != H3_GPU_U32 ||
        !h3_gpu_require_elements(gpu, pairs, (size_t)rows * 2,
                                 @"fused token pool pairs") ||
        TENSOR(pairs).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, norm_weight, width,
                             @"fused token pool norm") ||
        !h3_gpu_require_bf16(gpu, modulation, 1,
                             @"fused token pool modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows,
                                 @"fused token pool row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, elements,
                             @"fused token pool AdaLN output") ||
        !h3_gpu_require_command(gpu)) return 0;
    typedef struct {
        uint32_t input_offset, original_offset, baseline_offset;
        uint32_t rows, width, slots, shift_slot, scale_slot;
        float epsilon;
    } token_pool_adaln_args;
    token_pool_adaln_args args = {
        (uint32_t)input_offset, (uint32_t)original_offset,
        (uint32_t)baseline_offset, rows, width, slots, shift_slot,
        scale_slot, epsilon
    };
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, @"h3_token_pool_adaln_bf16");
    if (!pipeline) return 0;
    const NSUInteger threads = 256;
    if (pipeline.maxTotalThreadsPerThreadgroup < threads) {
        h3_gpu_set_error(gpu,
            @"fused token pool AdaLN needs a 256-thread threadgroup");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(pairs).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(baseline).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(baseline_indices).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(original).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:6];
        [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:7];
        [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:8];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:9];
        [encoder setBytes:&args length:sizeof(args) atIndex:10];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_token_expand_delta_bf16(
                           h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents, uint32_t rows,
                           uint32_t reduced_rows, uint32_t baseline_rows,
                           uint32_t width,
                           uint32_t exact_prefix_rows,
                           float update_scale) {
    H3GPU *gpu = GPU(opaque);
    size_t elements = (size_t)rows * width;
    size_t reduced_elements = (size_t)reduced_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!rows || !reduced_rows || reduced_rows > rows ||
        baseline_rows > reduced_rows || !width ||
        exact_prefix_rows > reduced_rows || elements > UINT32_MAX ||
        reduced_elements > UINT32_MAX || original_offset > UINT32_MAX ||
        elements > UINT32_MAX - original_offset ||
        baseline_offset > UINT32_MAX ||
        baseline_elements > UINT32_MAX - baseline_offset ||
        !original || TENSOR(original).dtype != H3_GPU_BF16 ||
        original_offset > TENSOR(original).elements ||
        elements > TENSOR(original).elements - original_offset ||
        !h3_gpu_require_bf16(gpu, output, elements,
                             @"token expand output") ||
        !h3_gpu_require_bf16(gpu, reduced, reduced_elements,
                             @"token expand reduced") ||
        !baseline || TENSOR(baseline).dtype != H3_GPU_BF16 ||
        baseline_offset > TENSOR(baseline).elements ||
        baseline_elements > TENSOR(baseline).elements - baseline_offset ||
        !h3_gpu_require_elements(gpu, baseline_indices, reduced_rows,
                                 @"token expand baseline indices") ||
        TENSOR(baseline_indices).dtype != H3_GPU_U32 ||
        !h3_gpu_require_elements(gpu, parents, rows,
                                 @"token expand parents") ||
        TENSOR(parents).dtype != H3_GPU_U32) return 0;
    typedef struct {
        uint32_t original_offset, baseline_offset;
        uint32_t rows, width, exact_prefix_rows;
        float update_scale;
    } token_expand_args;
    token_expand_args args = {
        (uint32_t)original_offset, (uint32_t)baseline_offset,
        rows, width, exact_prefix_rows, update_scale
    };
    return h3_gpu_dispatch_2d(gpu, @"h3_token_expand_delta_bf16", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(original).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(reduced).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(baseline).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(baseline_indices).buffer
                          offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(parents).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
        });
}

int h3_gpu_token_expand_adaln_bf16(
                           h3_gpu *opaque, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t rows, uint32_t reduced_rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t exact_prefix_rows, float update_scale,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t elements = (size_t)rows * width;
    size_t reduced_elements = (size_t)reduced_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!rows || !reduced_rows || reduced_rows > rows || !width ||
        width > 5376 || baseline_rows > reduced_rows ||
        exact_prefix_rows > reduced_rows || shift_slot >= slots ||
        scale_slot >= slots || elements > UINT32_MAX ||
        reduced_elements > UINT32_MAX || original_offset > UINT32_MAX ||
        elements > UINT32_MAX - original_offset ||
        baseline_offset > UINT32_MAX ||
        baseline_elements > UINT32_MAX - baseline_offset ||
        !original || TENSOR(original).dtype != H3_GPU_BF16 ||
        original_offset > TENSOR(original).elements ||
        elements > TENSOR(original).elements - original_offset ||
        !h3_gpu_require_bf16(gpu, reduced, reduced_elements,
                             @"fused token AdaLN reduced input") ||
        !baseline || TENSOR(baseline).dtype != H3_GPU_BF16 ||
        baseline_offset > TENSOR(baseline).elements ||
        baseline_elements > TENSOR(baseline).elements - baseline_offset ||
        !h3_gpu_require_elements(gpu, baseline_indices, reduced_rows,
                                 @"fused token AdaLN baseline indices") ||
        TENSOR(baseline_indices).dtype != H3_GPU_U32 ||
        !h3_gpu_require_elements(gpu, parents, rows,
                                 @"fused token AdaLN parents") ||
        TENSOR(parents).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, residual, elements,
                             @"fused token AdaLN residual") ||
        !h3_gpu_require_bf16(gpu, norm_weight, width,
                             @"fused token AdaLN norm") ||
        !h3_gpu_require_bf16(gpu, modulation, 1,
                             @"fused token AdaLN modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows,
                                 @"fused token AdaLN row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, elements,
                             @"fused token AdaLN output") ||
        !h3_gpu_require_command(gpu)) return 0;
    typedef struct {
        uint32_t original_offset, baseline_offset, rows, width;
        uint32_t exact_prefix_rows, slots, shift_slot, scale_slot;
        float update_scale, epsilon;
    } token_expand_adaln_args;
    token_expand_adaln_args args = {
        (uint32_t)original_offset, (uint32_t)baseline_offset, rows, width,
        exact_prefix_rows, slots, shift_slot, scale_slot,
        update_scale, epsilon
    };
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, @"h3_token_expand_adaln_bf16");
    if (!pipeline) return 0;
    const NSUInteger threads = 256;
    if (pipeline.maxTotalThreadsPerThreadgroup < threads) {
        h3_gpu_set_error(gpu,
            @"fused token AdaLN needs a 256-thread threadgroup");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(original).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(reduced).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(baseline).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(baseline_indices).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(parents).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:6];
        [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:7];
        [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:8];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:9];
        [encoder setBytes:&args length:sizeof(args) atIndex:10];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_euler_bf16(h3_gpu *opaque, h3_gpu_tensor *sample,
                      size_t sample_offset, const h3_gpu_tensor *last,
                      const h3_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio) {
    H3GPU *gpu = GPU(opaque);
    if (!sample || TENSOR(sample).dtype != H3_GPU_F32 ||
        sample_offset > TENSOR(sample).elements ||
        elements > TENSOR(sample).elements - sample_offset ||
        sample_offset > UINT32_MAX || elements > UINT32_MAX - sample_offset ||
        !h3_gpu_require_bf16(gpu, last, elements, @"Euler last velocity") ||
        !h3_gpu_require_bf16(gpu, previous, elements,
                             @"Euler previous velocity")) return 0;
    euler_args args = {(uint32_t)sample_offset, elements, delta, ratio};
    return h3_gpu_dispatch_1d(gpu, @"h3_euler_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(sample).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(last).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(previous).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_silu_mul_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate,
                         const h3_gpu_tensor *up, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, gate, elements, @"SiLU gate") ||
        !h3_gpu_require_bf16(gpu, up, elements, @"SiLU up") ||
        !h3_gpu_require_bf16(gpu, output, elements, @"SiLU product")) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_silu_mul_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(gate).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(up).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:3];
        });
}
