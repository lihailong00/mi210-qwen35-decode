#include "ggml-cuda.h"
#include "common.cuh"
#include "iq3-repack.cuh"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/common.cuh"
#if defined(GGML_USE_HIP)
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>
#include <chrono>
#include <map>
#include <tuple>
#endif // defined(GGML_USE_HIP)
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/batch-get.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/conv3d.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fattn-dec.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/mmvq-fused-host.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/gdn-decode.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml-cuda/top-k-mask.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str) \
    { static std::once_flag warn_flag; std::call_once(warn_flag, []() { GGML_LOG_WARN(str); }); }

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
static int ggml_cuda_get_physical_device(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_device;
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    // translate the (possibly virtual) device id to the physical CUDA device that backs it
    const int physical_device = ggml_cuda_get_physical_device(device);

    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (physical_device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(physical_device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static bool ggml_cuda_env_flag(const char * name) {
    const char * env = getenv(name);
    return env != nullptr && std::atoi(env) != 0;
}

static int ggml_cuda_env_int(const char * name, int default_value) {
    const char * env = getenv(name);
    return env != nullptr ? std::atoi(env) : default_value;
}

static bool ggml_cuda_env_set(const char * name) {
    return getenv(name) != nullptr;
}

static bool ggml_cuda_unified_memory_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_ENABLE_UNIFIED_MEMORY");
    return enabled;
}

static bool ggml_cuda_pinned_memory_disabled() {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_NO_PINNED");
    return disabled;
}

static bool ggml_cuda_register_host_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_REGISTER_HOST");
    return enabled;
}

static bool ggml_cuda_fusion_disabled() {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_FUSION");
    return disabled;
}

static bool ggml_cuda_p2p_enabled() {
    static const bool enabled = ggml_cuda_env_set("GGML_CUDA_P2P");
    return enabled;
}

static int ggml_cuda_op_offload_min_batch() {
    static const int min_batch = ggml_cuda_env_int("GGML_OP_OFFLOAD_MIN_BATCH", 32);
    return min_batch;
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (ggml_cuda_unified_memory_enabled()) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(const char * devName) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    // Extract the base architecture model ("gfx908:sramecc+:xnack-" -> "908")
    std::string archName = devName;
    archName.erase(0, 3);
    archName = archName.substr(0, archName.find(':'));

    const auto hex = [](const std::string & s) { return (int) strtoul(s.c_str(), nullptr, 16); };

    int cc = 0;

    // tease out the version information
    if (archName.size() > 8) {
        // versions labeled generic use '-' as delimiter
        if (archName.compare(archName.size() - 8, 8, "-generic") == 0) {
            archName.resize(archName.size() - 8);
            const size_t sep = archName.find('-');
            cc = 0x100 * hex(archName.substr(0, sep));
            if (sep != std::string::npos) {
                cc += 0x10 * hex(archName.substr(sep + 1));
            }
        }
    } else if (archName.size() >= 3) {
        // ordinary IDs are the version in hex
        cc = hex(archName);
    }

    return GGML_CUDA_CC_OFFSET_AMD + cc;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.physical_device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
    // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
    info.device_count = info.physical_device_count;

    const char * devices_env = getenv("GGML_CUDA_DEVICES");
    if (devices_env != nullptr && info.physical_device_count > 0) {
        const int requested = atoi(devices_env);
        if (requested > 0) {
            info.device_count = requested;
        } else {
            GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, devices_env);
        }
    }

    if (info.device_count > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n",
                      __func__, info.device_count, GGML_CUDA_MAX_DEVICES);
        info.device_count = GGML_CUDA_MAX_DEVICES;
    }

    // map each (virtual) device to a backing physical device (round-robin), assign each its index
    // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
    int physical_share_count[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(info.device_count == 0 || info.physical_device_count > 0);
    for (int id = 0; id < info.device_count; ++id) {
        info.devices[id].physical_device = id % info.physical_device_count;
        info.devices[id].virtual_index  = physical_share_count[info.devices[id].physical_device]++;
    }

    int64_t total_vram = 0;
    for (int id = 0; id < info.physical_device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.physical_device_count, (size_t)(total_vram / (1024 * 1024)));
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n",
                      __func__, info.device_count, info.physical_device_count);
    }
    total_vram = 0;

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        const int physical_id = info.devices[id].physical_device;

        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, physical_id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_id));

        // a virtual device owns only a share of its physical GPU's memory; report that share so the
        // logged per-device VRAM sums to the physical total above.
        GGML_ASSERT(physical_share_count[physical_id] > 0);
        info.devices[id].physical_share_count = physical_share_count[physical_id];
        const size_t device_vram = prop.totalGlobalMem / info.devices[id].physical_share_count;
        const size_t device_vram_mib = device_vram / (1024 * 1024);

        info.default_tensor_split[id] = total_vram;
        total_vram += device_vram;
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      device_vram_mib);
#elif defined(GGML_USE_MUSA)
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(physical_id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

#if defined(GGML_USE_HIP)
    // On CDNA2 (gfx90a) rocBLAS's own Tensile kernel selection is poor for several large f16->f32
    // GEMM shapes used by prefill (e.g. m=5120 n=4096 k=17408 runs at ~48 TFLOPS), while the
    // hipBLASLt backend picks ~140-160 TFLOPS kernels (prefill pp8192 +27%, pp2048 +38%).
    // rocBLAS reads ROCBLAS_USE_HIPBLASLT when the first handle is created, which happens lazily
    // after this point. Only applied when every device is CDNA2 and the user has not set it;
    // set ROCBLAS_USE_HIPBLASLT=0 to keep the rocBLAS kernels.
    {
        bool all_cdna2 = info.device_count > 0;
        for (int id = 0; id < info.device_count; ++id) {
            all_cdna2 = all_cdna2 && GGML_CUDA_CC_IS_CDNA2(info.devices[id].cc);
        }
        if (all_cdna2 && getenv("ROCBLAS_USE_HIPBLASLT") == nullptr) {
#if defined(_WIN32)
            _putenv_s("ROCBLAS_USE_HIPBLASLT", "1");
#else
            setenv("ROCBLAS_USE_HIPBLASLT", "1", 0);
#endif
        }
    }
#endif // defined(GGML_USE_HIP)

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    if (ggml_cuda_p2p_enabled()) {
        for (int id = 0; id < info.physical_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));
            for (int id_other = 0; id_other < info.physical_device_count; ++id_other) {
                if (id == id_other) {
                    continue;
                }
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
                if (can_access_peer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                }
            }
        }
    }

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    ggml_backend_cuda_context & ctx;
    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    ggml_cuda_pool_leg(ggml_backend_cuda_context & ctx, int device) :
        ctx(ctx),
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        clear_pool();
        GGML_ASSERT(pool_size == 0);
    }

    void clear_pool() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
    }

    // captured CUDA graphs can still use freed buffers, so they must be captured again
    void invalidate_cuda_graphs() {
#ifdef USE_CUDA_GRAPH
        int n_captured = 0;
        for (const auto & [key, graph] : ctx.cuda_graphs) {
            n_captured += graph->instances.size();
        }
        if (n_captured > 0) {
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: memory freed, %d captured CUDA graphs must be captured again\n", device, n_captured);
        }
        ctx.cuda_graphs.clear();
#endif // USE_CUDA_GRAPH
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        if (err == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            invalidate_cuda_graphs();
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
        invalidate_cuda_graphs();
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    int physical_device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        physical_device(ggml_cuda_get_physical_device(device)),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = physical_device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
            bool use_peer_access = ggml_cuda_p2p_enabled();
#if defined(GGML_USE_NCCL)
            use_peer_access = true;
#endif // defined(GGML_USE_NCCL)

            if (use_peer_access) {
                // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
                // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
                // allocations do not become peer-accessible from that alone, so access
                // must be granted explicitly here. With virtual devices, grant access
                // on the backing *physical* devices (deduplicated, since several
                // virtual devices can map to the same physical GPU).
                std::vector<CUmemAccessDesc> access_descs;
                bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
                const int device_count = ggml_cuda_info().device_count;
                for (int id = 0; id < device_count; ++id) {
                    const int id_physical = ggml_cuda_get_physical_device(id);
                    if (id_physical != physical_device) {
                        int can_access_peer = 0;
                        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                        if (!can_access_peer) {
                            continue;
                        }
                    }
                    if (physical_seen[id_physical]) {
                        continue;
                    }
                    physical_seen[id_physical] = true;
                    CUmemAccessDesc access = {};
                    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = id_physical;
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    access_descs.push_back(access);
                }
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size()));
            } else {
                // set access for non P2P
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = physical_device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, &access, 1));
            }

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(ggml_backend_cuda_context & ctx,
                                                                               int                         device,
                                                                               [[maybe_unused]] int        stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(ctx, device));
}

// Shared f16/bf16 copies of large F32 GEMM inputs (prefill).
// Several MUL_MATs of one graph read the same activation (DeltaNet in-projections qkv/z and beta/alpha, attention
// q/k/v, FFN gate/up) and the cuBLAS path converts src1 to the compute type for each of them. Here the converted copy
// is kept in a dedicated buffer (not from the pool, so no other op can overwrite it) and reused by the later
// consumers. The op that creates the activation can also write the copy itself (ggml_cuda_f16_share_produce, used by
// the fused rms_norm * weight), then no conversion pass is needed at all.
//
// An entry is identified by the exact src1 ggml_tensor pointer + data range + shape + strides + target type + stream.
// It is only created for tensors that are src[1] of MUL_MAT nodes of the graph being evaluated, so the pointer is a
// live graph tensor (never a stack slice). An entry is dropped
//  - at the start of every graph evaluation (all entries),
//  - before a node is executed whose output bytes [data, data + nbytes) intersect the bytes of the cached source
//    (covers in-place ops, allocator reuse and writes through views); for a fused node group this is also done for
//    every non-view member of the group after the group ran, except for the entry that was created by the group
//    for that very member (the copy of what the group just wrote).
// Copies are only looked up by MUL_MAT nodes that are executed on their own or as the first node of a fused group
// (MUL_MAT -> ADD folded into the GEMM, see ggml_cuda_try_fuse_gemm_add), i.e. after the checks above ran for every
// node that precedes them in the graph; never from a later member of a fused group. Within one graph evaluation the
// bytes of a tensor only change through a node of the graph that owns them, so a surviving entry is always the
// conversion of the current contents. A consumer only counts if ggml_cuda_mul_mat really runs it as a cuBLAS GEMM of
// that compute type (ggml_cuda_mul_mat_cublas_gemm_type). The cache is bypassed (no lookup, no copy written by the
// producer) while concurrent stream regions exist (GGML_CUDA_GRAPH_OPT), like the fusions: the entries carry no
// cross-stream ordering. GGML_CUDA_F16_SHARE_VERIFY=1 re-converts on every reuse and compares bitwise.
// Entries used while a stream is being captured are pinned (the captured graph references the buffer): pinned buffers
// are not freed before the context is destroyed (an outgrown one is retired), and a replayed graph contains the
// kernel that fills the buffer ahead of the consumers, so it never depends on the host-side state.
// GGML_CUDA_F16_SHARE=0 disables the cache, GGML_CUDA_RMS_NORM_F16=0 only the copy written by rms_norm.
// GGML_CUDA_FUSION_STATS=1 prints the reuse / fusion counters when the context is destroyed.
// The state belongs to the backend context (ggml_backend_cuda_context::f16_share) and is used by the thread that
// evaluates a graph on that context, like the rest of the context.
// HIP only: the cache and the copy written by rms_norm were developed and validated on the HIP backend alone and have
// never been built or run with CUDA, so they are disabled there (ggml_cuda_f16_share_enabled).
#define GGML_CUDA_F16_SHARE_ENTRIES 2
struct ggml_cuda_f16_share_entry {
    void              * buf    = nullptr;
    size_t              size   = 0; // capacity in bytes
    bool                pinned = false;
    bool                valid  = false;
    ggml_type           type   = GGML_TYPE_F16;
    const ggml_tensor * src    = nullptr;
    const char        * lo     = nullptr;
    const char        * hi     = nullptr;
    int64_t             ne[GGML_MAX_DIMS] = {};
    size_t              nb[GGML_MAX_DIMS] = {};
    cudaStream_t        stream = nullptr;
    uint64_t            stamp  = 0;
    uint64_t            epoch  = 0;
};
struct ggml_cuda_f16_share_state {
    const ggml_cgraph * cgraph  = nullptr;
    bool                counted = false;
    bool                in_fuse = false; // inside ggml_cuda_try_fuse: no lookups unless the fusion allows them, see above
    int                 n_valid = 0;
    uint64_t            clock   = 0;
    uint64_t            epoch   = 0; // advanced for every executed node (group)
    uint64_t            n_reused    = 0;
    uint64_t            n_converted = 0;
    uint64_t            n_produced  = 0;
    // the other large-batch fusions, counted here for GGML_CUDA_FUSION_STATS
    uint64_t            n_gemm_add       = 0;
    uint64_t            n_swiglu_f16     = 0;
    uint64_t            n_swiglu_f16_add = 0;
    uint64_t            n_conv_fuse      = 0;
    std::unordered_map<const ggml_tensor *, std::pair<int, int>> uses; // MUL_MAT consumers: f16, bf16
    ggml_cuda_f16_share_entry entries[GGML_CUDA_F16_SHARE_ENTRIES];
    std::vector<void *> retired; // outgrown buffers that captured graphs may still reference
};

static bool ggml_cuda_f16_share_enabled() {
#if defined(GGML_USE_HIP)
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_F16_SHARE");
        return e == nullptr || atoi(e) != 0;
    }();
    return enabled;
#else
    return false; // never built or validated with CUDA, see above
#endif // defined(GGML_USE_HIP)
}

static ggml_cuda_f16_share_state * ggml_cuda_f16_share_get(ggml_backend_cuda_context * ctx) {
    if (ctx->f16_share == nullptr) {
        ctx->f16_share = new ggml_cuda_f16_share_state();
    }
    return ctx->f16_share;
}

// start of a graph evaluation: nothing cached before is trusted
static void ggml_cuda_f16_share_begin(ggml_backend_cuda_context * ctx, const ggml_cgraph * cgraph) {
    ggml_cuda_f16_share_state * st = ggml_cuda_f16_share_get(ctx);
    st->cgraph  = cgraph;
    st->counted = false;
    if (st->n_valid != 0) {
        for (auto & e : st->entries) {
            e.valid = false;
        }
        st->n_valid = 0;
    }
}

// node is (about to be) written: drop the entries whose source bytes it overlaps
static void ggml_cuda_f16_share_invalidate(ggml_cuda_f16_share_state * st, const ggml_tensor * node, const bool keep_own) {
    const char * lo = (const char *) node->data;
    const char * hi = lo + ggml_nbytes(node);
    for (auto & e : st->entries) {
        if (e.valid && (lo == nullptr || (lo < e.hi && e.lo < hi))) {
            if (keep_own && e.src == node && e.epoch == st->epoch) {
                continue;
            }
            e.valid = false;
            st->n_valid--;
        }
    }
}

static void ggml_cuda_f16_share_release(ggml_backend_cuda_context * ctx) {
    const std::unique_ptr<ggml_cuda_f16_share_state> st(ctx->f16_share);
    ctx->f16_share = nullptr;
    if (!st) {
        return;
    }
    if (getenv("GGML_CUDA_FUSION_STATS") != nullptr) {
        // debugging / tests: how often each of the large-batch fusions ran in this context. Written to stderr and
        // not through GGML_LOG_*: it was asked for explicitly, and tools that install a null log callback
        // (llama-bench without -v) would drop it.
        fprintf(stderr, "fusion stats: f16_share %llu reused, %llu converted, %llu produced; gemm_add %llu; swiglu_f16 %llu; "
                "swiglu_f16_add %llu; conv_fuse %llu\n",
                (unsigned long long) st->n_reused, (unsigned long long) st->n_converted, (unsigned long long) st->n_produced,
                (unsigned long long) st->n_gemm_add, (unsigned long long) st->n_swiglu_f16,
                (unsigned long long) st->n_swiglu_f16_add, (unsigned long long) st->n_conv_fuse);
    }
    for (auto & e : st->entries) {
        if (e.buf != nullptr) {
            if (e.stream != nullptr) {
                CUDA_CHECK(cudaStreamSynchronize(e.stream));
            }
            CUDA_CHECK(cudaFree(e.buf));
        }
    }
    for (void * buf : st->retired) {
        CUDA_CHECK(cudaFree(buf));
    }
}

static bool ggml_cuda_f16_share_capturing(cudaStream_t stream) {
#if defined(GGML_USE_HIP)
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    CUDA_CHECK(hipStreamIsCapturing(stream, &capture_status));
    return capture_status != hipStreamCaptureStatusNone;
#else
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    return capture_status != cudaStreamCaptureStatusNone;
#endif
}

static bool ggml_cuda_f16_share_candidate(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && t->data != nullptr && ggml_nelements(t) >= (1 << 20) && ggml_is_contiguous(t);
}

// not while concurrent stream regions exist, see above
static bool ggml_cuda_f16_share_usable(ggml_backend_cuda_context & ctx) {
    return ggml_cuda_f16_share_enabled() && ctx.curr_stream_no == 0 && ctx.stream_context().concurrent_events.empty();
}

static ggml_type ggml_cuda_mul_mat_cublas_gemm_type(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);

// number of MUL_MAT nodes of the current graph that read t as src1 and that ggml_cuda_mul_mat runs as an f16 / bf16
// cuBLAS GEMM (the only consumers that convert src1 and can take the copy)
static std::pair<int, int> ggml_cuda_f16_share_uses(ggml_backend_cuda_context & ctx, ggml_cuda_f16_share_state * st, const ggml_tensor * t) {
    if (!st->counted) {
        st->uses.clear();
        for (int i = 0; i < st->cgraph->n_nodes; ++i) {
            const ggml_tensor * n = st->cgraph->nodes[i];
            if (n->op == GGML_OP_MUL_MAT && (n->flags & GGML_TENSOR_FLAG_COMPUTE) && n->src[0] != nullptr && n->src[1] != nullptr &&
                ggml_cuda_f16_share_candidate(n->src[1])) {
                const ggml_type gemm_type = ggml_cuda_mul_mat_cublas_gemm_type(ctx, n);
                if (gemm_type == GGML_TYPE_BF16) {
                    st->uses[n->src[1]].second++;
                } else if (gemm_type == GGML_TYPE_F16) {
                    st->uses[n->src[1]].first++;
                }
            }
        }
        st->counted = true;
    }
    const auto it = st->uses.find(t);
    return it == st->uses.end() ? std::pair<int, int>(0, 0) : it->second;
}

// A slot with a buffer of at least nbytes for a new entry, or nullptr. Its previous entry is dropped.
static ggml_cuda_f16_share_entry * ggml_cuda_f16_share_slot(ggml_cuda_f16_share_state * st, cudaStream_t stream, const size_t nbytes) {
    const bool capturing = ggml_cuda_f16_share_capturing(stream);

    // an unused slot that is large enough, else an unused one, else the least recently used one
    ggml_cuda_f16_share_entry * slot = nullptr;
    for (int pass = 0; pass < 3 && slot == nullptr; ++pass) {
        for (auto & e : st->entries) {
            if ((pass == 0 && !e.valid && e.size >= nbytes) || (pass == 1 && !e.valid) || pass == 2) {
                if (slot == nullptr || e.stamp < slot->stamp) {
                    slot = &e;
                }
            }
        }
    }
    if (slot->valid) {
        slot->valid = false;
        st->n_valid--;
    }
    if (slot->size < nbytes) {
        void * ptr = nullptr;
        if (cudaMalloc(&ptr, nbytes) != cudaSuccess) {
            (void) cudaGetLastError();
            return nullptr;
        }
        if (slot->buf != nullptr) {
            if (slot->pinned || capturing) {
                // a captured graph may reference it (and a stream cannot be synchronized while it is capturing)
                st->retired.push_back(slot->buf);
            } else {
                if (slot->stream != nullptr) {
                    CUDA_CHECK(cudaStreamSynchronize(slot->stream));
                }
                CUDA_CHECK(cudaFree(slot->buf));
            }
        }
        slot->buf    = ptr;
        slot->size   = nbytes;
        slot->pinned = false;
    }
    slot->pinned = slot->pinned || capturing;
    return slot;
}

static void ggml_cuda_f16_share_fill(ggml_cuda_f16_share_state * st, ggml_cuda_f16_share_entry * slot, const ggml_tensor * t,
        const ggml_type type, cudaStream_t stream) {
    slot->valid  = true;
    slot->type   = type;
    slot->src    = t;
    slot->lo     = (const char *) t->data;
    slot->hi     = slot->lo + ggml_nbytes(t);
    memcpy(slot->ne, t->ne, sizeof(slot->ne));
    memcpy(slot->nb, t->nb, sizeof(slot->nb));
    slot->stream = stream;
    slot->stamp  = ++st->clock;
    slot->epoch  = st->epoch;
    st->n_valid++;
}

static void ggml_cuda_f16_share_convert(const ggml_tensor * t, const ggml_type type, void * buf, cudaStream_t stream) {
    if (type == GGML_TYPE_F16) {
        ggml_get_to_fp16_cuda(GGML_TYPE_F32)(t->data, (half *) buf, ggml_nelements(t), stream);
    } else {
        ggml_get_to_bf16_cuda(GGML_TYPE_F32)(t->data, (nv_bfloat16 *) buf, ggml_nelements(t), stream);
    }
}

// Returns the f16 (type == GGML_TYPE_F16) or bf16 copy of src1, converting it if it is not cached, or nullptr if the
// cache cannot be used for this tensor; the caller then converts into a pool buffer as before.
static const void * ggml_cuda_f16_share_lookup(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const ggml_type type) {
    if ((type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) || !ggml_cuda_f16_share_candidate(src1) || !ggml_cuda_f16_share_usable(ctx)) {
        return nullptr;
    }
    ggml_cuda_f16_share_state * st = ggml_cuda_f16_share_get(&ctx);
    if (st->cgraph == nullptr || st->in_fuse) {
        return nullptr;
    }
    const int64_t      ne     = ggml_nelements(src1);
    const cudaStream_t stream = ctx.stream();
    const char *       lo     = (const char *) src1->data;

    for (auto & e : st->entries) {
        if (e.valid && e.src == src1 && e.type == type && e.lo == lo && e.hi == lo + ggml_nbytes(src1) && e.stream == stream &&
            memcmp(e.ne, src1->ne, sizeof(e.ne)) == 0 && memcmp(e.nb, src1->nb, sizeof(e.nb)) == 0) {
            e.stamp = ++st->clock;
            st->n_reused++;
            static const bool verify = getenv("GGML_CUDA_F16_SHARE_VERIFY") != nullptr;
            if (verify && !ggml_cuda_f16_share_capturing(stream)) {
                ggml_cuda_pool_alloc<uint16_t> tmp(ctx.pool(), ne);
                ggml_cuda_f16_share_convert(src1, type, tmp.get(), stream);
                std::vector<uint16_t> a(ne), b(ne);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaMemcpy(a.data(), tmp.get(), ne*sizeof(uint16_t), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(b.data(), e.buf,     ne*sizeof(uint16_t), cudaMemcpyDeviceToHost));
                if (memcmp(a.data(), b.data(), ne*sizeof(uint16_t)) != 0) {
                    GGML_ABORT("f16_share: stale %s copy of %s", ggml_type_name(type), src1->name);
                }
            }
            return e.buf;
        }
    }

    const std::pair<int, int> uses = ggml_cuda_f16_share_uses(ctx, st, src1);
    if ((type == GGML_TYPE_F16 ? uses.first : uses.second) < 2) {
        return nullptr;
    }
    ggml_cuda_f16_share_entry * slot = ggml_cuda_f16_share_slot(st, stream, ne*sizeof(uint16_t));
    if (slot == nullptr) {
        return nullptr;
    }
    ggml_cuda_f16_share_convert(src1, type, slot->buf, stream);
    ggml_cuda_f16_share_fill(st, slot, src1, type, stream);
    st->n_converted++;
    return slot->buf;
}

// For the op that is about to write the F32 tensor t (contiguous, in the current graph evaluation): returns a buffer
// that the op must fill with the f16 values of t in the same element order, or nullptr if no f16 copy is wanted.
// The call replaces a cache entry, so the op asks only once it knows that it can write the copy; if it cannot after
// all, it must call ggml_cuda_f16_share_cancel. Always nullptr in builds without HIP (ggml_cuda_f16_share_usable).
void * ggml_cuda_f16_share_produce(ggml_backend_cuda_context & ctx, const ggml_tensor * t) {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_RMS_NORM_F16");
        return e == nullptr || atoi(e) != 0;
    }();
    if (!enabled || !ggml_cuda_f16_share_candidate(t) || !ggml_cuda_f16_share_usable(ctx)) {
        return nullptr;
    }
    ggml_cuda_f16_share_state * st = ggml_cuda_f16_share_get(&ctx);
    if (st->cgraph == nullptr || ggml_cuda_f16_share_uses(ctx, st, t).first < 1) {
        return nullptr;
    }
    const cudaStream_t stream = ctx.stream();
    ggml_cuda_f16_share_entry * slot = ggml_cuda_f16_share_slot(st, stream, ggml_nelements(t)*sizeof(half));
    if (slot == nullptr) {
        return nullptr;
    }
    ggml_cuda_f16_share_fill(st, slot, t, GGML_TYPE_F16, stream);
    st->n_produced++;
    return slot->buf;
}

// Drops the entry that ggml_cuda_f16_share_produce created for t in the current node group (and only that one: not
// a copy that a MUL_MAT converted, nor a second entry).
void ggml_cuda_f16_share_cancel(ggml_backend_cuda_context & ctx, const ggml_tensor * t) {
    ggml_cuda_f16_share_state * st = ggml_cuda_f16_share_get(&ctx);
    for (auto & e : st->entries) {
        if (e.valid && e.src == t && e.type == GGML_TYPE_F16 && e.epoch == st->epoch) {
            e.valid = false;
            st->n_valid--;
            st->n_produced--;
            break;
        }
    }
}

// Allows lookups for the GEMM of a fused group for the lifetime of the object. Only for a MUL_MAT that is the first
// node of its group: the invalidation checks then ran for every node that precedes it, as for an unfused MUL_MAT.
struct ggml_cuda_f16_share_allow_lookup {
    ggml_cuda_f16_share_state * st;
    const bool                  in_fuse;
    ggml_cuda_f16_share_allow_lookup(ggml_backend_cuda_context * ctx) : st(ggml_cuda_f16_share_get(ctx)), in_fuse(st->in_fuse) {
        st->in_fuse = false;
    }
    ~ggml_cuda_f16_share_allow_lookup() {
        st->in_fuse = in_fuse;
    }
};

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    ggml_cuda_set_device(device);
    ggml_cuda_f16_share_release(this);
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                // per-stream caches keyed by the stream handle: free them before the handle can be reused
                ggml_cuda_set_device(i);
                ggml_cuda_mmvq_cache_release(i, streams[i][j]);
                ggml_cuda_mmvq_fused_release(i, streams[i][j]);
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
}


// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;
#if defined(GGML_USE_HIP)
    std::mutex codex_iq3_mutex;
    std::unordered_map<const void *, void *> codex_iq3_cache;
    std::map<std::pair<const void *, int>, void *> repack_cache;
#endif

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
#if defined(GGML_USE_HIP)
        if (!codex_iq3_cache.empty() || !repack_cache.empty()) ggml_cuda_set_device(device);
        for (auto & entry : codex_iq3_cache) CUDA_CHECK(hipFree(entry.second));
        for (auto & entry : repack_cache) CUDA_CHECK(hipFree(entry.second));
#endif
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

#if defined(GGML_USE_HIP)
static __global__ void codex_iq3_pack_kernel(const block_iq3_s * raw, uint32_t * codes,
                                           codex_iq3_meta * meta, size_t count) {
    const size_t gidx = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (gidx >= count*8) return;
    const size_t block = gidx / 8;
    const int group = gidx % 8;
    const auto & in = raw[block];
    const uint32_t high = in.qh[group];
#pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        const int j = 2*pair;
        const auto signs = in.signs[group*4+pair];
        uint32_t packed = 0;
#pragma unroll
        for (int side = 0; side < 2; side++) {
            const uint32_t index = in.qs[group*8+j+side] | (((high>>(j+side))&1u)<<8);
            const uint32_t values = iq3s_grid[index];
#pragma unroll
            for (int lane = 0; lane < 4; lane++) {
                int value = (values>>(8*lane))&255;
                if ((signs>>(4*side+lane))&1) value = -value;
                packed |= uint32_t((value+15)/2)<<(8*lane+4*side);
            }
        }
        codes[gidx*4+pair] = packed;
    }
    if (group == 0) {
        meta[block].d = in.d;
#pragma unroll
        for (int i = 0; i < 4; i++) meta[block].scales[i] = in.scales[i];
    }
}

static __global__ void codex_iq3_xxs_pack_kernel(const block_iq3_xxs * raw, uint32_t * codes,
                                               codex_iq3_meta * meta, size_t count) {
    const size_t gidx = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (gidx >= count*8) return;
    const size_t block = gidx/8;
    const int group = gidx%8;
    const auto & in = raw[block];
    const uint8_t * a = in.qs+64+4*group;
    const uint32_t aux = uint32_t(a[0]) | uint32_t(a[1])<<8 | uint32_t(a[2])<<16 | uint32_t(a[3])<<24;
#pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        const uint8_t s7 = aux>>(7*pair);
        const uint32_t signs = s7 ^ ((__popc(s7)&1u)<<7);
        uint32_t packed = 0;
#pragma unroll
        for (int side = 0; side < 2; side++) {
            const uint32_t v = iq3xxs_grid[in.qs[group*8+2*pair+side]];
#pragma unroll
            for (int lane = 0; lane < 4; lane++) {
                const uint32_t magnitude = (v>>(8*lane))&255u;
                const uint32_t index = (magnitude-4u)/8u + (((signs>>(4*side+lane))&1u)<<3);
                packed |= index<<(8*lane+4*side);
            }
        }
        codes[gidx*4+pair] = packed;
    }
    if (group == 0) {
        meta[block].d = in.d;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            meta[block].scales[i] = (in.qs[64+8*i+3]>>4) | (in.qs[64+8*i+7]&0xf0u);
        }
    }
}

const void * codex_iq3_get_packed(const ggml_tensor * tensor, int device, hipStream_t stream) {
    static const bool enabled = [] {
        const char * value = getenv("CODEX_IQ3_PREPACK");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    static const bool xxs_enabled = [] {
        const char * value = getenv("CODEX_IQ3_XXS_PREPACK");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    if (!tensor || tensor->view_src ||
            !((enabled && tensor->type == GGML_TYPE_IQ3_S) ||
              (xxs_enabled && tensor->type == GGML_TYPE_IQ3_XXS)) ||
            !tensor->buffer || !ggml_backend_buffer_is_cuda(tensor->buffer) ||
            ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            !ggml_is_matrix(tensor) || !ggml_is_contiguous(tensor) || tensor->ne[0] % 256 ||
            !GGML_CUDA_CC_IS_CDNA2(ggml_cuda_info().devices[device].cc)) {
        return nullptr;
    }
    auto * buffer = static_cast<ggml_backend_cuda_buffer_context *>(tensor->buffer->context);
    if (buffer->device != device) return nullptr;
    std::lock_guard<std::mutex> guard(buffer->codex_iq3_mutex);
    auto it = buffer->codex_iq3_cache.find(tensor->data);
    if (it != buffer->codex_iq3_cache.end()) return it->second;
    hipStreamCaptureStatus status;
    CUDA_CHECK(hipStreamIsCapturing(stream, &status));
    if (status != hipStreamCaptureStatusNone) return nullptr;
    const size_t count = ggml_nelements(tensor)/256;
    const size_t bytes = count*(32*sizeof(uint32_t)+sizeof(codex_iq3_meta));
    size_t free_bytes, total_bytes;
    CUDA_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    if (free_bytes < bytes + size_t(8)*1024*1024*1024) return nullptr;
    void * memory = nullptr;
    if (hipMalloc(&memory, bytes) != hipSuccess) {
        (void) hipGetLastError();
        return nullptr;
    }
    auto * codes = static_cast<uint32_t *>(memory);
    auto * meta = reinterpret_cast<codex_iq3_meta *>(codes+count*32);
    if (tensor->type == GGML_TYPE_IQ3_XXS) {
        codex_iq3_xxs_pack_kernel<<<(count*8+255)/256,256,0,stream>>>(
                static_cast<const block_iq3_xxs *>(tensor->data),codes,meta,count);
    } else {
        codex_iq3_pack_kernel<<<(count*8+255)/256,256,0,stream>>>(
                static_cast<const block_iq3_s *>(tensor->data),codes,meta,count);
    }
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipStreamSynchronize(stream));
    buffer->codex_iq3_cache.emplace(tensor->data,memory);
    return memory;
}

const void * ggml_cuda_get_repacked(const ggml_tensor * tensor, int device, hipStream_t stream, int kind, size_t bytes, ggml_cuda_repack_fn fn) {
    if (!tensor || tensor->view_src || !tensor->buffer || !ggml_backend_buffer_is_cuda(tensor->buffer) ||
            ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return nullptr;
    }
    auto * buffer = static_cast<ggml_backend_cuda_buffer_context *>(tensor->buffer->context);
    if (buffer->device != device) return nullptr;
    std::lock_guard<std::mutex> guard(buffer->codex_iq3_mutex);
    const auto key = std::make_pair((const void *) tensor->data, kind);
    auto it = buffer->repack_cache.find(key);
    if (it != buffer->repack_cache.end()) return it->second;
    hipStreamCaptureStatus status;
    CUDA_CHECK(hipStreamIsCapturing(stream, &status));
    if (status != hipStreamCaptureStatusNone) return nullptr;
    size_t free_bytes, total_bytes;
    CUDA_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    if (free_bytes < bytes + size_t(4)*1024*1024*1024) return nullptr;
    void * memory = nullptr;
    if (hipMalloc(&memory, bytes) != hipSuccess) {
        (void) hipGetLastError();
        return nullptr;
    }
    fn(tensor->data, memory, tensor, stream);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipStreamSynchronize(stream));
    buffer->repack_cache.emplace(key, memory);
    return memory;
}

bool ggml_cuda_has_repacked(const ggml_tensor * tensor, int kind) {
    if (!tensor || tensor->view_src || !tensor->buffer || !ggml_backend_buffer_is_cuda(tensor->buffer)) return false;
    auto * buffer = static_cast<ggml_backend_cuda_buffer_context *>(tensor->buffer->context);
    std::lock_guard<std::mutex> guard(buffer->codex_iq3_mutex);
    return buffer->repack_cache.find(std::make_pair((const void *) tensor->data, kind)) != buffer->repack_cache.end();
}
#endif

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemsetAsync((char *)tensor->data + original_size, 0, padded_size - original_size, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_n(ggml_backend_buffer_t buffer, struct ggml_tensor * const * tensors, const void * const * data, const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);

    for (size_t i = 0; i < n_tensors; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        CUDA_CHECK(cudaMemcpyAsync(
            (char *) tensors[i]->data + offsets[i],
            data[i],
            sizes[i],
            cudaMemcpyHostToDevice,
            cudaStreamPerThread));
    }

    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_n(ggml_backend_buffer_t buffer, const struct ggml_tensor * const * tensors, void * const * data, const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);

    if (!ggml_cuda_pinned_memory_disabled() &&
            ggml_cuda_get_tensor_n(ctx->device, tensors, data, offsets, sizes, n_tensors)) {
        return;
    }

    for (size_t i = 0; i < n_tensors; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        CUDA_CHECK(cudaMemcpyAsync(
            data[i],
            (const char *) tensors[i]->data + offsets[i],
            sizes[i],
            cudaMemcpyDeviceToHost,
            cudaStreamPerThread));
    }

    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cuda_buffer_cpy_tensor_n(ggml_backend_buffer_t buffer, const struct ggml_tensor * const * srcs, struct ggml_tensor * const * dsts, size_t n_tensors) {
    ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    for (size_t i = 0; i < n_tensors; i++) {
        ggml_backend_buffer_t src_buf = srcs[i]->view_src ? srcs[i]->view_src->buffer : srcs[i]->buffer;
        if (!ggml_backend_buffer_is_cuda(src_buf)) {
            return false;
        }
    }

#ifdef GGML_CUDA_NO_PEER_COPY
    for (size_t i = 0; i < n_tensors; i++) {
        ggml_backend_buffer_t src_buf = srcs[i]->view_src ? srcs[i]->view_src->buffer : srcs[i]->buffer;
        ggml_backend_buffer_t dst_buf = dsts[i]->view_src ? dsts[i]->view_src->buffer : dsts[i]->buffer;
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src_buf->context;
        ggml_backend_cuda_buffer_context * dst_ctx_i = (ggml_backend_cuda_buffer_context *)dst_buf->context;
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx_i->device);
        if (src_physical != dst_physical) {
            return false;
        }
    }
#endif

    ggml_cuda_set_device(dst_ctx->device);

    for (size_t i = 0; i < n_tensors; i++) {
        ggml_backend_buffer_t src_buf = srcs[i]->view_src ? srcs[i]->view_src->buffer : srcs[i]->buffer;
        ggml_backend_buffer_t dst_buf = dsts[i]->view_src ? dsts[i]->view_src->buffer : dsts[i]->buffer;
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src_buf->context;
        ggml_backend_cuda_buffer_context * dst_ctx_i = (ggml_backend_cuda_buffer_context *)dst_buf->context;
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx_i->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dsts[i]->data, srcs[i]->data, ggml_nbytes(srcs[i]), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            GGML_ABORT("peer copy not supported");
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dsts[i]->data, dst_physical, srcs[i]->data, src_physical, ggml_nbytes(srcs[i]), cudaStreamPerThread));
#endif
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    return true;
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
    /* .set_tensor_n    = */ ggml_backend_cuda_buffer_set_tensor_n,
    /* .get_tensor_n    = */ ggml_backend_cuda_buffer_get_tensor_n,
    /* .cpy_tensor_n    = */ ggml_backend_cuda_buffer_cpy_tensor_n,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
    using try_allreduce_fn = bool(*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **);

    std::vector<ggml_backend_t> backends;
    std::vector<int>            dev_ids;

    // Set by the init chain (comm_init_{nccl, internal, none}) to one of
    // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
    // internal needs `ar_pipeline`, butterfly needs nothing.  Per-call
    // failures return false; the meta backend's generic implementation then
    // handles that call.
    try_allreduce_fn            try_allreduce = nullptr;

    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;

    // Number of ranks in the communicator when each rank is a separate process. Zero unless
    // comm_init_rank created this context.
    int                         world_size = 0;

    // Number of ranks taking part in the reduction. For a communicator from comm_init_rank this is
    // world_size. Otherwise it is the number of backends this process holds. The NCCL reduction
    // uses it to choose the tensor size below which it reduces in F32 rather than converting to
    // BF16 first.
    size_t n_ranks() const {
        return world_size > 0 ? (size_t) world_size : backends.size();
    }

#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t>     comms;
#endif // GGML_USE_NCCL

    ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
        for (ncclComm_t comm : comms) {
            NCCL_CHECK(ncclCommDestroy(comm));
        }
#endif // GGML_USE_NCCL
        ggml_cuda_ar_pipeline_free(ar_pipeline);
    }
};

#ifdef GGML_USE_NCCL
// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32.
static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    const size_t n_backends = comm_ctx->backends.size();
    const size_t n_ranks    = comm_ctx->n_ranks();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    if ((n_ranks <= 2 && ne < 32768) || (n_ranks == 3 && ne < 131072) || (n_ranks >= 4 && ne < 262144)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
static bool ggml_backend_cuda_comm_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);

    const size_t n_backends = comm_ctx->backends.size();
    GGML_ASSERT(n_backends == 2);
    GGML_ASSERT(tensors[0] != nullptr);

    const int64_t   ne   = ggml_nelements(tensors[0]);
    const ggml_type type = tensors[0]->type;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int) type);
        return false;
    }

    if (ne == 0) {
        return true;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, i);
            return false;
        }
        if (ggml_nelements(tensors[i]) != ne || tensors[i]->type != type) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                           __func__, i, ggml_nelements(tensors[i]), (int) tensors[i]->type, ne, (int) type);
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i])) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64 " nbytes=%zu packed=%zu type=%d\n",
                           __func__, i, ne, ggml_nbytes(tensors[i]),
                           (size_t) ne * ggml_type_size(type) / ggml_blck_size(type), (int) type);
            return false;
        }
        if (((uintptr_t) tensors[i]->data & 0xF) != 0) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64 "\n",
                           __func__, i, tensors[i]->data, (int) type, ne);
            return false;
        }
        GGML_ASSERT((ggml_nbytes(tensors[i]) & 0xF) == 0);
    }

    return ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);
}
#endif // GGML_USE_NCCL

static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
}

static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
    return false;
}

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
    if (comm_ctx_v == nullptr) {
        return;
    }
    delete static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}

static void ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                  "falling back to meta-backend butterfly\n");
    ggml_backend_cuda_comm_init_none(ret);
}

static void ggml_backend_cuda_comm_init_nccl(ggml_backend_cuda_comm_context * ret) {
#ifdef GGML_USE_NCCL
    // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN("NCCL disabled: virtual devices in use; "
                      "falling back to internal AllReduce\n");
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }

    const size_t n = ret->dev_ids.size();
    ret->comms.resize(n);
    ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());
    if (rc == ncclSuccess) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
        return;
    }

    ret->comms.clear();
    GGML_LOG_WARN("NCCL init failed (%s); falling back to internal AllReduce\n",
                  ncclGetErrorString(rc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
    GGML_LOG_WARN("NCCL not compiled in; falling back to internal AllReduce.  "
                  "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL

    ggml_backend_cuda_comm_init_internal(ret);
}

// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.assign(backends, backends + n_backends);
    ret->dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->dev_ids.push_back(static_cast<ggml_backend_cuda_context *>(backends[i]->context)->device);
    }

    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
        // Platform default: Linux uses NCCL, otherwise (generally Windows) internal
#if defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
#else
        ggml_backend_cuda_comm_init_internal(ret);
#endif // defined(__linux__)
    } else {
        std::string env_str(env);
        if (env_str == "nccl") {
            ggml_backend_cuda_comm_init_nccl(ret);
        } else if (env_str == "internal") {
            ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "none") {
            ggml_backend_cuda_comm_init_none(ret);
        } else {
            GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env);
            ggml_backend_cuda_comm_init_none(ret);
        }
    }

    return ret;
}

#ifdef GGML_USE_NCCL
// Builds the communicator with ncclCommInitRank, for the case where each rank is its own
// process. Blocks until all world_size ranks have joined.
static void * ggml_backend_cuda_comm_init_rank(ggml_backend_t backend, int rank, int world_size,
                                               ggml_backend_comm_bootstrap_callback bootstrap_callback, void * bootstrap_callback_data) {
    if (world_size < 1 || rank < 0 || rank >= world_size) {
        GGML_LOG_ERROR("%s: invalid rank %d for world size %d\n", __func__, rank, world_size);
        return nullptr;
    }

    // Every rank sends whether its local checks passed, and rank 0 also sends the NCCL unique id.
    // So all ranks either join together or fail together, and no rank waits in ncclCommInitRank alone.
    struct bootstrap_msg {
        uint8_t      ok;
        ncclUniqueId id;
    };
    bootstrap_msg local;
    memset(&local, 0, sizeof(local));
    local.ok = 1;

    if (!ggml_backend_is_cuda(backend)) {
        GGML_LOG_ERROR("%s: backend is not a CUDA backend\n", __func__);
        local.ok = 0;
    } else {
        // NCCL requires one distinct physical GPU per rank, see ggml_backend_cuda_comm_init_nccl.
        const ggml_cuda_device_info & info = ggml_cuda_info();
        if (info.device_count > info.physical_device_count) {
            GGML_LOG_ERROR("%s: CUDA virtual devices are in use, which NCCL does not support\n", __func__);
            local.ok = 0;
        }
    }
    if (local.ok && rank == 0) {
        const ncclResult_t rc = ncclGetUniqueId(&local.id);
        if (rc != ncclSuccess) {
            GGML_LOG_ERROR("%s: ncclGetUniqueId failed (%s)\n", __func__, ncclGetErrorString(rc));
            local.ok = 0;
        }
    }

    std::vector<bootstrap_msg> all(world_size);
    if (!bootstrap_callback(bootstrap_callback_data, &local, all.data(), sizeof(local))) {
        GGML_LOG_ERROR("%s: bootstrap exchange failed on rank %d\n", __func__, rank);
        return nullptr;
    }
    for (int i = 0; i < world_size; i++) {
        if (!all[i].ok) {
            GGML_LOG_ERROR("%s: rank %d failed its setup checks\n", __func__, i);
            return nullptr;
        }
    }

    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.push_back(backend);
    ret->dev_ids.push_back(cuda_ctx->device);
    ret->world_size = world_size;
    ret->comms.resize(1);
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;

    // On DGX Spark, NCCL picks 64 channels by default, which is slower and uses more memory than 8.
    // NCCL_MAX_CTAS still overrides this.
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        config.maxCTAs = 8;
    }

    ggml_cuda_set_device(cuda_ctx->device);
    const ncclResult_t rc = ncclCommInitRankConfig(&ret->comms[0], world_size, all[0].id, rank, &config);
    if (rc != ncclSuccess) {
        GGML_LOG_ERROR("%s: ncclCommInitRankConfig failed for rank %d/%d (%s)\n",
                       __func__, rank, world_size, ncclGetErrorString(rc));
        ret->comms.clear();
        delete ret;
        return nullptr;
    }

    GGML_LOG_INFO("%s: joined NCCL communicator as rank %d/%d on device %d\n",
                  __func__, rank, world_size, cuda_ctx->device);
    return ret;
}
#endif // GGML_USE_NCCL

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->try_allreduce(comm_ctx, tensors);
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (ggml_cuda_pinned_memory_disabled()) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

static size_t ggml_backend_cuda_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // very large pinned allocations can fail (e.g. on Windows) and fall back to pageable memory, smaller ones succeed
    return 4ull*1024*1024*1024;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name            = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer        = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size        = */ ggml_backend_cuda_host_buffer_type_get_max_size,
            /* .get_alloc_size      = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp32_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_bf16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

#if defined(GGML_USE_HIP)
// Per-shape GEMM kernel selection for large prefill GEMMs on CDNA2 with the hipBLASLt backend.
// hipBLASLt's default choice is 3-10% slower than its best kernel for most prefill shapes. The first
// time a (m, k, n-bucket) shape is seen, a short list of known-good solution indices (those the
// library reports as valid for the problem) is timed against the default and the fastest is cached.
// The default is kept unless a candidate is at least 2% faster, so a library with different indices can
// only fall back to the default. Costs ~0.1-0.2 s once per shape. GGML_CUDA_GEMM_TUNE=0 disables it.
// Different kernels differ in fp32 summation order only.
// With beta != 0 (GEMM accumulating onto C, see ggml_cuda_try_fuse_gemm_add) the shape is tuned separately and the
// timing runs write to a scratch C, so the caller's C is accumulated onto exactly once.
static bool ggml_cuda_hip_gemm_tuned(cublasHandle_t handle, cudaStream_t stream, int cc,
        const int m, const int n, const int k,
        const void * alpha, const void * a, const int lda, const void * b, const int ldb,
        const void * beta, void * c, const int ldc) {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_GEMM_TUNE");
        const char * l = getenv("ROCBLAS_USE_HIPBLASLT");
        return (e == nullptr || atoi(e) != 0) && l != nullptr && atoi(l) == 1;
    }();
    if (!enabled || !GGML_CUDA_CC_IS_CDNA2(cc) || n < 2048 || (int64_t) m*k < (1 << 22)) {
        return false;
    }

    // with the rocBLAS backend of hipBLAS the hipBLAS handle is the rocBLAS handle
    rocblas_handle h = (rocblas_handle) handle;
    const auto run = [&](rocblas_gemm_algo algo, int32_t sol, void * c_out) {
        return rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none, m, n, k, alpha,
            a, rocblas_datatype_f16_r, lda, b, rocblas_datatype_f16_r, ldb, beta,
            c_out, rocblas_datatype_f32_r, ldc, c_out, rocblas_datatype_f32_r, ldc,
            rocblas_datatype_f32_r, algo, sol, 0) == rocblas_status_success;
    };

    const bool accumulate = *(const float *) beta != 0.0f;

    static std::mutex mutex;
    static std::map<std::tuple<int, int, int>, int32_t> cache;
    const auto key = std::make_tuple(m, k, (n >= 4096 ? 1 : 0) | (accumulate ? 2 : 0));

    int32_t sol = 0;
    bool    known;
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = cache.find(key);
        known = it != cache.end();
        if (known) {
            sol = it->second;
        }
    }

    if (!known) {
        static const int32_t candidates[] = { 107806, 107812, 107817, 107825, 108043, 108049, 108091, 108169, 108210, 108434 };

        rocblas_int n_valid = 0;
        std::vector<rocblas_int> valid;
        if (rocblas_gemm_ex_get_solutions(h, rocblas_operation_transpose, rocblas_operation_none, m, n, k, alpha,
                a, rocblas_datatype_f16_r, lda, b, rocblas_datatype_f16_r, ldb, beta,
                c, rocblas_datatype_f32_r, ldc, c, rocblas_datatype_f32_r, ldc,
                rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, 0, nullptr, &n_valid) == rocblas_status_success && n_valid > 0) {
            valid.resize(n_valid);
            if (rocblas_gemm_ex_get_solutions(h, rocblas_operation_transpose, rocblas_operation_none, m, n, k, alpha,
                    a, rocblas_datatype_f16_r, lda, b, rocblas_datatype_f16_r, ldb, beta,
                    c, rocblas_datatype_f32_r, ldc, c, rocblas_datatype_f32_r, ldc,
                    rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, 0, valid.data(), &n_valid) != rocblas_status_success) {
                valid.clear();
            }
        }

        // beta != 0: time on a scratch C (its values do not matter, they only have to stay finite)
        void * c_time = c;
        void * c_tmp  = nullptr;
        if (accumulate) {
            const size_t c_size = (size_t) ldc*n*sizeof(float);
            if (cudaMalloc(&c_tmp, c_size) != cudaSuccess) {
                c_tmp = nullptr;
            } else {
                CUDA_CHECK(cudaMemsetAsync(c_tmp, 0, c_size, stream));
                c_time = c_tmp;
            }
        }
        const bool can_time = !accumulate || c_tmp != nullptr;

        const auto time_it = [&](rocblas_gemm_algo algo, int32_t s) -> double {
            if (!run(algo, s, c_time)) {
                return -1.0;
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto t0 = std::chrono::steady_clock::now();
            run(algo, s, c_time);
            run(algo, s, c_time);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        };

        const double t_default = can_time ? time_it(rocblas_gemm_algo_standard, 0) : -1.0;
        double       t_best    = t_default;
        if (t_default > 0.0) {
            for (const int32_t cand : candidates) {
                if (std::find(valid.begin(), valid.end(), cand) == valid.end()) {
                    continue;
                }
                const double t = time_it(rocblas_gemm_algo_solution_index, cand);
                if (t > 0.0 && t < t_best) {
                    t_best = t;
                    sol    = cand;
                }
            }
            if (t_best > 0.98*t_default) {
                sol = 0;
            }
        }
        if (c_tmp != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaFree(c_tmp));
        }
        GGML_LOG_DEBUG("%s: m=%d n=%d k=%d beta=%d -> solution %d (%.3f ms vs default %.3f ms)\n", __func__, m, n, k,
                       accumulate ? 1 : 0, (int) sol, t_best*500.0, t_default*500.0);
        std::lock_guard<std::mutex> lock(mutex);
        cache[key] = sol;
    }

    if (sol == 0) {
        return false;
    }
    if (run(rocblas_gemm_algo_solution_index, sol, c)) {
        return true;
    }
    // The caller now runs the default GEMM on the same C. With beta != 0 that is only correct if the failed call
    // wrote nothing to C. Assumption: rocblas_gemm_ex rejects a solution index / argument set before it launches
    // a kernel, so a failure leaves C untouched (a call that failed after accumulating would be added twice).
    std::lock_guard<std::mutex> lock(mutex);
    cache[key] = 0;
    return false;
}
#endif // defined(GGML_USE_HIP)

// src1_converted: optional, the contiguous src1 data already converted to compute_type by the caller (same element
// order as src1), used instead of converting src1->data here (which is then not read at all).
// accumulate: dst += src0^T * src1 (GEMM with beta = 1) instead of dst = src0^T * src1. Only for the paths that write
// fp32 straight into dst with a single GEMM (see ggml_cuda_mul_mat_cublas_can_accumulate).
template<ggml_type compute_type>
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const void * src1_converted = nullptr, const bool accumulate = false) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    // F32 chunks can have padding between rows to preserve the original destination stride.
    GGML_ASSERT(ggml_is_contiguous(dst) ||
        (compute_type == GGML_TYPE_F32 && ggml_is_contiguous_1(dst)));

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ldc = nb1 / sizeof(float);
    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == src0_ts);
    int64_t s01 = nb01 / src0_ts;
    int64_t s02 = nb02 / src0_ts;
    int64_t s03 = nb03 / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == src1_ts);
    int64_t s11 = nb11 / src1_ts;
    int64_t s12 = nb12 / src1_ts;
    int64_t s13 = nb13 / src1_ts;

    float * dst_ddf = (float *) dst->data;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));

        if (ggml_is_contiguously_allocated(src0)) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }

    // f16/bf16 copy shared with the other GEMMs that read this activation (same layout as the contiguous conversion)
    const void * src1_shared = nullptr;
    if constexpr (compute_type != GGML_TYPE_F32) {
        if (src1_converted == nullptr) {
            src1_shared = ggml_cuda_f16_share_lookup(ctx, src1, compute_type);
        }
    }

    if (src1_converted != nullptr) {
        GGML_ASSERT(ggml_is_contiguous(src1));
        src1_ptr = (const cuda_t *) src1_converted;
        const size_t src1_bs = ggml_blck_size(src1->type);
        s11 *= src1_bs;
        s12 *= src1_bs;
        s13 *= src1_bs;
    } else if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else if (src1_shared != nullptr) {
        src1_ptr = (const cuda_t *) src1_shared;
    } else {
        src1_alloc.alloc(ggml_nelements(src1));

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }

    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    bool prefer_f32_output = false;
    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {
        prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_RDNA4_M(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }

    if (prefer_f32_output) {
        dst_ptr = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    } else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_ptr = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }

    if (accumulate) {
        GGML_ASSERT(dst_ptr == (char *) dst_ddf && cu_data_type == CUDA_R_32F && ne12 == 1 && ne13 == 1);
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha(); // 1.0f
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    (const float *) alpha, (const float *) src0_ptr, s01,
                                           (const float *) src1_ptr, s11,
                    (const float *) beta,  (float       *)  dst_ptr, ldc));
    } else if (ne12 == 1 && ne13 == 1) {
#if defined(GGML_USE_HIP)
        if (compute_type == GGML_TYPE_F16 && prefer_f32_output &&
            ggml_cuda_hip_gemm_tuned(cublas_h, main_stream, cc, ne01, ne11, ne10, alpha, src0_ptr, s01, src1_ptr, s11, beta, dst_ptr, ldc)) {
            // done with a tuned kernel
        } else
#endif // defined(GGML_USE_HIP)
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ldc,
                    cu_compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? s03 : s02;
        const int64_t smb = ne12 == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta,   dst_ptr, cu_data_type,   ldc, ne1*ldc, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        const size_t src_type_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        const dim3 block_dims(threads_x, threads_y);

        const dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_ptr,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                s02*src_type_size, s03*src_type_size,
                s12*src_type_size, s13*src_type_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ldc,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static ggml_type ggml_cuda_mul_mat_cublas_compute_type(const int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    const ggml_prec prec = (ggml_prec) ggml_get_op_params_i32(dst, 0);
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc)) {
        compute_type = GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_BF16 && !fast_bf16_hardware_available(cc)) {
        if (GGML_CUDA_CC_IS_AMD(cc) && src1->ne[1] > 32) {
            compute_type = GGML_TYPE_F32;
        }
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && src1->ne[1] > (cc >= GGML_CUDA_CC_VOLTA ? 8 : 128)) {
            compute_type = GGML_TYPE_F32;
        }
    }
    // F16 is the only compute type that can not satisfy a request for BF16
    if (prec == GGML_PREC_BF16 && compute_type == GGML_TYPE_F16) {
        compute_type = fast_bf16_hardware_available(cc) ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    } else if (prec == GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }

    // the override is parsed once: this function is also called by the fusion checks, several times per node
    static const ggml_type env_compute_type = [] {
        const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
        if (env_c != nullptr) {
            std::string env_cpp = env_c;
            for (char & c : env_cpp) {
                c = std::tolower(c);
            }
            if (env_cpp == "f32" || env_cpp == "fp32") {
                return GGML_TYPE_F32;
            } else if (env_cpp == "f16" || env_cpp == "fp16") {
                return GGML_TYPE_F16;
            } else if (env_cpp == "bf16") {
                return GGML_TYPE_BF16;
            } else if (env_cpp != "auto") {
                GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s\n", "ggml_cuda_mul_mat_cublas", env_cpp.c_str());
            }
        }
        return GGML_TYPE_COUNT; // no override
    }();
    if (env_compute_type != GGML_TYPE_COUNT) {
        compute_type = env_compute_type;
    }

    return compute_type;
}

// true if ggml_cuda_mul_mat_cublas runs this MUL_MAT as one fp16-input GEMM that writes fp32 straight into dst,
// so that it can accumulate onto dst (beta = 1)
static bool ggml_cuda_mul_mat_cublas_can_accumulate(const int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return GGML_CUDA_CC_IS_CDNA(cc) && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) &&
           src1->ne[2] == 1 && src1->ne[3] == 1 && src0->ne[2] == 1 && src0->ne[3] == 1 &&
           ggml_cuda_mul_mat_cublas_compute_type(cc, src0, src1, dst) == GGML_TYPE_F16;
}

// src1_f16 / accumulate: see ggml_cuda_mul_mat_cublas_impl, only for an fp16 GEMM
static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const void * src1_f16 = nullptr, const bool accumulate = false) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_type compute_type = ggml_cuda_mul_mat_cublas_compute_type(cc, src0, src1, dst);
    GGML_ASSERT(src1_f16 == nullptr || compute_type == GGML_TYPE_F16);
    GGML_ASSERT(!accumulate || ggml_cuda_mul_mat_cublas_can_accumulate(cc, src0, src1, dst));

    constexpr size_t max_src0_convert_size = 512ull * 1024 * 1024;
    const size_t src0_f32_size = ggml_nelements(src0) * sizeof(float);

    if (compute_type == GGML_TYPE_F32 &&
            (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
            src0_f32_size > max_src0_convert_size) {
        const size_t f32_row_size = src0_f32_size / src0->ne[1];
        const int64_t rows_per_chunk = std::max<int64_t>(1, (int64_t) (max_src0_convert_size / f32_row_size));

        if (rows_per_chunk < src0->ne[1]) {
            ggml_tensor src0_chunk = *src0;
            ggml_tensor dst_chunk = *dst;

            for (int64_t i01 = 0; i01 < src0->ne[1]; i01 += rows_per_chunk) {
                src0_chunk.ne[1] = std::min(rows_per_chunk, src0->ne[1] - i01);
                src0_chunk.data = (char *) src0->data + i01*src0->nb[1];

                dst_chunk.ne[0] = src0_chunk.ne[1];
                dst_chunk.data = (char *) dst->data + i01*dst->nb[0];

                ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, &src0_chunk, src1, &dst_chunk);
            }
            return;
        }
    }

    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst, src1_f16, accumulate);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
#if defined(GGML_USE_HIP)
        // IQ3_S + IQ3_XXS gate/up pairs: fused by the single-token iq3gemv kernel (mmvq-iq3.cu), which implements only
        // plain (non-swapped) SwiGLU; every other GLU op would reach the generic kernel with the wrong gate type
        if (!(is_mul_mat && !has_bias && !has_scale && ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
              ggml_get_op_params_i32(glu, 1) == 0 && ggml_cuda_mmvq_iq3_fuse_mixed_ok(ffn_up->src[0], ffn_gate->src[0]))) {
            return false;
        }
#else
        return false;
#endif
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc        = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }

    return use_mul_mat_vec_q;
}

static bool ggml_cuda_match_shared_expert(const ggml_cgraph * graph, int routed_idx, int shared_idx) {
    if (routed_idx + 2 >= graph->n_nodes || shared_idx + 2 >= graph->n_nodes || shared_idx < routed_idx + 3) {
        return false;
    }
    const int nodes[] = { routed_idx, routed_idx + 1, routed_idx + 2, shared_idx, shared_idx + 1, shared_idx + 2 };
    const ggml_op ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU,
                           GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU };
    const int outputs[] = { routed_idx + 2, shared_idx + 2 };
    if (!ggml_can_fuse_subgraph_ext(graph, nodes, 6, ops, outputs, 2)) {
        return false;
    }

    const ggml_tensor * routed = graph->nodes[routed_idx + 2];
    const ggml_tensor * shared = graph->nodes[shared_idx + 2];
    const ggml_tensor * gate = routed->src[0];
    const ggml_tensor * up = routed->src[1];
    const ggml_tensor * shared_gate = shared->src[0];
    const ggml_tensor * shared_up = shared->src[1];
    const auto is_pair = [&](const ggml_tensor * a, const ggml_tensor * b, int idx) {
        return (a == graph->nodes[idx] && b == graph->nodes[idx + 1]) ||
               (b == graph->nodes[idx] && a == graph->nodes[idx + 1]);
    };
    // only batch-size independent checks here: graph_optimize must produce the same graph topology for every ubatch
    // size, otherwise ggml-alloc has to re-reserve (and the scheduler to synchronize) at runtime.
    // the MMVQ batch size check is done in ggml_cuda_try_fuse
    if (!is_pair(gate, up, routed_idx) || !is_pair(shared_gate, shared_up, shared_idx) ||
            !ggml_cuda_should_fuse_mul_mat(up, gate, routed) ||
            !ggml_cuda_should_fuse_mul_mat(shared_up, shared_gate, shared) ||
            !up->src[0]->buffer ||
            !ggml_is_quantized(up->src[0]->type)) {
        return false;
    }
    const ggml_tensor * input = up->src[1];
    const ggml_tensor * weight = up->src[0];
    const ggml_tensor * shared_weight = shared_up->src[0];
    if (input->op != GGML_OP_RESHAPE || input->src[0] != shared_up->src[1] ||
            input->ne[1] != 1 || input->ne[3] != 1 || !ggml_is_contiguous(input) ||
            !ggml_is_contiguous(shared_up->src[1]) || !ggml_is_matrix(shared_up->src[1]) ||
            // same type for routed up, shared up and shared gate: the shared-expert kernel decodes all of them with one
            // type (the mixed IQ3 gate/up exception of ggml_cuda_should_fuse_mul_mat must not leak into this match)
            weight->type != shared_weight->type || shared_gate->src[0]->type != shared_weight->type ||
            weight->ne[0] != shared_weight->ne[0] ||
            weight->ne[1] != shared_weight->ne[1] || weight->nb[1] != shared_weight->nb[1] || weight->ne[3] != 1 ||
            !ggml_is_matrix(shared_weight) || !ggml_is_contiguous(shared_weight) ||
            !ggml_is_contiguous(shared_gate->src[0]) || !ggml_is_contiguous(routed) || !ggml_is_contiguous(shared)) {
        return false;
    }
    if (shared_weight->op != GGML_OP_NONE || shared_gate->src[0]->op != GGML_OP_NONE ||
            ggml_get_glu_op(routed) != ggml_get_glu_op(shared) ||
            ggml_get_op_params_f32(routed, 3) != ggml_get_op_params_f32(shared, 3)) {
        return false;
    }
    return true;
}

// Buffer- and batch-size independent, so graph_optimize produces the same graph topology for every ubatch size (see #29986).
static bool ggml_cuda_mul_mat_q_fusion_matches(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu, const int device) {
    const ggml_tensor * x_up = up->src[0];
    const ggml_tensor * y    = up->src[1];

    if ((up->op != GGML_OP_MUL_MAT && up->op != GGML_OP_MUL_MAT_ID) || !ggml_cuda_mmq_fusion_supported(x_up->type) || y->type != GGML_TYPE_F32) {
        return false;
    }

    // The fused kernel is only compiled for weights whose rows fill whole tiles (no fallback).
    if (ggml_cuda_mmq_needs_fallback(x_up->ne[1])) {
        return false;
    }

    if (!ggml_cuda_should_fuse_mul_mat(up, gate, glu)) {
        return false;
    }

    // The fused write-back applies SWIGLU_OAI with the default alpha and limit of ggml_cuda_op_swiglu_oai_single.
    if (ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU_OAI && (ggml_get_op_params_f32(glu, 2) != 1.702f || ggml_get_op_params_f32(glu, 3) != 7.0f)) {
        return false;
    }

    // NVIDIA only: the fused write-back relies on the MMA accumulator layout.
    return turing_mma_available(ggml_cuda_info().devices[device].cc);
}

static bool ggml_cuda_should_fuse_mul_mat_q(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu) {
    // A hinted matmul can be fused, but its unfused path may be faster (e.g. FWHT for GGML_HINT_SRC0_IS_HADAMARD).
    if (ggml_get_op_params_i32(up, 1) != GGML_HINT_NONE || ggml_get_op_params_i32(gate, 1) != GGML_HINT_NONE) {
        return false;
    }

    // The padding of a compute-buffer weight that is a view cannot be cleared, see ggml_cuda_mul_mat.
    const auto bad_padding_clear = [](const ggml_tensor * x) {
        return ggml_backend_buffer_get_usage(x->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            ggml_nbytes(x) != ggml_backend_buffer_get_alloc_size(x->buffer, x) && x->view_src;
    };
    const int device = ggml_cuda_get_device();
    if (!ggml_cuda_mul_mat_q_fusion_matches(up, gate, glu, device) || bad_padding_clear(up->src[0]) || bad_padding_clear(gate->src[0])) {
        return false;
    }

    // The batch size checks are done here, not in ggml_cuda_mul_mat_q_fusion_matches.
    const ggml_tensor * x_up = up->src[0];
    const ggml_tensor * y    = up->src[1];
    const int           cc   = ggml_cuda_info().devices[device].cc;

    // For MUL_MAT_ID: same kernel choice as ggml_cuda_mul_mat_id, with the nominal tiles of all experts (the per-expert token counts are only known on the GPU).
    const bool    is_id     = up->op == GGML_OP_MUL_MAT_ID;
    const int64_t ncols     = is_id ? y->ne[2]    : y->ne[1];
    const int64_t nchannels = is_id ? x_up->ne[2] : y->ne[2]*y->ne[3];
    const bool    use_mmvq  = is_id ? ncols <= get_mmvq_mmid_max_batch(x_up->type, cc) : ggml_cuda_should_use_mmvq(x_up->type, cc, ncols);
    if (use_mmvq || !ggml_cuda_should_use_mmq(x_up->type, cc, ncols, /*n_experts =*/ is_id ? x_up->ne[2] : 0)) {
        return false;
    }

    // The fused kernel needs one block per tile (no stream-k split), so only fuse if the tiling is efficient.
    return ggml_cuda_mmq_fusion_is_efficient(x_up->type, x_up->ne[1], ncols, nchannels, device);
}

// The kernel family ggml_cuda_mul_mat uses for a MUL_MAT (after the optional FWHT fast path). This is the one place
// that decides it: ggml_cuda_mul_mat dispatches on the result, and the fusions / the f16 copy cache that depend on a
// MUL_MAT being a cuBLAS GEMM ask ggml_cuda_mul_mat_cublas_gemm_type, so they cannot drift apart.
enum ggml_cuda_mul_mat_route {
    GGML_CUDA_MUL_MAT_ROUTE_CUBLAS,
    GGML_CUDA_MUL_MAT_ROUTE_MMVF,
    GGML_CUDA_MUL_MAT_ROUTE_MMVF_TRANSPOSED, // src0 is a row vector: mul_mat_vec_f with the operands swapped
    GGML_CUDA_MUL_MAT_ROUTE_MMF,
    GGML_CUDA_MUL_MAT_ROUTE_MMVQ,
    GGML_CUDA_MUL_MAT_ROUTE_MMQ,
};

static ggml_cuda_mul_mat_route ggml_cuda_mul_mat_get_route(const int device, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return GGML_CUDA_MUL_MAT_ROUTE_CUBLAS;
    }

    const int     cc        = ggml_cuda_info().devices[device].cc;
    const int     warp_size = ggml_cuda_info().devices[device].warp_size;
    const int64_t ne01      = src0->ne[1];
    const int64_t ne11      = src1->ne[1];

    if (ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        return GGML_CUDA_MUL_MAT_ROUTE_MMVF;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && dst->ne[2] == 1 && dst->ne[3] == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, warp_size, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        return GGML_CUDA_MUL_MAT_ROUTE_MMVF_TRANSPOSED;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        return GGML_CUDA_MUL_MAT_ROUTE_MMF;
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        return GGML_CUDA_MUL_MAT_ROUTE_MMVQ;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        return GGML_CUDA_MUL_MAT_ROUTE_MMQ;
    }
    return GGML_CUDA_MUL_MAT_ROUTE_CUBLAS;
}

// The compute type of the cuBLAS GEMM that ggml_cuda_mul_mat runs for the MUL_MAT node dst, or GGML_TYPE_COUNT if
// the node may take another path (FWHT or one of the custom kernels).
static ggml_type ggml_cuda_mul_mat_cublas_gemm_type(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (ggml_cuda_op_mul_mat_use_fwht(dst) ||
            ggml_cuda_mul_mat_get_route(ctx.device, src0, src1, dst) != GGML_CUDA_MUL_MAT_ROUTE_CUBLAS) {
        return GGML_TYPE_COUNT;
    }
    return ggml_cuda_mul_mat_cublas_compute_type(ggml_cuda_info().devices[ctx.device].cc, src0, src1, dst);
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (ggml_cuda_op_mul_mat_use_fwht(dst) && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }

    switch (ggml_cuda_mul_mat_get_route(ctx.device, src0, src1, dst)) {
        case GGML_CUDA_MUL_MAT_ROUTE_MMVF:
            ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
            break;
        case GGML_CUDA_MUL_MAT_ROUTE_MMVF_TRANSPOSED: {
            const int64_t ne11 = src1->ne[1];
            ggml_tensor dst_vec = *dst;
            dst_vec.ne[0] = ne11;
            dst_vec.ne[1] = 1;
            dst_vec.nb[1] = dst_vec.nb[0]*ne11;
            dst_vec.nb[2] = dst_vec.nb[1];
            dst_vec.nb[3] = dst_vec.nb[1];
            ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
        } break;
        case GGML_CUDA_MUL_MAT_ROUTE_MMF:
            ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
            break;
        case GGML_CUDA_MUL_MAT_ROUTE_MMVQ:
            ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
            break;
        case GGML_CUDA_MUL_MAT_ROUTE_MMQ:
            ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
            break;
        case GGML_CUDA_MUL_MAT_ROUTE_CUBLAS:
            ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
            break;
    }
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return true;
    }

    if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        if (ggml_is_quantized(src0->type)) {
            if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                return false;
            }
        } else if (GGML_CUDA_CC_IS_AMD(cc)) {
            return false;
        }
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
        return false;
    }

    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        return false;
    }

    return true;
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                if (ne2 <= mmvq_mmid_max) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }
            } else {
                if (GGML_CUDA_CC_IS_AMD(cc)) {
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                    return;
                }
            }
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc));
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        memcpy(dst_slice.op_params, dst->op_params, sizeof(dst_slice.op_params));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    if (!ggml_cuda_swiglu_q8(ctx, dst)) {
                        ggml_cuda_op_swiglu(ctx, dst);
                    }
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cuda_op_swiglu_clamp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_3D:
            ggml_cuda_op_conv3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_cuda_op_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_GDN_DECODE_PREP:
            ggml_cuda_op_gdn_decode_prep(ctx, dst);
            break;
        case GGML_OP_GDN_DECODE_OUT:
#if defined(GGML_USE_HIP)
            {
                // also produce the q8_1 activation of the following single-token GEMV (ssm_out)
                static const bool q8_out = [] {
                    const char * v = getenv("GGML_Q8_PRODUCERS");
                    const char * g = getenv("GGML_GDN_DECODE_Q8");
                    return (v == nullptr || v[0] != '0') && (g == nullptr || g[0] != '0');
                }();
                block_q8_1 * q8      = nullptr;
                int        * q8_sums = nullptr;
                if (q8_out && ggml_cuda_info().devices[ctx.device].warp_size == 64 &&
                        ggml_cuda_mmvq_cache_produce(ctx, dst, &q8, &q8_sums)) {
                    ggml_cuda_op_gdn_decode_out_q8(ctx, dst, q8, q8_sums);
                } else {
                    ggml_cuda_op_gdn_decode_out(ctx, dst);
                }
            }
#else
            ggml_cuda_op_gdn_decode_out(ctx, dst);
#endif // defined(GGML_USE_HIP)
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_lightning_indexer(ctx, dst);
            break;
        case GGML_OP_TOP_K_MASK:
            ggml_cuda_op_top_k_mask(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

// Batched small host->device uploads: the decode inputs (~10 tensors of 4 B .. 20 KB per token) were one async copy
// each, and every small H2D copy costs several us of GPU time (a blit kernel or a DMA transfer). Small
// set_tensor_async uploads are now staged (memcpy into pinned host memory, so the source may be reused right away)
// and flushed as one H2D copy plus one scatter kernel before the next operation on the stream.
// GGML_CUDA_NO_UPLOAD_BATCH=1 disables it.
struct ggml_cuda_upload_entry {
    void *   dst;
    uint32_t size;
    uint32_t off;
};

struct ggml_cuda_upload_batch {
    static constexpr size_t   max_entries = 64;
    static constexpr size_t   table_bytes = max_entries*sizeof(ggml_cuda_upload_entry);
    static constexpr size_t   capacity    = 256*1024; // staging bytes (table + data)
    static constexpr uint32_t max_upload  = 64*1024;  // larger uploads keep the direct copy

    uint8_t *   host      = nullptr; // pinned
    uint8_t *   dev       = nullptr;
    cudaEvent_t done      = nullptr; // last flush finished reading the host staging
    bool        in_flight = false;
    bool        failed    = false;
    size_t      used      = 0;
    std::vector<ggml_cuda_upload_entry> entries;
};

// experiment: GGML_CUDA_UPLOAD_BATCH_ALTERNATE=1 batches only every other graph compute (in-process A/B)
static int ggml_cuda_upload_alternate_counter = 0;
static bool ggml_cuda_upload_batch_disabled() {
    static const bool disabled  = getenv("GGML_CUDA_NO_UPLOAD_BATCH") != nullptr;
    static const bool alternate = getenv("GGML_CUDA_UPLOAD_BATCH_ALTERNATE") != nullptr;
    return disabled || (alternate && (ggml_cuda_upload_alternate_counter & 1));
}

static std::mutex ggml_cuda_upload_batch_mutex;
static std::unordered_map<const ggml_backend_cuda_context *, ggml_cuda_upload_batch> ggml_cuda_upload_batches;

static ggml_cuda_upload_batch * ggml_cuda_get_upload_batch(const ggml_backend_cuda_context * ctx, const bool create) {
    std::lock_guard<std::mutex> lock(ggml_cuda_upload_batch_mutex);
    auto it = ggml_cuda_upload_batches.find(ctx);
    if (it != ggml_cuda_upload_batches.end()) {
        return &it->second;
    }
    return create ? &ggml_cuda_upload_batches[ctx] : nullptr;
}

static __global__ void k_upload_scatter(const uint8_t * __restrict__ staging, const int table_bytes) {
    const ggml_cuda_upload_entry e = ((const ggml_cuda_upload_entry *) staging)[blockIdx.x];
    const uint8_t * src = staging + table_bytes + e.off;
    uint8_t       * dst = (uint8_t *) e.dst;
    if (((uintptr_t) dst % 16) == 0 && e.size % 16 == 0) {
        for (uint32_t i = 16*threadIdx.x; i < e.size; i += 16*blockDim.x) {
            *(uint4 *) (dst + i) = *(const uint4 *) (src + i);
        }
    } else {
        for (uint32_t i = threadIdx.x; i < e.size; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
}

static void ggml_cuda_upload_flush(ggml_backend_cuda_context * ctx) {
    ggml_cuda_upload_batch * ub = ggml_cuda_get_upload_batch(ctx, false);
    if (ub == nullptr || ub->entries.empty()) {
        return;
    }
    ggml_cuda_set_device(ctx->device);
    memcpy(ub->host, ub->entries.data(), ub->entries.size()*sizeof(ggml_cuda_upload_entry));
    cudaStream_t stream = ctx->stream();
    CUDA_CHECK(cudaMemcpyAsync(ub->dev, ub->host, ggml_cuda_upload_batch::table_bytes + ub->used, cudaMemcpyHostToDevice, stream));
    k_upload_scatter<<<(unsigned) ub->entries.size(), 256, 0, stream>>>(ub->dev, (int) ggml_cuda_upload_batch::table_bytes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ub->done, stream));
    ub->in_flight = true;
    ub->entries.clear();
    ub->used = 0;
}

// stage a small upload, false if it has to be done directly
static bool ggml_cuda_upload_stage(ggml_backend_cuda_context * ctx, void * dst, const void * data, const size_t size) {
    if (ggml_cuda_upload_batch_disabled() || size == 0 || size > ggml_cuda_upload_batch::max_upload) {
        return false;
    }
    ggml_cuda_upload_batch * ub = ggml_cuda_get_upload_batch(ctx, true);
    if (ub->failed) {
        return false;
    }
    if (ub->host == nullptr) {
        ggml_cuda_set_device(ctx->device);
        if (cudaMallocHost((void **) &ub->host, ggml_cuda_upload_batch::capacity) != cudaSuccess ||
            cudaMalloc((void **) &ub->dev, ggml_cuda_upload_batch::capacity) != cudaSuccess ||
            cudaEventCreateWithFlags(&ub->done, cudaEventDisableTiming) != cudaSuccess) {
            (void) cudaGetLastError();
            ub->failed = true;
            return false;
        }
    }
    const size_t size_al = GGML_PAD(size, 16);
    if (ub->entries.size() == ggml_cuda_upload_batch::max_entries ||
        ggml_cuda_upload_batch::table_bytes + ub->used + size_al > ggml_cuda_upload_batch::capacity) {
        ggml_cuda_upload_flush(ctx);
    }
    if (ub->in_flight) {
        // the previous batch may still be reading the host staging (normally long finished)
        CUDA_CHECK(cudaEventSynchronize(ub->done));
        ub->in_flight = false;
    }
    memcpy(ub->host + ggml_cuda_upload_batch::table_bytes + ub->used, data, size);
    ub->entries.push_back({ dst, (uint32_t) size, (uint32_t) ub->used });
    ub->used += size_al;
    return true;
}

static void ggml_cuda_upload_batch_free(ggml_backend_cuda_context * ctx) {
    std::lock_guard<std::mutex> lock(ggml_cuda_upload_batch_mutex);
    auto it = ggml_cuda_upload_batches.find(ctx);
    if (it == ggml_cuda_upload_batches.end()) {
        return;
    }
    ggml_cuda_upload_batch & ub = it->second;
    if (ub.host != nullptr) {
        ggml_cuda_set_device(ctx->device);
        if (ub.in_flight) {
            (void) cudaEventSynchronize(ub.done);
        }
        (void) cudaFreeHost(ub.host);
        (void) cudaFree(ub.dev);
        (void) cudaEventDestroy(ub.done);
    }
    ggml_cuda_upload_batches.erase(it);
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_upload_flush(cuda_ctx);
    ggml_cuda_upload_batch_free(cuda_ctx);
    delete cuda_ctx;
    delete backend;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    if (ggml_cuda_upload_stage(cuda_ctx, (char *) tensor->data + offset, data, size)) {
        return;
    }
    ggml_cuda_upload_flush(cuda_ctx);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_upload_flush(cuda_ctx);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_upload_flush(cuda_ctx);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_upload_flush(cuda_ctx);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;
    ggml_cuda_upload_flush(cuda_ctx_src);
    ggml_cuda_upload_flush(cuda_ctx_dst);

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_ctx_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(cuda_ctx_dst->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_cuda_upload_flush(cuda_ctx);

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_debug_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_GRAPH_DEBUG");
    return enabled;
}

static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
                if (ggml_cuda_graph_debug_enabled()) {
                    GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
                }
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    return cgraph->nodes[0];
}

static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    bool res = false;

    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        if (ggml_cuda_graph_debug_enabled()) {
            GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        }
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    // Check if the graph size has changed
    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            graph->node_props[i] = prop;
            res = true;
        }
    }

    return res;
}

static void ggml_cuda_graph_update_executable(cudaGraphExec_t & instance, cudaGraph_t cuda_graph) {
#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(instance, cuda_graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(instance, cuda_graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
        if (ggml_cuda_graph_debug_enabled()) {
            GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
        }

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(instance));
        instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&instance, cuda_graph, nullptr, nullptr, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}

// number of ggml nodes per CUDA graph chunk, 0 captures the whole ggml graph as one CUDA graph
static int ggml_cuda_graph_chunk_size() {
    static const int chunk_size = [] {
        const char * env = getenv("GGML_CUDA_GRAPH_CHUNK_SIZE");
        return env != nullptr ? atoi(env) : 512;
    }();
    return chunk_size;
}

// end the capture of chunk `idx`, update or create its executable and launch it
static void ggml_cuda_graph_launch_chunk(ggml_backend_cuda_context * cuda_ctx, ggml_cuda_graph * graph, size_t idx) {
    cudaGraph_t cuda_graph = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &cuda_graph));

    if (idx < graph->instances.size()) {
        ggml_cuda_graph_update_executable(graph->instances[idx], cuda_graph);
    } else {
        GGML_ASSERT(idx == graph->instances.size());
        graph->instances.emplace_back(nullptr);
        CUDA_CHECK(cudaGraphInstantiate(&graph->instances[idx], cuda_graph, nullptr, nullptr, 0));
    }
    CUDA_CHECK(cudaGraphDestroy(cuda_graph));

    CUDA_CHECK(cudaGraphLaunch(graph->instances[idx], cuda_ctx->stream()));
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox/multi shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE && mode != GGML_ROPE_TYPE_IMROPE) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles the norm/neox/mrope/imrope rope modes (not vision)
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE && mode != GGML_ROPE_TYPE_IMROPE) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    if (mode & GGML_ROPE_TYPE_MROPE) {
        static const bool no_mrope = getenv("GGML_CUDA_FUSED_ROPE_NO_MROPE") != nullptr;
        const int32_t * sections = (const int32_t *) rope->op_params + 11;
        if (no_mrope || sections[0] + sections[1] + sections[2] + sections[3] <= 0) {
            return false;
        }
    }

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0) {
        return false;
    }

    return true;
}

static bool ggml_cuda_attn_prep_fusion_disabled() {
    static const bool disabled = getenv("GGML_CUDA_DISABLE_ATTN_PREP_FUSION") != nullptr;
    return disabled;
}

// [data, data + size) of the allocation backing t (the root of a view chain)
static void ggml_cuda_tensor_root_range(const ggml_tensor * t, uintptr_t & lo, uintptr_t & hi) {
    while (t->view_src) {
        t = t->view_src;
    }
    lo = (uintptr_t) t->data;
    hi = lo + ggml_nbytes(t);
}

static bool ggml_cuda_ranges_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    uintptr_t a0, a1, b0, b1;
    ggml_cuda_tensor_root_range(a, a0, a1);
    ggml_cuda_tensor_root_range(b, b0, b1);
    return a0 < b1 && b0 < a1;
}

// Fused full-attention prep (qwen3.5/qwen3-next): matches, starting at node i, the real-node sequence
//   RMS_NORM(q view) MUL(w_q) ROPE(q) CONT(gate view) RMS_NORM(k) MUL(w_k) ROPE(k) SET_ROWS(view(rope k)) SET_ROWS(v)
// (views/no-ops in between are skipped). Fills nodes[9] and returns the number of nodes to skip, 0 if no match.
// The kernel reads all inputs and writes all outputs concurrently, so the allocated outputs (rope q, gate) must not
// share memory with any input; the model graph emits the CONT right after the q rope so that the q|gate projection
// (consumed by the CONT) and the k/v projections are still alive when those outputs are allocated.
static int ggml_cuda_try_attn_prep_fusion(const ggml_cgraph * cgraph, const int i, ggml_tensor ** nodes) {
    if (ggml_cuda_attn_prep_fusion_disabled()) {
        return 0;
    }

    static const ggml_op seq[9] = {
        GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_CONT,
        GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_SET_ROWS, GGML_OP_SET_ROWS,
    };

    int idx[9];
    int n = 0;
    for (int j = i; j < cgraph->n_nodes && n < 9; ++j) {
        ggml_tensor * node = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }
        if (node->op != seq[n]) {
            return 0;
        }
        idx[n]   = j;
        nodes[n] = node;
        n++;
    }
    if (n != 9) {
        return 0;
    }

    const ggml_tensor * rms_q  = nodes[0];
    const ggml_tensor * mul_q  = nodes[1];
    const ggml_tensor * rope_q = nodes[2];
    const ggml_tensor * cont   = nodes[3];
    const ggml_tensor * rms_k  = nodes[4];
    const ggml_tensor * mul_k  = nodes[5];
    const ggml_tensor * rope_k = nodes[6];
    const ggml_tensor * sr_k   = nodes[7];
    const ggml_tensor * sr_v   = nodes[8];

    // data flow
    if ((mul_q->src[0] != rms_q && mul_q->src[1] != rms_q) || rope_q->src[0] != mul_q ||
        (mul_k->src[0] != rms_k && mul_k->src[1] != rms_k) || rope_k->src[0] != mul_k) {
        return 0;
    }
    const ggml_tensor * wq = mul_q->src[0] == rms_q ? mul_q->src[1] : mul_q->src[0];
    const ggml_tensor * wk = mul_k->src[0] == rms_k ? mul_k->src[1] : mul_k->src[0];
    const ggml_tensor * q  = rms_q->src[0];
    const ggml_tensor * k  = rms_k->src[0];
    const ggml_tensor * g  = cont->src[0];
    const ggml_tensor * kv = sr_k->src[0];
    const ggml_tensor * v  = sr_v->src[0];

    // the k store must read the k rope output flattened over the heads
    if (kv->op != GGML_OP_VIEW || kv->view_src != rope_k || kv->view_offs != 0 || !ggml_is_contiguous(kv) ||
        !ggml_is_contiguous(rope_k) || kv->ne[0] != rope_k->ne[0]*rope_k->ne[1] || kv->ne[1] != rope_k->ne[2] ||
        kv->ne[2] != 1 || kv->ne[3] != 1) {
        return 0;
    }

    // shapes and types
    const int64_t hd = q->ne[0];
    const int64_t nh = q->ne[1];
    const int64_t nt = q->ne[2];
    const int64_t nkv = k->ne[1];
    if (!ggml_cuda_attn_prep_supported(hd, sr_k->type, sr_v->type)) {
        return 0;
    }
    for (const ggml_tensor * t : { q, k, g, v, wq, wk, (const ggml_tensor *) rms_q, mul_q, rope_q, cont, rms_k, mul_k, rope_k }) {
        if (t->type != GGML_TYPE_F32) {
            return 0;
        }
    }
    if (q->ne[3] != 1 || k->ne[0] != hd || k->ne[2] != nt || k->ne[3] != 1 ||
        g->ne[0] != hd || g->ne[1] != nh || g->ne[2] != nt || g->ne[3] != 1 ||
        v->ne[0] != hd*nkv || v->ne[1] != nt || v->ne[2] != 1 || v->ne[3] != 1 ||
        ggml_nelements(wq) != hd || ggml_nelements(wk) != hd || !ggml_is_contiguous(wq) || !ggml_is_contiguous(wk) ||
        !ggml_are_same_shape(rms_q, q) || !ggml_are_same_shape(mul_q, q) || !ggml_are_same_shape(rope_q, q) ||
        !ggml_are_same_shape(rms_k, k) || !ggml_are_same_shape(mul_k, k) || !ggml_are_same_shape(rope_k, k) ||
        !ggml_is_contiguous(cont) || cont->ne[0] != hd*nh || cont->ne[1] != nt) {
        return 0;
    }
    if (nkv > nh) {
        return 0;
    }
    // element-contiguous rows, 16-byte aligned vector access for every head row
    const auto aligned = [](const ggml_tensor * t) {
        return t->nb[0] == ggml_type_size(t->type) && ((uintptr_t) t->data % 16) == 0 &&
            t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0;
    };
    for (const ggml_tensor * t : { q, k, g, v, wq, wk, rope_q, cont }) {
        if (!aligned(t)) {
            return 0;
        }
    }
    // cache rows: [hd*nkv] per row, 2d destination, one i64 index per token
    for (const ggml_tensor * sr : { sr_k, sr_v }) {
        if (sr->ne[0] != hd*nkv || sr->ne[2] != 1 || sr->ne[3] != 1 || sr->nb[0] != ggml_type_size(sr->type) ||
            ((uintptr_t) sr->data % 16) != 0 || sr->nb[1] % 16 != 0 ||
            sr->src[1]->type != GGML_TYPE_I64 || sr->src[1]->ne[0] != nt || !ggml_is_contiguous(sr->src[1])) {
            return 0;
        }
    }

    // both ropes must be identical (positions, params) and supported by the fused kernel
    if (rope_q->src[1] != rope_k->src[1] || rope_q->src[2] != rope_k->src[2] ||
        memcmp(rope_q->op_params, rope_k->op_params, 16*sizeof(int32_t)) != 0) {
        return 0;
    }
    if (!ggml_cuda_should_fuse_rms_norm_mul_rope(rms_q, mul_q, rope_q) ||
        !ggml_cuda_should_fuse_rms_norm_mul_rope(rms_k, mul_k, rope_k)) {
        return 0;
    }
    const int n_dims = ((const int32_t *) rope_q->op_params)[1];
    if (n_dims > hd) {
        return 0;
    }

    // intermediates (norms, muls, k rope and its view) must not be used outside of the fused nodes
    int kv_idx = -1;
    for (int j = idx[6] + 1; j < idx[7]; ++j) {
        if (cgraph->nodes[j] == kv) {
            kv_idx = j;
            break;
        }
    }
    if (kv_idx < 0) {
        return 0;
    }
    const int     sub_idx[10] = { idx[0], idx[1], idx[2], idx[3], idx[4], idx[5], idx[6], kv_idx, idx[7], idx[8] };
    const ggml_op sub_ops[10] = {
        GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_CONT, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE,
        GGML_OP_VIEW, GGML_OP_SET_ROWS, GGML_OP_SET_ROWS,
    };
    const int     outs[4] = { idx[2], idx[3], idx[7], idx[8] };
    if (!ggml_can_fuse_subgraph_ext(cgraph, sub_idx, 10, sub_ops, outs, 4)) {
        return 0;
    }

    // memory hazards: outputs written by the kernel vs everything it reads
    const ggml_tensor * inputs[] = { q, k, g, v, wq, wk, rope_q->src[1], sr_k->src[1], sr_v->src[1], rope_q->src[2] };
    const ggml_tensor * outputs[] = { rope_q, cont, sr_k, sr_v };
    for (const ggml_tensor * o : outputs) {
        for (const ggml_tensor * in : inputs) {
            if (in != nullptr && ggml_cuda_ranges_overlap(o, in)) {
                return 0;
            }
        }
    }
    if (ggml_cuda_ranges_overlap(rope_q, cont)) {
        return 0;
    }

    return idx[8] - i;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// match ssm_scan + the strided cpy that scatters its state snapshots into the cache, so the kernel writes them and skips the cpy
static int ggml_cuda_try_ssm_scan_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_ssm_scan_fused_cache & fused_state_cpy) {
    const ggml_tensor * ssm = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the scan output must not be a graph output
    if (ssm->op != GGML_OP_SSM_SCAN || ssm->type != GGML_TYPE_F32 || (ssm->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const int64_t K = ggml_get_op_params_i32(ssm, 0); // snapshot slot count

    const ggml_tensor * s = ssm->src[0];
    const ggml_tensor * x = ssm->src[1];
    const ggml_tensor * A = ssm->src[3];

    const int64_t d_state = s->ne[0];
    const int64_t D       = d_state * s->ne[1] * x->ne[1]; // d_state * head_dim * n_head
    const int64_t n_tok   = x->ne[2];
    const int64_t n_seqs  = x->ne[3];

    // only the mamba-2 kernels (group scan and SSD) write to the cache; mamba-1 still uses the cpy
    if (A->nb[1] != sizeof(float) || (d_state != 96 && d_state != 128 && d_state != 256)) {
        return 0;
    }

    // the scan reads its input rows from the cache (picked by ids), so with more than one seq a seq can read a row that another seq writes in the same launch
    if (n_seqs != 1) {
        return 0;
    }

    const int64_t n_written = std::min<int64_t>(n_tok, K);
    const size_t  tail_off  = ggml_row_size(GGML_TYPE_F32, ggml_nelements(x));

    // snapshot cpy is the first real node after the scan (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the scan snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this scan's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != ssm || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D, the per-seq stride the kernel takes from src0->nb[3]
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback slot 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// expert group selection from build_moe_ffn (n_expert_groups > 1)
// src0/src1 index into this table, -1 is the (biased) probs, -2 is not checked
static const struct {
    ggml_op op;
    int     src0;
    int     src1;
} topk_moe_group_pattern[] = {
    { GGML_OP_RESHAPE,  -1, -2 }, //  0: [n_exp_per_group, n_expert_groups, n_tokens]
    { GGML_OP_RESHAPE,   0, -2 }, //  1: [1, n_exp_per_group, n_expert_groups, n_tokens]
    { GGML_OP_ARGSORT,   0, -2 }, //  2
    { GGML_OP_VIEW,      2, -2 }, //  3: top-2 per group
    { GGML_OP_GET_ROWS,  1,  3 }, //  4
    { GGML_OP_RESHAPE,   4, -2 }, //  5
    { GGML_OP_SUM_ROWS,  5, -2 }, //  6: group scores
    { GGML_OP_RESHAPE,   6, -2 }, //  7
    { GGML_OP_ARGSORT,   7, -2 }, //  8
    { GGML_OP_VIEW,      8, -2 }, //  9: top n_group_used groups
    { GGML_OP_GET_ROWS,  0,  9 }, // 10
    { GGML_OP_FILL,      0, -2 }, // 11
    { GGML_OP_SET_ROWS, 10,  9 }, // 12: unused groups -> -inf
    { GGML_OP_RESHAPE,  12, -2 }, // 13: [n_expert, n_tokens]
};

static bool ggml_cuda_topk_moe_group_fusion(const struct ggml_cgraph * cgraph, int node_idx, const ggml_tensor * selection, ggml_cuda_topk_moe_args & args) {
#if defined(GGML_USE_HIP) && HIP_VERSION < 60300000
    GGML_UNUSED_VARS(cgraph, node_idx, selection, args);
    return false;
#else
    const int n_ops = (int) std::size(topk_moe_group_pattern);
    if (node_idx + n_ops > cgraph->n_nodes) {
        return false;
    }

    ggml_tensor ** nodes = cgraph->nodes + node_idx;
    auto src_ok = [&](const ggml_tensor * src, int idx) {
        return idx == -2 || src == (idx == -1 ? selection : nodes[idx]);
    };

    for (int i = 0; i < n_ops; i++) {
        const auto & p = topk_moe_group_pattern[i];
        if (nodes[i]->op != p.op || !src_ok(nodes[i]->src[0], p.src0) || !src_ok(nodes[i]->src[1], p.src1)) {
            return false;
        }
    }

    const int n_groups     = nodes[0]->ne[1];
    const int n_group_used = nodes[9]->ne[0];

    if (nodes[3]->ne[0] != 2 || nodes[12]->src[2] != nodes[11] || ggml_get_op_params_f32(nodes[11], 0) != -INFINITY ||
            n_groups > WARP_SIZE) {
        return false;
    }

    args.n_groups     = n_groups;
    args.n_group_used = n_group_used;
    return true;
#endif // defined(GGML_USE_HIP) && HIP_VERSION < 60300000
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;
    args.n_groups        = 0;
    args.n_group_used    = 0;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        const ggml_tensor * selection = nodes[node_idx - 2];
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == selection) {
            args.prob_bias = true;
            selection = nodes[node_idx];
            node_idx++;
        }

        if (ggml_cuda_topk_moe_group_fusion(cgraph, node_idx, selection, args)) {
            node_idx += (int) std::size(topk_moe_group_pattern);
            selection = nodes[node_idx - 1];
        }

        // RESHAPE/ADD/group mask -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT || nodes[node_idx]->src[0] != selection) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }

        // kept groups must hold all used experts
        if (args.n_groups > 0 && nodes[node_idx]->ne[0] > args.n_group_used * (probs_reshaped->ne[1] / args.n_groups)) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}


static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    // scale_out*sigmoid(scale_in*post) as the hc_post weights (qwen4exp hc_combine)
    std::initializer_list<enum ggml_op> hc_post_gate_ops = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST };

    if (is_equal(hc_post_gate_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 3 })) {
        const ggml_tensor * scale_in  = cgraph->nodes[node_idx];
        const ggml_tensor * sigmoid   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scale_out = cgraph->nodes[node_idx + 2];

        // the chain must feed hc_post->src[2]; a bias on either scale is not handled
        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 2, 2}}) &&
            ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
            ggml_get_op_params_f32(scale_in, 1) == 0.0f && ggml_get_op_params_f32(scale_out, 1) == 0.0f &&
            scale_in->src[0]->type == GGML_TYPE_F32 && scale_out->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(scale_in->src[0], scale_out)) {
            int out_nodes[] = { node_idx + 3 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * scale    = cgraph->nodes[node_idx+1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        float bias;
        memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));

        return bias == 0.0f && scale->type == GGML_TYPE_F32;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16 && unary->type != GGML_TYPE_BF16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        // the fused softcap kernel is F32 only
        if (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            return false;
        }

        // TODO: make the fused softcap kernel stride aware, it reads src and writes dst as dense
        if (!ggml_is_contiguous(scale->src[0]) || !ggml_is_contiguous(scale2)) {
            return false;
        }

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

// try and fuse nodes and return the number of nodes to skip
// flash attention (single query) followed by sigmoid(gate) * attn (qwen3.5 gated attention): the gating and the q8_1 of
// the gated output for the attn_output GEMV are done by the FA combine kernel (one launch less per layer).
// Matches FLASH_ATTN_EXT, views, UNARY(SIGMOID, gate), MUL(view(FA), SIGMOID); returns the nodes to skip or 0.
static int ggml_cuda_try_fattn_gate_fusion(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const int i) {
    static const bool disabled = getenv("GGML_CUDA_DISABLE_FA_GATE_FUSION") != nullptr;
    if (disabled) {
        return 0;
    }
    ggml_tensor * fa = cgraph->nodes[i];
    const ggml_tensor * Q = fa->src[0];
    if (Q->ne[1] != 1 || Q->ne[3] != 1 || fa->type != GGML_TYPE_F32 || !ggml_is_contiguous(fa) ||
        (fa->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    int iu = -1;
    int im = -1;
    for (int j = i + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            // only views of the FA output may sit in between
            const ggml_tensor * r = n;
            while (r->view_src) {
                r = r->view_src;
            }
            if (r != fa || ggml_node_get_use_count(cgraph, j) != 1 || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                return 0;
            }
            continue;
        }
        if (iu < 0) {
            if (n->op != GGML_OP_UNARY || ggml_get_unary_op(n) != GGML_UNARY_OP_SIGMOID) {
                return 0;
            }
            iu = j;
            continue;
        }
        if (n->op != GGML_OP_MUL) {
            return 0;
        }
        im = j;
        break;
    }
    if (iu < 0 || im < 0) {
        return 0;
    }
    ggml_tensor * un  = cgraph->nodes[iu];
    ggml_tensor * mul = cgraph->nodes[im];
    const ggml_tensor * gate  = un->src[0];
    const ggml_tensor * other = mul->src[0] == un ? mul->src[1] : (mul->src[1] == un ? mul->src[0] : nullptr);
    if (other == nullptr) {
        return 0;
    }
    const ggml_tensor * root = other;
    while (root->view_src) {
        root = root->view_src;
    }
    const int64_t n = ggml_nelements(fa);
    if (root != fa || other->data != fa->data || !ggml_is_contiguous(other) || ggml_nelements(other) != n ||
        gate->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(gate) || !ggml_is_contiguous(un) || !ggml_is_contiguous(mul) ||
        ggml_nelements(gate) != n || ggml_nelements(un) != n || ggml_nelements(mul) != n ||
        ggml_node_get_use_count(cgraph, i) != 1 || ggml_node_get_use_count(cgraph, iu) != 1 ||
        (un->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    // short KV: one-launch split-KV decode with the gate + q8_1 epilogue (fattn-dec.cu)
    if (ggml_cuda_fattn_dec_gate(*cuda_ctx, fa, (const float *) gate->data, mul)) {
        return im - i;
    }

    ggml_cuda_fattn_gate_epilogue ep;
    ep.ctx      = cuda_ctx;
    ep.gate     = (const float *) gate->data;
    ep.mul_node = mul;
    ggml_cuda_fattn_epilogue = &ep;
    ggml_cuda_flash_attn_ext(*cuda_ctx, fa);
    ggml_cuda_fattn_epilogue = nullptr;
    if (!ep.applied) {
        // no combine step in this call: gate separately, exactly like the UNARY+MUL fusion below
        if (!ggml_cuda_unary_mul_q8(*cuda_ctx, un, mul)) {
            ggml_cuda_op_unary_mul(*cuda_ctx, un, mul);
        }
    }
    return im - i;
}

#if defined(GGML_USE_HIP)
// Single-token BF16 thin GEMV + elementwise epilogue chain (qwen35 ssm_beta / ssm_alpha):
//   MUL_MAT -> RESHAPE -> UNARY(SIGMOID)                          epi 1
//   MUL_MAT -> RESHAPE -> ADD(bias) -> UNARY(SOFTPLUS) -> MUL(a)  epi 2
// Returns the number of nodes of the chain starting at k (0: no match).
static int ggml_cuda_bf16_epi_chain(const ggml_cgraph * cgraph, const int k, const ggml_tensor * src1_req, ggml_cuda_bf16_epi_job & job) {
    if (k + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * mm = cgraph->nodes[k];
    if (mm->op != GGML_OP_MUL_MAT || mm->src[0]->type != GGML_TYPE_BF16 || mm->src[1]->type != GGML_TYPE_F32 ||
            mm->type != GGML_TYPE_F32 || mm->src[0]->ne[1] > 4096 || ggml_nelements(mm) != mm->src[0]->ne[1] ||
            ggml_nelements(mm->src[1]) != mm->src[0]->ne[0] || (src1_req && mm->src[1] != src1_req)) {
        return 0;
    }
    const int64_t N = mm->ne[0];
    const ggml_tensor * rs = cgraph->nodes[k + 1];
    if (rs->op != GGML_OP_RESHAPE || rs->src[0] != mm) {
        return 0;
    }
    // contiguous f32 with N elements (any shape, e.g. the [1, N] beta reshape)
    auto vec_ok = [N](const ggml_tensor * t) {
        return t != nullptr && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) == N;
    };
    // operands of the elementwise ADD / MUL: additionally ne0 == N, so there is no broadcasting
    auto row_ok = [N, &vec_ok](const ggml_tensor * t) {
        return vec_ok(t) && t->ne[0] == N;
    };
    ggml_tensor * n2 = cgraph->nodes[k + 2];
    if (n2->op == GGML_OP_UNARY && ggml_get_unary_op(n2) == GGML_UNARY_OP_SIGMOID && n2->src[0] == rs && vec_ok(n2)) {
        const int out = k + 2;
        if (!ggml_can_fuse_subgraph(cgraph, k, { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_UNARY }, { out }) ||
                !ggml_cuda_check_fusion_memory_ranges(cgraph, k, 3, &out, 1)) {
            return 0;
        }
        job = { mm->src[0], n2, nullptr, nullptr, 1 };
        return 3;
    }
    if (n2->op == GGML_OP_ADD && k + 4 < cgraph->n_nodes) {
        const ggml_tensor * b  = n2->src[0] == rs ? n2->src[1] : (n2->src[1] == rs ? n2->src[0] : nullptr);
        const ggml_tensor * n3 = cgraph->nodes[k + 3];
        ggml_tensor       * n4 = cgraph->nodes[k + 4];
        if (!row_ok(b) || !row_ok(n2) || n3->op != GGML_OP_UNARY || ggml_get_unary_op(n3) != GGML_UNARY_OP_SOFTPLUS ||
                n3->src[0] != n2 || !row_ok(n3) || n4->op != GGML_OP_MUL || !row_ok(n4)) {
            return 0;
        }
        const ggml_tensor * a = n4->src[0] == n3 ? n4->src[1] : (n4->src[1] == n3 ? n4->src[0] : nullptr);
        if (!row_ok(a)) {
            return 0;
        }
        const int out = k + 4;
        if (!ggml_can_fuse_subgraph(cgraph, k, { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL }, { out }) ||
                !ggml_cuda_check_fusion_memory_ranges(cgraph, k, 5, &out, 1)) {
            return 0;
        }
        job = { mm->src[0], n4, (const float *) b->data, (const float *) a->data, 2 };
        return 5;
    }
    return 0;
}

// Linear-attention gated output norm:
//   RMS_NORM(x) -> MUL(w) -> [views] -> MUL_MAT (z GEMV) -> [views] -> UNARY(SILU)(z) -> MUL(normed, silu(z))
// The z GEMV does not depend on the norm, so it is computed first; then one kernel computes
// silu(z) * (rms_norm(x) * w) and its q8_1 for the ssm_out GEMV (2 launches -> 1). Returns the nodes to skip.
static int ggml_cuda_try_fuse_gated_norm(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const int i) {
    const int n = cgraph->n_nodes;
    if (i + 4 >= n) {
        return 0;
    }
    ggml_tensor * norm = cgraph->nodes[i];
    ggml_tensor * mul  = cgraph->nodes[i + 1];
    if (norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || (mul->src[0] != norm && mul->src[1] != norm)) {
        return 0;
    }
    const ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    int j = i + 2;
    while (j < n && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
        j++;
    }
    if (j >= n || cgraph->nodes[j]->op != GGML_OP_MUL_MAT) {
        return 0;
    }
    ggml_tensor * mm = cgraph->nodes[j];
    j++;
    while (j < n && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
        j++;
    }
    if (j + 1 >= n) {
        return 0;
    }
    ggml_tensor * un = cgraph->nodes[j];
    ggml_tensor * gm = cgraph->nodes[j + 1];
    if (un->op != GGML_OP_UNARY || ggml_get_unary_op(un) != GGML_UNARY_OP_SILU || gm->op != GGML_OP_MUL ||
            !((gm->src[0] == mul && gm->src[1] == un) || (gm->src[0] == un && gm->src[1] == mul))) {
        return 0;
    }
    auto base = [](const ggml_tensor * t) {
        while (t->view_src) {
            t = t->view_src;
        }
        return t;
    };
    const ggml_tensor * z = un->src[0];
    if (base(z) != mm) {
        return 0;
    }
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        if (mm->src[s] && (base(mm->src[s]) == norm || base(mm->src[s]) == mul)) {
            return 0;
        }
    }
    const int flags = norm->flags | mul->flags | un->flags;
    if ((flags & GGML_TENSOR_FLAG_OUTPUT) || !(norm->flags & mul->flags & un->flags & gm->flags & mm->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            !ggml_node_has_n_uses(cgraph, i, 1) || !ggml_node_has_n_uses(cgraph, i + 1, 1) || !ggml_node_has_n_uses(cgraph, j, 1)) {
        return 0;
    }
    const ggml_tensor * x = norm->src[0];
    if (!ggml_cuda_rms_norm_gated_q8_supported(x, w, z, gm)) {
        return 0;
    }
    // memory: the early z GEMV must not overwrite x; dst may alias x / z only exactly (elementwise in place)
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data, * a1 = a0 + ggml_nbytes(a);
        const char * b0 = (const char *) b->data, * b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    if (overlap(mm, x) || (overlap(gm, x) && gm->data != x->data) || (overlap(gm, z) && gm->data != z->data)) {
        return 0;
    }
    if (!ggml_cuda_compute_forward(*cuda_ctx, mm)) {
        GGML_ABORT("%s: z GEMV %s failed", __func__, mm->name);
    }
    float eps;
    memcpy(&eps, norm->op_params, sizeof(float));
    ggml_cuda_rms_norm_gated_q8(*cuda_ctx, x, w, eps, z, gm);
    return j + 1 - i;
}

// one launch for the chain at i and, if the next non-view node starts another chain on the same activation, that one too
static int ggml_cuda_try_fuse_bf16_epi(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const int i) {
    ggml_cuda_bf16_epi_job jobs[2];
    const int l1 = ggml_cuda_bf16_epi_chain(cgraph, i, nullptr, jobs[0]);
    if (l1 == 0) {
        return 0;
    }
    int j = i + l1;
    while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
        j++;
    }
    const int l2 = j < cgraph->n_nodes ? ggml_cuda_bf16_epi_chain(cgraph, j, cgraph->nodes[i]->src[1], jobs[1]) : 0;
    if (!ggml_cuda_mmvf_bf16_thin_epi(*cuda_ctx, cgraph->nodes[i]->src[1], jobs, l2 > 0 ? 2 : 1)) {
        return 0;
    }
    return l2 > 0 ? j + l2 - 1 - i : l1 - 1;
}

// Single-token GEMVs that read the same activation back to back, with only views/no-ops in between (qkv + z of the
// linear-attention layers, q/k/v of the attention layers): one multi-output launch (ggml_cuda_mul_mat_vec_q_multi).
// The group is the longest prefix (2..3 GEMVs) the module accepts; returns the number of nodes to skip.
static int ggml_cuda_try_fuse_mmvq_multi(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const int i) {
    const ggml_tensor * node = cgraph->nodes[i];
    const ggml_tensor * src1 = node->src[1];
    if (src1->type != GGML_TYPE_F32 || src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || !ggml_is_quantized(node->src[0]->type)) {
        return 0;
    }
    auto is_gemv = [&](const ggml_tensor * t) {
        return t->op == GGML_OP_MUL_MAT && t->src[1] == src1 && (t->flags & GGML_TENSOR_FLAG_COMPUTE) &&
               t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) == t->src[0]->ne[1] &&
               t->src[0]->ne[0] == node->src[0]->ne[0];
    };
    if (!is_gemv(node)) {
        return 0;
    }
    const ggml_tensor * w[3] = { node->src[0], nullptr, nullptr };
    ggml_tensor *       d[3] = { cgraph->nodes[i], nullptr, nullptr };
    int idx[3] = { i, i, i };
    int n = 1;
    for (int j = i + 1; j < cgraph->n_nodes && n < 3; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(t)) {
            continue;
        }
        if (!is_gemv(t)) {
            break;
        }
        w[n] = t->src[0]; d[n] = t; idx[n] = j; n++;
    }
    for (; n >= 2; --n) {
        if (ggml_cuda_mul_mat_vec_q_multi(*cuda_ctx, w, src1, d, n)) {
            return idx[n - 1] - i;
        }
    }
    return 0;
}
#endif // defined(GGML_USE_HIP)

#if defined(GGML_USE_HIP)
// iq4k-r3: consecutive single-token MUL_MATs (only views/reshapes in between) that read the same activation and have
// IQ4_XS/Q4_K weights (mixed allowed) -> one grouped launch (attention q/k/v, linear-attention qkv + z, mixed-type
// gate + up). A same-type gate/up pair that the GLU fusion takes is left to it, and a group is not formed if the node
// after it consumes a member with ADD/MUL/GLU (residual / scale / GLU fusions keep priority).
static int ggml_cuda_try_iq4k_group(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const int i) {
    ggml_tensor * n0 = cgraph->nodes[i];
    if (!ggml_cuda_mmvq_iq4k_group_ok(n0)) {
        return 0;
    }
    ggml_tensor * mm[4] = { n0, nullptr, nullptr, nullptr };
    int n = 1, last = i, j = i + 1;
    for (; j < cgraph->n_nodes && n < 4; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(t)) {
            continue;
        }
        if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || t->op != GGML_OP_MUL_MAT || t->src[1] != n0->src[1] ||
                !ggml_cuda_mmvq_iq4k_group_ok(t)) {
            break;
        }
        mm[n++] = t;
        last = j;
    }
    if (n < 2) {
        return 0;
    }
    // the first compute node after the group must not be an epilogue another fusion would merge with a member
    int nx = last + 1;
    while (nx < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[nx])) {
        nx++;
    }
    if (nx < cgraph->n_nodes) {
        const ggml_tensor * t = cgraph->nodes[nx];
        auto is_member = [&](const ggml_tensor * s) {
            for (; s; s = s->view_src) {
                for (int k = 0; k < n; ++k) {
                    if (s == mm[k]) {
                        return true;
                    }
                }
                if (s->op != GGML_OP_VIEW && s->op != GGML_OP_RESHAPE) {
                    break;
                }
            }
            return false;
        };
        const bool uses = (t->src[0] && is_member(t->src[0])) || (t->src[1] && is_member(t->src[1]));
        if (uses && t->op == GGML_OP_GLU) {
            if (n == 2 && ggml_cuda_should_fuse_mul_mat(t->src[1], t->src[0], t)) {
                return 0;   // same-type gate/up: the fused SwiGLU GEMV is better
            }
        } else if (uses && (t->op == GGML_OP_ADD || t->op == GGML_OP_MUL)) {
            return 0;
        }
    }
    int outs[4];
    int no = 0;
    for (int k = i; k <= last; ++k) {
        for (int m = 0; m < n; ++m) {
            if (cgraph->nodes[k] == mm[m]) {
                outs[no++] = k;
            }
        }
    }
    if (no != n || !ggml_cuda_check_fusion_memory_ranges(cgraph, i, last - i + 1, outs, no)) {
        return 0;
    }
    if (!ggml_cuda_mmvq_iq4k_group(*cuda_ctx, mm, n)) {
        return 0;
    }
    return last - i;
}

// r4-mixgroup: exactly two consecutive single-token MUL_MATs (only views/no-ops in between) that read the same activation,
// one with IQ3_S/IQ3_XXS and one with IQ4_XS/Q4_K weights (linear-attention qkv + z, mixed-type FFN gate + up; the SwiGLU
// of a gate/up pair stays a separate node) -> one launch (ggml_cuda_mmvq_mix). Runs before the iq4k / iq3 same-module
// groupings and claims only these mixed pairs: no group if a third GEMV on the activation follows (q/k/v triples keep
// their k+v grouping) or if the next node consumes a member with ADD/MUL (residual / scale fusions keep priority).
static int ggml_cuda_try_mix_group(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const int i) {
    ggml_tensor * n0 = cgraph->nodes[i];
    if (!ggml_cuda_mmvq_mix_member(n0)) {
        return 0;
    }
    int j = i + 1;
    while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
        j++;
    }
    if (j >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * n1 = cgraph->nodes[j];
    if ((n1->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || n1->op != GGML_OP_MUL_MAT || n1->src[1] != n0->src[1] ||
            !ggml_cuda_mmvq_mix_member(n1)) {
        return 0;
    }
    int nx = j + 1;
    while (nx < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[nx])) {
        nx++;
    }
    ggml_tensor * mm[2] = { n0, n1 };
    if (nx < cgraph->n_nodes) {
        ggml_tensor * t = cgraph->nodes[nx];
        if (t->op == GGML_OP_MUL_MAT && t->src[1] == n0->src[1]) {
            return 0;
        }
        // gate/up pair + its SWIGLU node in one launch (the GLU output and its q8_1 come from the pair kernel)
        if (t->op == GGML_OP_GLU && (t->flags & GGML_TENSOR_FLAG_COMPUTE) && t->src[0] && t->src[1] &&
                ((t->src[0] == n0 && t->src[1] == n1) || (t->src[0] == n1 && t->src[1] == n0)) &&
                !ggml_cuda_should_fuse_mul_mat(t->src[1], t->src[0], t)) {
            const int outs3[3] = { i, j, nx };
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, nx - i + 1, outs3, 3) && ggml_cuda_mmvq_mix(*cuda_ctx, mm, 2, t)) {
                return nx - i;
            }
        }
        auto is_member = [&](const ggml_tensor * s) {
            for (; s; s = s->view_src) {
                if (s == n0 || s == n1) {
                    return true;
                }
                if (s->op != GGML_OP_VIEW && s->op != GGML_OP_RESHAPE) {
                    break;
                }
            }
            return false;
        };
        const bool uses = (t->src[0] && is_member(t->src[0])) || (t->src[1] && is_member(t->src[1]));
        if (uses && (t->op == GGML_OP_ADD || t->op == GGML_OP_MUL)) {
            return 0;
        }
        if (uses && t->op == GGML_OP_GLU && t->src[0] && t->src[1] && ggml_cuda_should_fuse_mul_mat(t->src[1], t->src[0], t)) {
            return 0;   // a fusable gate/up pair: the fused SwiGLU GEMV is better
        }
    }
    const int outs[2] = { i, j };
    if (!ggml_cuda_check_fusion_memory_ranges(cgraph, i, j - i + 1, outs, 2)) {
        return 0;
    }
    if (!ggml_cuda_mmvq_mix(*cuda_ctx, mm, 2)) {
        return 0;
    }
    return j - i;
}
#endif

#if defined(GGML_USE_HIP)
// Conv-input fusion (kernels and layout: ssm-conv.cu). Called for a CONCAT node; when the node is the conv input of
// a delta-net layer and nothing needs its full output, marks it in the context: ggml_cuda_op_concat() then writes
// only the conv-state columns and ggml_cuda_op_ssm_conv() computes conv + SiLU from the concat sources.
// A plan is always consumed by its SSM_CONV node before the next CONCAT is looked at: no computed CONCAT may lie
// between the two (checked below), the SSM_CONV + SILU pair is known to fuse, and ggml_cuda_op_ssm_conv() aborts if
// it is reached for the planned concat in any other way. A plan left over from an aborted graph evaluation is
// dropped at the start of the next one (ggml_backend_cuda_graph_compute).
static void ggml_cuda_plan_conv_fuse(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int i) {
    GGML_ASSERT(cuda_ctx->conv_fuse_concat == nullptr);

    const ggml_tensor * concat = cgraph->nodes[i];
    const ggml_tensor * st     = concat->src[0];
    const ggml_tensor * x      = concat->src[1];

    if (!ggml_cuda_conv_fuse_enabled() || !cuda_ctx->stream_context().concurrent_events.empty()) {
        return;
    }
    if (ggml_get_op_params_i32(concat, 0) != 0 || concat->type != GGML_TYPE_F32 || st->type != GGML_TYPE_F32 ||
        x->type != GGML_TYPE_F32 || (concat->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || concat->view_src != nullptr) {
        return; // a concat that is itself a view writes into a tensor whose other readers are not tracked here
    }

    const int64_t n_t = x->ne[0];
    const int64_t C   = concat->ne[1];
    const int64_t ne0 = concat->ne[0];
    const int64_t n_s = concat->ne[2];
    const size_t  fs  = sizeof(float);

    auto aligned = [](const ggml_tensor * t) { return ((uintptr_t) t->data & 15) == 0; };

    // src0: contiguous state [3, C, n_s], src1: transposed view of a contiguous token-major [C, n_t, n_s]
    if (n_t < 256 || C % 4 != 0 || concat->ne[3] != 1 || st->ne[0] != 3 || st->ne[1] != C || st->ne[2] != n_s ||
        x->ne[1] != C || x->ne[2] != n_s || x->ne[3] != 1 || x->nb[1] != fs || x->nb[0] != C*fs || x->nb[2] != n_t*C*fs ||
        !ggml_is_contiguous(st) || !ggml_is_contiguous(concat) || !aligned(st) || !aligned(x) || !aligned(concat)) {
        return;
    }

    // the SSM_CONV + SILU pair that consumes the concat, with no other CONCAT in between: the context holds one
    // plan, and the prep kernel must not replace a concat whose plan would be dropped before its SSM_CONV runs
    int j = -1;
    for (int k = i + 1; k < cgraph->n_nodes - 1 && k <= i + 32; ++k) {
        if (cgraph->nodes[k]->op == GGML_OP_CONCAT) {
            break;
        }
        if (cgraph->nodes[k]->op == GGML_OP_SSM_CONV && cgraph->nodes[k]->src[0] == concat) {
            j = k;
            break;
        }
    }
    if (j < 0 || !ggml_cuda_can_fuse(cgraph, j, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        return;
    }
    const ggml_tensor * w   = cgraph->nodes[j]->src[1];
    const ggml_tensor * out = cgraph->nodes[j + 1];
    if ((cgraph->nodes[j]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || (out->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return;
    }
    if (w->type != GGML_TYPE_F32 || w->ne[0] != 4 || w->ne[1] != C || w->nb[0] != fs || w->nb[1] != 4*fs || !aligned(w) ||
        !ggml_is_contiguous(out) || out->ne[0] != C || out->ne[1] != n_t || out->ne[2] != n_s || out->ne[3] != 1 || !aligned(out)) {
        return;
    }

    // every other reader must be a view of the trailing columns (the conv state update), and there must be one:
    // the columns in front of the first viewed one hold the halo rows.
    // cgraph may be one split of a larger graph (ggml_backend_sched), and a reader in another split is not among
    // its nodes. The use counts are those of the whole graph (a split is a ggml_graph_view of it), so every direct
    // reference to the concat has to be accounted for by the nodes seen here: the SSM_CONV and the views of it.
    int64_t col0   = ne0;
    int     n_refs = 0;
    for (int k = 0; k < cgraph->n_nodes; ++k) {
        const ggml_tensor * n = cgraph->nodes[k];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] != concat) {
                continue;
            }
            if (k != j && n->view_src != concat) {
                return;
            }
            n_refs++;
        }
        if (n->view_src != concat) {
            continue;
        }
        const size_t offs = (const char *) n->data - (const char *) concat->data;
        if (n->op != GGML_OP_VIEW || offs % fs != 0 || offs >= concat->nb[1] || n->nb[0] != fs) {
            return;
        }
        const int64_t c0 = offs/fs;
        // [cols, C, n_s] rows of the concat, or [cols, windows, C, n_s] overlapping windows one column apart
        const bool rows = n->nb[1] == concat->nb[1] && n->nb[2] == concat->nb[2] && c0 + n->ne[0] <= ne0 &&
                          n->ne[1] <= C && n->ne[2] <= n_s && n->ne[3] == 1;
        const bool wins = n->nb[1] == fs && n->nb[2] == concat->nb[1] && n->nb[3] == concat->nb[2] &&
                          c0 + n->ne[0] + n->ne[1] - 1 <= ne0 && n->ne[2] <= C && n->ne[3] <= n_s;
        if (!rows && !wins) {
            return;
        }
        col0 = std::min(col0, c0);
    }
    if (n_refs != ggml_node_get_use_count(cgraph, i)) {
        return; // read by a node outside this (split) graph
    }
    // the geometry check also guarantees col0 >= 12: the old state in columns [0, 3) and the halo rows in
    // [4, col0) lie in front of the columns that the views read
    if (col0 >= ne0 || ne0 - col0 > 64 || !ggml_cuda_conv_fuse_geom_ok(n_t, C, col0)) {
        return;
    }

    // x must still be intact when the SSM_CONV node runs, and the output may only alias x exactly
    auto overlap = [](const char * a, const size_t na, const char * b, const size_t nb) { return a < b + nb && b < a + na; };
    const char * xd = (const char *) x->data;
    const char * od = (const char *) out->data;
    const size_t xn = n_s*n_t*C*fs;
    if ((od != xd && overlap(od, xn, xd, xn)) || overlap(od, xn, (const char *) concat->data, ggml_nbytes(concat)) ||
        overlap(od, xn, (const char *) w->data, ggml_nbytes(w))) {
        return;
    }
    for (int k = i; k < j; ++k) {
        const ggml_tensor * n = cgraph->nodes[k];
        if (ggml_cuda_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (overlap((const char *) n->data, ggml_nbytes(n), xd, xn)) {
            return;
        }
    }

    cuda_ctx->conv_fuse_concat = concat;
    cuda_ctx->conv_fuse_col0   = col0;
    ggml_cuda_f16_share_get(cuda_ctx)->n_conv_fuse++;
}
#endif // defined(GGML_USE_HIP)

#if defined(GGML_USE_HIP)
// MUL_MAT (large cuBLAS GEMM) -> ADD(residual, mul_mat) where the ADD is computed in place on the residual:
// one GEMM with beta = 1 that accumulates onto the residual, no separate add kernel (which would read both
// 4*n_embd*n_tokens operands and write the sum). The result is residual + W*x with the residual as the first
// term of each fp32 sum instead of the last, i.e. it differs from the separate add in rounding only.
//
// Legal only if
//  - the MUL_MAT result has no other consumer (and is not an output or viewed),
//  - only views / no-ops sit between the two nodes, so nothing reads the residual after the GEMM started writing,
//  - the ADD result is placed on the residual (same data pointer), either by ggml-alloc (the residual then has no
//    later consumer; the graph has to ask for this by passing the residual as the first ADD operand) or because the
//    graph adds in place (ggml_add_inplace: later readers of the residual get the sum either way),
//  - neither GEMM input overlaps the residual.
// GGML_CUDA_GEMM_ADD_FUSION=0 disables it.
//
// Returns the index of the ADD node that can be folded into the MUL_MAT node i, or -1.
static int ggml_cuda_match_gemm_add(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool enabled = ggml_cuda_env_int("GGML_CUDA_GEMM_ADD_FUSION", 1) != 0;
    if (!enabled) {
        return -1;
    }
    const ggml_tensor * mm = cgraph->nodes[i];
    // The batch size below which other MUL_MAT paths / fusions are in use anyway. Graphs that want this fusion have to
    // build add(residual, mul_mat) for at least this many tokens: keep it in sync with the n_tokens threshold for the
    // residual-first operand order in src/models/qwen35.cpp.
    if (mm->ne[1] < 256 || mm->ne[2] != 1 || mm->ne[3] != 1 || mm->view_src != nullptr ||
        (mm->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, i) != 1) {
        return -1;
    }
    int j = i + 1;
    const ggml_tensor * mm_out = mm; // the MUL_MAT result or a reshape of it
    for (; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (!ggml_cuda_is_view_or_noop(n)) {
            break;
        }
        if (n->op == GGML_OP_RESHAPE && n->src[0] == mm_out && n->view_src == mm && n->view_offs == 0 &&
            ggml_node_get_use_count(cgraph, j) == 1 && !(n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            mm_out = n;
        }
    }
    if (j >= cgraph->n_nodes) {
        return -1;
    }
    const ggml_tensor * add = cgraph->nodes[j];
    if (add->op != GGML_OP_ADD || !(add->flags & GGML_TENSOR_FLAG_COMPUTE) || add->type != GGML_TYPE_F32 ||
        (add->src[0] == mm_out) == (add->src[1] == mm_out)) {
        return -1;
    }
    const ggml_tensor * res = add->src[0] == mm_out ? add->src[1] : add->src[0];
    if (res->type != GGML_TYPE_F32 || !ggml_are_same_shape(res, add) || !ggml_are_same_shape(mm_out, add) ||
        !ggml_is_contiguous(res) || !ggml_is_contiguous(add) || !ggml_is_contiguous(mm_out) ||
        ggml_nelements(add) != ggml_nelements(mm)) {
        return -1;
    }
    // in place on the residual: placed there by ggml-alloc, or explicitly (ggml_add_inplace: a view of the residual)
    if (add->view_src != nullptr && add->view_src != (res->view_src ? res->view_src : res)) {
        return -1;
    }
    if (add->data != res->data || add->buffer != res->buffer || add->data == mm->data) {
        return -1;
    }
    const auto overlaps = [&](const ggml_tensor * t) {
        const char * a0 = (const char *) add->data;
        const char * a1 = a0 + ggml_nbytes(add);
        const char * t0 = (const char *) t->data;
        const char * t1 = t0 + ggml_backend_buft_get_alloc_size(t->buffer->buft, t);
        return t0 < a1 && a0 < t1;
    };
    if (overlaps(mm->src[0]) || overlaps(mm->src[1]) || overlaps(mm)) {
        return -1;
    }
    if (ggml_backend_buffer_get_usage(mm->src[0]->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return -1; // activation x activation: not a weight GEMM
    }
    // a single fp16 cuBLAS GEMM that writes fp32 straight into its output
    const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
    if (ggml_cuda_mul_mat_cublas_gemm_type(*cuda_ctx, mm) != GGML_TYPE_F16 ||
        !ggml_cuda_mul_mat_cublas_can_accumulate(cc, mm->src[0], mm->src[1], mm)) {
        return -1;
    }
    return j;
}

static int ggml_cuda_try_fuse_gemm_add(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const int i) {
    const int j = ggml_cuda_match_gemm_add(cuda_ctx, cgraph, i);
    if (j < 0) {
        return 0;
    }
    ggml_tensor * mm = cgraph->nodes[i];

    // the MUL_MAT is the first node of the group: its input may come from the shared f16 copies as usual
    ggml_cuda_f16_share_allow_lookup allow_lookup(cuda_ctx);
    allow_lookup.st->n_gemm_add++;

    ggml_tensor acc = *mm;
    acc.data = cgraph->nodes[j]->data;
    ggml_cuda_mul_mat_cublas(*cuda_ctx, mm->src[0], mm->src[1], &acc, /*src1_f16 =*/ nullptr, /*accumulate =*/ true);
    return j - i;
}

// GLU(SWIGLU, separate f32 gate/up) -> MUL_MAT(W, glu) where the MUL_MAT is an f16 GEMM and the GLU has no other
// consumer. There is no batch threshold of its own: the fusion applies whenever ggml_cuda_mul_mat would run the
// MUL_MAT as an f16 cuBLAS GEMM (ggml_cuda_mul_mat_cublas_gemm_type), in practice for large batches such as the
// ffn_down of a prefill ubatch, because ggml_cuda_mul_mat_get_route gives smaller ones to the custom kernels.
// The gated kernel writes the f16 GEMM input directly (bit-identical to converting its f32 result) into a pool buffer
// that lives until the GEMM is enqueued; glu->data is neither written nor read. Saves the f32 store of the GLU and the
// f32 load of the conversion (GGML_CUDA_SWIGLU_F16=0 disables it).
// If the MUL_MAT in turn feeds an ADD that can be folded into the GEMM (ggml_cuda_match_gemm_add: the ffn_down
// residual), all three nodes are done here: the same f16 buffer, GEMM with beta = 1 onto the residual. Each of the two
// parts can be disabled on its own; without this one the MUL_MAT -> ADD pair is fused when the MUL_MAT node is reached.
// Returns the number of nodes to skip.
static int ggml_cuda_try_fuse_swiglu_f16(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const int i) {
    static const bool enabled = ggml_cuda_env_int("GGML_CUDA_SWIGLU_F16", 1) != 0;
    ggml_tensor * glu = cgraph->nodes[i];
    if (!enabled || i + 1 >= cgraph->n_nodes || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || !glu->src[1] ||
            glu->type != GGML_TYPE_F32) {
        return 0;
    }
    ggml_tensor * mm = cgraph->nodes[i + 1];
    if (mm->op != GGML_OP_MUL_MAT || mm->src[1] != glu || mm->src[0] == glu ||
            mm->ne[2] != 1 || mm->ne[3] != 1 || mm->src[0]->ne[2] != 1 || mm->src[0]->ne[3] != 1 ||
            !ggml_is_contiguous(mm) || !ggml_is_contiguous(glu) ||
            ggml_cuda_mul_mat_cublas_gemm_type(*cuda_ctx, mm) != GGML_TYPE_F16) {
        return 0;
    }
    // the GLU must not be an output and must have the MUL_MAT as its only consumer
    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_GLU, GGML_OP_MUL_MAT }, { i + 1 })) {
        return 0;
    }
    // Memory: the gated kernel finishes (stream order) before the GEMM writes mm->data, exactly like the unfused
    // sequence, so mm->data may alias gate/up/glu. Nothing reads glu->data afterwards (no other consumer).
    // With the ADD folded in, the GEMM writes the residual instead, which overlaps neither GEMM input nor mm
    // (ggml_cuda_match_gemm_add); gate/up are fully read before it starts in any case.
    const int j = ggml_cuda_match_gemm_add(cuda_ctx, cgraph, i + 1);

    ggml_cuda_pool_alloc<half> glu_f16(cuda_ctx->pool(), ggml_nelements(glu));
    if (!ggml_cuda_op_swiglu_f16(*cuda_ctx, glu, glu_f16.get())) {
        return 0;
    }

    if (j > 0) {
        ggml_cuda_f16_share_get(cuda_ctx)->n_swiglu_f16_add++;
        ggml_tensor acc = *mm;
        acc.data = cgraph->nodes[j]->data;
        ggml_cuda_mul_mat_cublas(*cuda_ctx, mm->src[0], glu, &acc, glu_f16.get(), /*accumulate =*/ true);
        return j - i;
    }
    ggml_cuda_f16_share_get(cuda_ctx)->n_swiglu_f16++;
    ggml_cuda_mul_mat_cublas(*cuda_ctx, mm->src[0], glu, mm, glu_f16.get());
    return 1;
}
#endif // defined(GGML_USE_HIP)

static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

    if (ggml_cuda_fusion_disabled()) {
        return 0;
    }

    ggml_tensor * node = cgraph->nodes[i];

#if defined(GGML_USE_HIP)
    if (node->op == GGML_OP_CONCAT) {
        ggml_cuda_plan_conv_fuse(cuda_ctx, cgraph, i);
        return 0;
    }
    if (node->op == GGML_OP_GLU && cuda_ctx->stream_context().concurrent_events.empty()) {
        const int nodes_to_skip = ggml_cuda_try_fuse_swiglu_f16(cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }
#endif // defined(GGML_USE_HIP)

    // like the other fusions added with it: not with concurrent stream regions (GGML_CUDA_GRAPH_OPT)
    if (node->op == GGML_OP_FLASH_ATTN_EXT && cuda_ctx->stream_context().concurrent_events.empty()) {
        const int nodes_to_skip = ggml_cuda_try_fattn_gate_fusion(cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

#if defined(GGML_USE_HIP)
    // CONT(strided view) -> UNARY(SIGMOID/SILU) -> MUL (attention output gate): read the gate through the view
    if (node->op == GGML_OP_CONT && i + 2 < cgraph->n_nodes && cuda_ctx->stream_context().concurrent_events.empty()) {
        ggml_tensor * un = cgraph->nodes[i + 1];
        ggml_tensor * mu = cgraph->nodes[i + 2];
        const int out = i + 2;
        if (un->op == GGML_OP_UNARY && un->src[0] == node && mu->op == GGML_OP_MUL && (mu->src[0] == un || mu->src[1] == un) &&
                ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_CONT, GGML_OP_UNARY, GGML_OP_MUL }, { out }) &&
                ggml_cuda_check_fusion_memory_ranges(cgraph, i, 3, &out, 1) &&
                ggml_cuda_cont_unary_mul_q8(*cuda_ctx, node, un, mu)) {
            return 2;
        }
    }
    if (node->op == GGML_OP_RMS_NORM && cuda_ctx->stream_context().concurrent_events.empty()) {
        const int nodes_to_skip = ggml_cuda_try_fuse_gated_norm(cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }
    if (node->op == GGML_OP_MUL_MAT && cuda_ctx->stream_context().concurrent_events.empty()) {
        const int nodes_to_skip = ggml_cuda_try_fuse_gemm_add(cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }
    if (node->op == GGML_OP_MUL_MAT && node->src[0]->type == GGML_TYPE_BF16 && cuda_ctx->stream_context().concurrent_events.empty()) {
        const int nodes_to_skip = ggml_cuda_try_fuse_bf16_epi(cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }
    if (node->op == GGML_OP_MUL_MAT && cuda_ctx->stream_context().concurrent_events.empty()) {
        int nodes_to_skip = ggml_cuda_try_mix_group(cuda_ctx, cgraph, i);   // mixed IQ3 + IQ4_XS/Q4_K pairs first
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
        nodes_to_skip = ggml_cuda_try_iq4k_group(cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }
#endif

    // full-attention prep chain (q/k norm + rope, gate copy, k/v cache stores) as one kernel. Not inside concurrent
    // stream regions (GGML_CUDA_GRAPH_OPT): the kernel reads k/v, which other streams of the region may still produce.
    if (node->op == GGML_OP_RMS_NORM && cuda_ctx->stream_context().concurrent_events.empty()) {
        ggml_tensor * prep_nodes[9];
        const int nodes_to_skip = ggml_cuda_try_attn_prep_fusion(cgraph, i, prep_nodes);
        if (nodes_to_skip > 0) {
            static int verify_left = getenv("GGML_CUDA_ATTN_PREP_VERIFY") ? atoi(getenv("GGML_CUDA_ATTN_PREP_VERIFY")) : 0;
            if (verify_left > 0) {
                // debug: run the unfused nodes, snapshot the outputs, run the fused kernel, compare bitwise (no graphs)
                verify_left--;
                const ggml_tensor * outs[4] = { prep_nodes[2], prep_nodes[3], prep_nodes[7], prep_nodes[8] };
                std::vector<std::vector<uint8_t>> ref(4), got(4);
                if (getenv("GGML_CUDA_ATTN_PREP_VERIFY_BASE")) {
                    // reference = the kernels the baseline used for this chain
                    ggml_cuda_op_rms_norm_fused(*cuda_ctx, prep_nodes[0], prep_nodes[1]);
                    ggml_cuda_op_rope(*cuda_ctx, prep_nodes[2]);
                    GGML_ASSERT(ggml_cuda_compute_forward(*cuda_ctx, prep_nodes[3]));
                    ggml_cuda_op_rms_norm_fused(*cuda_ctx, prep_nodes[4], prep_nodes[5]);
                    ggml_cuda_op_rope_fused(*cuda_ctx, prep_nodes[6], prep_nodes[7]);
                    GGML_ASSERT(ggml_cuda_compute_forward(*cuda_ctx, prep_nodes[8]));
                } else {
                    for (int j = i; j <= i + nodes_to_skip; ++j) {
                        ggml_tensor * n = cgraph->nodes[j];
                        if (!ggml_cuda_is_view_or_noop(n)) {
                            GGML_ASSERT(ggml_cuda_compute_forward(*cuda_ctx, n));
                        }
                    }
                }
                CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
                for (int k = 0; k < 4; ++k) {
                    ref[k].resize(ggml_nbytes(outs[k]));
                    CUDA_CHECK(cudaMemcpy(ref[k].data(), outs[k]->data, ref[k].size(), cudaMemcpyDeviceToHost));
                }
                ggml_cuda_op_attn_prep_fused(*cuda_ctx, prep_nodes);
                CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
                for (int k = 0; k < 4; ++k) {
                    got[k].resize(ggml_nbytes(outs[k]));
                    CUDA_CHECK(cudaMemcpy(got[k].data(), outs[k]->data, got[k].size(), cudaMemcpyDeviceToHost));
                    size_t ndiff = 0, first = 0;
                    const size_t ts = ggml_type_size(outs[k]->type);
                    for (size_t e = 0; e < got[k].size() / ts; ++e) {
                        if (memcmp(&got[k][e*ts], &ref[k][e*ts], ts) != 0) {
                            if (ndiff == 0) {
                                first = e;
                            }
                            ndiff++;
                        }
                    }
                    if (ndiff > 0) {
                        float r = 0.0f, g = 0.0f;
                        if (ts == 4) {
                            memcpy(&r, &ref[k][first*4], 4); memcpy(&g, &got[k][first*4], 4);
                        } else {
                            r = GGML_FP16_TO_FP32(*(const ggml_fp16_t *) &ref[k][first*2]);
                            g = GGML_FP16_TO_FP32(*(const ggml_fp16_t *) &got[k][first*2]);
                        }
                        fprintf(stderr, "attn_prep verify %s: %zu/%zu differ, first %zu ref %.9g got %.9g\n",
                            outs[k]->name, ndiff, got[k].size() / ts, first, r, g);
                    }
                }
                fprintf(stderr, "attn_prep verify done (%s)\n", prep_nodes[0]->name);
                return nodes_to_skip;
            }
            ggml_cuda_op_attn_prep_fused(*cuda_ctx, prep_nodes);
            return nodes_to_skip;
        }
    }

    if (node->op == GGML_OP_MUL_MAT_ID && cuda_ctx->stream_context().concurrent_events.empty() &&
            ggml_cuda_match_shared_expert(cgraph, i, i + 3) &&
            ggml_cuda_should_fuse_mul_mat_vec_q(cgraph->nodes[i + 2]->src[1])) {
        const int outputs[] = { i + 2, i + 5 };
        if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 6, outputs, 2)) {
            ggml_tensor * routed = cgraph->nodes[i + 2];
            ggml_tensor * shared = cgraph->nodes[i + 5];
            const ggml_tensor * up = routed->src[1];
            ggml_cuda_mm_fusion_args_host fusion{};
            fusion.gate = routed->src[0]->src[0];
            fusion.glu_op = ggml_get_glu_op(routed);
            fusion.glu_limit = ggml_get_op_params_f32(routed, 3);
            fusion.shared_up = shared->src[1]->src[0];
            fusion.shared_gate = shared->src[0]->src[0];
            fusion.shared_dst = shared;
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, up->src[0], up->src[1], up->src[2], routed, &fusion);
            return 5;
        }
    }

    if (node->op == GGML_OP_MUL) {
        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1)) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
                return match.node_count - 1;
            }
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    // ssm_scan -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_SSM_SCAN) {
        ggml_cuda_ssm_scan_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_ssm_scan_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused ssm_scan snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_ssm_scan_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_args args;
        const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
        std::vector<ggml_op>    ops;
        ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

        if (can_fuse) {
            const ggml_tensor * logits  = node->src[0];
            ggml_tensor *       weights = nullptr;
            ggml_tensor *       ids     = nullptr;
            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * clamp   = nullptr;
            const ggml_tensor * scale   = nullptr;

            if (!args.delayed_softmax) {
                int out_nodes[2];  // nodes which can't be elided

                if (args.sigmoid) {
                    ops.insert(ops.end(), { GGML_OP_UNARY });
                } else if (args.sqrt_softplus) {
                    ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                } else {
                    ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                }
                ops.push_back(GGML_OP_RESHAPE);
                if (args.prob_bias) {
                    ops.push_back(GGML_OP_ADD);
                    bias = cgraph->nodes[i + ops.size() - 1]->src[1];
                }
                if (args.n_groups > 0) {
                    for (const auto & p : topk_moe_group_pattern) {
                        ops.push_back(p.op);
                    }
                }
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                out_nodes[0] = i + ops.size() - 2;
                ids = cgraph->nodes[out_nodes[0]];

                if (args.norm) {
                    ops.insert(ops.end(),
                               { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                    clamp = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            } else if (!args.norm && !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                weights                     = cgraph->nodes[i + 5];
                ids                         = cgraph->nodes[i + 1];
                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                int out_nodes[2] = { i + 1, i + 5 };
                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }

    //RoPE + view + set-rows
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor * scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_q(up, gate, glu)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

#if defined(GGML_USE_HIP)
    // GEMVs on the same activation (qkv + z, k + v) in one launch; after the gate/up GLU fusion, which takes precedence
    if (node->op == GGML_OP_MUL_MAT && cuda_ctx->stream_context().concurrent_events.empty()) {
        const int nodes_to_skip = ggml_cuda_try_fuse_mmvq_multi(cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }
#endif

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        if (!ggml_cuda_rms_norm_mul_q8(*cuda_ctx, node, cgraph->nodes[i + 1])) {
            // the only caller that lets rms_norm write the shared f16 copy of its result: the group is exactly these
            // two nodes, so the copy is the conversion of what mul's data holds when the group is done
            ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1], /*share_f16 =*/ true);
        }
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
        ggml_cuda_op_rms_norm_scale_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        if (!ggml_cuda_unary_mul_q8(*cuda_ctx, node, cgraph->nodes[i + 1])) {
            ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1]);
        }
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST }, { GGML_UNARY_OP_SIGMOID })) {
        ggml_cuda_op_dsv4_hc_post(*cuda_ctx, cgraph->nodes[i + 3], node, cgraph->nodes[i + 2]);
        return 3;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    return 0;
}

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void * graph_key) {
    bool graph_evaluated_or_captured = false;

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_f16_share_state * f16_share = ggml_cuda_f16_share_get(cuda_ctx);

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;

#ifdef USE_CUDA_GRAPH
    ggml_cuda_graph * graph      = use_cuda_graph ? cuda_ctx->cuda_graph(graph_key) : nullptr;
    const int         chunk_size = ggml_cuda_graph_chunk_size();
    int               chunk_end  = chunk_size;
    size_t            n_chunks   = 0;
#endif // USE_CUDA_GRAPH

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
#ifdef USE_CUDA_GRAPH
                // fused nodes are skipped together, so a cut never splits a fusion
                // a concurrent region must join before its capture can end
                if (use_cuda_graph && chunk_size > 0 && i >= chunk_end && !is_concurrent_event_active) {
                    ggml_cuda_graph_launch_chunk(cuda_ctx, graph, n_chunks++);
                    CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
                    chunk_end = i + chunk_size;
                }
#endif // USE_CUDA_GRAPH
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

                prev_i = i;

                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                f16_share->epoch++;
                if (f16_share->n_valid != 0) {
                    ggml_cuda_f16_share_invalidate(f16_share, node, false);
                }

                f16_share->in_fuse = true;
                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
                f16_share->in_fuse = false;

                if (nodes_to_skip != 0) {
                    // a fused group may write the data of any of its (non-view) members
                    for (int k = 1; k <= nodes_to_skip && f16_share->n_valid != 0; ++k) {
                        if (!ggml_cuda_is_view_or_noop(cgraph->nodes[i + k])) {
                            ggml_cuda_f16_share_invalidate(f16_share, cgraph->nodes[i + k], true);
                        }
                    }
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    i += nodes_to_skip;
                    continue;
                }
#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
        }

#ifdef USE_CUDA_GRAPH
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            ggml_cuda_graph_launch_chunk(cuda_ctx, graph, n_chunks++);

            // drop the chunks left over from a longer graph
            for (size_t i = n_chunks; i < graph->instances.size(); ++i) {
                CUDA_CHECK(cudaGraphExecDestroy(graph->instances[i]));
            }
            graph->instances.resize(n_chunks);
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph && !cuda_graph_update_required) {
        // the chunks were launched during capture, otherwise launch them here
        for (cudaGraphExec_t instance : graph->instances) {
            CUDA_CHECK(cudaGraphLaunch(instance, cuda_ctx->stream()));
        }
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->instances.empty()) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
            if (!graph->disable_due_to_gpu_arch && ggml_cuda_graph_debug_enabled()) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);
    ggml_cuda_upload_flush(cuda_ctx); // before any capture begins
    ggml_cuda_upload_alternate_counter++;
#if defined(GGML_USE_HIP)
    ggml_cuda_mmvq_cache_reset(*cuda_ctx);
    cuda_ctx->conv_fuse_concat = nullptr; // a plan never outlives the graph evaluation it was made in
#endif
    ggml_cuda_f16_share_begin(cuda_ctx, cgraph);

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;

#ifdef USE_CUDA_GRAPH
    graph_key = ggml_cuda_graph_get_key(cgraph);

    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    if (graph->is_enabled()) {
        const bool graph_compatible = ggml_cuda_graph_check_compability(cgraph);
        if (graph_compatible) {
            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph);

            if (!graph->warmup_complete) {
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (!properties_changed) {
                    graph->warmup_complete = true;
                    if (ggml_cuda_graph_debug_enabled()) {
                        GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    }
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
                // else: properties changed or first call - execute directly (use_cuda_graph stays false)
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    // Properties changed - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    if (ggml_cuda_graph_debug_enabled()) {
                        GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                    }
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = graph->instances.empty();
                }
            }
        }
    }
#endif // USE_CUDA_GRAPH

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_cuda_upload_flush(cuda_ctx);

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_cuda_upload_flush(cuda_ctx);

    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    const bool disable_fusion = ggml_cuda_fusion_disabled();

    auto add_alloc_deps = [&](size_t start, size_t last_node) {

        for (size_t i = start; i < last_node; ++i) {
            params->add_alloc_dep(params->user_data, cgraph->nodes[i], cgraph->nodes[last_node]);

            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (cgraph->nodes[i]->src[j]) {
                    params->add_alloc_dep(params->user_data, cgraph->nodes[i]->src[j], cgraph->nodes[last_node]);
                }
            }
        }
    };

    if (!disable_fusion) {
        // add alloc deps for performance positive fusions. This may increase the overall compute buffer size.
        // TODO: consolidate fusion paths in graph_optimize and graph_compute
        ggml_cuda_set_device(cuda_ctx->device);
        for (int i = 0; i + 5 < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            for (int j = i + 3; j + 2 < cgraph->n_nodes; ++j) {
                if (cgraph->nodes[j]->op == GGML_OP_MUL_MAT_ID && cgraph->nodes[j + 1]->op == GGML_OP_MUL_MAT_ID) {
                    break;
                }
                if (cgraph->nodes[j]->op != GGML_OP_MUL_MAT || !ggml_cuda_match_shared_expert(cgraph, i, j)) {
                    continue;
                }
                // Group both outputs before allocation so the shared result cannot alias intervening nodes.
                std::rotate(cgraph->nodes + i + 3, cgraph->nodes + j, cgraph->nodes + j + 3);
                ggml_tensor * up = cgraph->nodes[i + 2]->src[1];
                params->add_alloc_dep(params->user_data, up->src[1], cgraph->nodes[i + 5]);
                params->add_alloc_dep(params->user_data, up->src[2], cgraph->nodes[i + 5]);
                i += 5;
                break;
            }
        }
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            // Keep the inputs of the fused gate/up/GLU MMQ kernel alive until the GLU so that they cannot alias its output (under op offload the weights are buffer copies).
            if (i + 2 < cgraph->n_nodes && cgraph->nodes[i + 2]->op == GGML_OP_GLU) {
                ggml_tensor * glu  = cgraph->nodes[i + 2];
                ggml_tensor * gate = glu->src[0];
                ggml_tensor * up   = glu->src[1];
                const bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) || (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);
                if (ok && ggml_cuda_mul_mat_q_fusion_matches(up, gate, glu, cuda_ctx->device)) {
                    add_alloc_deps(i, i + 2);
                    i += 2;
                    continue;
                }
            }

            ggml_cuda_moe_weighted_reduction_match match;
            if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
                if (match.expert_scale != nullptr) {
                    params->add_alloc_dep(
                        params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
                }
                i += match.node_count - 1;
            }

            if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
                    cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
                ggml_cuda_topk_moe_args args;
                const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
                std::vector<ggml_op>    ops;
                ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

                const ggml_tensor * node = cgraph->nodes[i];

                if (can_fuse) {
                    const ggml_tensor * logits  = node->src[0];
                    ggml_tensor *       weights = nullptr;
                    ggml_tensor *       ids     = nullptr;

                    if (!args.delayed_softmax) {
                        int out_nodes[2];  // nodes which can't be elided

                        if (args.sigmoid) {
                            ops.insert(ops.end(), { GGML_OP_UNARY });
                        } else if (args.sqrt_softplus) {
                            ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                        } else {
                            ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                        }
                        const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                        if (args.prob_bias) {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                                    GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 4;
                        } else {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 3;
                        }
                        ids = cgraph->nodes[out_nodes[0]];

                        if (args.norm) {
                            ops.insert(ops.end(),
                                       { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                        }
                        if (args.scale) {
                            ops.insert(ops.end(), { GGML_OP_SCALE });
                        }

                        weights      = cgraph->nodes[i + ops.size() - 1];
                        out_nodes[1] = i + ops.size() - 1;

                        if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                                ggml_cuda_should_use_topk_moe(node, logits, weights, ids)) {

                            add_alloc_deps(i, i + ops.size());
                            i += ops.size() - 1;
                        }
                    }
                }
            }
        }
    }

#ifdef USE_CUDA_GRAPH
    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph) {
        return;
    }

    ggml_cuda_set_device(cuda_ctx->device);

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

static std::string ggml_cuda_device_description(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device)));

    const ggml_cuda_device_info & info = ggml_cuda_info();
    std::string description = prop.name;
    if (info.device_count > info.physical_device_count) {
        description += " (dev p" + std::to_string(info.devices[device].physical_device) +
                       "/v" + std::to_string(info.devices[device].virtual_index) + ")";
    }
    return description;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    snprintf(description, description_size, "%s", ggml_cuda_device_description(device).c_str());
}

static int ggml_cuda_physical_device_share_count(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(device);
    *free  /= share_count;
    *total /= share_count;
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (!ggml_cuda_register_host_enabled()) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (!ggml_cuda_register_host_enabled()) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(err));
        *free = 0;
        *total = 0;
        return;
    }

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__) && !defined(GGML_USE_HIP)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    // Check if UMA is explicitly enabled via environment variable
    bool is_uma = prop.integrated > 0 || ggml_cuda_unified_memory_enabled();

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__) && !defined(GGML_USE_HIP)

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
    *free  /= share_count;
    *total /= share_count;
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    return prop.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    const bool host_buffer = !ggml_cuda_pinned_memory_disabled();
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    if (op->src[0]->type == GGML_TYPE_BF16 && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
                        return false;
                    }
                    return op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    if (op->src[0]->type == GGML_TYPE_BF16 &&
                            (ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI || ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_CLAMP)) {
                        return false;
                    }
                    if (op->src[0]->type == GGML_TYPE_F16 && ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI) {
                        return false;
                    }
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false; // TODO this could in principle be implemented though currently there is no use case.
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16 && !ggml_cuda_op_mul_mat_use_fwht(op)) {
                    return false;
                }
                if (op->op == GGML_OP_MUL_MAT_ID && ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ4_XS:
                        return true;
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                        // 32-value sub-blocks, the row size does not guarantee
                        // the QK_K super-blocks the get_rows kernel iterates on
                        return op->src[0]->ne[0] % QK_K == 0;
                    default:
                        return false;
                }
            }
        case GGML_OP_GET_ROWS_BACK:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
        case GGML_OP_SET_ROWS:
            {
                return (
                           (
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                               op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                               op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                               op->src[0]->type == GGML_TYPE_F32
                           ) || (
                               op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F16
                           )
                       ) &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            }
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            }
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            }
        case GGML_OP_DUP:
                return true;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            return true;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            }
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                const int32_t dim = op->op_params[0];
                return src0_type == src1_type &&
                       src0_type == op->type &&
                       (
                           (
                               ggml_is_quantized(src0_type) &&
                               (
                                   (
                                       dim == 3 &&
                                       ggml_is_contiguous(op->src[0]) &&
                                       ggml_is_contiguous(op->src[1])
                                   ) || (
                                       dim != 3 &&
                                       ggml_is_contiguous_to_3(op->src[0]) &&
                                       ggml_is_contiguous_to_3(op->src[1])
                                   )
                               ) &&
                               op->src[0]->ne[0] % ggml_blck_size(src0_type) == 0 &&
                               op->src[1]->ne[0] % ggml_blck_size(src0_type) == 0
                           ) || (
                               !ggml_is_quantized(src0_type) &&
                               ggml_blck_size(src0_type) == 1 &&
                               (
                                   ggml_type_size(src0_type) == 1 ||
                                   ggml_type_size(src0_type) == 2 ||
                                   ggml_type_size(src0_type) == 4 ||
                                   ggml_type_size(src0_type) == 8
                               )
                           )
                       );
            }
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            }
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            }
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_SCALE:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16) && op->type == op->src[0]->type &&
                   ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_CLAMP:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            if (op->src[0]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_BF16 || op->type == GGML_TYPE_BF16) {
                return op->src[0]->type == GGML_TYPE_BF16 && op->type == GGML_TYPE_BF16 &&
                    (op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_F32);
            }
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type         == GGML_TYPE_F32 || op->type         == GGML_TYPE_F16);
        case GGML_OP_SSM_SCAN: {
            const int32_t K = ggml_get_op_params_i32(op, 0);

            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 96 || d_state == 128 || d_state == 256) && d_head % 16 == 0)
                const int64_t d_state = op->src[0]->ne[0];
                return (d_state == 96 || d_state == 128 || d_state == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                if (K > 1) {
                    return false;
                }

                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]));
        case GGML_OP_CONV_3D:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
        case GGML_OP_CONV_2D_DW:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguously_allocated(op->src[0]);
        case GGML_OP_TOP_K:
            return op->src[0]->ne[0] <= INT_MAX;
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            {
                // bitonic path: the padded row must fit in shared memory
                int64_t ncols_pad = 1;
                while (ncols_pad < op->src[0]->ne[0]) {
                    ncols_pad *= 2;
                }
                return ncols_pad * sizeof(int) <= ggml_cuda_info().devices[dev_ctx->device].smpb;
            }
#else
            return op->src[0]->ne[0] <= INT_MAX;
#endif
        case GGML_OP_SUM_ROWS:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            return true;
        case GGML_OP_GDN_DECODE_PREP:
        case GGML_OP_GDN_DECODE_OUT:
            return ggml_cuda_gdn_decode_supported(op);
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);
        case GGML_OP_TOP_K_MASK:
            return op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32;

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(dev);
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_FA_QUANTS
        features.push_back({ "FA_QUANTS", GGML_CUDA_FA_QUANTS });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
#ifdef GGML_USE_NCCL
    if (strcmp(name, "ggml_backend_comm_init_rank") == 0) {
        return (void *)ggml_backend_cuda_comm_init_rank;
    }
#endif // GGML_USE_NCCL
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = ggml_cuda_op_offload_min_batch();

            const ggml_cuda_device_info & info = ggml_cuda_info();
            const bool virtual_devices = info.device_count > info.physical_device_count;

            for (int i = 0; i < info.device_count; i++) {
                const int physical_id = info.devices[i].physical_device;

                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);
                dev_ctx->description = ggml_cuda_device_description(i);

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id));
                dev_ctx->pci_bus_id = pci_bus_id;
                if (virtual_devices) {
                    // make the pci bus id unique for virtual devices
                    dev_ctx->pci_bus_id += "-v" + std::to_string(i);
                }
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
