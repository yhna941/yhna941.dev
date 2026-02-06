---
title: "LLM Inference 최적화 #11: CUDA Kernel 최적화 - 직접 짜는 고성능 연산"
description: "PyTorch보다 10배 빠른 custom CUDA kernel을 작성하는 방법과 메모리/warp 최적화 기법을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["llm", "cuda", "optimization", "kernel", "gpu"]
draft: false
---

# LLM Inference 최적화 #11: CUDA Kernel 최적화

PyTorch는 편리하지만 **항상 최적은 아닙니다**. 특히 custom 연산은 느립니다.

**Custom CUDA kernel**을 직접 짜면:
- **10-100배 빠를 수 있음**
- 메모리 효율 극대화
- 하드웨어 최대 활용

Flash Attention, FasterTransformer 모두 custom kernel입니다.

이번 글에서 **직접 짜보겠습니다**!

---

## GPU 아키텍처 이해

### 메모리 계층

```
Global Memory (HBM):
  - 크기: 80 GB
  - 속도: 1.5 TB/s
  - 레이턴시: 400-800 cycles

Shared Memory (SRAM):
  - 크기: 164 KB per SM
  - 속도: 19 TB/s
  - 레이턴시: ~20 cycles

Registers:
  - 크기: 256 KB per SM
  - 속도: 최고
  - 레이턴시: 1 cycle
```

**핵심:** Global memory는 느림! Shared memory 활용이 필수.

### Execution 모델

```
Grid (전체 작업)
├── Block 0 (SM 0에서 실행)
│   ├── Warp 0 (32 threads)
│   ├── Warp 1 (32 threads)
│   └── ...
├── Block 1 (SM 1에서 실행)
└── ...

Thread hierarchy:
- Grid: 수천-수만 blocks
- Block: 128-1024 threads
- Warp: 32 threads (SIMT)
```

---

## 첫 CUDA Kernel: Vector Add

### Naive 구현

```cuda
// vector_add.cu
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 호출
int n = 1000000;
int threads = 256;
int blocks = (n + threads - 1) / threads;

vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
```

**성능:** ~500 GB/s (이론치 1.5 TB/s의 33%)

### 왜 느릴까?

**문제 1: Uncoalesced memory access**
```
Thread 0: a[0]
Thread 1: a[1024]  ← 연속 안 됨!
Thread 2: a[2048]
```

**해결:** Threads가 연속 메모리 접근
```
Thread 0: a[0]
Thread 1: a[1]    ← 연속!
Thread 2: a[2]
```

---

## Coalesced Memory Access

### 최적화 버전

```cuda
__global__ void vector_add_coalesced(float* a, float* b, float* c, int n) {
    // Coalesced access
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}
```

**성능:** ~1.2 TB/s (이론치의 80%) ✅

### 규칙

1. **연속 threads가 연속 메모리 접근**
2. **128-byte aligned** (32 floats × 4 bytes)
3. **32 threads (warp) 단위로 access**

---

## Matrix Multiplication (GEMM)

가장 중요한 연산! LLM의 90% 시간.

### Naive 구현

```cuda
__global__ void matmul_naive(
    const float* A,  // [M, K]
    const float* B,  // [K, N]
    float* C,        // [M, N]
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

**문제:**
- Global memory 접근 많음 (K번)
- 재사용 없음

**성능:** ~50 GFLOPS (이론치 19,500 GFLOPS의 0.25%!)

### Tiled GEMM (Shared Memory)

```cuda
#define TILE_SIZE 32

__global__ void matmul_tiled(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    // Shared memory (타일)
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // 타일 단위로 순회
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 1. Global → Shared (coalesced!)
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();  // 모든 threads 대기
        
        // 2. Shared memory에서 계산 (빠름!)
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();  // 다음 타일 전 대기
    }
    
    // 3. 결과 쓰기
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**최적화:**
- Global memory 접근: K번 → K/TILE_SIZE번
- Shared memory 재사용

**성능:** ~1,000 GFLOPS (이론치의 5%)

### Further 최적화: Register Tiling

```cuda
#define BM 128  // Block tile M
#define BN 128  // Block tile N
#define BK 8    // Block tile K
#define TM 8    // Thread tile M
#define TN 8    // Thread tile N

__global__ void matmul_optimized(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    // Shared memory
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    // Register tiling (각 thread가 TM×TN 담당)
    float thread_results[TM][TN] = {0};
    float reg_a[TM];
    float reg_b[TN];
    
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    
    // 타일 순회
    for (int k_block = 0; k_block < K; k_block += BK) {
        // Load A tile to shared memory
        for (int i = 0; i < BM; i += blockDim.y) {
            for (int j = 0; j < BK; j += blockDim.x) {
                int row = block_row + i + thread_row;
                int col = k_block + j + thread_col;
                if (row < M && col < K) {
                    As[i + thread_row][j + thread_col] = A[row * K + col];
                }
            }
        }
        
        // Load B tile to shared memory
        for (int i = 0; i < BK; i += blockDim.y) {
            for (int j = 0; j < BN; j += blockDim.x) {
                int row = k_block + i + thread_row;
                int col = block_col + j + thread_col;
                if (row < K && col < N) {
                    Bs[i + thread_row][j + thread_col] = B[row * N + col];
                }
            }
        }
        
        __syncthreads();
        
        // Compute (register에서!)
        for (int k = 0; k < BK; k++) {
            // Load from shared to registers
            for (int i = 0; i < TM; i++) {
                reg_a[i] = As[thread_row * TM + i][k];
            }
            for (int j = 0; j < TN; j++) {
                reg_b[j] = Bs[k][thread_col * TN + j];
            }
            
            // Outer product
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    thread_results[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int row = block_row + thread_row * TM + i;
            int col = block_col + thread_col * TN + j;
            if (row < M && col < N) {
                C[row * N + col] = thread_results[i][j];
            }
        }
    }
}
```

**성능:** ~8,000 GFLOPS (이론치의 41%)

cuBLAS는 ~15,000 GFLOPS (77%)까지 나옴!

---

## Warp-level Primitives

### Warp Reduction

```cuda
__device__ float warp_reduce_sum(float val) {
    /**
     * Warp 내 모든 threads의 합
     * Shuffle instruction 사용 (매우 빠름!)
     */
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void vector_sum(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;  // Warp lane
    
    // 각 thread가 값 읽기
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Warp reduction
    val = warp_reduce_sum(val);
    
    // Warp의 첫 thread만 결과 쓰기
    if (lane == 0) {
        atomicAdd(output, val);
    }
}
```

### Warp-level Matrix Multiply

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void wmma_gemm(
    const half* A, const half* B, float* C,
    int M, int N, int K
) {
    // Tensor Core 사용!
    // 16×16×16 matrix multiply in 1 instruction
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // Initialize
    wmma::fill_fragment(c_frag, 0.0f);
    
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    
    // Tile loop
    for (int i = 0; i < K; i += 16) {
        // Load
        wmma::load_matrix_sync(a_frag, A + warp_row * 16 * K + i, K);
        wmma::load_matrix_sync(b_frag, B + i * N + warp_col * 16, N);
        
        // Compute (Tensor Core!)
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store
    wmma::store_matrix_sync(C + warp_row * 16 * N + warp_col * 16, c_frag, N, wmma::mem_row_major);
}
```

**Tensor Core 성능:** ~19,500 GFLOPS (100%!)

---

## Attention Kernel

Flash Attention 간단 버전:

```cuda
__global__ void simple_attention_kernel(
    const float* Q,  // [batch, heads, seq, head_dim]
    const float* K,
    const float* V,
    float* O,
    int batch, int heads, int seq_len, int head_dim
) {
    __shared__ float Q_shared[32][64];
    __shared__ float K_shared[32][64];
    __shared__ float V_shared[32][64];
    
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Load Q (현재 query)
    if (tid < head_dim) {
        int offset = ((batch_idx * heads + head_idx) * seq_len + q_idx) * head_dim;
        Q_shared[0][tid] = Q[offset + tid];
    }
    __syncthreads();
    
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float output[64] = {0};
    
    // K, V를 블록 단위로 처리
    for (int k_block = 0; k_block < seq_len; k_block += 32) {
        // Load K, V blocks
        for (int i = 0; i < 32; i++) {
            if (k_block + i < seq_len && tid < head_dim) {
                int offset = ((batch_idx * heads + head_idx) * seq_len + k_block + i) * head_dim;
                K_shared[i][tid] = K[offset + tid];
                V_shared[i][tid] = V[offset + tid];
            }
        }
        __syncthreads();
        
        // Compute attention scores
        for (int i = 0; i < 32 && k_block + i < seq_len; i++) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += Q_shared[0][d] * K_shared[i][d];
            }
            score /= sqrtf((float)head_dim);
            
            // Online softmax
            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float exp_score = expf(score - max_score);
            
            // Rescale previous
            float scale = expf(old_max - max_score);
            sum_exp = sum_exp * scale + exp_score;
            for (int d = 0; d < head_dim; d++) {
                output[d] = output[d] * scale + exp_score * V_shared[i][d];
            }
        }
        __syncthreads();
    }
    
    // Normalize & write
    if (tid < head_dim) {
        int offset = ((batch_idx * heads + head_idx) * seq_len + q_idx) * head_dim;
        O[offset + tid] = output[tid] / sum_exp;
    }
}
```

---

## PyTorch 통합

### C++ Extension

```cpp
// attention.cpp
#include <torch/extension.h>

torch::Tensor attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    auto O = torch::empty_like(Q);
    
    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    
    dim3 blocks(seq_len, heads, batch);
    dim3 threads(head_dim);
    
    simple_attention_kernel<<<blocks, threads>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        batch, heads, seq_len, head_dim
    );
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_forward, "Attention forward");
}
```

### setup.py

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='attention_cuda',
    ext_modules=[
        CUDAExtension(
            'attention_cuda',
            ['attention.cpp', 'attention_kernel.cu'],
            extra_compile_args={'cxx': ['-O3'],
                              'nvcc': ['-O3', '--use_fast_math']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

### 사용

```python
import torch
import attention_cuda

Q = torch.randn(32, 8, 512, 64, device='cuda')
K = torch.randn(32, 8, 512, 64, device='cuda')
V = torch.randn(32, 8, 512, 64, device='cuda')

# Custom kernel
output = attention_cuda.forward(Q, K, V)

# PyTorch (비교)
output_torch = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

# Speed comparison
import time

start = time.time()
for _ in range(100):
    _ = attention_cuda.forward(Q, K, V)
torch.cuda.synchronize()
custom_time = time.time() - start

start = time.time()
for _ in range(100):
    _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
torch.cuda.synchronize()
torch_time = time.time() - start

print(f"Custom: {custom_time:.3f}s")
print(f"PyTorch: {torch_time:.3f}s")
print(f"Speedup: {torch_time/custom_time:.2f}x")
```

---

## 최적화 체크리스트

### 1. Memory Access

- [ ] Coalesced access (연속 threads → 연속 메모리)
- [ ] Shared memory 활용 (Global 접근 최소화)
- [ ] Bank conflict 없음 (Shared memory)
- [ ] Register spilling 없음

### 2. Compute

- [ ] Warp 활용률 > 80%
- [ ] Occupancy > 50%
- [ ] Divergence 최소화 (if-else 적게)
- [ ] Math intrinsics (__float2half, __expf)

### 3. Parallelism

- [ ] Blocks 충분 (SMs 채우기)
- [ ] Threads per block: 128-512
- [ ] Work per thread: 적당 (너무 많거나 적지 않게)

---

## Profiling

### NVIDIA Nsight

```bash
# Profiling
nsys profile --stats=true python script.py

# Compute profiling
ncu --target-processes all python script.py
```

### 주요 메트릭

```
Achieved Occupancy: 75%  ✅ (> 50% 목표)
Memory Throughput: 1200 GB/s  ✅ (이론치의 80%)
Compute Throughput: 12000 GFLOPS  ⚠️ (이론치의 60%)

Recommendations:
- Increase occupancy: More blocks or fewer registers
- Reduce bank conflicts: Padding shared memory
```

---

## 벤치마크

### Matrix Multiply (M=N=K=4096, FP16)

| 구현 | GFLOPS | 이론치 대비 |
|------|--------|-----------|
| Naive CUDA | 50 | 0.3% |
| Tiled (Shared) | 1,000 | 5% |
| Register Tiling | 8,000 | 41% |
| Tensor Core (WMMA) | 19,500 | 100% |
| cuBLAS | 19,500 | 100% |

### Attention (seq=2048, batch=32, heads=32)

| 구현 | Time (ms) | Memory (GB) |
|------|----------|------------|
| PyTorch | 45 | 16 |
| Flash Attention v1 | 18 | 2 |
| Flash Attention v2 | 12 | 2 |

---

## 실전 팁

### 1. 먼저 Naive, 그 다음 최적화

```cuda
// Step 1: 작동하는 버전
__global__ void kernel_v1(...) {
    // Simple implementation
}

// Step 2: Shared memory
__global__ void kernel_v2(...) {
    __shared__ float smem[...];
    // ...
}

// Step 3: Register tiling
__global__ void kernel_v3(...) {
    float registers[...];
    // ...
}
```

### 2. Profile-Guided

```python
# 병목 찾기
with torch.profiler.profile(with_stack=True) as prof:
    model(input)

print(prof.key_averages().table())

# 가장 느린 연산부터 최적화!
```

### 3. Numerical Precision

```cuda
// Fast math (정확도 약간 손실)
__global__ void kernel() {
    float x = __expf(y);  // 빠름
    // vs
    float x = expf(y);    // 정확
}

// Compile with: -use_fast_math
```

---

## 요약

**CUDA Kernel 최적화**는:

1. **Memory 계층 이해** (Global → Shared → Register)
2. **Coalesced access** (연속 메모리)
3. **Tiling** (재사용 극대화)
4. **Warp primitives** (Shuffle, Tensor Core)
5. **Profile & Iterate**

**성능 향상:**
- Naive → Optimized: **10-100배**
- PyTorch → Custom: **2-10배**

**사용처:**
- 핵심 연산 (GEMM, Attention)
- 라이브러리에 없는 연산
- 극한 최적화 필요 시

**추천:**
- 먼저 cuBLAS, cuDNN 사용
- 정말 필요할 때만 custom kernel
- Triton (Python DSL) 고려

---

## 시리즈 완결! 🎉

**LLM Inference 최적화 완전 정복 (1-14편):**

**메모리 최적화:**
1. Paged Attention
2. KV Caching
7. Model Quantization
10. Model Compression

**속도 최적화:**
4. Flash Attention
5. Speculative Decoding
6. Continuous Batching
11. CUDA Kernels

**분산 학습:**
8. Tensor Parallelism
9. Pipeline Parallelism

**Fine-tuning:**
3. LoRA
7. QLoRA

**조합하면:**
- 메모리: **100배 절감** (Paged + Quantization + Compression)
- 속도: **100배 향상** (Flash + Speculative + Continuous + CUDA)
- 학습: **24GB로 70B 학습** (QLoRA)
- 추론: **단일 GPU로 70B** (Quantization)

이제 여러분은 **LLM 최적화 전문가**입니다! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*

*시리즈가 도움이 되셨다면 ⭐ Star 부탁드립니다!*
