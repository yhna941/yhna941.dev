---
title: "LLM Inference 최적화 #1: Paged Attention이 뭐길래?"
description: "vLLM의 핵심 기술인 Paged Attention을 이해하고, 어떻게 메모리 효율을 10배 높이는지 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["llm", "optimization", "vllm", "attention", "memory"]
draft: false
---

# LLM Inference 최적화 #1: Paged Attention

LLM을 서빙하다 보면 **메모리가 금방 찹니다**. GPT-3 규모의 모델을 돌리면 GPU 메모리 대부분이 **KV Cache**에 잡아먹히죠.

vLLM은 이 문제를 **Paged Attention**으로 해결했습니다. 메모리 효율을 **10배** 높이고, 처리량(throughput)을 **24배** 올렸어요.

어떻게 가능했을까요?

---

## 문제: KV Cache가 메모리를 잡아먹는다

### Transformer Attention 복습

Transformer의 attention 계산:

```
Attention(Q, K, V) = softmax(QK^T / √d) V
```

- **Q** (Query): 현재 토큰
- **K** (Key): 모든 이전 토큰들
- **V** (Value): 모든 이전 토큰들

**문제는 K와 V입니다.**

### 예시: "Hello, how are you?" 생성

```
Step 1: "Hello"
  K = [K_Hello]
  V = [V_Hello]

Step 2: "Hello," → ","
  K = [K_Hello, K_,]
  V = [V_Hello, V_,]

Step 3: "Hello, how" → "how"
  K = [K_Hello, K_,, K_how]
  V = [V_Hello, V_,, V_how]

...
```

토큰을 생성할 때마다 K, V가 늘어납니다. 이걸 **KV Cache**라고 부릅니다.

### 메모리 계산

**모델**: LLaMA-13B (40 layers, hidden 5120)
**배치 크기**: 1
**시퀀스 길이**: 2048

```
KV Cache 크기 = 2 (K+V) × 40 (layers) × 5120 (hidden) × 2048 (seq) × 2 (fp16)
             = 1.6 GB
```

**단 한 개의 요청**이 1.6GB를 먹습니다!

배치 크기 16이면? **25.6GB**. A100 80GB도 3개 배치면 끝이에요.

---

## 기존 방식의 문제점

### 1. 메모리 단편화 (Fragmentation)

기존 방식은 각 요청마다 **연속된 메모리**를 할당합니다.

```
Request 1: [████████████████] 2048 tokens (full)
Request 2: [████████░░░░░░░░] 512 tokens (not full)
Request 3: [██████████░░░░░░] 1024 tokens (not full)
```

Request 2, 3은 최대 길이만큼 메모리를 예약하지만, 실제론 일부만 씁니다.

**남은 공간은 낭비됩니다.**

### 2. 동적 길이를 처리 못함

사용자마다 생성 길이가 다릅니다:
- "Hello" → 짧음 (10 tokens)
- "Explain quantum physics" → 김 (500 tokens)

하지만 **미리 최대 길이를 할당**해야 하니 낭비가 심합니다.

### 3. Batching 효율 떨어짐

```
Batch 1: [2048, 512, 1024, 128] tokens
```

가장 긴 2048에 맞춰서 모두 2048만큼 할당 → **76% 낭비**

---

## 해결책: Paged Attention

아이디어는 간단합니다:

> **운영체제의 가상 메모리처럼 메모리를 페이지 단위로 관리하자!**

### 핵심 개념

**1. 페이지 단위 할당**

연속된 큰 메모리 대신, 작은 **페이지(block)**로 나눕니다.

```
Page size = 16 tokens

기존:
Request 1: [████████████████████████████████] 2048 tokens

Paged Attention:
Request 1: [████] [████] [████] ... [████]
           page1  page2  page3      page128
```

### 2. 비연속 메모리 사용

페이지들은 물리적으로 떨어져 있어도 됩니다.

```
Physical Memory:
[Page A] [Free] [Page C] [Free] [Page B] [Free]

Logical View (Request 1):
[Page A] → [Page B] → [Page C]
```

OS의 페이지 테이블처럼, **매핑 테이블**로 관리합니다.

### 3. 동적 할당

필요할 때만 페이지를 추가합니다.

```
Step 1: "Hello" (5 tokens)
  [████░░░░░░░░░░░░]  1 page (16 tokens)

Step 10: 50 tokens 생성
  [████████████████] [████████████████] [████████████████]
   page 1 (full)      page 2 (full)      page 3 (partial)
```

---

## vLLM 구현

### Block Table (페이지 테이블)

각 요청마다 **Block Table**을 유지합니다.

```python
class Sequence:
    def __init__(self):
        self.tokens = []
        self.block_table = []  # 물리 블록 ID들
    
    def append_token(self, token):
        self.tokens.append(token)
        
        # 현재 블록이 꽉 찼으면 새 블록 할당
        if len(self.tokens) % BLOCK_SIZE == 1:
            new_block = allocate_block()
            self.block_table.append(new_block)
```

### Block Manager

메모리 풀을 관리합니다.

```python
class BlockSpaceManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks = {}
    
    def allocate(self, seq_id):
        """새 블록 할당"""
        if not self.free_blocks:
            raise OutOfMemoryError()
        
        block_id = self.free_blocks.pop()
        self.allocated_blocks[seq_id] = block_id
        return block_id
    
    def free(self, seq_id):
        """블록 해제"""
        block_id = self.allocated_blocks.pop(seq_id)
        self.free_blocks.append(block_id)
```

### Attention 계산

기존 attention과 동일하지만, **불연속 메모리**를 읽습니다.

```python
def paged_attention(query, key_cache, value_cache, block_table):
    """
    query: [batch, num_heads, head_dim]
    key_cache: [num_blocks, block_size, num_heads, head_dim]
    value_cache: [num_blocks, block_size, num_heads, head_dim]
    block_table: [batch, max_num_blocks] - 각 시퀀스의 블록 ID들
    """
    batch_size = query.shape[0]
    outputs = []
    
    for i in range(batch_size):
        # 이 시퀀스의 블록들 가져오기
        blocks = block_table[i]
        
        # 각 블록에서 K, V 수집
        keys = []
        values = []
        for block_id in blocks:
            keys.append(key_cache[block_id])
            values.append(value_cache[block_id])
        
        # Attention 계산 (표준 방식)
        K = torch.cat(keys, dim=0)  # [seq_len, num_heads, head_dim]
        V = torch.cat(values, dim=0)
        
        scores = query[i] @ K.transpose(-2, -1) / sqrt(d)
        attn = softmax(scores, dim=-1)
        output = attn @ V
        
        outputs.append(output)
    
    return torch.stack(outputs)
```

실제로는 CUDA 커널로 최적화되어 있습니다!

---

## CUDA 커널 최적화

### 문제: 비연속 메모리 읽기는 느리다

```cuda
// 나이브 구현: 블록마다 메모리 접근
for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
    int block_id = block_table[block_idx];
    // key_cache[block_id]에서 읽기 → 캐시 미스 많음
}
```

### 해결책: Fused Kernel

모든 블록을 **한 번에** 처리하는 CUDA 커널:

```cuda
__global__ void paged_attention_kernel(
    const float* Q,           // [batch, heads, head_dim]
    const float* K_cache,     // [num_blocks, block_size, heads, head_dim]
    const float* V_cache,     // [num_blocks, block_size, heads, head_dim]
    const int* block_table,   // [batch, max_blocks]
    float* output,            // [batch, heads, head_dim]
    int block_size
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    // Shared memory에 Q 로드
    __shared__ float Q_shared[HEAD_DIM];
    if (tid < HEAD_DIM) {
        Q_shared[tid] = Q[batch_idx * num_heads * HEAD_DIM + 
                          head_idx * HEAD_DIM + tid];
    }
    __syncthreads();
    
    // 각 블록 순회
    float attn_sum = 0.0f;
    for (int block_idx = 0; block_idx < max_blocks; block_idx++) {
        int physical_block = block_table[batch_idx * max_blocks + block_idx];
        if (physical_block < 0) break;  // 유효한 블록 끝
        
        // 이 블록의 모든 토큰에 대해 attention
        for (int token_idx = 0; token_idx < block_size; token_idx++) {
            // K와 내적
            float score = 0.0f;
            for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
                int k_idx = physical_block * block_size * num_heads * HEAD_DIM +
                           token_idx * num_heads * HEAD_DIM +
                           head_idx * HEAD_DIM + d;
                score += Q_shared[d] * K_cache[k_idx];
            }
            
            // Reduce across threads
            score = warp_reduce_sum(score);
            
            // Softmax 분자 계산
            float exp_score = expf(score / sqrtf(HEAD_DIM));
            attn_sum += exp_score;
            
            // V와 곱하기 (누적)
            // ...
        }
    }
    
    // Softmax 정규화 & 출력
    // ...
}
```

### 핵심 최적화

1. **Shared Memory**: Q를 공유 메모리에 캐싱
2. **Coalesced Access**: K, V를 연속으로 읽기
3. **Warp Reduction**: 스레드 간 합산 병렬화
4. **Fused Operation**: Attention 전체를 한 커널에서

---

## 성능 비교

### 메모리 사용량

**Scenario**: LLaMA-13B, 배치 크기 64, 평균 시퀀스 길이 512

| 방식 | 메모리 사용량 |
|------|--------------|
| Naive (고정 할당) | 102 GB |
| Paged Attention | 12 GB |

**8.5배 절약!**

### Throughput

| 방식 | 처리량 (requests/sec) |
|------|----------------------|
| HuggingFace Transformers | 0.8 |
| FasterTransformer | 4.5 |
| vLLM (Paged Attention) | 24.0 |

**30배 향상!**

---

## 추가 최적화: Copy-on-Write

### 문제: Prefix 공유

같은 시스템 프롬프트를 여러 요청이 공유합니다:

```
Request 1: "You are a helpful assistant. What is AI?"
Request 2: "You are a helpful assistant. Explain quantum physics."
Request 3: "You are a helpful assistant. Write a poem."
```

"You are a helpful assistant"는 **모두 같은데** 각자 메모리를 씁니다.

### 해결책: Block 공유

```python
class Sequence:
    def __init__(self, prefix_blocks=None):
        self.block_table = prefix_blocks.copy() if prefix_blocks else []
        self.num_shared_blocks = len(self.block_table)
    
    def append_token(self, token):
        # 공유 블록에 쓰려고 하면 복사 (Copy-on-Write)
        if self.num_shared_blocks > 0:
            last_shared = self.block_table[-1]
            new_block = copy_block(last_shared)
            self.block_table[-1] = new_block
            self.num_shared_blocks -= 1
```

### 효과

시스템 프롬프트가 100 토큰이고, 1000개 요청이면:

- **Before**: 100 × 1000 = 100,000 토큰 저장
- **After**: 100 토큰 저장 (공유)

**1000배 절약!**

---

## 구현 예제 (간단 버전)

전체 vLLM은 복잡하니, 핵심만 구현해봅시다.

```python
import torch
import torch.nn.functional as F

class PagedKVCache:
    def __init__(self, num_blocks, block_size, num_layers, num_heads, head_dim):
        self.num_blocks = num_blocks
        self.block_size = block_size
        
        # Physical memory pool
        self.key_cache = torch.zeros(
            num_layers, num_blocks, block_size, num_heads, head_dim,
            dtype=torch.float16, device='cuda'
        )
        self.value_cache = torch.zeros_like(self.key_cache)
        
        # Free block list
        self.free_blocks = list(range(num_blocks))
    
    def allocate_block(self):
        if not self.free_blocks:
            raise RuntimeError("Out of memory")
        return self.free_blocks.pop()
    
    def free_block(self, block_id):
        self.free_blocks.append(block_id)
    
    def write(self, layer_id, block_id, slot_id, key, value):
        """블록의 특정 슬롯에 K, V 쓰기"""
        self.key_cache[layer_id, block_id, slot_id] = key
        self.value_cache[layer_id, block_id, slot_id] = value
    
    def read(self, layer_id, block_table):
        """블록 테이블에서 K, V 읽기"""
        keys = []
        values = []
        for block_id in block_table:
            keys.append(self.key_cache[layer_id, block_id])
            values.append(self.value_cache[layer_id, block_id])
        
        # [num_blocks * block_size, num_heads, head_dim]
        K = torch.cat(keys, dim=0)
        V = torch.cat(values, dim=0)
        return K, V


class PagedAttention:
    def __init__(self, kv_cache, num_heads, head_dim):
        self.kv_cache = kv_cache
        self.num_heads = num_heads
        self.head_dim = head_dim
    
    def forward(self, query, layer_id, block_table, seq_len):
        """
        query: [num_heads, head_dim]
        block_table: [num_blocks]
        seq_len: 실제 토큰 수
        """
        # KV Cache에서 읽기
        K, V = self.kv_cache.read(layer_id, block_table)
        
        # 실제 길이만큼만 사용
        K = K[:seq_len]  # [seq_len, num_heads, head_dim]
        V = V[:seq_len]
        
        # Attention 계산
        query = query.unsqueeze(0)  # [1, num_heads, head_dim]
        
        scores = torch.matmul(query, K.transpose(-2, -1))  # [1, num_heads, seq_len]
        scores = scores / (self.head_dim ** 0.5)
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)  # [1, num_heads, head_dim]
        
        return output.squeeze(0)


# 사용 예제
def generate_with_paged_attention():
    # 초기화
    kv_cache = PagedKVCache(
        num_blocks=1024,
        block_size=16,
        num_layers=32,
        num_heads=32,
        head_dim=128
    )
    
    attention = PagedAttention(kv_cache, num_heads=32, head_dim=128)
    
    # 시퀀스 상태
    block_table = []
    seq_len = 0
    
    # 토큰 생성 루프
    for step in range(100):
        # 새 블록 필요?
        if seq_len % kv_cache.block_size == 0:
            new_block = kv_cache.allocate_block()
            block_table.append(new_block)
        
        # Forward pass (각 레이어마다)
        for layer_id in range(32):
            # Query 계산 (모델에서)
            query = get_query(layer_id)  # [num_heads, head_dim]
            
            # Paged Attention
            output = attention.forward(query, layer_id, block_table, seq_len)
            
            # KV Cache에 저장
            key, value = compute_kv(output)
            block_id = block_table[-1]
            slot_id = seq_len % kv_cache.block_size
            kv_cache.write(layer_id, block_id, slot_id, key, value)
        
        # 다음 토큰 생성
        next_token = sample_token(output)
        seq_len += 1
        
        if next_token == EOS:
            break
    
    # 메모리 해제
    for block_id in block_table:
        kv_cache.free_block(block_id)
```

---

## 실전 vLLM 사용

### 설치

```bash
pip install vllm
```

### 기본 사용

```python
from vllm import LLM, SamplingParams

# 모델 로드
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# 프롬프트들
prompts = [
    "Write a poem about AI:",
    "Explain quantum physics:",
    "Tell me a joke:",
]

# 샘플링 파라미터
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512
)

# 배치 생성
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### 고급 설정

```python
llm = LLM(
    model="meta-llama/Llama-2-13b-hf",
    tensor_parallel_size=2,     # 2 GPU 사용
    dtype="float16",
    gpu_memory_utilization=0.9, # GPU 메모리 90% 사용
    block_size=16,              # 페이지 크기
    max_num_seqs=256,           # 최대 배치 크기
)
```

---

## 한계와 트레이드오프

### 1. 추가 오버헤드

블록 테이블 관리와 불연속 메모리 읽기는 약간의 오버헤드가 있습니다.

**But**: 메모리 절약으로 더 큰 배치를 쓸 수 있어서 **전체 성능은 향상**

### 2. 블록 크기 선택

- **작은 블록** (8): 메모리 효율 ↑, 오버헤드 ↑
- **큰 블록** (32): 메모리 효율 ↓, 오버헤드 ↓

**vLLM 기본값: 16** (좋은 밸런스)

### 3. CUDA 커널 복잡도

표준 attention보다 구현이 복잡합니다. 하지만 vLLM이 다 해줘서 사용자는 신경 안 써도 됨!

---

## 요약

**Paged Attention**은:

1. **메모리를 페이지(블록) 단위로 관리**
2. **비연속 메모리 사용 가능**
3. **동적 할당으로 낭비 최소화**
4. **Copy-on-Write로 prefix 공유**

결과:
- **메모리: 8-10배 절약**
- **처리량: 20-30배 향상**
- **더 큰 배치, 더 긴 시퀀스 가능**

---

## 다음 글

**5편: KV Caching 완전 정복**
- KV Cache가 정확히 뭐길래?
- 메모리 레이아웃
- Multi-head attention 최적화

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
