---
title: "LLM Inference 최적화 #9: Pipeline Parallelism - 레이어를 파이프라인처럼 흘려보내기"
description: "여러 GPU에 레이어를 순차 배치하고 micro-batching으로 효율을 극대화하는 Pipeline Parallelism을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["llm", "optimization", "parallelism", "pipeline", "distributed"]
draft: false
---

# LLM Inference 최적화 #9: Pipeline Parallelism

Tensor Parallelism은 통신이 많습니다. 레이어마다 All-reduce가 필요하죠.

**Pipeline Parallelism**은 다른 접근입니다:
- **레이어를 GPU에 순차 배치**
- **Activation만 GPU 간 전송**
- **Micro-batching으로 병렬화**
- **통신: 레이어당 1번**

결과:
- 통신량: Tensor Parallel의 1/10
- GPU 활용률: 50-80% (bubble 최소화 필요)
- 구현: 비교적 간단

---

## 문제: Naive Pipeline의 Bubble

### Naive 방식

```
GPU 0: [Layer 1-8]
GPU 1: [Layer 9-16]
GPU 2: [Layer 17-24]
GPU 3: [Layer 25-32]

Timeline:
GPU 0: [████████]░░░░░░░░░░░░░░░░░░░░░░░░
GPU 1: ░░░░░░░░[████████]░░░░░░░░░░░░░░░░
GPU 2: ░░░░░░░░░░░░░░░░[████████]░░░░░░░░
GPU 3: ░░░░░░░░░░░░░░░░░░░░░░░░[████████]

Bubble: 75% idle time!
```

**문제:**
- GPU가 순차적으로만 동작
- 3개 GPU는 놀고 있음
- 활용률: 25%

---

## 해결책: Micro-batching

### 핵심 아이디어

> **배치를 작은 micro-batch로 나누고, 파이프라인처럼 흘려보낸다**

```
Batch 32 → 4 micro-batches (size 8 each)

Timeline:
GPU 0: [MB1][MB2][MB3][MB4]
GPU 1:     [MB1][MB2][MB3][MB4]
GPU 2:         [MB1][MB2][MB3][MB4]
GPU 3:             [MB1][MB2][MB3][MB4]

Bubble reduced!
```

### 예시 (4 GPUs, 4 micro-batches)

```
Time:  0  1  2  3  4  5  6  7
GPU 0: 1  2  3  4  .  .  .  .
GPU 1: .  1  2  3  4  .  .  .
GPU 2: .  .  1  2  3  4  .  .
GPU 3: .  .  .  1  2  3  4  .

Legend:
  1,2,3,4: Micro-batch ID
  .: Bubble (idle)

Bubble: 3/(7) = 43% (better!)
```

### Bubble 계산

```
num_stages = 4 (GPUs)
num_microbatches = m

Bubble fraction = (num_stages - 1) / (m + num_stages - 1)

m=1:  3/4 = 75%
m=4:  3/7 = 43%
m=8:  3/11 = 27%
m=16: 3/19 = 16%
```

**m을 늘리면 bubble 감소!**

---

## GPipe Schedule

### 1F1B (One Forward One Backward)

효율적인 스케줄:

```python
# GPipe schedule
def gpipe_schedule(num_stages, num_microbatches):
    schedule = []
    
    # Warmup: Fill pipeline (forward only)
    for i in range(num_stages - 1):
        schedule.append(('F', i))  # Forward micro-batch i
    
    # Steady: 1F1B (one forward, one backward)
    for i in range(num_microbatches - num_stages + 1):
        schedule.append(('F', i + num_stages - 1))
        schedule.append(('B', i))
    
    # Cooldown: Empty pipeline (backward only)
    for i in range(num_stages - 1):
        schedule.append(('B', num_microbatches - num_stages + 1 + i))
    
    return schedule

# Example: 4 stages, 8 micro-batches
schedule = gpipe_schedule(4, 8)
# Stage 0: F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
# Stage 1:    F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 B6 B7
# ...
```

### 메모리 효율

**GPipe:**
- Forward pass의 activation을 모두 저장 (backward 위해)
- 메모리: O(num_microbatches × layers_per_stage)

**문제:** m이 크면 메모리 부족!

---

## PipeDream-Flush (Improved)

### 개선된 스케줄

```python
def pipedream_flush_schedule(num_stages, num_microbatches):
    """
    메모리 효율적인 스케줄
    Forward와 Backward를 빨리 연결
    """
    schedule = []
    in_flight = 0  # Forward done, backward not done
    
    for step in range(num_microbatches + num_stages - 1):
        # Forward if possible
        if step < num_microbatches:
            schedule.append(('F', step))
            in_flight += 1
        
        # Backward if possible
        backward_idx = step - (num_stages - 1)
        if backward_idx >= 0:
            schedule.append(('B', backward_idx))
            in_flight -= 1
    
    return schedule
```

**특징:**
- Activation을 빨리 해제 (메모리 ↓)
- Bubble은 비슷

---

## 구현

### 1. 기본 Pipeline Stage

```python
import torch
import torch.distributed.rpc as rpc

class PipelineStage(nn.Module):
    def __init__(self, layers, stage_id, num_stages):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.is_first = (stage_id == 0)
        self.is_last = (stage_id == num_stages - 1)
    
    def forward(self, x):
        """단일 micro-batch forward"""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def forward_microbatch(self, micro_batch):
        """Micro-batch 처리"""
        # Forward
        output = self.forward(micro_batch)
        
        # 다음 stage로 전송
        if not self.is_last:
            next_stage = self.stage_id + 1
            rpc.rpc_async(
                f"worker{next_stage}",
                PipelineStage.forward_microbatch,
                args=(output,)
            )
        
        return output


def create_pipeline(model, num_stages):
    """모델을 stages로 분할"""
    layers = list(model.children())
    layers_per_stage = len(layers) // num_stages
    
    stages = []
    for i in range(num_stages):
        start_idx = i * layers_per_stage
        end_idx = start_idx + layers_per_stage if i < num_stages - 1 else len(layers)
        stage_layers = layers[start_idx:end_idx]
        stages.append(PipelineStage(stage_layers, i, num_stages))
    
    return stages
```

### 2. Micro-batch 처리

```python
class PipelineParallelEngine:
    def __init__(self, model, num_stages, num_microbatches):
        self.stages = create_pipeline(model, num_stages)
        self.num_microbatches = num_microbatches
        self.stage_id = dist.get_rank()
    
    def split_batch(self, batch, num_splits):
        """배치를 micro-batches로 분할"""
        batch_size = batch.size(0)
        micro_batch_size = batch_size // num_splits
        
        micro_batches = []
        for i in range(num_splits):
            start = i * micro_batch_size
            end = start + micro_batch_size
            micro_batches.append(batch[start:end])
        
        return micro_batches
    
    def forward(self, batch):
        """Pipeline forward pass"""
        micro_batches = self.split_batch(batch, self.num_microbatches)
        
        # 현재 stage
        stage = self.stages[self.stage_id]
        
        # Activation 저장 (backward 위해)
        activations = []
        
        # GPipe schedule 실행
        schedule = gpipe_schedule(len(self.stages), self.num_microbatches)
        
        for action, mb_idx in schedule:
            if action == 'F':
                # Forward
                if self.stage_id == 0:
                    # First stage: Use input
                    input_mb = micro_batches[mb_idx]
                else:
                    # Receive from previous stage
                    input_mb = self.recv_activation()
                
                output_mb = stage(input_mb)
                activations.append(output_mb)
                
                if self.stage_id < len(self.stages) - 1:
                    # Send to next stage
                    self.send_activation(output_mb)
            
            elif action == 'B':
                # Backward
                if self.stage_id == len(self.stages) - 1:
                    # Last stage: Compute loss gradient
                    grad_output = self.compute_loss_gradient(activations[mb_idx])
                else:
                    # Receive gradient from next stage
                    grad_output = self.recv_gradient()
                
                # Backward pass
                activation = activations[mb_idx]
                activation.backward(grad_output)
                
                if self.stage_id > 0:
                    # Send gradient to previous stage
                    self.send_gradient(activation.grad)
        
        # Aggregate results
        if self.stage_id == len(self.stages) - 1:
            outputs = torch.cat([a.detach() for a in activations], dim=0)
            return outputs
    
    def send_activation(self, tensor):
        """다음 stage로 activation 전송"""
        next_rank = self.stage_id + 1
        dist.send(tensor, dst=next_rank)
    
    def recv_activation(self):
        """이전 stage에서 activation 수신"""
        prev_rank = self.stage_id - 1
        tensor = torch.empty_like(...)  # Shape must be known
        dist.recv(tensor, src=prev_rank)
        return tensor
    
    def send_gradient(self, tensor):
        """이전 stage로 gradient 전송"""
        prev_rank = self.stage_id - 1
        dist.send(tensor, dst=prev_rank)
    
    def recv_gradient(self):
        """다음 stage에서 gradient 수신"""
        next_rank = self.stage_id + 1
        tensor = torch.empty_like(...)
        dist.recv(tensor, src=next_rank)
        return tensor
```

### 3. DeepSpeed Pipeline

```python
from deepspeed.pipe import PipelineModule, LayerSpec

# 모델 정의 (layers as specs)
layers = [
    LayerSpec(nn.Linear, 768, 3072),
    LayerSpec(nn.GELU),
    LayerSpec(nn.Linear, 3072, 768),
    # ... repeat 32 times
]

# Pipeline 생성
model = PipelineModule(
    layers=layers,
    num_stages=4,  # 4 GPUs
    partition_method='uniform'  # or 'parameters'
)

# DeepSpeed 초기화
engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config={
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 1,
        "pipeline": {
            "pipe_partitioned": True,
            "grad_partitioned": True
        }
    }
)

# 학습
for batch in dataloader:
    loss = engine.train_batch(batch)
```

---

## Interleaved Pipeline (GPipe-1F1B)

### 문제: Bubble이 여전히 큼

```
4 stages, 8 micro-batches:
Bubble = 3/11 = 27%
```

### 해결책: Interleaving

각 GPU가 여러 stage를 담당:

```
# Standard
GPU 0: Layers 1-8
GPU 1: Layers 9-16
GPU 2: Layers 17-24
GPU 3: Layers 25-32

# Interleaved (2-way)
GPU 0: Layers 1-4, 17-20
GPU 1: Layers 5-8, 21-24
GPU 2: Layers 9-12, 25-28
GPU 3: Layers 13-16, 29-32
```

**효과:**
- Bubble 감소
- 메모리 약간 증가

```python
def create_interleaved_pipeline(model, num_stages, num_model_chunks):
    """
    Interleaved pipeline 생성
    
    Args:
        num_stages: GPU 개수
        num_model_chunks: 각 GPU가 담당하는 chunk 수
    """
    layers = list(model.children())
    total_chunks = num_stages * num_model_chunks
    layers_per_chunk = len(layers) // total_chunks
    
    stage_layers = [[] for _ in range(num_stages)]
    
    for chunk_id in range(total_chunks):
        stage_id = chunk_id % num_stages
        start_idx = chunk_id * layers_per_chunk
        end_idx = start_idx + layers_per_chunk
        stage_layers[stage_id].extend(layers[start_idx:end_idx])
    
    return stage_layers
```

---

## 메모리 관리

### Activation Checkpointing

메모리 줄이기:

```python
class CheckpointedLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x):
        # Activation을 저장 안 함
        # Backward 시 recompute
        return torch.utils.checkpoint.checkpoint(self.layer, x)
```

**Trade-off:**
- 메모리: ↓↓
- 계산: ↑ (recompute)
- 전체 속도: 비슷 (메모리 병목 해소)

### Selective Checkpointing

일부만 checkpoint:

```python
def create_checkpointed_pipeline(layers, checkpoint_every=4):
    """N개 layer마다 checkpoint"""
    wrapped = []
    for i, layer in enumerate(layers):
        if i % checkpoint_every == 0:
            wrapped.append(CheckpointedLayer(layer))
        else:
            wrapped.append(layer)
    return wrapped
```

---

## 통신 최적화

### 1. Tensor Compression

Activation 압축:

```python
def compress_activation(tensor, bits=8):
    """8-bit로 압축해서 전송"""
    # Quantize
    scale = tensor.abs().max() / 127
    quantized = (tensor / scale).round().to(torch.int8)
    
    # Send quantized + scale
    return quantized, scale

def decompress_activation(quantized, scale):
    """복원"""
    return quantized.float() * scale
```

### 2. Overlapping

통신과 계산 겹치기:

```python
def overlapped_forward(stage, input_tensor):
    """
    계산과 통신 동시 진행
    """
    # 1. 이전 stage에서 수신 시작 (비동기)
    recv_handle = dist.irecv(input_tensor, src=stage.prev_rank, async_op=True)
    
    # 2. 다른 작업 (예: normalization)
    # ...
    
    # 3. 수신 완료 대기
    recv_handle.wait()
    
    # 4. Forward
    output = stage(input_tensor)
    
    # 5. 다음 stage로 전송 시작 (비동기)
    send_handle = dist.isend(output, dst=stage.next_rank, async_op=True)
    
    # 6. 다른 작업
    # ...
    
    # 7. 전송 완료 대기
    send_handle.wait()
    
    return output
```

---

## Hybrid: Pipeline + Tensor Parallelism

최고의 성능:

```
16 GPUs = 4 pipeline stages × 4 tensor parallel per stage

GPU  0,1,2,3:  Layers 1-8   (TP=4)
GPU  4,5,6,7:  Layers 9-16  (TP=4)
GPU 8,9,10,11: Layers 17-24 (TP=4)
GPU 12,13,14,15: Layers 25-32 (TP=4)
```

**장점:**
- Pipeline: 통신 적음
- Tensor Parallel: Bubble 적음
- 최고 효율!

```python
# DeepSpeed 3D parallelism
config = {
    "pipeline": {
        "stages": 4,
        "micro_batches": 16
    },
    "tensor_parallel": {
        "size": 4
    },
    "data_parallel": {
        "size": 2
    }
}

# 총 GPU: 4 × 4 × 2 = 32
```

---

## 벤치마크

### GPT-3 175B, 64 GPUs

| 방식 | 처리량 (samples/s) | GPU 효율 |
|------|-------------------|---------|
| Tensor Parallel (64) | 85 | 60% |
| Pipeline (8 stages) | 120 | 45% |
| **Hybrid (PP=8, TP=8)** | **280** | **75%** |

### Bubble 비교

| Schedule | Bubble | 메모리 |
|----------|--------|--------|
| Naive | 75% | 낮음 |
| GPipe | 27% (m=8) | 높음 |
| PipeDream-Flush | 27% | 중간 |
| Interleaved | 15% | 높음 |

---

## 실전 예제

### Megatron-LM Style

```python
from megatron import get_args
from megatron.model import GPTModel
from megatron.training import train

# 설정
args = get_args()
args.pipeline_model_parallel_size = 4
args.tensor_model_parallel_size = 2
args.micro_batch_size = 4
args.global_batch_size = 32

# 모델
model = GPTModel(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    vocab_size=50257,
    max_position_embeddings=2048
)

# 학습
train(model)
```

### Fairscale

```python
from fairscale.nn import Pipe

# 모델 → Sequential
layers = [
    nn.Linear(768, 3072),
    nn.GELU(),
    nn.Linear(3072, 768),
    # ... 32 times
]

model = nn.Sequential(*layers)

# Pipeline wrapping
model = Pipe(
    model,
    balance=[8, 8, 8, 8],  # Layers per GPU
    chunks=8,              # Micro-batches
    checkpoint='except_last'  # Activation checkpointing
)

# 사용
outputs = model(inputs)
```

---

## Best Practices

### 1. Micro-batch 크기 선택

```python
# Rule of thumb
num_microbatches = 4 × num_pipeline_stages

# 예: 8 stages → 32 micro-batches
```

### 2. Balance 조정

```python
# 각 stage가 비슷한 시간 소요하도록
balance = [7, 8, 9, 8]  # Layer counts

# 자동 balancing
balance = auto_balance(model, num_stages, profile=True)
```

### 3. Checkpoint 전략

```python
# 큰 레이어는 checkpoint
if layer.num_parameters() > threshold:
    layer = CheckpointedLayer(layer)
```

---

## 요약

**Pipeline Parallelism**은:

1. **레이어를 GPU에 순차 배치**
2. **Micro-batching**으로 병렬화
3. **통신: Activation만** (Tensor Parallel보다 적음)
4. **Bubble: 15-30%** (최적화 시)
5. **구현: 비교적 간단**

**핵심 기법:**
- GPipe schedule (1F1B)
- Interleaved pipeline
- Activation checkpointing
- Hybrid (PP + TP)

**사용처:**
- Large model (70B+)
- 통신 대역폭 낮을 때
- Tensor Parallel과 조합

---

## 다음 글

**13편: Model Compression**
- Pruning (가지치기)
- Distillation (지식 증류)
- 정확도 유지하며 크기 줄이기

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
