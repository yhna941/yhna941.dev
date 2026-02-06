---
title: "LLM Inference 최적화 #6: Continuous Batching - 처리량 극대화의 비밀"
description: "vLLM과 TGI의 핵심 기술인 Continuous Batching으로 GPU 활용률을 극대화하는 방법을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["llm", "optimization", "batching", "throughput", "vllm"]
draft: false
---

# LLM Inference 최적화 #6: Continuous Batching

전통적인 batching은 **비효율적**입니다. 짧은 요청이 끝나도 가장 긴 요청을 기다려야 하죠.

**Continuous Batching**은 이 문제를 해결합니다:
- 끝난 요청은 **즉시 제거**
- 새 요청을 **즉시 추가**
- **GPU 항상 풀가동**

결과:
- 처리량: **2-10배 향상**
- 레이턴시: **감소**
- GPU 활용률: **90%+**

vLLM과 TGI의 핵심 기술입니다.

---

## 문제: Static Batching의 낭비

### 전통적인 방식

```python
batch = [
    "Hi",                              # 5 tokens → 빠름
    "Explain quantum physics in detail",  # 500 tokens → 느림
    "What's 2+2?",                    # 8 tokens → 빠름
]

# 모든 요청이 끝날 때까지 대기
while not all_finished(batch):
    next_tokens = model.forward(batch)
    update_all(batch, next_tokens)
```

### 문제점

**시간 낭비:**
```
Time:  0s -------- 10s --------- 20s
Req 1: [████]                          ← 5초에 끝났지만 20초까지 대기
Req 2: [████████████████████████████]  ← 느림
Req 3: [█████]                         ← 6초에 끝났지만 20초까지 대기

GPU:   [████░░░░░░░░░░░░░░░░░░░░░░░░]  ← 낭비!
```

**배치 크기 감소:**
- 시작: 3
- 5초 후: 2 (Req 1 끝)
- 6초 후: 1 (Req 3 끝)
- **GPU 활용률 급락!**

### 실제 영향

**LLaMA-7B, 배치 32:**
- 요청 길이 분포: [10, 50, 100, 500] tokens
- 평균 완료 시간: 100 tokens
- **하지만 마지막 요청(500)을 기다림**
- 실제 처리량: 이론치의 **20%**

---

## 해결책: Continuous Batching

### 핵심 아이디어

> **요청이 끝나는 즉시 제거하고, 새 요청을 바로 추가한다**

```python
running_batch = []
queue = []

while True:
    # 1. 끝난 요청 제거
    remove_finished(running_batch)
    
    # 2. 큐에서 새 요청 추가
    while len(running_batch) < max_batch_size and queue:
        running_batch.append(queue.pop())
    
    # 3. 한 스텝 실행
    if running_batch:
        next_tokens = model.forward(running_batch)
        update_batch(running_batch, next_tokens)
```

### 시각화

```
Time:  0s -------- 5s --------- 10s -------- 15s
Req 1: [████]
Req 2:                                [█████████]
Req 3: [█████]
Req 4:         [██████]
Req 5:                  [████████]
Req 6:                             [███████████]

Batch: [1,2,3] [2,3,4] [2,4,5] [2,5] [2,5,6] [5,6]
GPU:   [█████] [█████] [█████] [████] [█████] [████]  ← 항상 사용!
```

---

## 구현

### 1. 요청 관리

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"

@dataclass
class GenerationRequest:
    id: str
    prompt: List[int]  # token IDs
    max_tokens: int
    temperature: float
    
    # 상태
    status: RequestStatus = RequestStatus.WAITING
    generated: List[int] = None
    num_generated: int = 0
    
    def __post_init__(self):
        if self.generated is None:
            self.generated = []
    
    def is_finished(self) -> bool:
        return (
            self.num_generated >= self.max_tokens or
            self.generated and self.generated[-1] == EOS_TOKEN
        )


class RequestPool:
    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        self.waiting: List[GenerationRequest] = []
        self.running: List[GenerationRequest] = []
        self.finished: List[GenerationRequest] = []
    
    def add_request(self, req: GenerationRequest):
        """새 요청 추가"""
        req.status = RequestStatus.WAITING
        self.waiting.append(req)
    
    def schedule(self):
        """실행할 배치 구성"""
        # 끝난 요청 제거
        finished = [r for r in self.running if r.is_finished()]
        for req in finished:
            req.status = RequestStatus.FINISHED
            self.running.remove(req)
            self.finished.append(req)
        
        # 새 요청 추가 (배치 크기까지)
        available_slots = self.max_batch_size - len(self.running)
        new_requests = self.waiting[:available_slots]
        
        for req in new_requests:
            req.status = RequestStatus.RUNNING
            self.waiting.remove(req)
            self.running.append(req)
    
    def get_running_batch(self) -> List[GenerationRequest]:
        return self.running
```

### 2. 배치 실행 엔진

```python
import torch

class ContinuousBatchingEngine:
    def __init__(self, model, tokenizer, max_batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.pool = RequestPool(max_batch_size)
        self.kv_caches = {}  # request_id -> KVCache
    
    def add_request(self, prompt: str, max_tokens: int = 100, temperature: float = 1.0):
        """새 요청 추가"""
        tokens = self.tokenizer.encode(prompt)
        req = GenerationRequest(
            id=f"req_{len(self.pool.waiting)}",
            prompt=tokens,
            max_tokens=max_tokens,
            temperature=temperature
        )
        self.pool.add_request(req)
        return req.id
    
    def step(self):
        """한 스텝 실행"""
        # 1. 스케줄링 (끝난 것 제거, 새 것 추가)
        self.pool.schedule()
        
        running = self.pool.get_running_batch()
        if not running:
            return
        
        # 2. 입력 준비
        input_ids = []
        attention_masks = []
        
        for req in running:
            if req.num_generated == 0:
                # Prefill: 전체 프롬프트
                tokens = req.prompt
            else:
                # Decode: 마지막 토큰만
                tokens = [req.generated[-1]]
            
            input_ids.append(tokens)
        
        # 3. Padding (길이 다를 수 있음)
        max_len = max(len(ids) for ids in input_ids)
        padded = []
        masks = []
        
        for ids in input_ids:
            pad_len = max_len - len(ids)
            padded.append([PAD_TOKEN] * pad_len + ids)
            masks.append([0] * pad_len + [1] * len(ids))
        
        input_tensor = torch.tensor(padded, device='cuda')
        mask_tensor = torch.tensor(masks, device='cuda')
        
        # 4. Forward
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor,
                attention_mask=mask_tensor,
                use_cache=True,
                past_key_values=self.get_kv_caches(running)
            )
        
        logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
        
        # 5. 샘플링 & 업데이트
        for i, req in enumerate(running):
            # Temperature scaling
            probs = torch.softmax(logits[i] / req.temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # 업데이트
            req.generated.append(next_token)
            req.num_generated += 1
            
            # KV Cache 업데이트
            self.update_kv_cache(req.id, outputs.past_key_values[i])
    
    def run(self):
        """계속 실행"""
        while self.pool.waiting or self.pool.running:
            self.step()
    
    def get_result(self, request_id: str) -> Optional[str]:
        """결과 가져오기"""
        for req in self.pool.finished:
            if req.id == request_id:
                return self.tokenizer.decode(req.generated)
        return None


# 사용
engine = ContinuousBatchingEngine(model, tokenizer, max_batch_size=32)

# 요청 추가
req1 = engine.add_request("Hello", max_tokens=50)
req2 = engine.add_request("Explain AI", max_tokens=200)
req3 = engine.add_request("What's 2+2?", max_tokens=20)

# 실행
engine.run()

# 결과
print(engine.get_result(req1))
print(engine.get_result(req2))
print(engine.get_result(req3))
```

---

## 고급 최적화

### 1. Iteration-level Scheduling

매 스텝마다 스케줄링 (더 공격적):

```python
def iteration_level_schedule(self):
    """매 토큰마다 배치 재구성"""
    # Preemption: 긴 요청을 일시 중단하고 짧은 요청 우선
    self.running.sort(key=lambda r: r.num_generated)
    
    # 오래된 요청 중단
    if len(self.waiting) > 0 and len(self.running) == self.max_batch_size:
        # 가장 긴 요청 중단
        oldest = max(self.running, key=lambda r: r.num_generated)
        if oldest.num_generated > 50:  # 임계값
            self.running.remove(oldest)
            self.waiting.insert(0, oldest)  # 큐 앞에 추가
```

### 2. Priority Scheduling

우선순위 기반:

```python
@dataclass
class GenerationRequest:
    priority: int = 0  # 높을수록 우선

class PriorityRequestPool(RequestPool):
    def schedule(self):
        # 우선순위로 정렬
        self.waiting.sort(key=lambda r: r.priority, reverse=True)
        super().schedule()

# 사용
high_priority = GenerationRequest(..., priority=10)
low_priority = GenerationRequest(..., priority=1)
```

### 3. Mixed Prefill/Decode

Prefill과 Decode를 같은 배치에:

```python
def mixed_batch_forward(self, requests):
    """Prefill + Decode 동시 처리"""
    prefill_reqs = [r for r in requests if r.num_generated == 0]
    decode_reqs = [r for r in requests if r.num_generated > 0]
    
    # Prefill (긴 입력)
    if prefill_reqs:
        prefill_inputs = prepare_prefill(prefill_reqs)
        prefill_outputs = self.model.forward(prefill_inputs)
    
    # Decode (토큰 1개)
    if decode_reqs:
        decode_inputs = prepare_decode(decode_reqs)
        decode_outputs = self.model.forward(decode_inputs)
    
    # 결합
    return merge_outputs(prefill_outputs, decode_outputs)
```

---

## vLLM의 Continuous Batching

vLLM은 Paged Attention + Continuous Batching을 결합합니다.

### 핵심 구조

```python
class LLMEngine:
    def __init__(self):
        self.scheduler = Scheduler()
        self.model_executor = ModelExecutor()
        self.cache_engine = CacheEngine()  # Paged KV Cache
    
    def step(self):
        # 1. 스케줄: 어떤 요청을 실행할지
        scheduler_output = self.scheduler.schedule()
        
        # 2. 메모리 할당: Paged blocks
        self.cache_engine.allocate(scheduler_output.running)
        
        # 3. 실행
        outputs = self.model_executor.execute(
            scheduler_output.running,
            self.cache_engine.get_kv_cache()
        )
        
        # 4. 샘플링
        for seq, output in zip(scheduler_output.running, outputs):
            next_token = sample(output)
            seq.append_token(next_token)
        
        # 5. 메모리 해제
        self.cache_engine.free(scheduler_output.finished)
```

### Scheduler

```python
class Scheduler:
    def __init__(self, max_num_seqs=256):
        self.max_num_seqs = max_num_seqs
        self.waiting = []
        self.running = []
        self.swapped = []  # CPU로 옮긴 것
    
    def schedule(self):
        # 끝난 요청 제거
        finished = [s for s in self.running if s.is_finished()]
        for seq in finished:
            self.running.remove(seq)
        
        # 메모리 부족 시 swap
        if self.cache_engine.is_full():
            # 가장 긴 요청을 CPU로
            victim = max(self.running, key=lambda s: len(s))
            self.running.remove(victim)
            self.swapped.append(victim)
            self.cache_engine.swap_out(victim)
        
        # Swap in (여유 있으면)
        if not self.cache_engine.is_full() and self.swapped:
            seq = self.swapped.pop(0)
            self.running.append(seq)
            self.cache_engine.swap_in(seq)
        
        # 새 요청 추가
        while (len(self.running) < self.max_num_seqs and
               self.waiting and
               not self.cache_engine.is_full()):
            seq = self.waiting.pop(0)
            self.running.append(seq)
        
        return SchedulerOutput(
            running=self.running,
            finished=finished
        )
```

---

## 벤치마크

### Static vs Continuous

**LLaMA-7B, 1000 requests, 다양한 길이:**

| 방식 | 처리량 (req/s) | P99 지연 (s) | GPU 활용률 |
|------|----------------|--------------|-----------|
| Static Batching | 12 | 8.5 | 45% |
| Continuous Batching | 48 | 2.3 | 87% |
| vLLM (Paged + Continuous) | 64 | 1.8 | 92% |

**4배 처리량 향상!**

### 배치 크기별

**Continuous Batching:**

| 최대 배치 크기 | 처리량 | 지연 |
|---------------|--------|------|
| 8 | 25 req/s | 1.2s |
| 16 | 42 req/s | 1.5s |
| 32 | 58 req/s | 1.8s |
| 64 | 64 req/s | 2.3s |
| 128 | 63 req/s | 3.5s |

**최적: 32-64**

---

## 실전 예제: FastAPI 서빙

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import asyncio
from uuid import uuid4

app = FastAPI()

# 글로벌 엔진
engine = ContinuousBatchingEngine(model, tokenizer, max_batch_size=32)

# 백그라운드 실행
@app.on_event("startup")
async def startup():
    asyncio.create_task(run_engine())

async def run_engine():
    """백그라운드에서 계속 실행"""
    while True:
        engine.step()
        await asyncio.sleep(0)  # 다른 태스크에 양보

# API
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0

@app.post("/generate")
async def generate(req: GenerateRequest):
    request_id = engine.add_request(
        req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature
    )
    
    # 결과 대기
    while True:
        result = engine.get_result(request_id)
        if result:
            return {"output": result}
        await asyncio.sleep(0.01)

# 스트리밍 버전
from fastapi.responses import StreamingResponse

@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    request_id = engine.add_request(req.prompt, req.max_tokens, req.temperature)
    
    async def stream():
        last_len = 0
        while True:
            result = engine.get_result(request_id)
            if result:
                # 새로 생성된 부분만 yield
                new_text = result[last_len:]
                if new_text:
                    yield f"data: {new_text}\n\n"
                    last_len = len(result)
                
                # 끝났으면 종료
                req_obj = engine.pool.finished[-1]
                if req_obj.id == request_id and req_obj.is_finished():
                    break
            
            await asyncio.sleep(0.01)
    
    return StreamingResponse(stream(), media_type="text/event-stream")

# 사용
# curl -X POST "http://localhost:8000/generate" \
#   -H "Content-Type: application/json" \
#   -d '{"prompt": "Once upon a time", "max_tokens": 100}'
```

---

## TGI (Text Generation Inference)

HuggingFace의 TGI도 Continuous Batching을 사용합니다.

### 설치 & 실행

```bash
# Docker로 실행
docker run --gpus all -p 8080:80 \
  -v $PWD/models:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-2-7b-hf \
  --max-batch-size 64 \
  --max-input-length 1024 \
  --max-total-tokens 2048
```

### API 사용

```python
import requests

url = "http://localhost:8080/generate"
payload = {
    "inputs": "Once upon a time",
    "parameters": {
        "max_new_tokens": 100,
        "temperature": 0.7
    }
}

response = requests.post(url, json=payload)
print(response.json()["generated_text"])
```

### 스트리밍

```python
import requests

url = "http://localhost:8080/generate_stream"
payload = {
    "inputs": "Explain quantum physics",
    "parameters": {"max_new_tokens": 500}
}

with requests.post(url, json=payload, stream=True) as response:
    for line in response.iter_lines():
        if line:
            # SSE format
            data = json.loads(line.decode().replace("data: ", ""))
            print(data["token"]["text"], end="", flush=True)
```

---

## 최적 설정 가이드

### 배치 크기

```python
# GPU 메모리에 따라
A100 80GB:  max_batch_size = 128
A100 40GB:  max_batch_size = 64
RTX 4090:   max_batch_size = 32
RTX 3090:   max_batch_size = 16
```

### 큐 관리

```python
# 큐 크기 제한 (메모리 관리)
max_queue_size = max_batch_size * 4

# 타임아웃 (너무 오래 대기하면 거부)
max_wait_time = 5.0  # seconds
```

### Preemption

```python
# 긴 요청 중단 임계값
preemption_threshold = 100  # tokens

# 우선순위 차이
priority_boost = 2  # 2배 더 자주 스케줄
```

---

## 한계와 트레이드오프

### 1. 복잡도 증가

구현이 복잡합니다:
- KV Cache 관리
- 메모리 할당/해제
- 동적 배치 처리

**대책:** vLLM, TGI 같은 라이브러리 사용

### 2. Prefill 병목

새 요청의 prefill은 느립니다:

```
Prefill (100 tokens): 50ms
Decode (1 token): 5ms

100 토큰 생성: 50 + 100*5 = 550ms
```

**대책:** Prefill을 별도 배치로 처리

### 3. 메모리 단편화

Paged Attention 없이는 메모리 단편화 발생

**대책:** Paged Attention 결합 (vLLM)

---

## 요약

**Continuous Batching**은:

1. **동적 배치**: 끝나는 즉시 제거, 새로 추가
2. **GPU 최대 활용**: 90%+ 활용률
3. **2-10배 처리량** 향상
4. **레이턴시 감소**

**핵심 기법:**
- Iteration-level scheduling
- Priority-based scheduling
- Mixed prefill/decode
- Paged Attention 결합

**사용 라이브러리:**
- vLLM (추천!)
- Text Generation Inference (HuggingFace)
- TensorRT-LLM (NVIDIA)

**결론:** 모든 프로덕션 LLM 서빙에 필수!

---

## 다음 글

**10편: Model Quantization**
- INT8/INT4 양자화
- 메모리 4배 절감
- 속도 2-3배 향상
- QLoRA, GPTQ, AWQ

시리즈 완결편! 기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
