---
title: "LLM Inference 최적화 #5: Speculative Decoding - 작은 모델로 큰 모델 가속하기"
description: "Draft 모델로 추론 속도를 2-3배 높이는 Speculative Decoding의 원리와 실전 구현을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["llm", "optimization", "speculative-decoding", "inference", "speed"]
draft: false
---

# LLM Inference 최적화 #5: Speculative Decoding

LLM은 한 번에 **한 토큰씩** 생성합니다. 병렬화가 안 되니 느립니다.

**Speculative Decoding**은 이 문제를 해결합니다:
- 작은 모델이 **여러 토큰을 추측**
- 큰 모델이 **한 번에 검증**
- **2-3배 빠름**
- **결과는 동일** (무손실!)

마법 같지만 수학적으로 보장됩니다.

---

## 문제: Autoregressive는 느리다

### 순차 생성

```
Step 1: "The" → "cat"
Step 2: "The cat" → "is"
Step 3: "The cat is" → "sleeping"
...
```

각 단계마다:
1. 전체 모델 실행 (70B 파라미터!)
2. 1개 토큰 생성
3. 다음 단계

**100 토큰 생성 = 100번 모델 실행**

### GPU 활용률이 낮음

```
GPU Utilization during inference:
[████░░░░░░░░░░░░░░░░] 20%
```

**왜?**
- Memory-bound (계산보다 메모리 읽기가 병목)
- 배치 크기 1 (한 토큰씩)
- 병렬화 불가

---

## 해결책: Speculative Decoding

### 핵심 아이디어

> **작은 모델(draft)이 여러 토큰을 빠르게 추측하고,**  
> **큰 모델(target)이 한 번에 검증한다**

```
Draft model (1B):  "The cat is sleeping on the"  ← 빠름 (6 tokens)
Target model (70B): "The cat is sleeping"  ← 검증 (4 tokens 승인)

결과: 1번의 target 실행으로 4 토큰 생성!
```

### 왜 빠른가?

**Standard (4 tokens):**
```
Target("The") → "cat"
Target("The cat") → "is"
Target("The cat is") → "sleeping"
Target("The cat is sleeping") → "on"

총: 4번 실행
```

**Speculative (4 tokens):**
```
Draft("The") → "cat is sleeping on the"  (빠름)
Target("The", candidates=["cat", "is", "sleeping", "on", "the"])
  → ["cat", "is", "sleeping"] 승인, ["on", "the"] 거부

총: Draft 6번 + Target 1번
```

Draft가 70B보다 **10-20배 빠르니** 전체적으로 빠릅니다!

---

## 알고리즘

### Step-by-step

**1. Draft 단계 (추측)**
```python
def draft_phase(prompt, draft_model, K=5):
    """작은 모델로 K개 토큰 추측"""
    tokens = [prompt]
    
    for _ in range(K):
        next_token = draft_model.sample(tokens)
        tokens.append(next_token)
    
    return tokens  # [prompt, t1, t2, ..., tK]
```

**2. Target 단계 (검증)**
```python
def target_phase(tokens, target_model):
    """큰 모델로 한 번에 검증"""
    # 모든 prefix에 대해 확률 계산
    probs = target_model.forward(tokens)  # [K+1, vocab_size]
    
    accepted = []
    for i in range(len(tokens) - 1):
        draft_token = tokens[i + 1]
        target_prob = probs[i]
        
        if should_accept(draft_token, target_prob):
            accepted.append(draft_token)
        else:
            # 거부: target에서 새로 샘플링
            new_token = target_model.sample(target_prob)
            accepted.append(new_token)
            break  # 이후는 무효
    
    return accepted
```

### 수학적 보장

**핵심 질문:** 어떻게 결과가 정확히 같을까?

**답:** Modified Rejection Sampling

**Draft 확률:** p(x)  
**Target 확률:** q(x)

**Accept 확률:**
```
α(x) = min(1, q(x) / p(x))
```

**거부 시 재샘플링:**
```
q'(x) = max(0, q(x) - p(x)) / Z
```

이렇게 하면 **수학적으로 q(x)와 동일한 분포**!

---

## 구현 예제

### 1. 기본 구현

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SpeculativeDecoder:
    def __init__(self, draft_model, target_model, tokenizer):
        self.draft = draft_model
        self.target = target_model
        self.tokenizer = tokenizer
    
    def generate(self, prompt, max_tokens=100, K=5):
        """
        Args:
            prompt: 입력 텍스트
            max_tokens: 생성할 최대 토큰 수
            K: Draft가 추측할 토큰 수
        """
        tokens = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        generated = []
        
        while len(generated) < max_tokens:
            # 1. Draft phase
            draft_tokens = self.draft_phase(tokens, K)
            
            # 2. Target phase (검증)
            accepted, new_token = self.target_phase(tokens, draft_tokens)
            
            # 3. 승인된 토큰들 추가
            generated.extend(accepted)
            tokens = torch.cat([tokens, torch.tensor([accepted]).to('cuda')], dim=-1)
            
            # 4. 거부된 경우 새 토큰 추가
            if new_token is not None:
                generated.append(new_token)
                tokens = torch.cat([tokens, torch.tensor([[new_token]]).to('cuda')], dim=-1)
            
            # EOS 체크
            if generated[-1] == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(generated)
    
    def draft_phase(self, tokens, K):
        """Draft model로 K개 토큰 추측"""
        draft_tokens = []
        current = tokens.clone()
        
        with torch.no_grad():
            for _ in range(K):
                logits = self.draft(current).logits[:, -1, :]
                next_token = torch.multinomial(
                    torch.softmax(logits, dim=-1), 
                    num_samples=1
                )
                draft_tokens.append(next_token.item())
                current = torch.cat([current, next_token], dim=-1)
        
        return draft_tokens
    
    def target_phase(self, tokens, draft_tokens):
        """Target model로 검증"""
        # 모든 draft token을 한 번에 처리
        all_tokens = torch.cat([
            tokens,
            torch.tensor([draft_tokens]).to('cuda')
        ], dim=-1)
        
        with torch.no_grad():
            logits = self.target(all_tokens).logits[0]  # [seq_len, vocab_size]
        
        accepted = []
        new_token = None
        
        for i, draft_token in enumerate(draft_tokens):
            # Target의 확률 분포
            target_probs = torch.softmax(logits[tokens.shape[1] + i - 1], dim=-1)
            
            # Draft의 확률 (재계산)
            draft_logits = self.draft(
                torch.cat([tokens, torch.tensor([draft_tokens[:i]]).to('cuda')], dim=-1)
            ).logits[:, -1, :]
            draft_probs = torch.softmax(draft_logits, dim=-1)
            
            # Rejection sampling
            accept_prob = min(1.0, 
                target_probs[draft_token].item() / 
                (draft_probs[0, draft_token].item() + 1e-10)
            )
            
            if torch.rand(1).item() < accept_prob:
                accepted.append(draft_token)
            else:
                # 거부: target에서 새로 샘플링
                adjusted_probs = torch.clamp(target_probs - draft_probs[0], min=0)
                adjusted_probs = adjusted_probs / adjusted_probs.sum()
                new_token = torch.multinomial(adjusted_probs, num_samples=1).item()
                break
        
        return accepted, new_token


# 사용
draft_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B", device_map="cuda")
target_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

decoder = SpeculativeDecoder(draft_model, target_model, tokenizer)
output = decoder.generate("Once upon a time", max_tokens=100, K=5)
print(output)
```

### 2. 최적화된 버전 (배치 처리)

```python
class FastSpeculativeDecoder:
    def target_phase_batched(self, tokens, draft_tokens):
        """배치로 한 번에 처리"""
        K = len(draft_tokens)
        
        # 모든 prefix를 배치로 구성
        # [prefix, prefix+t1, prefix+t1+t2, ...]
        batch = []
        for i in range(K):
            batch.append(torch.cat([
                tokens,
                torch.tensor([draft_tokens[:i+1]]).to('cuda')
            ], dim=-1))
        
        # Padding
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
        
        # 한 번에 실행!
        with torch.no_grad():
            logits = self.target(batch).logits  # [K, seq_len, vocab_size]
        
        # 각 위치에서 확률 추출
        target_probs = torch.softmax(logits[:, -1, :], dim=-1)  # [K, vocab_size]
        
        # 검증 (벡터화)
        draft_probs = self.get_draft_probs_batched(tokens, draft_tokens)
        accept_probs = torch.minimum(
            torch.ones(K),
            target_probs[torch.arange(K), draft_tokens] / 
            (draft_probs[torch.arange(K), draft_tokens] + 1e-10)
        )
        
        # 첫 거부 지점까지 승인
        random_vals = torch.rand(K)
        accepted_mask = random_vals < accept_probs
        first_reject = torch.where(~accepted_mask)[0]
        
        if len(first_reject) > 0:
            accept_until = first_reject[0].item()
            accepted = draft_tokens[:accept_until]
            # 새 토큰 샘플링
            adjusted = torch.clamp(
                target_probs[accept_until] - draft_probs[accept_until],
                min=0
            )
            new_token = torch.multinomial(adjusted / adjusted.sum(), 1).item()
        else:
            accepted = draft_tokens
            new_token = None
        
        return accepted, new_token
```

---

## 성능 분석

### 이론적 속도업

**K개 토큰 추측, 평균 α개 승인:**

```
Speedup = α / (K * t_draft + t_target)

where:
  α: 평균 승인 토큰 수 (acceptance rate)
  K: Draft 추측 수
  t_draft: Draft 시간 (작음)
  t_target: Target 시간 (김)
```

**예시:**
- K = 5
- α = 3 (60% acceptance)
- t_draft = 0.1ms
- t_target = 10ms

```
Standard: 3 tokens = 30ms
Speculative: 3 tokens = 5*0.1 + 10 = 10.5ms

Speedup: 30 / 10.5 = 2.86x
```

### 실제 벤치마크

**LLaMA-7B (target) + TinyLlama-1B (draft):**

| K | Acceptance Rate | Tokens/sec | Speedup |
|---|-----------------|------------|---------|
| 3 | 65% | 42 | 1.8x |
| 5 | 60% | 58 | 2.5x |
| 7 | 55% | 63 | 2.7x |
| 10 | 50% | 61 | 2.6x |

**최적 K = 5-7**

---

## 고급 기법

### 1. Tree-based Speculative Decoding

**아이디어:** 여러 후보를 트리로 탐색

```
                    "The"
                   /  |  \
                cat  dog  bird
               / |    |     |
             is sat  ran  flew
```

Draft가 여러 경로를 생성 → Target이 한 번에 검증

**장점:** Acceptance rate ↑  
**단점:** 메모리 ↑

```python
def tree_draft(prompt, draft_model, tree_depth=2, branching=3):
    """트리 구조로 후보 생성"""
    root = TreeNode(prompt)
    queue = [root]
    
    for level in range(tree_depth):
        new_queue = []
        for node in queue:
            # 각 노드에서 branching개 후보 생성
            top_k = draft_model.top_k(node.tokens, k=branching)
            for token in top_k:
                child = TreeNode(node.tokens + [token])
                node.children.append(child)
                new_queue.append(child)
        queue = new_queue
    
    return root
```

### 2. Multi-draft Models

여러 작은 모델을 사용:

```python
drafts = [
    TinyLlama_1B,
    TinyLlama_1B_finetuned,
    Pythia_1B
]

# 각 draft가 후보 생성
candidates = []
for draft in drafts:
    candidates.extend(draft.generate(prompt, K=3))

# Target이 모든 후보 검증
best_path = target.verify(candidates)
```

### 3. Adaptive K

Acceptance rate에 따라 K 조정:

```python
class AdaptiveSpeculativeDecoder:
    def __init__(self, draft, target, K_min=3, K_max=10):
        self.K = K_min
        self.K_min = K_min
        self.K_max = K_max
        self.acceptance_history = []
    
    def generate_step(self, tokens):
        # Draft
        draft_tokens = self.draft_phase(tokens, self.K)
        
        # Verify
        accepted, new_token = self.target_phase(tokens, draft_tokens)
        
        # K 조정
        acceptance_rate = len(accepted) / self.K
        self.acceptance_history.append(acceptance_rate)
        
        if acceptance_rate > 0.7:
            self.K = min(self.K + 1, self.K_max)
        elif acceptance_rate < 0.4:
            self.K = max(self.K - 1, self.K_min)
        
        return accepted, new_token
```

---

## Draft Model 선택

### 기준

**1. 크기 비율**
- Target 70B → Draft 1-7B
- 10-70배 작아야 효과

**2. 품질**
- 너무 나쁘면 acceptance rate ↓
- 적당한 품질 필요

**3. 같은 토크나이저**
- 필수!

### 추천 조합

| Target | Draft | Speedup |
|--------|-------|---------|
| LLaMA-70B | LLaMA-7B | 2.3x |
| LLaMA-70B | TinyLlama-1B | 2.1x |
| GPT-3.5 | GPT-2 | 1.8x |
| Mixtral-8x7B | Mistral-7B | 2.5x |

### Fine-tuning Draft

Target 스타일에 맞춰 draft를 fine-tune:

```python
# Target의 출력으로 draft 학습
def train_draft_on_target(draft, target, dataset):
    for prompt in dataset:
        with torch.no_grad():
            target_output = target.generate(prompt)
        
        # Draft가 target 모방
        loss = draft.train_step(prompt, target_output)
```

**결과:** Acceptance rate 60% → 75%

---

## 실전 예제: 챗봇 서빙

```python
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 모델 로드
draft = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B", device_map="cuda:0")
target = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf", device_map="cuda:1")
decoder = SpeculativeDecoder(draft, target, tokenizer)

class Message(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.post("/generate")
async def generate(msg: Message):
    # Speculative decoding
    output = decoder.generate(
        msg.prompt,
        max_tokens=msg.max_tokens,
        K=5
    )
    
    return {"output": output}

# 사용
# curl -X POST "http://localhost:8000/generate" \
#   -H "Content-Type: application/json" \
#   -d '{"prompt": "Explain AI", "max_tokens": 100}'
```

---

## 한계와 트레이드오프

### 1. Acceptance Rate 의존

낮은 acceptance rate → 속도업 감소

```
Acceptance 30%: 1.5x
Acceptance 50%: 2.2x
Acceptance 70%: 3.0x
```

**대책:** Draft fine-tuning

### 2. 메모리 증가

Draft + Target 모두 메모리에:

```
70B model: 140 GB
+ 7B draft: 14 GB
Total: 154 GB
```

**대책:** Quantization (4-bit target)

### 3. Draft Overhead

Draft가 너무 크면 오히려 느림:

```
70B + 30B draft: 1.2x (별로)
70B + 7B draft: 2.5x (좋음)
70B + 1B draft: 2.3x (좋음)
```

---

## 다른 기법과 비교

| 기법 | 속도업 | 품질 | 메모리 |
|------|--------|------|--------|
| Speculative Decoding | 2-3x | 100% | 1.1x |
| Flash Attention | 2-4x | 100% | 0.1x |
| Quantization | 2x | 98% | 0.25x |
| Pruning | 1.5x | 95% | 0.5x |
| **All Combined** | **10x+** | 98% | 0.5x |

---

## 요약

**Speculative Decoding**은:

1. **작은 모델이 추측**, 큰 모델이 검증
2. **무손실**: 결과는 정확히 동일
3. **2-3배 속도 향상**
4. **메모리 증가**: Draft 모델 추가

**핵심:**
- K = 5-7 최적
- Acceptance rate 중요 (60% 이상)
- Draft fine-tuning으로 개선

**사용처:**
- 대규모 모델 서빙 (70B+)
- 레이턴시 중요한 챗봇
- 비용 절감 (같은 처리량, 적은 GPU)

---

## 다음 글

**9편: Continuous Batching**
- 동적 배치 처리
- 처리량 극대화
- vLLM, TGI 동작 원리

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
