---
title: "LLM Post-training #2: DPO - Reward Model 없이 직접 학습하기"
description: "Direct Preference Optimization (DPO)로 RLHF의 복잡성 없이 간단하고 안정적으로 모델을 정렬하는 방법을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["llm", "dpo", "alignment", "post-training", "preference-learning"]
draft: false
---

# LLM Post-training #2: DPO

RLHF의 문제:
- **3단계** 파이프라인 (SFT → Reward → PPO)
- Reward model 학습 필요
- PPO 불안정 (hyperparameter 민감)
- 느림

**DPO (Direct Preference Optimization)**는 혁신적입니다:
- **1단계**만! (SFT → DPO)
- Reward model 불필요
- 안정적 (supervised learning처럼)
- 빠름 (2-3배)

결과:
- Zephyr-7B: DPO로 GPT-3.5 능가
- Starling-7B: DPO로 GPT-4 근접
- 간단하면서 강력!

---

## 핵심 아이디어

### RLHF의 복잡성

```
1. Reward Model 학습
   r(x, y) = RewardModel(x, y)

2. PPO로 Policy 최적화
   max E[r(x, y) - β·KL(π||π_ref)]
   
문제: 두 단계, 불안정, 느림
```

### DPO의 간결함

```
Preference data만 있으면:
  (x, y_w, y_l)  where y_w > y_l

직접 학습:
  max P(y_w > y_l | x)
  
장점: 한 단계, 안정, 빠름
```

---

## 수학적 유도

### 1. RLHF Objective

Reward를 최대화하되, reference model과 너무 멀어지지 않기:

```
π* = argmax E_{x~D, y~π(y|x)} [r(x,y) - β·log(π(y|x)/π_ref(y|x))]
```

### 2. Optimal Policy

이 objective의 최적해:

```
π*(y|x) = π_ref(y|x) · exp(r(x,y)/β) / Z(x)

where Z(x) = Σ_y π_ref(y|x) · exp(r(x,y)/β)
```

### 3. Reward 역산

위 식을 정리하면:

```
r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x)
```

### 4. Bradley-Terry Model

Preference 확률:

```
P(y_w > y_l | x) = σ(r(x,y_w) - r(x,y_l))
```

Reward를 대입:

```
P(y_w > y_l | x) = σ(β·log(π*(y_w|x)/π_ref(y_w|x)) - β·log(π*(y_l|x)/π_ref(y_l|x)))
```

**핵심:** Z(x) 항이 소거됨!

### 5. DPO Loss

```
L_DPO = -E[(x,y_w,y_l)~D] [log σ(β·log(π_θ(y_w|x)/π_ref(y_w|x)) - β·log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

간단히:

```
L_DPO = -log σ(β·(log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)))
```

**Reward model 없이 직접 학습!**

---

## 구현

### Naive 버전

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_model,
    reference_model,
    prompt,
    chosen_response,
    rejected_response,
    beta=0.1
):
    """
    DPO loss 계산
    
    Args:
        policy_model: 학습할 모델 (θ)
        reference_model: 참조 모델 (frozen)
        prompt: 입력
        chosen_response: 선호 답변 (y_w)
        rejected_response: 비선호 답변 (y_l)
        beta: KL penalty 계수
    """
    # Tokenize
    chosen_tokens = tokenizer(prompt + chosen_response, return_tensors="pt")
    rejected_tokens = tokenizer(prompt + rejected_response, return_tensors="pt")
    
    # Log probabilities
    with torch.no_grad():
        ref_chosen_logprobs = reference_model(**chosen_tokens).logits.log_softmax(-1)
        ref_rejected_logprobs = reference_model(**rejected_tokens).logits.log_softmax(-1)
    
    policy_chosen_logprobs = policy_model(**chosen_tokens).logits.log_softmax(-1)
    policy_rejected_logprobs = policy_model(**rejected_tokens).logits.log_softmax(-1)
    
    # Gather log probs for actual tokens
    chosen_logprobs = policy_chosen_logprobs.gather(-1, chosen_tokens.input_ids.unsqueeze(-1)).squeeze(-1).sum()
    rejected_logprobs = policy_rejected_logprobs.gather(-1, rejected_tokens.input_ids.unsqueeze(-1)).squeeze(-1).sum()
    
    ref_chosen_logprobs = ref_chosen_logprobs.gather(-1, chosen_tokens.input_ids.unsqueeze(-1)).squeeze(-1).sum()
    ref_rejected_logprobs = ref_rejected_logprobs.gather(-1, rejected_tokens.input_ids.unsqueeze(-1)).squeeze(-1).sum()
    
    # Log ratios
    chosen_ratio = chosen_logprobs - ref_chosen_logprobs
    rejected_ratio = rejected_logprobs - ref_rejected_logprobs
    
    # DPO loss
    loss = -F.logsigmoid(beta * (chosen_ratio - rejected_ratio))
    
    return loss
```

### 실전 구현 (TRL)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# 모델 로드
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-sft")
ref_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-sft")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-sft")

# 데이터 로드
dataset = load_dataset("Anthropic/hh-rlhf")

# 데이터 포맷
def format_dataset(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }

dataset = dataset.map(format_dataset)

# DPO 설정
training_args = DPOConfig(
    output_dir="./llama2-7b-dpo",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    beta=0.1,  # KL penalty
    logging_steps=10,
    save_strategy="epoch",
    remove_unused_columns=False
)

# Trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer
)

# 학습
dpo_trainer.train()
```

---

## Beta 파라미터

### 역할

```
β = 0.01: 거의 변화 없음 (안전)
β = 0.1:  적당한 변화 (권장)
β = 1.0:  큰 변화 (위험)
```

### 선택 가이드

```python
# 작은 모델 (7B)
beta = 0.1

# 중간 모델 (13B-30B)
beta = 0.05

# 큰 모델 (70B+)
beta = 0.01

# 도메인 shift 큰 경우
beta = 0.2
```

---

## DPO vs RLHF

### 학습 시간

**RLHF:**
```
SFT:          10 hours
Reward Model: 5 hours
PPO:          20 hours
Total:        35 hours
```

**DPO:**
```
SFT: 10 hours
DPO: 8 hours
Total: 18 hours  (2배 빠름!)
```

### 메모리

**RLHF:**
```
Policy model:    7B params
Value model:     7B params
Reference model: 7B params
Reward model:    7B params
Total:           28B (4 models!)
```

**DPO:**
```
Policy model:    7B params
Reference model: 7B params
Total:           14B (2 models)
```

### 안정성

**RLHF PPO:**
```python
# Hyperparameters
ppo_epochs = 4
clip_range = 0.2
vf_coef = 0.5
entropy_coef = 0.01
gae_lambda = 0.95
target_kl = 0.1
# ... 많음!

# 불안정하면 발산
```

**DPO:**
```python
# Hyperparameters
learning_rate = 5e-7
beta = 0.1
# 끝!

# Supervised learning처럼 안정
```

---

## 개선 버전들

### 1. IPO (Identity Preference Optimization)

**문제:** DPO는 logit 차이에 민감

**해결:**
```
L_IPO = E[(r(x,y_w) - r(x,y_l) - 1)^2]

간단한 MSE loss!
```

```python
def ipo_loss(policy_logprobs, ref_logprobs, beta=0.1):
    log_ratio = policy_logprobs - ref_logprobs
    loss = (log_ratio - 1) ** 2
    return loss.mean()
```

### 2. KTO (Kahneman-Tversky Optimization)

**문제:** Pairwise comparison 데이터 수집 어려움

**해결:** Binary feedback만 사용
```
Data: (x, y, label)
  label ∈ {좋음, 나쁨}

L_KTO = E[loss(y, label)]
```

```python
def kto_loss(
    policy_logprobs,
    ref_logprobs,
    label,  # 0 or 1
    beta=0.1
):
    log_ratio = policy_logprobs - ref_logprobs
    
    if label == 1:  # 좋은 답변
        loss = -F.logsigmoid(beta * log_ratio)
    else:  # 나쁜 답변
        loss = -F.logsigmoid(-beta * log_ratio)
    
    return loss
```

### 3. ORPO (Odds Ratio Preference Optimization)

**문제:** Reference model 필요 (메모리)

**해결:** Reference 없이 학습
```
L_ORPO = L_SFT + λ·L_OR

where:
  L_OR = log(odds(y_w)/odds(y_l))
  odds(y) = p(y)/(1-p(y))
```

```python
def orpo_loss(
    logits,
    chosen_tokens,
    rejected_tokens,
    lambda_coef=0.1
):
    # SFT loss
    sft_loss = F.cross_entropy(logits, chosen_tokens)
    
    # Odds ratio loss
    chosen_probs = F.softmax(logits, dim=-1).gather(-1, chosen_tokens)
    rejected_probs = F.softmax(logits, dim=-1).gather(-1, rejected_tokens)
    
    chosen_odds = chosen_probs / (1 - chosen_probs + 1e-8)
    rejected_odds = rejected_probs / (1 - rejected_probs + 1e-8)
    
    or_loss = -torch.log(chosen_odds / rejected_odds).mean()
    
    return sft_loss + lambda_coef * or_loss
```

---

## 실전 팁

### 1. 데이터 품질

```python
# 좋은 preference data
{
  "prompt": "Explain quantum entanglement",
  "chosen": "Quantum entanglement is a phenomenon where...",  # 상세, 정확
  "rejected": "It's when particles are connected"  # 짧고 불충분
}

# 나쁜 preference data
{
  "prompt": "What's 2+2?",
  "chosen": "4",
  "rejected": "5"  # 너무 명확, 학습 가치 낮음
}
```

**규칙:**
- Margin이 적당히 있어야 함
- 명백한 차이보다 미묘한 차이
- 다양한 측면 (정확도, 유용성, 안전성)

### 2. Learning Rate

```python
# DPO는 매우 작은 LR 필요
learning_rate = 5e-7  # RLHF보다 10배 작음

# 큰 모델은 더 작게
if model_size >= 70B:
    learning_rate = 1e-7
```

### 3. 평가

```python
# 학습 중 모니터링
metrics = {
    "chosen_reward": chosen_ratio.mean(),
    "rejected_reward": rejected_ratio.mean(),
    "reward_margin": (chosen_ratio - rejected_ratio).mean(),
    "reward_accuracy": (chosen_ratio > rejected_ratio).float().mean()
}

# Reward margin > 0 유지
# Reward accuracy > 60% 목표
```

---

## 벤치마크

### Zephyr-7B (DPO)

| 모델 | Method | MT-Bench | AlpacaEval |
|------|--------|----------|------------|
| Llama-2-7B-chat | RLHF | 6.27 | - |
| Mistral-7B-Instruct | - | 6.84 | - |
| **Zephyr-7B-beta** | **DPO** | **7.34** | **90.6%** |

DPO가 RLHF보다 좋음!

### Starling-7B (DPO)

| 모델 | MT-Bench | AlpacaEval 2.0 |
|------|----------|----------------|
| GPT-4-Turbo | 9.32 | 50.0% |
| Claude-3-Opus | 9.00 | 40.5% |
| **Starling-LM-7B** | **8.09** | **36.6%** |
| Llama-2-70B-chat | 6.86 | 13.9% |

7B 모델이 70B 능가!

---

## DPO의 한계

### 1. Length Bias

DPO는 긴 답변 선호:

```python
# 문제
chosen = "Short answer."
rejected = "Very very very long but wrong answer..."

# DPO는 rejected에 높은 확률 (길이 때문)
```

**해결:**
- Length-normalized rewards
- Explicit length penalty

```python
def length_normalized_dpo_loss(
    chosen_logprobs,
    rejected_logprobs,
    chosen_length,
    rejected_length,
    beta=0.1
):
    # Normalize by length
    chosen_logprobs = chosen_logprobs / chosen_length
    rejected_logprobs = rejected_logprobs / rejected_length
    
    log_ratio = beta * (chosen_logprobs - rejected_logprobs)
    loss = -F.logsigmoid(log_ratio)
    
    return loss
```

### 2. Reward Hacking

DPO도 reward hacking 가능:

```python
# 모델이 특정 패턴 학습
"I'm happy to help! ..." → 높은 확률
(실제 내용 상관없이)
```

**해결:**
- 다양한 데이터
- Iterative DPO

### 3. Out-of-distribution

Training data와 다른 입력에 약함:

```python
# Training: 영어 대화
# Test: 코드 생성 → 성능 하락
```

**해결:**
- 다양한 도메인 데이터
- Domain-specific DPO

---

## 고급 기법

### 1. Iterative DPO

```python
# Round 1
model_v1 = dpo_train(sft_model, preference_data_v1)

# Generate new data with v1
new_data = generate_preference_data(model_v1)

# Round 2
model_v2 = dpo_train(model_v1, new_data)

# Repeat...
```

### 2. Multi-objective DPO

여러 목표 동시 최적화:

```python
# Helpfulness + Harmlessness + Honesty
loss = (
    w1 * dpo_loss(helpful_data) +
    w2 * dpo_loss(harmless_data) +
    w3 * dpo_loss(honest_data)
)
```

### 3. Conditional DPO

조건부 학습:

```python
# Persona-specific
loss = dpo_loss(
    prompt="[Friendly] " + user_input,
    chosen=friendly_response,
    rejected=formal_response
)
```

---

## 실전 예제: 전체 파이프라인

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DPOTrainer
from datasets import load_dataset

# 1. SFT
print("Stage 1: Supervised Fine-Tuning")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
sft_dataset = load_dataset("yahma/alpaca-cleaned")

sft_trainer = SFTTrainer(
    model=base_model,
    train_dataset=sft_dataset["train"],
    max_seq_length=512,
    packing=True
)
sft_trainer.train()
sft_trainer.save_model("./llama2-7b-sft")

# 2. DPO
print("Stage 2: Direct Preference Optimization")
sft_model = AutoModelForCausalLM.from_pretrained("./llama2-7b-sft")
ref_model = AutoModelForCausalLM.from_pretrained("./llama2-7b-sft")
dpo_dataset = load_dataset("Anthropic/hh-rlhf")

dpo_trainer = DPOTrainer(
    model=sft_model,
    ref_model=ref_model,
    train_dataset=dpo_dataset["train"],
    beta=0.1,
    max_length=512,
    max_prompt_length=256
)
dpo_trainer.train()
dpo_trainer.save_model("./llama2-7b-dpo")

# 3. Evaluation
print("Stage 3: Evaluation")
model = AutoModelForCausalLM.from_pretrained("./llama2-7b-dpo")
tokenizer = AutoTokenizer.from_pretrained("./llama2-7b-dpo")

test_prompts = [
    "Explain quantum mechanics simply",
    "Write a poem about AI",
    "How to make a website?"
]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
```

---

## 요약

**DPO**는:

1. **간단**: Reward model 불필요
2. **빠름**: RLHF보다 2배 빠름
3. **안정**: Supervised learning처럼
4. **효과적**: Zephyr, Starling 등 SOTA

**Loss:**
```
L_DPO = -log σ(β·log(π_θ(y_w)/π_ref(y_w)) - β·log(π_θ(y_l)/π_ref(y_l)))
```

**핵심:**
- Preference data만 있으면 됨
- Reference model과 비교
- Beta로 변화량 조절

**개선:**
- IPO: MSE loss
- KTO: Binary feedback
- ORPO: Reference 없이

**다음 글:**
- **GRPO**: Group-based reward
- **Online RL**: 실시간 피드백

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
