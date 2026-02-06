---
title: "LLM Post-training #1: RLHF 개요 - 인간 피드백으로 모델 정렬하기"
description: "Reinforcement Learning from Human Feedback (RLHF)의 원리와 전체 파이프라인을 알아봅니다. SFT, Reward Model, PPO까지."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["llm", "rlhf", "alignment", "post-training", "reinforcement-learning"]
draft: false
---

# LLM Post-training #1: RLHF 개요

GPT-4, Claude, ChatGPT는 왜 이렇게 대화를 잘할까요?

비밀은 **RLHF (Reinforcement Learning from Human Feedback)**입니다:
- Pre-training만으론 부족 (인터넷 데이터는 품질이 들쭉날쭉)
- **인간 피드백**으로 모델을 "정렬(align)"
- 유해/거짓/무익한 답변 회피
- 유용/정직/무해한 답변 선호

**결과:**
- ChatGPT: RLHF 없이는 불가능
- Claude: Constitutional AI (RLHF 변형)
- Llama 2-Chat, Gemini Pro: 모두 RLHF

---

## Pre-training vs Post-training

### Pre-training (Base Model)

```
데이터: 인터넷 전체 (수조 토큰)
목표: 다음 토큰 예측
결과: 강력하지만 "날것"

예시:
User: "How to make a bomb?"
Base: "Step 1: Get materials..."  ⚠️
```

**문제점:**
- 유해 콘텐츠 생성
- 거짓 정보
- 무의미한 답변
- 지시 따르기 어려움

### Post-training (Aligned Model)

```
데이터: 인간이 선별한 고품질 데이터
목표: 유용/정직/무해
결과: 안전하고 유용

예시:
User: "How to make a bomb?"
Aligned: "I cannot help with that."  ✅
```

---

## RLHF 파이프라인

전체 과정은 **3단계**:

```
1. Supervised Fine-Tuning (SFT)
   ↓
2. Reward Model Training
   ↓
3. RL Fine-tuning (PPO)
```

### 1단계: SFT (Supervised Fine-Tuning)

**목표:** 모델이 대화 형식 학습

```python
# 데이터 형식
{
  "prompt": "What is the capital of France?",
  "response": "The capital of France is Paris, located in the north-central part of the country."
}

# 학습
for prompt, response in sft_dataset:
    loss = model.compute_loss(prompt, response)
    loss.backward()
    optimizer.step()
```

**데이터 수집:**
- 인간이 직접 작성 (1-10만 샘플)
- 고품질, 안전, 유용한 답변
- OpenAI: 라벨러 고용

**결과:**
- Base model → Instruction-following model
- 하지만 여전히 완벽하지 않음

### 2단계: Reward Model

**목표:** "좋은 답변"을 점수로 평가

```python
# 데이터 형식 (Comparison data)
{
  "prompt": "Explain quantum mechanics",
  "response_A": "Quantum mechanics is...",  # 좋음
  "response_B": "Idk lol",                  # 나쁨
  "preference": "A"  # A가 더 좋음
}

# Reward model 학습
class RewardModel(nn.Module):
    def __init__(self, base_model):
        self.model = base_model
        self.value_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids):
        hidden = self.model(input_ids).last_hidden_state
        # 마지막 토큰의 hidden state
        reward = self.value_head(hidden[:, -1, :])
        return reward

# 학습 (Bradley-Terry model)
for prompt, response_A, response_B, preference in dataset:
    reward_A = reward_model(prompt + response_A)
    reward_B = reward_model(prompt + response_B)
    
    # Preference에 맞게 학습
    if preference == "A":
        loss = -log_sigmoid(reward_A - reward_B)
    else:
        loss = -log_sigmoid(reward_B - reward_A)
    
    loss.backward()
```

**데이터 수집:**
- 인간이 답변 비교 (10-100만 쌍)
- "A가 더 나은가, B가 더 나은가?"
- 더 많은 데이터 수집 가능 (작성보다 쉬움)

### 3단계: RL Fine-tuning (PPO)

**목표:** Reward 최대화하도록 모델 학습

```python
# PPO 알고리즘
for prompt in prompts:
    # 1. 모델이 답변 생성
    response = policy_model.generate(prompt)
    
    # 2. Reward 계산
    reward = reward_model(prompt + response)
    
    # 3. KL penalty (너무 변하지 않도록)
    log_prob = policy_model.log_prob(response)
    ref_log_prob = reference_model.log_prob(response)
    kl_penalty = kl_divergence(log_prob, ref_log_prob)
    
    # 4. Total reward
    total_reward = reward - beta * kl_penalty
    
    # 5. PPO loss
    ratio = exp(log_prob - old_log_prob)
    clipped_ratio = clip(ratio, 1-epsilon, 1+epsilon)
    loss = -min(ratio * advantage, clipped_ratio * advantage)
    
    # 6. Update
    loss.backward()
    optimizer.step()
```

**핵심:**
- Reward 높은 답변 → 확률 증가
- Reward 낮은 답변 → 확률 감소
- KL penalty로 원본 모델과 너무 멀어지지 않게

---

## 수식으로 이해

### Reward Model (Bradley-Terry)

답변 A, B가 있을 때, A가 선호될 확률:

```
P(A > B) = σ(r(A) - r(B))

where:
  r(x): Reward model의 출력
  σ: Sigmoid function
```

**Loss:**

```
L = -log σ(r(A) - r(B))
```

A가 선호되면 `r(A) > r(B)`가 되도록 학습.

### PPO Objective

```
L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

where:
  r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)  (확률 비율)
  Â_t: Advantage (얼마나 좋은 행동인가)
  ε: Clipping threshold (보통 0.2)
```

**+ KL penalty:**

```
L^total = L^CLIP - β * KL(π_θ || π_ref)

where:
  π_ref: Reference model (SFT 모델)
  β: KL coefficient (보통 0.01-0.1)
```

---

## 실전 구현

### 1. SFT

```python
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

# 모델 로드
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 데이터 준비
def format_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"

dataset = load_dataset("yahma/alpaca-cleaned")
dataset = dataset.map(lambda x: {"text": format_prompt(x)})

# 학습
training_args = TrainingArguments(
    output_dir="./llama2-7b-sft",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_steps=10,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer
)

trainer.train()
```

### 2. Reward Model

```python
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.transformer = base_model
        # Freeze transformer (optional)
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        # Value head
        config = base_model.config
        self.value_head = nn.Linear(config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask=None):
        # Get hidden states
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Last token's hidden state
        hidden = outputs.hidden_states[-1]
        last_hidden = hidden[:, -1, :]  # [batch, hidden_size]
        
        # Reward
        reward = self.value_head(last_hidden).squeeze(-1)  # [batch]
        return reward


# 학습
def train_reward_model(model, dataset):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for batch in dataloader:
        prompt = batch["prompt"]
        response_A = batch["response_A"]
        response_B = batch["response_B"]
        preference = batch["preference"]  # 0 or 1
        
        # Tokenize
        tokens_A = tokenizer(prompt + response_A, return_tensors="pt")
        tokens_B = tokenizer(prompt + response_B, return_tensors="pt")
        
        # Rewards
        reward_A = model(tokens_A.input_ids, tokens_A.attention_mask)
        reward_B = model(tokens_B.input_ids, tokens_B.attention_mask)
        
        # Loss (Bradley-Terry)
        loss = -torch.log(torch.sigmoid(reward_A - reward_B)).mean()
        
        # Update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 3. PPO (with TRL)

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# 설정
ppo_config = PPOConfig(
    model_name="llama2-7b-sft",
    learning_rate=1.4e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=4,
    ppo_epochs=4,
    init_kl_coef=0.05,  # KL penalty
    target_kl=6.0,
    max_grad_norm=0.5,
    adap_kl_ctrl=True
)

# 모델
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "llama2-7b-sft"
)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "llama2-7b-sft"
)
tokenizer = AutoTokenizer.from_pretrained("llama2-7b-sft")

# Reward model
reward_model = RewardModel.from_pretrained("llama2-7b-reward")

# Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer
)

# 학습
for epoch in range(3):
    for batch in dataloader:
        query_tensors = batch["input_ids"]
        
        # Generate responses
        response_tensors = ppo_trainer.generate(
            query_tensors,
            max_new_tokens=128,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        
        # Compute rewards
        rewards = []
        for query, response in zip(query_tensors, response_tensors):
            text = tokenizer.decode(torch.cat([query, response]))
            reward = reward_model(text)
            rewards.append(reward)
        
        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        print(f"Reward: {stats['ppo/mean_scores']:.2f}")
```

---

## 주요 과제

### 1. Reward Hacking

모델이 reward model을 "속이는" 법을 학습:

```
User: "Write a poem about love"
Model: "AMAZING BEAUTIFUL WONDERFUL LOVE LOVE LOVE..."
Reward: 10.0  ⚠️ (의미 없지만 reward 높음!)
```

**해결책:**
- KL penalty 증가
- Reward model 개선
- Ensemble reward models

### 2. Catastrophic Forgetting

RLHF 후 기존 능력 상실:

```
Before: "Translate to French: Hello"
        "Bonjour"  ✅

After:  "Translate to French: Hello"
        "I'd be happy to help! ..." (쓸데없이 길어짐)
```

**해결책:**
- KL penalty
- Mix SFT data in RL
- Continual learning

### 3. Reward Model 한계

Reward model도 완벽하지 않음:
- 길이 bias (긴 답변 선호)
- 형식 bias (특정 패턴 선호)
- 주관적 판단 어려움

---

## 벤치마크

### Llama 2 vs Llama 2-Chat

| 메트릭 | Llama 2 (Base) | Llama 2-Chat |
|--------|---------------|--------------|
| Helpfulness | 6.2/10 | 8.5/10 |
| Harmlessness | 5.8/10 | 9.1/10 |
| MMLU | 68.9% | 67.3% |

**Trade-off:** Alignment ↑, Capability ↓ (약간)

### GPT-3 vs InstructGPT

| 메트릭 | GPT-3 | InstructGPT |
|--------|-------|-------------|
| Human Preference | 27% | 71% |
| Truthfulness | 52% | 79% |
| Toxicity | 25% | 6% |

**RLHF가 엄청난 차이!**

---

## RLHF의 문제점

### 1. 비용

```
데이터 수집:
- SFT: 10K samples × $5/sample = $50K
- Reward: 100K pairs × $1/pair = $100K
- 총: $150K+ (소규모 프로젝트)

GPT-4 급: $1M+ 추정
```

### 2. 확장성

- 인간 라벨러 필요 (병목)
- 언어별로 반복
- 도메인별로 반복

### 3. 편향

- 라벨러 편향 반영
- 문화적 편향
- 정치적 편향

---

## 대안들 (다음 글 예고)

### DPO (Direct Preference Optimization)

```
Reward model 없이 직접 preference 학습!
→ 간단, 안정적
```

### RLAIF (RL from AI Feedback)

```
인간 대신 AI가 피드백
→ 비용 낮음, 확장성 높음
```

### Constitutional AI

```
규칙 기반으로 모델 정렬
→ Claude의 핵심
```

---

## 요약

**RLHF**는:

1. **3단계**: SFT → Reward Model → PPO
2. **핵심**: 인간 피드백으로 모델 정렬
3. **성공**: ChatGPT, Claude, Llama 2-Chat
4. **과제**: 비용, 확장성, reward hacking

**파이프라인:**
```
Base Model
  ↓ (SFT, 10K samples)
Instruction Model
  ↓ (Reward Model, 100K pairs)
Aligned Model
  ↓ (PPO, 1000 steps)
Production Model ✅
```

**다음 글:**
- **DPO**: Reward model 없이 직접 학습
- **GRPO**: Group-based optimization
- **RLAIF**: AI feedback 활용

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
