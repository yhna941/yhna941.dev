---
title: "LLM Post-training #4: RLAIF - AI 피드백으로 확장 가능한 정렬"
description: "Reinforcement Learning from AI Feedback (RLAIF)로 인간 라벨러 없이 대규모로 모델을 정렬하는 방법을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["llm", "rlaif", "alignment", "post-training", "ai-feedback"]
draft: false
---

# LLM Post-training #4: RLAIF

RLHF의 병목:
- **인간 라벨러 비용** ($100K+)
- **확장성** (언어/도메인마다 반복)
- **속도** (사람은 느림)

**RLAIF (RL from AI Feedback)**는 해결책:
- **AI가 피드백** (GPT-4, Claude 등)
- **무한 확장** (데이터 무제한)
- **빠름** (API 호출)
- **저렴** ($1K)

결과:
- Claude 2: Constitutional AI (RLAIF 변형)
- Llama 3: AI feedback 활용
- Google: RLAIF = RLHF 성능

---

## 핵심 아이디어

### RLHF (Human)

```
1. 인간이 답변 A, B 비교
   "A가 더 유용함"
   
2. Reward model 학습
   r(A) > r(B)
   
3. RL로 reward 최대화
```

**문제:** 인간 필요 (비용, 시간)

### RLAIF (AI)

```
1. AI가 답변 A, B 비교
   "A is more helpful because..."
   
2. Reward model 학습
   r(A) > r(B)
   
3. RL로 reward 최대화
```

**장점:** AI 무제한 (확장 가능)

---

## 파이프라인

### 1단계: AI Annotator

```python
def ai_annotate(prompt, response_A, response_B):
    """AI가 선호도 판단"""
    
    annotation_prompt = f"""
You are an expert evaluator. Compare two responses.

Prompt: {prompt}

Response A: {response_A}

Response B: {response_B}

Which response is better? Consider:
- Helpfulness
- Harmlessness
- Honesty
- Clarity

Answer with "A" or "B" and explain why.
"""
    
    # AI 모델 호출 (GPT-4, Claude 등)
    result = ai_model.generate(annotation_prompt)
    
    # Parse
    if "Response A" in result or result.startswith("A"):
        preference = "A"
    else:
        preference = "B"
    
    return {
        "preference": preference,
        "explanation": result
    }
```

### 2단계: Reward Model 학습

```python
# RLHF와 동일!
class RewardModel(nn.Module):
    # ... (이전과 같음)

# AI preference로 학습
for batch in ai_preference_data:
    prompt = batch["prompt"]
    response_A = batch["response_A"]
    response_B = batch["response_B"]
    preference = batch["preference"]  # AI가 선택
    
    reward_A = reward_model(prompt + response_A)
    reward_B = reward_model(prompt + response_B)
    
    if preference == "A":
        loss = -log_sigmoid(reward_A - reward_B)
    else:
        loss = -log_sigmoid(reward_B - reward_A)
    
    loss.backward()
```

### 3단계: RL Fine-tuning

```python
# RLHF/GRPO와 동일
ppo_trainer = PPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    reward_model=reward_model  # AI feedback로 학습된 것!
)

ppo_trainer.train()
```

---

## AI Annotator 설계

### 1. Zero-shot

```python
prompt = """
Compare these responses. Which is better?

Prompt: {user_question}
Response A: {response_A}
Response B: {response_B}

Choose A or B.
"""

# 간단하지만 일관성 낮음
```

### 2. Few-shot

```python
prompt = """
You are an expert evaluator.

Example 1:
Prompt: "What is 2+2?"
Response A: "4"
Response B: "Idk"
Better: A (correct and concise)

Example 2:
Prompt: "Explain AI"
Response A: "AI is artificial intelligence..."
Response B: "AI AI AI AI" 
Better: A (informative)

Now evaluate:
Prompt: {user_question}
Response A: {response_A}
Response B: {response_B}

Which is better and why?
"""

# 일관성 향상!
```

### 3. Chain-of-Thought

```python
prompt = """
Evaluate step-by-step:

1. Helpfulness: Which response better answers the question?
2. Harmlessness: Which is safer?
3. Honesty: Which is more truthful?
4. Clarity: Which is clearer?

Response A: {response_A}
Response B: {response_B}

Analysis:
1. Helpfulness: [your analysis]
2. Harmlessness: [your analysis]
3. Honesty: [your analysis]
4. Clarity: [your analysis]

Conclusion: [A or B] is better because...
"""

# 최고 품질!
```

---

## Constitutional AI (Anthropic)

Claude의 핵심 기술!

### 원리

**Constitution**: 규칙 집합

```python
CONSTITUTION = [
    "Choose the response that is more helpful and harmless",
    "Avoid responses that are illegal or unethical",
    "Prefer responses that are honest and acknowledge uncertainty",
    "Choose responses that are clearer and more informative"
]
```

### 알고리즘

```python
def constitutional_ai(response, constitution):
    """
    Constitutional AI feedback
    """
    critiques = []
    
    # 1. Critique phase
    for principle in constitution:
        critique_prompt = f"""
Principle: {principle}

Response: {response}

Does this response violate the principle?
If yes, how should it be revised?
"""
        critique = ai_model.generate(critique_prompt)
        critiques.append(critique)
    
    # 2. Revision phase
    revision_prompt = f"""
Original response: {response}

Critiques:
{'\n'.join(critiques)}

Revise the response to address all critiques:
"""
    
    revised_response = ai_model.generate(revision_prompt)
    
    return revised_response


# 반복 개선
response = initial_response
for _ in range(3):  # 3 iterations
    response = constitutional_ai(response, CONSTITUTION)
```

### Self-improvement

```python
# 1. 모델이 자기 출력 평가
response_A = model.generate(prompt)
response_B = model.generate(prompt)

# 2. 모델이 자기 비교
preference = model.evaluate(response_A, response_B, CONSTITUTION)

# 3. Preference data로 학습
# (Bootstrapping!)
```

---

## 실전 구현

### 전체 파이프라인

```python
import anthropic
from transformers import AutoModelForCausalLM
from trl import PPOTrainer

class RLAIFTrainer:
    def __init__(
        self,
        policy_model,
        ref_model,
        ai_judge_model="claude-3-opus",
        constitution=None
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.ai_judge = anthropic.Anthropic()
        self.constitution = constitution or DEFAULT_CONSTITUTION
    
    def generate_preference_data(self, prompts, num_pairs=2):
        """AI로 preference data 생성"""
        preference_data = []
        
        for prompt in prompts:
            # 1. Generate responses
            responses = []
            for _ in range(num_pairs):
                response = self.policy.generate(prompt)
                responses.append(response)
            
            # 2. AI judges
            for i in range(len(responses)):
                for j in range(i+1, len(responses)):
                    response_A = responses[i]
                    response_B = responses[j]
                    
                    # AI annotation
                    preference = self.ai_annotate(
                        prompt,
                        response_A,
                        response_B
                    )
                    
                    preference_data.append({
                        "prompt": prompt,
                        "chosen": response_A if preference == "A" else response_B,
                        "rejected": response_B if preference == "A" else response_A
                    })
        
        return preference_data
    
    def ai_annotate(self, prompt, response_A, response_B):
        """AI가 선호도 판단 (Constitutional AI 스타일)"""
        
        eval_prompt = f"""
You are an expert evaluator following these principles:

{chr(10).join(f"- {p}" for p in self.constitution)}

Compare these responses:

User: {prompt}

Response A: {response_A}

Response B: {response_B}

Which response better follows the principles? 
Answer with "A" or "B" and explain.
"""
        
        # Claude API 호출
        message = self.ai_judge.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=500,
            messages=[{"role": "user", "content": eval_prompt}]
        )
        
        result = message.content[0].text
        
        # Parse
        preference = "A" if "Response A" in result[:50] else "B"
        
        return preference
    
    def train_reward_model(self, preference_data):
        """Reward model 학습"""
        reward_model = RewardModel(self.policy)
        
        optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-5)
        
        for epoch in range(3):
            for batch in DataLoader(preference_data, batch_size=4):
                # ... (이전과 동일)
                pass
        
        return reward_model
    
    def train_policy(self, reward_model, prompts):
        """RL로 policy 학습"""
        ppo_trainer = PPOTrainer(
            model=self.policy,
            ref_model=self.ref,
            reward_model=reward_model
        )
        
        for epoch in range(3):
            for prompt in prompts:
                # ... (PPO/GRPO)
                pass


# 사용
DEFAULT_CONSTITUTION = [
    "Be helpful and informative",
    "Be harmless and avoid toxic content",
    "Be honest and acknowledge limitations",
    "Be clear and well-structured"
]

policy_model = AutoModelForCausalLM.from_pretrained("llama2-7b-sft")
ref_model = AutoModelForCausalLM.from_pretrained("llama2-7b-sft")

trainer = RLAIFTrainer(
    policy_model=policy_model,
    ref_model=ref_model,
    ai_judge_model="claude-3-opus",
    constitution=DEFAULT_CONSTITUTION
)

# 1. Generate preference data
prompts = load_prompts()
preference_data = trainer.generate_preference_data(prompts, num_pairs=4)

# 2. Train reward model
reward_model = trainer.train_reward_model(preference_data)

# 3. Train policy
trainer.train_policy(reward_model, prompts)
```

---

## AI Judge 선택

### GPT-4

```python
import openai

def gpt4_judge(prompt, response_A, response_B):
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{
            "role": "user",
            "content": f"Compare:\nA: {response_A}\nB: {response_B}\nWhich is better?"
        }]
    )
    
    return "A" if "A" in response.choices[0].message.content[:10] else "B"
```

**장점:** 강력, 다목적  
**단점:** 비용 ($0.01/1K tokens)

### Claude

```python
import anthropic

def claude_judge(prompt, response_A, response_B):
    client = anthropic.Anthropic()
    
    message = client.messages.create(
        model="claude-3-opus-20240229",
        messages=[{
            "role": "user",
            "content": f"Compare:\nA: {response_A}\nB: {response_B}"
        }]
    )
    
    return "A" if "A" in message.content[0].text[:10] else "B"
```

**장점:** Constitutional AI에 최적  
**단점:** 비용 비슷

### Open-source (자체 모델)

```python
def self_judge(model, prompt, response_A, response_B):
    """자기 자신이 평가"""
    judge_prompt = f"Compare:\nA: {response_A}\nB: {response_B}\nBetter:"
    
    output = model.generate(judge_prompt, max_tokens=5)
    
    return "A" if "A" in output else "B"
```

**장점:** 무료, 빠름  
**단점:** 품질 낮을 수 있음

---

## 벤치마크

### Google 연구 (2023)

**결과: RLAIF ≈ RLHF**

| 메트릭 | Human Feedback | AI Feedback |
|--------|---------------|-------------|
| Win Rate | 50% | 49.8% |
| Helpfulness | 7.8/10 | 7.7/10 |
| Harmlessness | 8.5/10 | 8.6/10 |
| Cost | $50K | $500 |

**AI feedback으로 충분!**

### Claude (Constitutional AI)

| 모델 | Method | Harmlessness | Helpfulness |
|------|--------|--------------|-------------|
| Claude 1 | RLHF | 75% | 82% |
| **Claude 2** | **Constitutional AI** | **95%** | **88%** |

**Constitutional AI가 더 나음!**

---

## 비용 비교

### RLHF (Human)

```
라벨러: 10명
시간: 40시간/주, 4주
비용: $25/hour

총: 10 × 40 × 4 × $25 = $40,000

+ 데이터 수집 플랫폼: $10,000

= $50,000
```

### RLAIF (AI)

```
API 비용:
- 100K comparisons
- $0.03 per comparison (GPT-4)

총: 100K × $0.03 = $3,000

+ 개발 시간: $2,000

= $5,000

(10배 저렴!)
```

---

## 한계와 해결

### 1. AI Judge 편향

**문제:** AI도 편향 있음

```python
# GPT-4는 verbose 선호
Response A: "The capital is Paris."
Response B: "The capital of France, located in the northern part..."
GPT-4 prefers: B  (길이 편향!)
```

**해결:**
```python
# Multiple judges
judgments = [
    gpt4_judge(A, B),
    claude_judge(A, B),
    llama_judge(A, B)
]

# Majority vote
final = max(set(judgments), key=judgments.count)
```

### 2. Self-preference Bias

**문제:** 모델이 자기 출력 선호

```python
# Llama가 Llama 출력 선호
llama_output vs gpt_output
→ Llama judge → Llama wins (편향!)
```

**해결:**
```python
# Blind evaluation (출처 숨김)
# Cross-evaluation (다른 모델이 평가)
```

### 3. Reward Hacking

**문제:** AI judge 속이기

```python
# AI가 "helpful" 키워드 선호 발견
Model learns: "I'm happy to help! ..."
(내용 관계없이)
```

**해결:**
```python
# Diverse judges
# Adversarial testing
# Human spot-check
```

---

## 고급 기법

### 1. Iterative RLAIF

```python
# Round 1
preference_v1 = generate_with_gpt4()
model_v1 = train(preference_v1)

# Round 2 (모델 개선)
preference_v2 = generate_with_model_v1()  # Self-improvement
model_v2 = train(preference_v2)

# Round 3
preference_v3 = generate_with_ensemble([gpt4, claude, model_v2])
model_v3 = train(preference_v3)
```

### 2. Hierarchical Feedback

```python
# Multi-level constitution
LEVEL_1 = ["Safety first"]
LEVEL_2 = ["Helpfulness", "Clarity"]
LEVEL_3 = ["Style", "Tone"]

# Sequential evaluation
score = 0
if passes(LEVEL_1):
    score += evaluate(LEVEL_2)
    if passes(LEVEL_2):
        score += evaluate(LEVEL_3)
```

### 3. Synthetic Data Augmentation

```python
# AI가 데이터 생성
prompts = gpt4.generate_diverse_prompts(num=10000)

# AI가 답변 생성
for prompt in prompts:
    good_response = gpt4.generate(prompt, temperature=0.7, principle="helpful")
    bad_response = gpt4.generate(prompt, temperature=0.9, principle="harmful")
    
    preference_data.append({
        "prompt": prompt,
        "chosen": good_response,
        "rejected": bad_response
    })
```

---

## 실전 팁

### 1. Constitution 설계

```python
# 좋은 constitution
GOOD = [
    "Provide accurate, factual information",  # 구체적
    "Acknowledge when uncertain",  # 측정 가능
    "Use clear, simple language"  # 명확
]

# 나쁜 constitution
BAD = [
    "Be good",  # 너무 모호
    "Don't be bad",  # 부정형
    "Make everyone happy"  # 불가능
]
```

### 2. Judge Calibration

```python
# Human baseline과 비교
human_prefs = load_human_preferences()
ai_prefs = generate_ai_preferences()

# Agreement rate
agreement = (human_prefs == ai_prefs).mean()

if agreement < 0.7:
    # AI judge 개선 필요
    calibrate_judge()
```

### 3. Cost 최적화

```python
# Cascade evaluation
def cascade_judge(A, B):
    # 1. Cheap model first
    cheap_result = llama_judge(A, B)
    confidence = cheap_result['confidence']
    
    if confidence > 0.9:
        return cheap_result['preference']
    
    # 2. Expensive model if uncertain
    return gpt4_judge(A, B)

# 90% cases: cheap
# 10% cases: expensive
# → 10배 cost reduction
```

---

## 요약

**RLAIF**는:

1. **AI가 피드백**: 인간 불필요
2. **확장 가능**: 무제한 데이터
3. **저렴**: 10배 이상 ($50K → $5K)
4. **빠름**: API 호출

**핵심:**
- Constitutional AI (Claude)
- AI judge (GPT-4, Claude)
- Self-improvement

**성능:**
- RLAIF ≈ RLHF (Google 연구)
- Constitutional AI > RLHF (Claude)

**한계:**
- AI judge 편향
- Self-preference bias
- Reward hacking

**해결:**
- Multiple judges
- Blind evaluation
- Human spot-check

---

## RLHF 시리즈 완결! 🎉

**Post-training 완전 정복 (1-4편):**

1. **RLHF**: 인간 피드백 (표준)
2. **DPO**: Reward model 불필요
3. **GRPO**: Group baseline (효율)
4. **RLAIF**: AI 피드백 (확장)

**비교:**

| 방법 | 비용 | 시간 | 샘플 효율 | 성능 |
|------|------|------|----------|------|
| RLHF | $$$$$ | 느림 | 보통 | ⭐⭐⭐⭐⭐ |
| DPO | $$$ | 빠름 | 보통 | ⭐⭐⭐⭐ |
| GRPO | $$$$ | 빠름 | 높음 | ⭐⭐⭐⭐⭐ |
| **RLAIF** | **$** | **매우 빠름** | **높음** | ⭐⭐⭐⭐⭐ |

**추천:**
- 리소스 풍부: RLHF or GRPO
- 빠르게 시작: DPO
- 확장 필요: **RLAIF** ✅

---

## 다음 시리즈

**System Design** - 대규모 시스템 설계!

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
