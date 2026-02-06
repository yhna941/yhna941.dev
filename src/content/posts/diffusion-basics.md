---
title: "Diffusion Models #1: 기초 - DDPM과 노이즈 제거 원리"
description: "Stable Diffusion의 핵심인 Diffusion 모델의 수학적 원리와 구현을 완전히 이해합니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["diffusion", "ddpm", "generative-models", "stable-diffusion", "pytorch"]
draft: false
---

# Diffusion Models #1: 기초

**"노이즈에서 이미지를 만든다"**

```
Random Noise → ... → Beautiful Image
[Static]           [Mona Lisa]
```

이게 어떻게 가능할까?

이번 글:
- Diffusion의 직관
- Forward/Reverse Process
- DDPM 수학
- 완전한 구현

---

## Diffusion이란?

### 비유: 물감 확산

**Forward (확산):**

```
깨끗한 물 → 물감 한 방울 → 점점 퍼짐 → 완전히 섞임
[Clear]                                     [Muddy]
```

**Reverse (역확산):**

```
섞인 물 → 점점 분리 → 물감 응집 → 깨끗한 물
[Muddy]                              [Clear]
```

### 이미지 생성

**Forward Process (학습 시):**

```
실제 이미지 x₀
→ 약간 노이즈 x₁
→ 더 많은 노이즈 x₂
→ ...
→ 완전한 노이즈 xₜ
```

**Reverse Process (생성 시):**

```
Random Noise xₜ
→ 약간 덜 노이즈 xₜ₋₁
→ 더 선명 xₜ₋₂
→ ...
→ 실제 이미지 x₀
```

---

## Forward Process (확산)

### 수학

**한 스텝씩 노이즈 추가:**

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

- $x_t$: t 시점의 이미지
- $\beta_t$: 노이즈 스케줄 (0.0001 → 0.02)
- $\mathcal{N}$: 정규분포

**쉽게:**

```python
x_t = sqrt(1 - β_t) * x_{t-1} + sqrt(β_t) * ε

여기서:
- x_{t-1}: 이전 이미지
- β_t: 노이즈 양
- ε ~ N(0, I): 랜덤 노이즈
```

### 중요한 성질: 한 번에 점프!

**T 스텝 반복 대신:**

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)
$$

여기서:
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$

**코드:**

```python
import torch
import torch.nn as nn
import numpy as np

class ForwardDiffusion:
    def __init__(self, T=1000):
        """
        T: Total timesteps
        """
        self.T = T
        
        # Beta schedule (linear)
        self.betas = torch.linspace(0.0001, 0.02, T)
        
        # Alpha
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # For convenience
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Sample x_t from q(x_t | x_0)
        
        x_0: (batch, channels, height, width)
        t: (batch,) - timestep for each sample
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Extract coefficients for each timestep
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(x_0.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        # x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
        
        return x_t, noise

# 사용
diffusion = ForwardDiffusion(T=1000)

# 원본 이미지
x_0 = torch.randn(4, 3, 32, 32)  # Batch=4, RGB, 32x32

# t=500에서 샘플링
t = torch.tensor([500, 500, 500, 500])
x_t, noise = diffusion.q_sample(x_0, t)

print(x_t.shape)  # (4, 3, 32, 32)
```

### 시각화

```python
import matplotlib.pyplot as plt

def visualize_forward_process(image, diffusion, steps=[0, 50, 100, 250, 500, 999]):
    """이미지가 점점 노이즈로 변하는 과정"""
    fig, axes = plt.subplots(1, len(steps), figsize=(15, 3))
    
    for idx, t in enumerate(steps):
        t_tensor = torch.tensor([t])
        x_t, _ = diffusion.q_sample(image.unsqueeze(0), t_tensor)
        
        # Denormalize and show
        img = x_t.squeeze(0).permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        axes[idx].imshow(img)
        axes[idx].set_title(f't={t}')
        axes[idx].axis('off')
    
    plt.show()
```

---

## Reverse Process (생성)

### 목표

**배우고 싶은 것:**

$$
p_\theta(x_{t-1} | x_t)
$$

"노이즈에서 이전 단계를 예측"

### 문제: 직접 학습 불가능

**이유:**

```
p(x_{t-1} | x_t)를 직접 모델링? 
→ x_t만 보고 x_{t-1} 예측? 정보 부족!

해결책:
x_0 (원본)도 조건으로!
→ p(x_{t-1} | x_t, x_0)
```

### 핵심 통찰 (Ho et al., 2020)

**조건부 분포는 정규분포:**

$$
q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)
$$

여기서:

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
$$

**문제:** $x_0$를 모름!

**해결:** $x_0$를 예측하는 모델 학습!

### Noise Prediction (DDPM)

**$x_0$ 직접 예측 대신, 노이즈 $\epsilon$ 예측:**

$$
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t} \epsilon)
$$

**모델:**

```python
ε_θ(x_t, t) → 예측된 노이즈
```

**Loss:**

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

"실제 노이즈와 예측 노이즈의 차이"

---

## U-Net 구조

**DDPM의 backbone:**

```
        Encoder                    Decoder
Input ──────┐                    ┌────── Output
    │       │                    │       │
  Conv   ┌──▼──┐              ┌──▼──┐  Conv
    │    │Down │              │ Up  │    │
  Conv   │     │              │     │  Conv
    │    └──┬──┘              └──▲──┘    │
  Pool      │                    │     Upsample
    │    ┌──▼──┐    Middle   ┌──┴──┐    │
    └────┤Down ├─────────────►│ Up  ├────┘
         └─────┘              └─────┘
         
Skip connections (concat)
```

**구현:**

```python
class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        """
        t: (batch,)
        return: (batch, dim)
        """
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb).to(t.device)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ResBlock(nn.Module):
    """Residual block with time embedding"""
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Linear(time_dim, out_channels)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.act = nn.SiLU()
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, t_emb):
        """
        x: (batch, in_channels, H, W)
        t_emb: (batch, time_dim)
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # Add time embedding
        t_emb = self.time_mlp(t_emb)
        h = h + t_emb[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.skip(x)

class UNet(nn.Module):
    """U-Net for noise prediction"""
    def __init__(self, in_channels=3, out_channels=3, time_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        # Encoder
        self.enc1 = ResBlock(in_channels, 64, time_dim)
        self.enc2 = ResBlock(64, 128, time_dim)
        self.enc3 = ResBlock(128, 256, time_dim)
        
        self.pool = nn.MaxPool2d(2)
        
        # Middle
        self.middle = ResBlock(256, 256, time_dim)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3 = ResBlock(256 + 256, 128, time_dim)  # +256 from skip
        
        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2 = ResBlock(128 + 128, 64, time_dim)
        
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = ResBlock(64 + 64, 64, time_dim)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x, t):
        """
        x: (batch, 3, H, W) - noisy image
        t: (batch,) - timestep
        return: (batch, 3, H, W) - predicted noise
        """
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encoder
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)
        
        # Middle
        m = self.middle(self.pool(e3), t_emb)
        
        # Decoder
        d3 = self.up3(m)
        d3 = torch.cat([d3, e3], dim=1)  # Skip connection
        d3 = self.dec3(d3, t_emb)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2, t_emb)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1, t_emb)
        
        # Output (predicted noise)
        out = self.out(d1)
        return out

# 모델 크기
model = UNet()
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")  # ~35M
```

---

## Training

```python
class DDPMTrainer:
    def __init__(self, model, diffusion, device='cuda'):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.device = device
    
    def train_step(self, x_0):
        """
        x_0: (batch, 3, H, W) - real images
        """
        batch_size = x_0.size(0)
        
        # 1. Random timestep for each sample
        t = torch.randint(0, self.diffusion.T, (batch_size,), device=self.device)
        
        # 2. Add noise
        noise = torch.randn_like(x_0)
        x_t, _ = self.diffusion.q_sample(x_0, t, noise)
        
        # 3. Predict noise
        noise_pred = self.model(x_t, t)
        
        # 4. Loss (MSE)
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def train(self, dataloader, epochs=100, lr=1e-4):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for images, _ in dataloader:
                images = images.to(self.device)
                
                # Train step
                loss = self.train_step(images)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

# Train
diffusion = ForwardDiffusion(T=1000)
model = UNet()
trainer = DDPMTrainer(model, diffusion)

trainer.train(train_loader, epochs=100)
```

---

## Sampling (생성)

```python
@torch.no_grad()
def ddpm_sample(model, diffusion, shape=(4, 3, 32, 32), device='cuda'):
    """
    Generate images from noise
    shape: (batch, channels, height, width)
    """
    model.eval()
    
    # Start from pure noise
    x_t = torch.randn(shape, device=device)
    
    # Reverse process
    for t in reversed(range(diffusion.T)):
        # Current timestep
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        # Predict noise
        noise_pred = model(x_t, t_batch)
        
        # Compute x_{t-1}
        alpha_t = diffusion.alphas[t]
        alpha_bar_t = diffusion.alphas_cumprod[t]
        
        # Mean
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred
        )
        
        if t > 0:
            # Add noise (except last step)
            beta_t = diffusion.betas[t]
            noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(beta_t) * noise
        else:
            x_t = mean
    
    return x_t

# Generate images
samples = ddpm_sample(model, diffusion, shape=(16, 3, 32, 32))
print(samples.shape)  # (16, 3, 32, 32)
```

### 시각화

```python
import torchvision

def visualize_samples(samples):
    """Display generated images"""
    # Denormalize
    samples = (samples + 1) / 2  # [-1, 1] → [0, 1]
    samples = samples.clamp(0, 1)
    
    # Make grid
    grid = torchvision.utils.make_grid(samples, nrow=4)
    
    # Show
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.show()

# Generate and show
samples = ddpm_sample(model, diffusion, shape=(16, 3, 32, 32))
visualize_samples(samples)
```

---

## DDIM (빠른 샘플링)

**DDPM 문제:** 1000 스텝 필요 → 느림!

**DDIM (2020):** 50 스텝으로 같은 품질!

```python
@torch.no_grad()
def ddim_sample(model, diffusion, shape, steps=50, eta=0.0):
    """
    Fast sampling with DDIM
    steps: number of sampling steps (much less than T)
    eta: 0=deterministic, 1=stochastic (DDPM)
    """
    # Select timesteps
    skip = diffusion.T // steps
    timesteps = torch.arange(0, diffusion.T, skip).flip(0)
    
    x_t = torch.randn(shape, device='cuda')
    
    for i, t in enumerate(timesteps):
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device='cuda')
        
        # Predict noise
        noise_pred = model(x_t, t_batch)
        
        # Get α
        alpha_bar_t = diffusion.alphas_cumprod[t]
        
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            alpha_bar_prev = diffusion.alphas_cumprod[t_prev]
        else:
            alpha_bar_prev = torch.tensor(1.0)
        
        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        
        # DDIM update
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * \
                torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        
        direction = torch.sqrt(1 - alpha_bar_prev - sigma**2) * noise_pred
        
        x_t = torch.sqrt(alpha_bar_prev) * x_0_pred + direction
        
        if sigma > 0:
            x_t += sigma * torch.randn_like(x_t)
    
    return x_t

# 20배 빠름!
samples = ddim_sample(model, diffusion, shape=(16, 3, 32, 32), steps=50)
```

---

## 요약

**Diffusion Models:**

1. **Forward**: 이미지 → 노이즈 (확산)
2. **Reverse**: 노이즈 → 이미지 (역확산)
3. **학습**: 노이즈 예측 모델
4. **생성**: 반복적 노이즈 제거

**핵심 수식:**

$$
\mathcal{L} = \mathbb{E}\left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

**장점:**
- 고품질 생성
- 학습 안정적
- Likelihood 계산 가능

**단점:**
- 샘플링 느림 (DDIM으로 해결)

**다음 글:**
- **Stable Diffusion**: Latent Diffusion, CLIP
- **Conditional Generation**: Text-to-Image
- **ControlNet**: 조건부 제어

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
