---
title: "System Design #1: 대규모 시스템 설계 기초 - 확장성의 원칙"
description: "수백만 사용자를 지원하는 시스템을 설계하는 기본 원칙과 확장 전략을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["system-design", "scalability", "architecture", "distributed-systems"]
draft: false
---

# System Design #1: 대규모 시스템 설계 기초

**"YouTube는 어떻게 매일 10억 시간의 동영상을 스트리밍할까?"**

**"Twitter는 어떻게 초당 수만 개의 트윗을 처리할까?"**

답은 **확장 가능한 시스템 설계**입니다.

이번 시리즈에서:
- 확장성 원칙
- 실제 시스템 사례
- Trade-off 분석
- 실전 설계 패턴

---

## 확장성이란?

### 정의

> **부하가 증가해도 성능을 유지하는 능력**

```
사용자 10명 → 1ms 응답
사용자 1000명 → 1ms 응답 ✅
사용자 1M명 → 1ms 응답 ✅
```

### 왜 중요?

**실패 사례:**
```
2016 Pokemon GO 출시
- 예상: 10M users
- 실제: 50M users (첫 주)
- 결과: 서버 다운, 3일간 접속 불가
```

**성공 사례:**
```
2023 ChatGPT 출시
- 5일만에 1M users
- 2개월만에 100M users
- 결과: 안정적 서비스 유지
```

---

## 확장 방법

### Vertical Scaling (Scale Up)

**더 강력한 서버:**

```
Before:
- CPU: 4 cores
- RAM: 16 GB
- Disk: 500 GB SSD

After:
- CPU: 32 cores
- RAM: 256 GB
- Disk: 2 TB NVMe
```

**장점:**
- 간단 (설정 변경 없음)
- 코드 수정 불필요
- 즉시 적용

**단점:**
- 한계 있음 (물리적 제약)
- 비용 급증 (비선형)
- Single Point of Failure

### Horizontal Scaling (Scale Out)

**더 많은 서버:**

```
Before:
1 server × 32 cores = 32 cores

After:
8 servers × 4 cores = 32 cores
```

**장점:**
- 무한 확장 가능
- 비용 선형
- 고가용성 (서버 하나 죽어도 OK)

**단점:**
- 복잡 (분산 시스템)
- 코드 수정 필요
- 일관성 문제

---

## 기본 아키텍처 진화

### 1단계: 단일 서버

```
┌─────────┐
│ Client  │
└────┬────┘
     │
┌────▼────────────────┐
│   Web Server        │
│   ├─ App            │
│   └─ Database       │
└─────────────────────┘
```

**특징:**
- 모든 것이 한 곳
- 간단, 관리 쉬움
- ~1000 users

**한계:**
- CPU/RAM 부족
- DB 병목
- 다운타임 = 전체 중단

### 2단계: Database 분리

```
┌─────────┐
│ Client  │
└────┬────┘
     │
┌────▼────────┐     ┌──────────┐
│ Web Server  │────▶│ Database │
│             │     │          │
└─────────────┘     └──────────┘
```

**개선:**
- 독립적 확장
- Web: CPU 집약
- DB: Memory 집약

**~10K users**

### 3단계: Load Balancer

```
┌─────────┐
│ Client  │
└────┬────┘
     │
┌────▼────────────┐
│ Load Balancer   │
└─┬──────┬────────┘
  │      │
┌─▼──┐ ┌─▼──┐    ┌──────────┐
│Web1│ │Web2│───▶│ Database │
└────┘ └────┘    └──────────┘
```

**개선:**
- 트래픽 분산
- 고가용성
- Rolling deploy

**~100K users**

### 4단계: Database Replication

```
┌────────────┐
│   LB       │
└─┬────────┬─┘
┌─▼──┐   ┌─▼──┐
│Web1│   │Web2│
└─┬──┘   └─┬──┘
  │        │
  ├────────┤
  │        │
┌─▼────────▼────┐
│ Master DB      │
└────┬───────────┘
     │ Replication
   ┌─┴──┬────┬───┐
┌──▼─┐┌─▼─┐┌─▼─┐
│Slv1││Slv2││Slv3│
└────┘└───┘└───┘
```

**개선:**
- Read 분산 (95%+ reads)
- Write: Master
- Read: Slaves
- Failover 가능

**~1M users**

### 5단계: Cache

```
┌────────────┐
│   LB       │
└─┬────────┬─┘
┌─▼──┐   ┌─▼──┐
│Web1│   │Web2│
└─┬──┘   └─┬──┘
  │        │
  ├────────┤
  │        │
┌─▼────────▼─┐
│   Cache    │ (Redis/Memcached)
│   (Memory) │
└─┬──────────┘
  │ Cache miss
┌─▼──────────┐
│ Database   │
└────────────┘
```

**개선:**
- 응답 속도: 100ms → 1ms
- DB 부하 ↓↓
- Hot data in memory

**~10M users**

### 6단계: CDN (Content Delivery Network)

```
┌─────────┐
│ Client  │
└────┬────┘
     │
   ┌─▼─────────┐
   │   CDN     │ (Static files)
   │ ├─ Images │
   │ ├─ CSS/JS │
   │ └─ Videos │
   └─┬─────────┘
     │ Origin miss
┌────▼──────┐
│    LB     │
└───────────┘
```

**개선:**
- Static content: CDN
- Dynamic content: Server
- 지연시간 ↓ (edge servers)
- 대역폭 비용 ↓

**~100M users**

### 7단계: Stateless Architecture

```
┌────────────┐
│   LB       │
└─┬────────┬─┘
┌─▼──┐   ┌─▼──┐
│Web1│   │Web2│ (Stateless)
└─┬──┘   └─┬──┘
  │        │
  ├────────┼─────┐
  │        │     │
┌─▼────────▼─┐ ┌─▼──────────┐
│  Cache     │ │ Session DB │
└────────────┘ └────────────┘
```

**개선:**
- Session: 외부 저장
- 서버 interchangeable
- Auto-scaling 가능

---

## 핵심 개념

### 1. Load Balancing

**알고리즘:**

```python
# Round Robin
servers = ['server1', 'server2', 'server3']
current = 0

def get_server():
    global current
    server = servers[current]
    current = (current + 1) % len(servers)
    return server

# Least Connections
def get_server_least_conn():
    return min(servers, key=lambda s: s.active_connections)

# Weighted
weights = {'server1': 5, 'server2': 3, 'server3': 2}
def get_server_weighted():
    # Pick based on weights
    pass
```

**Health Check:**

```python
def health_check(server):
    try:
        response = requests.get(f"{server}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# 주기적 체크
for server in servers:
    if not health_check(server):
        servers.remove(server)  # Pool에서 제거
```

### 2. Caching

**전략:**

```python
# 1. Cache-Aside (Lazy Loading)
def get_user(user_id):
    # 1. Cache 확인
    user = cache.get(f"user:{user_id}")
    if user:
        return user  # Cache hit
    
    # 2. DB 조회
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    
    # 3. Cache 저장
    cache.set(f"user:{user_id}", user, ttl=3600)
    
    return user

# 2. Write-Through
def update_user(user_id, data):
    # 1. DB 업데이트
    db.update("UPDATE users SET ... WHERE id = ?", user_id, data)
    
    # 2. Cache 업데이트
    cache.set(f"user:{user_id}", data, ttl=3600)

# 3. Write-Behind (Write-Back)
def update_user_async(user_id, data):
    # 1. Cache만 업데이트
    cache.set(f"user:{user_id}", data)
    
    # 2. 비동기로 DB 업데이트
    queue.enqueue("update_user_db", user_id, data)
```

**Eviction Policy:**

```
LRU (Least Recently Used): 가장 오래 안 쓴 것
LFU (Least Frequently Used): 가장 적게 쓴 것
FIFO: 먼저 들어온 것
TTL: 시간 만료
```

### 3. Database Scaling

**Master-Slave Replication:**

```python
class Database:
    def __init__(self):
        self.master = connect("master-db")
        self.slaves = [
            connect("slave-1"),
            connect("slave-2"),
            connect("slave-3")
        ]
    
    def write(self, query):
        # 모든 write는 master
        return self.master.execute(query)
    
    def read(self, query):
        # Read는 slaves 중 하나
        slave = random.choice(self.slaves)
        return slave.execute(query)
```

**Sharding (수평 분할):**

```python
# User ID 기반 sharding
def get_shard(user_id):
    shard_count = 4
    shard_id = hash(user_id) % shard_count
    return shards[shard_id]

# 예: user_id = 12345
# shard_id = hash(12345) % 4 = 2
# → shard_2에 저장

# Range-based sharding
def get_shard_range(user_id):
    if user_id < 10000:
        return shard_0
    elif user_id < 20000:
        return shard_1
    # ...
```

---

## 예제: URL Shortener (bit.ly)

### 요구사항

```
기능:
- 긴 URL → 짧은 URL 변환
- 짧은 URL → 원본 URL 리다이렉트

규모:
- 100M URLs/month (쓰기)
- 10B redirects/month (읽기)
- 읽기:쓰기 = 100:1
```

### 설계

**1. URL 인코딩:**

```python
import hashlib
import base62

def shorten_url(long_url):
    # Hash
    hash_value = hashlib.md5(long_url.encode()).hexdigest()
    
    # Base62 encoding (a-z, A-Z, 0-9)
    short_code = base62.encode(int(hash_value[:8], 16))[:7]
    
    # 예: "https://example.com/..." → "aB3xY9z"
    return f"http://short.ly/{short_code}"
```

**2. 아키텍처:**

```
┌──────────┐
│  Client  │
└────┬─────┘
     │
┌────▼─────────┐
│     CDN      │ (리다이렉트 캐시)
└────┬─────────┘
     │
┌────▼─────────┐
│  API Server  │
└─┬──────────┬─┘
  │          │
┌─▼────┐  ┌─▼────────┐
│Cache │  │ Database │
│(Redis)  │ (Sharded) │
└──────┘  └──────────┘
```

**3. 데이터 모델:**

```sql
CREATE TABLE urls (
    short_code VARCHAR(7) PRIMARY KEY,
    long_url VARCHAR(2048) NOT NULL,
    created_at TIMESTAMP,
    access_count BIGINT DEFAULT 0,
    INDEX idx_created (created_at)
);

-- Sharding key: short_code
-- 4 shards: hash(short_code) % 4
```

**4. API:**

```python
# Shorten
@app.post("/api/shorten")
def shorten(long_url: str):
    # 1. Check cache
    cached = cache.get(f"long:{long_url}")
    if cached:
        return {"short_url": cached}
    
    # 2. Generate short code
    short_code = generate_short_code(long_url)
    
    # 3. Store in DB
    shard = get_shard(short_code)
    shard.execute(
        "INSERT INTO urls (short_code, long_url) VALUES (?, ?)",
        short_code, long_url
    )
    
    # 4. Cache
    cache.set(f"long:{long_url}", short_code, ttl=86400)
    cache.set(f"short:{short_code}", long_url, ttl=86400)
    
    return {"short_url": f"http://short.ly/{short_code}"}

# Redirect
@app.get("/{short_code}")
def redirect(short_code: str):
    # 1. Check cache (99% hit rate)
    long_url = cache.get(f"short:{short_code}")
    if long_url:
        return RedirectResponse(long_url)
    
    # 2. DB query
    shard = get_shard(short_code)
    result = shard.query(
        "SELECT long_url FROM urls WHERE short_code = ?",
        short_code
    )
    
    if not result:
        return {"error": "Not found"}, 404
    
    long_url = result[0]['long_url']
    
    # 3. Cache
    cache.set(f"short:{short_code}", long_url, ttl=86400)
    
    # 4. Async: Update access count
    queue.enqueue("increment_count", short_code)
    
    return RedirectResponse(long_url)
```

**5. 성능:**

```
Capacity:
- 10B requests/month = 3.8K QPS
- Peak: 10K QPS

With:
- 10 API servers (1K QPS each)
- Redis cluster (100K QPS)
- 4 DB shards (1K QPS each)
- CDN (edge caching)

Result:
- 평균 응답시간: <10ms
- Cache hit rate: 99%
- Availability: 99.99%
```

---

## Back-of-the-Envelope 계산

### Twitter 예시

**요구사항:**
```
- 300M active users
- 평균 2 tweets/day/user
- 평균 1 tweet = 140 characters = 280 bytes
- 평균 follow 200명
```

**계산:**

```
Write:
- 300M users × 2 tweets/day = 600M tweets/day
- 600M / 86400 seconds = 6944 tweets/sec
- Peak (3x): 20K tweets/sec

Storage (per day):
- 600M tweets × 280 bytes = 168 GB/day
- Per year: 168 GB × 365 = 61 TB/year

Read (timeline):
- 300M users × 10 timeline views/day = 3B views/day
- 3B / 86400 = 34K QPS
- Peak (3x): 100K QPS

Fanout:
- 1 tweet → 200 followers
- 20K tweets/sec × 200 = 4M writes/sec (timelines)
```

**아키텍처 결정:**
```
- Write: 20K QPS → Sharded DB (10 shards)
- Read: 100K QPS → Heavy caching (Redis)
- Fanout: Async queue (RabbitMQ/Kafka)
```

---

## 요약

**확장성 원칙:**

1. **Stateless**: 서버는 상태 없이
2. **Horizontal**: 서버 추가로 확장
3. **Cache**: 자주 쓰는 데이터는 메모리에
4. **Async**: 무거운 작업은 비동기로
5. **Partition**: 데이터는 분산해서

**진화 단계:**
```
Single Server
→ DB 분리
→ Load Balancer
→ Replication
→ Cache
→ CDN
→ Sharding
→ Multi-datacenter
```

**다음 글:**
- **Database Design**: RDBMS vs NoSQL
- **Caching Strategies**: 심화
- **Message Queue**: 비동기 처리

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
