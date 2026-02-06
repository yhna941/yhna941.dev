---
title: "System Design #3: Caching 전략 - Redis 심화와 Cache Invalidation"
description: "대규모 시스템의 성능을 극대화하는 캐싱 전략과 실전 패턴을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["system-design", "caching", "redis", "performance", "scalability"]
draft: false
---

# System Design #3: Caching 전략

**"Cache가 없으면 죽는다"**

성능 비교:
```
Without Cache:
- Response: 500ms
- DB load: 100%
- Cost: $$$

With Cache:
- Response: 5ms (100배 빠름!)
- DB load: 10%
- Cost: $
```

---

## Cache 계층

### 전체 아키텍처

```
┌─────────┐
│ Client  │
└────┬────┘
     │
┌────▼─────────┐
│ CDN (Static) │ 🌍 Edge locations
└────┬─────────┘
     │
┌────▼─────────┐
│ API Server   │
└─┬────────┬───┘
  │        │
┌─▼────┐ ┌─▼─────────┐
│Redis │ │Application│ 💾 In-memory
│      │ │Cache      │
└─┬────┘ └───────────┘
  │
┌─▼────────┐
│ Database │ 💿 Disk
└──────────┘
```

**계층별 특성:**

```
L1 - Application Cache:
- Latency: 0.1ms
- Size: 100MB
- Scope: Single process

L2 - Redis:
- Latency: 1ms
- Size: 100GB
- Scope: All servers

L3 - Database:
- Latency: 10-100ms
- Size: 10TB
- Scope: Persistent
```

---

## Redis 기초

### 데이터 구조

```python
import redis

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# 1. String (가장 기본)
r.set('user:1000:name', 'John')
r.get('user:1000:name')  # 'John'

# TTL (Time To Live)
r.setex('session:abc', 3600, 'user_data')  # 1시간 후 자동 삭제

# 2. Hash (객체 저장)
r.hset('user:1000', mapping={
    'name': 'John',
    'email': 'john@example.com',
    'age': 30
})
r.hgetall('user:1000')  # {'name': 'John', 'email': '...', 'age': '30'}

# 3. List (큐, 스택)
r.lpush('queue:tasks', 'task1', 'task2', 'task3')
r.rpop('queue:tasks')  # 'task1' (FIFO)

# 4. Set (중복 제거)
r.sadd('user:1000:followers', '1001', '1002', '1003')
r.sismember('user:1000:followers', '1001')  # True

# 5. Sorted Set (리더보드)
r.zadd('leaderboard', {'user1': 100, 'user2': 250, 'user3': 150})
r.zrevrange('leaderboard', 0, 9)  # Top 10
```

### 실전 패턴

**User Session:**

```python
class SessionManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 86400  # 24시간
    
    def create_session(self, session_id, user_id):
        """Create user session"""
        session_key = f"session:{session_id}"
        self.redis.hset(session_key, mapping={
            'user_id': user_id,
            'created_at': time.time()
        })
        self.redis.expire(session_key, self.ttl)
    
    def get_user(self, session_id):
        """Get user from session"""
        session_key = f"session:{session_id}"
        return self.redis.hget(session_key, 'user_id')
    
    def extend_session(self, session_id):
        """Extend session TTL"""
        session_key = f"session:{session_id}"
        self.redis.expire(session_key, self.ttl)
```

**Rate Limiting:**

```python
class RateLimiter:
    def __init__(self, redis_client, max_requests=100, window=60):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window = window  # seconds
    
    def is_allowed(self, user_id):
        """Check if request is allowed"""
        key = f"rate_limit:{user_id}"
        
        # Increment counter
        count = self.redis.incr(key)
        
        # Set expiry on first request
        if count == 1:
            self.redis.expire(key, self.window)
        
        return count <= self.max_requests

# 사용
limiter = RateLimiter(redis_client, max_requests=100, window=60)

if limiter.is_allowed(user_id):
    # Process request
    pass
else:
    # 429 Too Many Requests
    return {"error": "Rate limit exceeded"}
```

**Leaderboard:**

```python
class Leaderboard:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.key = "leaderboard:global"
    
    def update_score(self, user_id, score):
        """Update user score"""
        self.redis.zadd(self.key, {user_id: score})
    
    def increment_score(self, user_id, delta):
        """Increment user score"""
        self.redis.zincrby(self.key, delta, user_id)
    
    def get_top(self, n=10):
        """Get top N users"""
        # ZREVRANGE: highest to lowest
        top_users = self.redis.zrevrange(
            self.key, 0, n-1, withscores=True
        )
        return [(user, int(score)) for user, score in top_users]
    
    def get_rank(self, user_id):
        """Get user rank (0-indexed)"""
        rank = self.redis.zrevrank(self.key, user_id)
        return rank + 1 if rank is not None else None
    
    def get_score(self, user_id):
        """Get user score"""
        score = self.redis.zscore(self.key, user_id)
        return int(score) if score else 0

# 사용
lb = Leaderboard(redis_client)
lb.update_score('user1', 1000)
lb.increment_score('user1', 50)

print(lb.get_top(10))  # [(user, score), ...]
print(lb.get_rank('user1'))  # 1
```

---

## Caching 전략

### 1. Cache-Aside (Lazy Loading)

**가장 일반적:**

```python
def get_user(user_id):
    cache_key = f"user:{user_id}"
    
    # 1. Try cache
    user = cache.get(cache_key)
    if user:
        return user  # Cache hit
    
    # 2. Cache miss - fetch from DB
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    
    # 3. Store in cache
    cache.set(cache_key, user, ttl=3600)
    
    return user
```

**장점:**
- 간단
- 필요한 것만 캐싱
- Fault tolerant (cache 죽어도 동작)

**단점:**
- Cold start (처음엔 느림)
- Cache miss penalty

### 2. Write-Through

**쓸 때마다 캐시 업데이트:**

```python
def update_user(user_id, data):
    cache_key = f"user:{user_id}"
    
    # 1. Update DB
    db.update("UPDATE users SET ... WHERE id = ?", user_id, data)
    
    # 2. Update cache
    cache.set(cache_key, data, ttl=3600)
    
    return data
```

**장점:**
- 항상 최신 데이터
- Cache miss 적음

**단점:**
- 쓰기 느림 (2번 write)
- 안 읽는 데이터도 캐싱

### 3. Write-Behind (Write-Back)

**비동기 쓰기:**

```python
import asyncio
from collections import deque

class WriteBehindCache:
    def __init__(self, cache, db, batch_size=100, flush_interval=5):
        self.cache = cache
        self.db = db
        self.write_queue = deque()
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Background flusher
        asyncio.create_task(self.flush_worker())
    
    def set(self, key, value):
        # 1. Update cache immediately
        self.cache.set(key, value)
        
        # 2. Queue DB write
        self.write_queue.append((key, value))
        
        # 3. Flush if batch full
        if len(self.write_queue) >= self.batch_size:
            self.flush()
    
    def flush(self):
        """Flush queue to DB"""
        while self.write_queue:
            key, value = self.write_queue.popleft()
            self.db.update(key, value)
    
    async def flush_worker(self):
        """Periodic flush"""
        while True:
            await asyncio.sleep(self.flush_interval)
            self.flush()
```

**장점:**
- 쓰기 빠름 (비동기)
- Batch write (효율적)

**단점:**
- 데이터 손실 위험
- 복잡

### 4. Refresh-Ahead

**만료 전 갱신:**

```python
class RefreshAheadCache:
    def __init__(self, cache, db, ttl=3600, refresh_threshold=0.8):
        self.cache = cache
        self.db = db
        self.ttl = ttl
        self.refresh_threshold = refresh_threshold
    
    def get(self, key):
        # Get from cache
        value, remaining_ttl = self.cache.get_with_ttl(key)
        
        if value is None:
            # Cache miss
            value = self.db.get(key)
            self.cache.set(key, value, ttl=self.ttl)
        elif remaining_ttl < self.ttl * self.refresh_threshold:
            # Refresh in background
            asyncio.create_task(self.refresh(key))
        
        return value
    
    async def refresh(self, key):
        """Background refresh"""
        value = self.db.get(key)
        self.cache.set(key, value, ttl=self.ttl)
```

---

## Cache Invalidation

**"컴퓨터 과학의 가장 어려운 문제"**

### 전략

#### 1. TTL (Time To Live)

```python
# 간단하지만 stale data 가능
cache.set('user:1000', user_data, ttl=3600)  # 1시간
```

**언제 사용:**
- 약간의 staleness 허용
- 데이터 자주 변경 안 됨

#### 2. Explicit Invalidation

```python
def update_user(user_id, data):
    # 1. Update DB
    db.update(user_id, data)
    
    # 2. Invalidate cache
    cache.delete(f"user:{user_id}")
    
    # 3. Invalidate related caches
    cache.delete(f"user:{user_id}:posts")
    cache.delete(f"feed:{user_id}")
```

**주의:** Cascading invalidation!

#### 3. Cache Stampede 방지

**문제:**

```
Cache expires
→ 1000 requests hit DB simultaneously
→ DB overload!
```

**해결: Lock**

```python
import threading

class CacheWithLock:
    def __init__(self, cache, db):
        self.cache = cache
        self.db = db
        self.locks = {}
    
    def get(self, key):
        # Try cache
        value = self.cache.get(key)
        if value:
            return value
        
        # Acquire lock for this key
        if key not in self.locks:
            self.locks[key] = threading.Lock()
        
        lock = self.locks[key]
        
        with lock:
            # Double-check after acquiring lock
            value = self.cache.get(key)
            if value:
                return value
            
            # Fetch from DB (only one thread does this)
            value = self.db.get(key)
            self.cache.set(key, value, ttl=3600)
            
            return value
```

**Redis Lock:**

```python
def get_with_redis_lock(key, redis_client, db):
    # Try cache
    value = redis_client.get(key)
    if value:
        return value
    
    # Try to acquire lock
    lock_key = f"lock:{key}"
    lock_acquired = redis_client.set(lock_key, '1', nx=True, ex=10)
    
    if lock_acquired:
        # I got the lock - fetch from DB
        value = db.get(key)
        redis_client.set(key, value, ex=3600)
        redis_client.delete(lock_key)
        return value
    else:
        # Someone else is fetching - wait and retry
        time.sleep(0.1)
        return get_with_redis_lock(key, redis_client, db)
```

#### 4. Probabilistic Early Expiration

**랜덤하게 미리 만료:**

```python
import random

def get_with_early_expiration(key, cache, db, base_ttl=3600):
    value, remaining_ttl = cache.get_with_ttl(key)
    
    if value is None:
        # Cache miss
        value = db.get(key)
        cache.set(key, value, ttl=base_ttl)
        return value
    
    # Probabilistic refresh
    # More likely as expiration approaches
    delta = base_ttl - remaining_ttl
    probability = delta * random.random() / base_ttl
    
    if random.random() < probability:
        # Refresh
        value = db.get(key)
        cache.set(key, value, ttl=base_ttl)
    
    return value
```

---

## Multi-Layer Caching

```python
class MultiLevelCache:
    def __init__(self, l1_cache, l2_redis, db):
        """
        l1_cache: Local in-memory cache (LRU)
        l2_redis: Redis (shared)
        db: Database
        """
        self.l1 = l1_cache
        self.l2 = l2_redis
        self.db = db
    
    def get(self, key):
        # L1 (local)
        value = self.l1.get(key)
        if value:
            return value
        
        # L2 (Redis)
        value = self.l2.get(key)
        if value:
            self.l1.set(key, value)  # Populate L1
            return value
        
        # L3 (Database)
        value = self.db.get(key)
        self.l2.set(key, value, ttl=3600)  # Populate L2
        self.l1.set(key, value)  # Populate L1
        
        return value
    
    def set(self, key, value):
        # Invalidate all levels
        self.l1.delete(key)
        self.l2.delete(key)
        
        # Update DB
        self.db.update(key, value)
```

---

## Redis 고급 기능

### 1. Pub/Sub (Cache Invalidation)

```python
# Publisher (when data changes)
def update_user(user_id, data):
    db.update(user_id, data)
    
    # Notify all servers
    redis_client.publish('cache_invalidation', f'user:{user_id}')

# Subscriber (on each server)
def cache_invalidation_listener():
    pubsub = redis_client.pubsub()
    pubsub.subscribe('cache_invalidation')
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            key = message['data']
            local_cache.delete(key)
            print(f"Invalidated: {key}")

# Run subscriber in background
threading.Thread(target=cache_invalidation_listener, daemon=True).start()
```

### 2. Lua Scripts (Atomic Operations)

```python
# Atomic compare-and-set
lua_script = """
local current = redis.call('GET', KEYS[1])
if current == ARGV[1] then
    redis.call('SET', KEYS[1], ARGV[2])
    return 1
else
    return 0
end
"""

script = redis_client.register_script(lua_script)

# Usage
success = script(keys=['user:1000:version'], args=[old_version, new_version])
if success:
    print("Updated!")
else:
    print("Conflict!")
```

### 3. Redis Streams (Event Sourcing)

```python
# Producer
redis_client.xadd('events', {
    'type': 'user_updated',
    'user_id': '1000',
    'timestamp': time.time()
})

# Consumer
last_id = '0'
while True:
    events = redis_client.xread({'events': last_id}, block=1000)
    
    for stream, messages in events:
        for message_id, data in messages:
            process_event(data)
            last_id = message_id
```

---

## 실전 예제: Twitter Timeline

```python
class TwitterTimeline:
    def __init__(self, redis_client, db):
        self.redis = redis_client
        self.db = db
    
    def post_tweet(self, user_id, tweet_id):
        """User posts a tweet"""
        # 1. Save to DB
        self.db.insert_tweet(user_id, tweet_id)
        
        # 2. Fan-out to followers' timelines (Redis)
        followers = self.db.get_followers(user_id)
        
        for follower_id in followers:
            # Add to follower's timeline (Sorted Set)
            self.redis.zadd(
                f"timeline:{follower_id}",
                {tweet_id: time.time()}
            )
            
            # Keep only recent 1000 tweets
            self.redis.zremrangebyrank(
                f"timeline:{follower_id}",
                0, -1001
            )
    
    def get_timeline(self, user_id, limit=20):
        """Get user's timeline"""
        timeline_key = f"timeline:{user_id}"
        
        # Get from Redis (sorted by timestamp)
        tweet_ids = self.redis.zrevrange(timeline_key, 0, limit-1)
        
        if not tweet_ids:
            # Cold start - load from DB
            tweet_ids = self.db.get_timeline(user_id, limit)
            
            # Populate cache
            for tweet_id in tweet_ids:
                self.redis.zadd(
                    timeline_key,
                    {tweet_id: self.db.get_tweet_time(tweet_id)}
                )
        
        # Fetch tweet details (batch)
        tweets = self.db.get_tweets(tweet_ids)
        
        return tweets
```

---

## 요약

**Caching 전략:**

1. **Cache-Aside**: 가장 일반적
2. **Write-Through**: 항상 최신
3. **Write-Behind**: 쓰기 빠름
4. **Refresh-Ahead**: Proactive

**Cache Invalidation:**

1. **TTL**: 간단
2. **Explicit**: 정확
3. **Lock**: Stampede 방지
4. **Probabilistic**: 분산

**Redis 활용:**
- Session
- Rate Limiting
- Leaderboard
- Pub/Sub
- Streams

**핵심:**

> "Cache는 약이지만, 잘못 쓰면 독!"

**다음 글:**
- **Message Queue**: Kafka, RabbitMQ
- **Microservices**: 서비스 분리
- **API Gateway**: 라우팅, 인증

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
