---
title: "System Design #2: Database 설계 - RDBMS vs NoSQL 선택 가이드"
description: "대규모 시스템에서 적절한 데이터베이스를 선택하고 설계하는 방법을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["system-design", "database", "sql", "nosql", "scalability"]
draft: false
---

# System Design #2: Database 설계

**"SQL을 쓸까, NoSQL을 쓸까?"**

가장 많이 받는 질문입니다. 답은: **"상황에 따라 다릅니다."**

이번 글에서:
- RDBMS vs NoSQL 비교
- 실제 사용 사례
- Scaling 전략
- 하이브리드 접근

---

## RDBMS (SQL)

### 특징

```sql
-- Schema 정의
CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP
);

CREATE TABLE posts (
    id BIGINT PRIMARY KEY,
    user_id BIGINT REFERENCES users(id),
    title VARCHAR(255),
    content TEXT,
    created_at TIMESTAMP
);

-- ACID 보장
BEGIN TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;  -- 모두 성공 또는 모두 실패
```

**장점:**
- **ACID**: 트랜잭션 보장
- **Schema**: 데이터 무결성
- **JOIN**: 복잡한 쿼리
- **표준**: SQL은 어디서나

**단점:**
- **Scaling**: 수평 확장 어려움
- **Schema 변경**: 비용 큼
- **Performance**: 복잡한 JOIN은 느림

### 언제 사용?

```
✅ 금융 거래 (ACID 필수)
✅ 사용자 관리 (관계 복잡)
✅ 재고 관리 (일관성 중요)
✅ ERP, CRM

❌ 소셜 미디어 피드 (읽기 많음)
❌ 로그 데이터 (쓰기 많음)
❌ 실시간 분석
```

---

## NoSQL

### 1. Document Store (MongoDB, Couchbase)

```javascript
// Schema-less
{
  "_id": ObjectId("..."),
  "name": "John Doe",
  "email": "john@example.com",
  "posts": [
    {
      "title": "First Post",
      "content": "...",
      "tags": ["tech", "coding"]
    }
  ],
  "followers": [123, 456, 789]
}

// 유연한 구조
{
  "_id": ObjectId("..."),
  "name": "Jane",
  "bio": "Developer",
  // posts 없어도 OK!
}
```

**장점:**
- Schema 유연
- Nested data
- Horizontal scaling

**사용처:**
- CMS (Content Management)
- User profiles
- Catalogs

### 2. Key-Value Store (Redis, DynamoDB)

```python
# Simple operations
cache.set("user:123", json.dumps(user_data))
cache.get("user:123")

# Expiration
cache.setex("session:abc", 3600, session_data)  # 1시간

# Atomic operations
cache.incr("page_views:homepage")
```

**장점:**
- 매우 빠름 (O(1))
- 간단
- 메모리 기반

**사용처:**
- Cache
- Session store
- Rate limiting
- Leaderboards

### 3. Column Store (Cassandra, HBase)

```sql
-- Wide columns
CREATE TABLE events (
    user_id bigint,
    event_time timestamp,
    event_type text,
    data map<text, text>,
    PRIMARY KEY (user_id, event_time)
) WITH CLUSTERING ORDER BY (event_time DESC);

-- Query
SELECT * FROM events 
WHERE user_id = 123 
AND event_time > '2024-01-01';
```

**장점:**
- 쓰기 최적화
- 시계열 데이터
- Petabyte scale

**사용처:**
- Logging
- Analytics
- IoT data
- Time series

### 4. Graph DB (Neo4j, Neptune)

```cypher
// Social network
CREATE (john:User {name: "John"})
CREATE (jane:User {name: "Jane"})
CREATE (john)-[:FOLLOWS]->(jane)

// Query: John의 친구의 친구
MATCH (john:User {name: "John"})-[:FOLLOWS]->()-[:FOLLOWS]->(fof)
RETURN fof.name
```

**장점:**
- 관계 탐색 빠름
- 추천 시스템
- 소셜 네트워크

**사용처:**
- LinkedIn connections
- Fraud detection
- Knowledge graphs

---

## SQL vs NoSQL 비교

| 특징 | RDBMS | NoSQL |
|------|-------|-------|
| Schema | 고정 | 유연 |
| Scaling | Vertical | Horizontal |
| ACID | 강함 | 약함 (BASE) |
| JOIN | 지원 | 제한적 |
| 일관성 | 즉시 | 최종 |
| 사용 | 구조화 데이터 | 비구조화 데이터 |

---

## CAP Theorem

**불가능한 삼각형:**

```
   Consistency
      /\
     /  \
    /    \
   /  CA  \
  /________\
AP          CP
```

**동시에 3개 모두 불가능!**

### CA (Consistency + Availability)

```
- RDBMS (단일 노드)
- 파티션 허용 안 함
```

### CP (Consistency + Partition Tolerance)

```
- MongoDB (strong consistency)
- HBase
- Redis (single instance)

파티션 발생 시: 일부 노드 unavailable
```

### AP (Availability + Partition Tolerance)

```
- Cassandra
- DynamoDB
- Couchbase

파티션 발생 시: 일관성 잠시 포기
```

### 선택

```python
# 금융: CP (일관성 > 가용성)
if transaction_system:
    return "CP"  # PostgreSQL, MongoDB

# 소셜 미디어: AP (가용성 > 일관성)
if social_feed:
    return "AP"  # Cassandra, DynamoDB

# 단일 노드: CA
if small_scale:
    return "CA"  # MySQL, PostgreSQL
```

---

## Sharding (분할)

### Horizontal Partitioning

**User ID 기반:**

```python
def get_shard(user_id):
    num_shards = 4
    return user_id % num_shards

# user_id = 12345
# shard = 12345 % 4 = 1
# → Shard 1에 저장
```

**Range-based:**

```python
# 0-25M: Shard 0
# 25M-50M: Shard 1
# 50M-75M: Shard 2
# 75M-100M: Shard 3

def get_shard_range(user_id):
    if user_id < 25_000_000:
        return 0
    elif user_id < 50_000_000:
        return 1
    # ...
```

**Hash-based:**

```python
import hashlib

def get_shard_hash(user_id):
    hash_value = hashlib.md5(str(user_id).encode()).hexdigest()
    return int(hash_value, 16) % num_shards

# 균등 분산!
```

### Consistent Hashing

```python
class ConsistentHash:
    def __init__(self, nodes):
        self.ring = {}
        self.sorted_keys = []
        
        for node in nodes:
            for i in range(150):  # Virtual nodes
                key = self.hash(f"{node}:{i}")
                self.ring[key] = node
                self.sorted_keys.append(key)
        
        self.sorted_keys.sort()
    
    def hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def get_node(self, item):
        if not self.ring:
            return None
        
        key = self.hash(str(item))
        
        # Binary search
        idx = bisect.bisect_right(self.sorted_keys, key)
        if idx == len(self.sorted_keys):
            idx = 0
        
        return self.ring[self.sorted_keys[idx]]

# 사용
ch = ConsistentHash(["shard1", "shard2", "shard3", "shard4"])
shard = ch.get_node(user_id)
```

**장점:**
- 노드 추가/제거 시 최소 이동
- 균등 분산

---

## Replication

### Master-Slave

```
        ┌─────────┐
        │ Master  │ (Write)
        └────┬────┘
             │ Replication
      ┌──────┼──────┐
      │      │      │
  ┌───▼──┐┌──▼──┐┌──▼──┐
  │Slave1││Slave2││Slave3│ (Read)
  └──────┘└─────┘└─────┘
```

**특징:**
- 모든 write → Master
- Read → Slaves
- Async replication (약간 lag)

**구현:**

```python
class Database:
    def __init__(self):
        self.master = connect("master-db")
        self.slaves = [
            connect("slave-1"),
            connect("slave-2"),
            connect("slave-3")
        ]
        self.current_slave = 0
    
    def write(self, query):
        return self.master.execute(query)
    
    def read(self, query, consistent=False):
        if consistent:
            # Strong consistency: Read from master
            return self.master.execute(query)
        
        # Eventually consistent: Read from slave
        slave = self.slaves[self.current_slave]
        self.current_slave = (self.current_slave + 1) % len(self.slaves)
        return slave.execute(query)
```

### Multi-Master

```
  ┌─────────┐     ┌─────────┐
  │ Master1 │◄───►│ Master2 │
  └────┬────┘     └────┬────┘
       │               │
     Slaves          Slaves
```

**특징:**
- 양쪽 write 가능
- 충돌 해결 필요
- 지역별 배치 (Geo-distributed)

**충돌 해결:**

```python
# Last Write Wins (LWW)
def resolve_conflict(version_a, version_b):
    return version_a if version_a.timestamp > version_b.timestamp else version_b

# Vector Clocks
def merge_versions(version_a, version_b):
    if dominates(version_a.vector, version_b.vector):
        return version_a
    elif dominates(version_b.vector, version_a.vector):
        return version_b
    else:
        # Conflict: 사용자 개입 필요
        return ask_user_to_resolve(version_a, version_b)
```

---

## Indexing

### B-Tree Index (기본)

```sql
-- Create index
CREATE INDEX idx_user_email ON users(email);

-- Query (fast!)
SELECT * FROM users WHERE email = 'john@example.com';
-- Index seek: O(log n)

-- Without index
-- Full table scan: O(n)
```

### Composite Index

```sql
-- Create
CREATE INDEX idx_user_created ON users(status, created_at);

-- Good
SELECT * FROM users WHERE status = 'active' AND created_at > '2024-01-01';

-- Bad (index not used)
SELECT * FROM users WHERE created_at > '2024-01-01';
-- status는 앞에 와야 index 사용!
```

### Covering Index

```sql
-- Index includes all columns
CREATE INDEX idx_user_cover ON users(email, name, created_at);

-- Query
SELECT email, name, created_at FROM users WHERE email = 'john@example.com';
-- Index만으로 해결! (No table access)
```

### Full-Text Search

```sql
-- PostgreSQL
CREATE INDEX idx_posts_search ON posts USING GIN(to_tsvector('english', title || ' ' || content));

-- Query
SELECT * FROM posts 
WHERE to_tsvector('english', title || ' ' || content) @@ to_tsquery('postgres & database');
```

**Elasticsearch (전문 검색):**

```python
from elasticsearch import Elasticsearch

es = Elasticsearch()

# Index
es.index(
    index="posts",
    id=post_id,
    body={
        "title": "PostgreSQL Tutorial",
        "content": "Learn about PostgreSQL...",
        "tags": ["database", "sql"]
    }
)

# Search
results = es.search(
    index="posts",
    body={
        "query": {
            "multi_match": {
                "query": "postgres database",
                "fields": ["title^2", "content", "tags"]  # title 2배 가중치
            }
        }
    }
)
```

---

## 실전 사례: Instagram

### 요구사항

```
- 1B+ photos
- 100M+ users
- 500M+ requests/day
- Sub-second response
```

### 설계

**1. PostgreSQL (Metadata)**

```sql
-- Users
CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    username VARCHAR(50) UNIQUE,
    created_at TIMESTAMP
);

-- Photos
CREATE TABLE photos (
    id BIGINT PRIMARY KEY,
    user_id BIGINT REFERENCES users(id),
    url VARCHAR(255),
    created_at TIMESTAMP,
    INDEX idx_user_photos (user_id, created_at)
);

-- Sharded by user_id
-- 100 shards
```

**2. Cassandra (Feed)**

```sql
-- User timeline (home feed)
CREATE TABLE user_feed (
    user_id BIGINT,
    photo_id BIGINT,
    posted_at TIMESTAMP,
    PRIMARY KEY (user_id, posted_at)
) WITH CLUSTERING ORDER BY (posted_at DESC);

-- Fast read: SELECT * FROM user_feed WHERE user_id = ? LIMIT 20;
```

**3. Redis (Cache)**

```python
# Hot users' feeds
cache.set(f"feed:{user_id}", json.dumps(feed), ttl=300)

# Like counts
cache.incr(f"likes:{photo_id}")

# Session
cache.setex(f"session:{token}", 3600, user_id)
```

**4. S3 (Photos)**

```python
# Upload
s3.upload_file(
    photo_data,
    bucket="instagram-photos",
    key=f"{user_id}/{photo_id}.jpg"
)

# CDN
photo_url = f"https://cdn.instagram.com/{user_id}/{photo_id}.jpg"
```

### 아키텍처

```
┌──────────┐
│  Client  │
└────┬─────┘
     │
┌────▼──────┐
│    CDN    │ (Photos)
└────┬──────┘
     │
┌────▼──────────┐
│ Load Balancer │
└───┬───────────┘
    │
┌───▼────────┬──────────┬─────────┐
│  API       │  API     │  API    │
│ Server 1   │ Server 2 │ Server 3│
└───┬────────┴──────────┴────┬────┘
    │                        │
┌───▼──────┐            ┌────▼─────┐
│  Redis   │            │Cassandra │
│ (Cache)  │            │ (Feed)   │
└──────────┘            └──────────┘
    │                        │
┌───▼──────────────────┬─────▼────┐
│   PostgreSQL         │    S3    │
│   (Metadata)         │ (Photos) │
│   100 shards         │          │
└──────────────────────┴──────────┘
```

---

## 요약

**선택 가이드:**

```python
def choose_database(requirements):
    if requirements.acid and requirements.relations:
        return "PostgreSQL/MySQL"
    
    if requirements.flexible_schema:
        return "MongoDB"
    
    if requirements.high_writes and requirements.time_series:
        return "Cassandra"
    
    if requirements.graph_traversal:
        return "Neo4j"
    
    if requirements.cache:
        return "Redis"
    
    # 하이브리드!
    return "Multiple databases"
```

**핵심 원칙:**

1. **Polyglot Persistence**: 여러 DB 조합
2. **Right tool for the job**: 적재적소
3. **Start simple**: 처음엔 단순하게
4. **Scale when needed**: 필요할 때 확장

**다음 글:**
- **Caching Strategies**: Redis 심화
- **Message Queue**: Kafka, RabbitMQ
- **Microservices**: 서비스 분리

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
