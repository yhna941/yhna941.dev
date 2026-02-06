---
title: "System Design #4: Message Queue - Kafka와 이벤트 기반 아키텍처"
description: "대규모 비동기 처리와 마이크로서비스 통신의 핵심인 Message Queue를 완전히 이해합니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["system-design", "message-queue", "kafka", "rabbitmq", "event-driven"]
draft: false
---

# System Design #4: Message Queue

**"비동기가 답이다"**

동기 처리의 문제:
```
User → [API] → Email → SMS → Notification → DB
                ↓ 5s 대기...
         User still waiting...
```

비동기 처리:
```
User → [API] → Queue → Response (10ms)
                ↓
         [Workers] → Email, SMS, Notification (백그라운드)
```

---

## Message Queue란?

### 정의

> **서비스 간 비동기 통신을 위한 중간 저장소**

```
┌──────────┐      ┌───────┐      ┌──────────┐
│ Producer │─────▶│ Queue │─────▶│ Consumer │
└──────────┘      └───────┘      └──────────┘
```

### 언제 사용?

**1. 비동기 처리:**

```python
# Without Queue (동기)
def create_user(data):
    user = db.save_user(data)
    send_email(user.email)  # 2초
    send_sms(user.phone)    # 1초
    update_analytics()      # 1초
    return user  # 4초 후 응답!

# With Queue (비동기)
def create_user(data):
    user = db.save_user(data)
    
    # Enqueue background tasks
    queue.publish('send_email', {'email': user.email})
    queue.publish('send_sms', {'phone': user.phone})
    queue.publish('update_analytics', {'user_id': user.id})
    
    return user  # 즉시 응답!
```

**2. Load Leveling (부하 평준화):**

```
Traffic spike:
10,000 requests/sec
        ↓
    [Queue] (버퍼)
        ↓
 100 requests/sec (Workers가 처리 가능한 속도)
```

**3. Decoupling (서비스 분리):**

```
Before (강결합):
Order Service → Payment Service (직접 호출)
→ Payment 죽으면 Order도 죽음!

After (느슨한 결합):
Order Service → Queue → Payment Service
→ Payment 죽어도 Order는 동작
→ Queue에 쌓이고 나중에 처리
```

---

## 주요 Message Queue

### 비교

| | RabbitMQ | Kafka | Redis Pub/Sub |
|---|---|---|---|
| 타입 | Message Broker | Event Streaming | Cache + Pub/Sub |
| 처리량 | 20K msg/s | 1M msg/s | 1M msg/s |
| 영속성 | 선택적 | 디스크 저장 | 메모리만 |
| 순서 보장 | Queue 내 | Partition 내 | 없음 |
| 사용처 | Task queue | Event log | Real-time notifications |

---

## RabbitMQ

### 기본 개념

```
Producer → Exchange → Queue → Consumer
```

**Exchange Types:**

1. **Direct**: Routing key 정확히 매칭
2. **Fanout**: 모든 queue로 broadcast
3. **Topic**: Pattern matching
4. **Headers**: Header 기반

### 구현

```python
import pika
import json

class RabbitMQProducer:
    def __init__(self, host='localhost'):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host)
        )
        self.channel = self.connection.channel()
    
    def publish(self, queue_name, message):
        """Publish message to queue"""
        # Declare queue (idempotent)
        self.channel.queue_declare(queue=queue_name, durable=True)
        
        # Publish
        self.channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Persistent
            )
        )
        print(f"Published: {message}")
    
    def close(self):
        self.connection.close()

class RabbitMQConsumer:
    def __init__(self, host='localhost'):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host)
        )
        self.channel = self.connection.channel()
    
    def consume(self, queue_name, callback):
        """Consume messages from queue"""
        self.channel.queue_declare(queue=queue_name, durable=True)
        
        def wrapped_callback(ch, method, properties, body):
            message = json.loads(body)
            
            try:
                # Process message
                callback(message)
                
                # Acknowledge
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                # Reject and requeue
                print(f"Error: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        
        self.channel.basic_qos(prefetch_count=1)  # Fair dispatch
        self.channel.basic_consume(
            queue=queue_name,
            on_message_callback=wrapped_callback
        )
        
        print(f"Consuming from {queue_name}...")
        self.channel.start_consuming()

# Producer
producer = RabbitMQProducer()
producer.publish('email_queue', {
    'to': 'user@example.com',
    'subject': 'Welcome!',
    'body': 'Thanks for signing up'
})

# Consumer
def send_email(message):
    print(f"Sending email to {message['to']}")
    # Email sending logic...
    time.sleep(2)  # Simulate work

consumer = RabbitMQConsumer()
consumer.consume('email_queue', send_email)
```

### Work Queue Pattern

```python
# Multiple workers for parallel processing
import threading

def worker(worker_id):
    consumer = RabbitMQConsumer()
    
    def process(message):
        print(f"Worker {worker_id} processing: {message}")
        time.sleep(1)
    
    consumer.consume('tasks', process)

# Start 3 workers
for i in range(3):
    thread = threading.Thread(target=worker, args=(i,))
    thread.start()
```

---

## Kafka

### 핵심 개념

**구조:**

```
Topic: "user_events"
  ├─ Partition 0: [msg1, msg2, msg3, ...]
  ├─ Partition 1: [msg4, msg5, msg6, ...]
  └─ Partition 2: [msg7, msg8, msg9, ...]

Consumer Group: "analytics"
  ├─ Consumer 1 → Partition 0
  ├─ Consumer 2 → Partition 1
  └─ Consumer 3 → Partition 2
```

**특징:**

1. **Partitioning**: 병렬 처리
2. **Replication**: 내구성
3. **Retention**: 메시지 보관 (default 7일)
4. **Offset**: 각 consumer가 독립적으로 읽기 위치 추적

### 구현

```python
from kafka import KafkaProducer, KafkaConsumer
import json

class KafkaMessageProducer:
    def __init__(self, bootstrap_servers=['localhost:9092']):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    
    def send(self, topic, message, key=None):
        """Send message to topic"""
        future = self.producer.send(
            topic,
            value=message,
            key=key.encode('utf-8') if key else None
        )
        
        # Wait for confirmation (optional)
        metadata = future.get(timeout=10)
        print(f"Sent to {metadata.topic} partition {metadata.partition}")
    
    def close(self):
        self.producer.flush()
        self.producer.close()

class KafkaMessageConsumer:
    def __init__(
        self,
        topics,
        group_id,
        bootstrap_servers=['localhost:9092']
    ):
        self.consumer = KafkaConsumer(
            *topics,
            group_id=group_id,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',  # or 'latest'
            enable_auto_commit=False  # Manual commit for reliability
        )
    
    def consume(self, callback):
        """Consume messages"""
        for message in self.consumer:
            try:
                callback(message.value)
                
                # Commit offset
                self.consumer.commit()
            except Exception as e:
                print(f"Error: {e}")
                # Don't commit - will retry

# Producer
producer = KafkaMessageProducer()

producer.send('user_events', {
    'event': 'user_signup',
    'user_id': '12345',
    'timestamp': time.time()
}, key='12345')  # Key for partitioning

# Consumer
consumer = KafkaMessageConsumer(
    topics=['user_events'],
    group_id='analytics'
)

def process_event(event):
    print(f"Processing: {event}")
    # Analytics logic...

consumer.consume(process_event)
```

### Exactly-Once Semantics

```python
class ExactlyOnceProcessor:
    def __init__(self, kafka_consumer, kafka_producer, db):
        self.consumer = kafka_consumer
        self.producer = kafka_producer
        self.db = db
    
    def process_message(self, message):
        """Idempotent processing"""
        message_id = message['id']
        
        # 1. Check if already processed
        if self.db.is_processed(message_id):
            print(f"Already processed: {message_id}")
            return
        
        # 2. Process message
        result = self.do_processing(message)
        
        # 3. Store result + mark as processed (atomic transaction)
        with self.db.transaction():
            self.db.save_result(result)
            self.db.mark_processed(message_id)
        
        # 4. Produce output
        self.producer.send('output_topic', result)
        
        # 5. Commit offset
        self.consumer.commit()
```

---

## Event-Driven Architecture

### 이벤트 기반 마이크로서비스

```
┌─────────────┐
│ User Service│
└──────┬──────┘
       │ Publishes: "user.created"
       ↓
┌─────────────┐
│ Event Bus   │ (Kafka)
└──┬──┬───┬───┘
   │  │   │
   ↓  ↓   ↓
  ┌──┐┌──┐┌──────┐
  │E││S││N│
  │m││M││o│
  │a││S││t│
  │i││ ││i│
  │l││S││f│
  └──┘└──┘└──────┘
```

### Event Schema

```python
# Event definition
class UserCreatedEvent:
    def __init__(self, user_id, email, name, timestamp):
        self.event_type = 'user.created'
        self.user_id = user_id
        self.email = email
        self.name = name
        self.timestamp = timestamp
    
    def to_dict(self):
        return {
            'event_type': self.event_type,
            'user_id': self.user_id,
            'email': self.email,
            'name': self.name,
            'timestamp': self.timestamp
        }

# Publisher (User Service)
class UserService:
    def __init__(self, kafka_producer):
        self.producer = kafka_producer
    
    def create_user(self, data):
        # 1. Create user in DB
        user = db.insert_user(data)
        
        # 2. Publish event
        event = UserCreatedEvent(
            user_id=user.id,
            email=user.email,
            name=user.name,
            timestamp=time.time()
        )
        
        self.producer.send('user_events', event.to_dict())
        
        return user

# Subscriber (Email Service)
class EmailService:
    def __init__(self, kafka_consumer):
        self.consumer = kafka_consumer
    
    def start(self):
        def handle_event(event):
            if event['event_type'] == 'user.created':
                self.send_welcome_email(
                    event['email'],
                    event['name']
                )
        
        self.consumer.consume(handle_event)
    
    def send_welcome_email(self, email, name):
        print(f"Sending welcome email to {email}")
        # Email logic...
```

---

## Patterns

### 1. Retry with Exponential Backoff

```python
import time
import random

class RetryableConsumer:
    def __init__(self, consumer, max_retries=5):
        self.consumer = consumer
        self.max_retries = max_retries
    
    def consume_with_retry(self, callback):
        for message in self.consumer:
            self.process_with_retry(message, callback)
    
    def process_with_retry(self, message, callback):
        for attempt in range(self.max_retries):
            try:
                callback(message.value)
                self.consumer.commit()
                return
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Max retries reached - send to DLQ
                    self.send_to_dlq(message)
                    return
                
                # Exponential backoff
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retry {attempt + 1} after {sleep_time}s")
                time.sleep(sleep_time)
    
    def send_to_dlq(self, message):
        """Send to Dead Letter Queue"""
        dlq_producer.send('dead_letter_queue', message.value)
        print(f"Sent to DLQ: {message}")
```

### 2. Circuit Breaker

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e
    
    def on_success(self):
        self.failures = 0
        self.state = 'CLOSED'
    
    def on_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold:
            self.state = 'OPEN'
            print("Circuit breaker opened!")

# 사용
breaker = CircuitBreaker()

def process_message(message):
    breaker.call(external_api_call, message)
```

### 3. Saga Pattern (Distributed Transaction)

```python
class OrderSaga:
    def __init__(self, event_bus):
        self.bus = event_bus
    
    def create_order(self, order_data):
        """
        Saga steps:
        1. Reserve inventory
        2. Process payment
        3. Update order status
        
        If any step fails → compensate previous steps
        """
        saga_id = generate_id()
        
        # Step 1: Reserve inventory
        self.bus.publish('inventory.reserve', {
            'saga_id': saga_id,
            'product_id': order_data['product_id'],
            'quantity': order_data['quantity']
        })
        
        # Saga coordinator handles rest...
    
    def handle_inventory_reserved(self, event):
        """Inventory reserved - proceed to payment"""
        saga_id = event['saga_id']
        
        self.bus.publish('payment.process', {
            'saga_id': saga_id,
            'amount': event['amount']
        })
    
    def handle_payment_failed(self, event):
        """Payment failed - compensate (release inventory)"""
        saga_id = event['saga_id']
        
        self.bus.publish('inventory.release', {
            'saga_id': saga_id
        })
    
    def handle_payment_succeeded(self, event):
        """All steps succeeded - complete order"""
        saga_id = event['saga_id']
        
        self.bus.publish('order.complete', {
            'saga_id': saga_id
        })
```

---

## 실전 예제: E-commerce Order Processing

```python
# Order Service (Orchestrator)
class OrderService:
    def __init__(self, kafka_producer):
        self.producer = kafka_producer
    
    def place_order(self, order_data):
        order_id = db.create_order(order_data)
        
        # Publish event
        self.producer.send('orders', {
            'event': 'order.placed',
            'order_id': order_id,
            'user_id': order_data['user_id'],
            'items': order_data['items'],
            'total': order_data['total']
        })
        
        return order_id

# Inventory Service
class InventoryService:
    def __init__(self, kafka_consumer, kafka_producer):
        self.consumer = kafka_consumer
        self.producer = kafka_producer
    
    def start(self):
        def handle(event):
            if event['event'] == 'order.placed':
                self.reserve_inventory(event)
        
        self.consumer.consume(handle)
    
    def reserve_inventory(self, order):
        try:
            for item in order['items']:
                db.decrement_stock(item['product_id'], item['quantity'])
            
            # Success
            self.producer.send('orders', {
                'event': 'inventory.reserved',
                'order_id': order['order_id']
            })
        except InsufficientStock:
            # Failure
            self.producer.send('orders', {
                'event': 'inventory.failed',
                'order_id': order['order_id'],
                'reason': 'Out of stock'
            })

# Payment Service
class PaymentService:
    def __init__(self, kafka_consumer, kafka_producer):
        self.consumer = kafka_consumer
        self.producer = kafka_producer
    
    def start(self):
        def handle(event):
            if event['event'] == 'inventory.reserved':
                self.process_payment(event)
        
        self.consumer.consume(handle)
    
    def process_payment(self, order):
        try:
            payment_gateway.charge(
                order['user_id'],
                order['total']
            )
            
            self.producer.send('orders', {
                'event': 'payment.succeeded',
                'order_id': order['order_id']
            })
        except PaymentFailed as e:
            self.producer.send('orders', {
                'event': 'payment.failed',
                'order_id': order['order_id'],
                'reason': str(e)
            })

# Notification Service
class NotificationService:
    def __init__(self, kafka_consumer):
        self.consumer = kafka_consumer
    
    def start(self):
        def handle(event):
            if event['event'] == 'payment.succeeded':
                self.send_confirmation(event)
            elif event['event'] in ['inventory.failed', 'payment.failed']:
                self.send_failure_notification(event)
        
        self.consumer.consume(handle)
    
    def send_confirmation(self, order):
        send_email(
            order['user_id'],
            f"Order {order['order_id']} confirmed!"
        )
```

---

## 요약

**Message Queue 장점:**

1. **비동기**: 빠른 응답
2. **Decoupling**: 서비스 독립성
3. **Load Leveling**: 부하 분산
4. **Reliability**: 메시지 보장

**RabbitMQ vs Kafka:**

```
RabbitMQ:
- Task queue
- 복잡한 routing
- 낮은 latency

Kafka:
- Event streaming
- 높은 처리량
- 장기 보관
- Replay 가능
```

**핵심 패턴:**
- Retry with backoff
- Circuit breaker
- Saga (distributed transaction)
- Dead letter queue

**다음 글:**
- **Microservices**: 서비스 분리
- **API Gateway**: 라우팅, 인증
- **Service Mesh**: Istio, Linkerd

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
