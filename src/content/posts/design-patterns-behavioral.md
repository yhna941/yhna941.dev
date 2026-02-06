---
title: "Design Patterns #3: 행위 패턴 - Observer, Strategy, Command"
description: "객체 간 상호작용과 책임 분배를 다루는 행위 디자인 패턴을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["design-patterns", "oop", "software-engineering", "python", "behavioral-patterns"]
draft: false
---

# Design Patterns #3: 행위 패턴

**"객체들이 어떻게 협력할 것인가?"**

행위 패턴 (Behavioral Patterns)은:
- 객체 간 통신
- 책임 분배
- 알고리즘 캡슐화
- 느슨한 결합

---

## Observer Pattern

### 문제

**상태 변화 알림:**

```python
# Bad: 강한 결합
class Subject:
    def __init__(self):
        self.state = 0
    
    def set_state(self, state):
        self.state = state
        # 직접 호출 (결합!)
        observer1.update(state)
        observer2.update(state)
        # 새 observer 추가 시 수정 필요
```

### 해결

**Observer 등록/알림:**

```python
from abc import ABC, abstractmethod

# Observer interface
class Observer(ABC):
    @abstractmethod
    def update(self, state):
        pass

# Subject
class Subject:
    def __init__(self):
        self._observers = []
        self._state = None
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self._state)
    
    def set_state(self, state):
        self._state = state
        self.notify()  # 자동 알림

# Concrete Observers
class EmailObserver(Observer):
    def update(self, state):
        print(f"Email: State changed to {state}")

class LogObserver(Observer):
    def update(self, state):
        print(f"Log: State = {state}")

class MetricsObserver(Observer):
    def update(self, state):
        print(f"Metrics: Recording {state}")

# 사용
subject = Subject()

email_obs = EmailObserver()
log_obs = LogObserver()
metrics_obs = MetricsObserver()

subject.attach(email_obs)
subject.attach(log_obs)
subject.attach(metrics_obs)

subject.set_state(10)
# Email: State changed to 10
# Log: State = 10
# Metrics: Recording 10

subject.detach(email_obs)
subject.set_state(20)
# Log: State = 20
# Metrics: Recording 20
```

### Push vs Pull

```python
# Push: Subject가 데이터 전송
class PushSubject:
    def notify(self):
        for observer in self._observers:
            observer.update(self._state)  # 데이터 전송

# Pull: Observer가 데이터 요청
class PullSubject:
    def notify(self):
        for observer in self._observers:
            observer.update(self)  # self 전송

class PullObserver:
    def update(self, subject):
        state = subject.get_state()  # 필요한 것만 가져감
        print(f"Pulled state: {state}")
```

### 실전 예제: Event System

```python
class Event:
    def __init__(self, name, data):
        self.name = name
        self.data = data

class EventBus:
    """중앙 이벤트 관리"""
    def __init__(self):
        self._listeners = {}  # {event_name: [listeners]}
    
    def subscribe(self, event_name, listener):
        if event_name not in self._listeners:
            self._listeners[event_name] = []
        self._listeners[event_name].append(listener)
    
    def unsubscribe(self, event_name, listener):
        if event_name in self._listeners:
            self._listeners[event_name].remove(listener)
    
    def publish(self, event):
        if event.name in self._listeners:
            for listener in self._listeners[event.name]:
                listener.handle(event)

# Listeners
class UserActivityListener:
    def handle(self, event):
        print(f"User activity: {event.data}")

class AnalyticsListener:
    def handle(self, event):
        print(f"Analytics: Tracking {event.name}")

class NotificationListener:
    def handle(self, event):
        if event.name == "user.signup":
            print(f"Welcome email sent to {event.data['email']}")

# 사용
bus = EventBus()

bus.subscribe("user.signup", UserActivityListener())
bus.subscribe("user.signup", AnalyticsListener())
bus.subscribe("user.signup", NotificationListener())

# 이벤트 발생
bus.publish(Event("user.signup", {"email": "user@example.com"}))
# User activity: {'email': 'user@example.com'}
# Analytics: Tracking user.signup
# Welcome email sent to user@example.com
```

---

## Strategy Pattern

### 문제

**알고리즘 선택:**

```python
# Bad: if-else 지옥
class PaymentProcessor:
    def process(self, method, amount):
        if method == "credit_card":
            # Credit card logic
            pass
        elif method == "paypal":
            # PayPal logic
            pass
        elif method == "crypto":
            # Crypto logic
            pass
        # 새 방법 추가 시 수정 필요
```

### 해결

**Strategy로 캡슐화:**

```python
from abc import ABC, abstractmethod

# Strategy interface
class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount):
        pass

# Concrete Strategies
class CreditCardStrategy(PaymentStrategy):
    def __init__(self, card_number):
        self.card_number = card_number
    
    def pay(self, amount):
        print(f"Paid ${amount} with credit card {self.card_number}")

class PayPalStrategy(PaymentStrategy):
    def __init__(self, email):
        self.email = email
    
    def pay(self, amount):
        print(f"Paid ${amount} via PayPal ({self.email})")

class CryptoStrategy(PaymentStrategy):
    def __init__(self, wallet):
        self.wallet = wallet
    
    def pay(self, amount):
        print(f"Paid ${amount} with crypto (wallet: {self.wallet})")

# Context
class PaymentProcessor:
    def __init__(self, strategy: PaymentStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: PaymentStrategy):
        self._strategy = strategy
    
    def process_payment(self, amount):
        self._strategy.pay(amount)

# 사용
processor = PaymentProcessor(CreditCardStrategy("1234-5678"))
processor.process_payment(100)

# 전략 변경
processor.set_strategy(PayPalStrategy("user@example.com"))
processor.process_payment(50)

# 런타임 선택
if user.prefers_crypto:
    strategy = CryptoStrategy(user.wallet)
else:
    strategy = CreditCardStrategy(user.card)

processor.set_strategy(strategy)
processor.process_payment(200)
```

### 실전 예제: Sorting

```python
class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data):
        pass

class QuickSort(SortStrategy):
    def sort(self, data):
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)

class MergeSort(SortStrategy):
    def sort(self, data):
        if len(data) <= 1:
            return data
        mid = len(data) // 2
        left = self.sort(data[:mid])
        right = self.sort(data[mid:])
        return self._merge(left, right)
    
    def _merge(self, left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

class DataSorter:
    def __init__(self, strategy: SortStrategy):
        self._strategy = strategy
    
    def sort(self, data):
        return self._strategy.sort(data)

# 사용 (데이터 크기에 따라)
data = [3, 1, 4, 1, 5, 9, 2, 6]

if len(data) < 1000:
    sorter = DataSorter(QuickSort())
else:
    sorter = DataSorter(MergeSort())

sorted_data = sorter.sort(data)
```

### Python의 함수로 Strategy

```python
# Strategy as functions
def credit_card_payment(amount, card_number):
    print(f"Paid ${amount} with card {card_number}")

def paypal_payment(amount, email):
    print(f"Paid ${amount} via PayPal {email}")

# Context
class SimplePaymentProcessor:
    def __init__(self, payment_func):
        self.payment_func = payment_func
    
    def process(self, amount, *args):
        self.payment_func(amount, *args)

# 사용
processor = SimplePaymentProcessor(credit_card_payment)
processor.process(100, "1234-5678")

processor.payment_func = paypal_payment
processor.process(50, "user@example.com")
```

---

## Command Pattern

### 문제

**요청을 객체화:**

```python
# Bad: 직접 호출
button.click()  # 무엇을 할지 하드코딩
```

### 해결

**Command로 캡슐화:**

```python
from abc import ABC, abstractmethod

# Command interface
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass
    
    @abstractmethod
    def undo(self):
        pass

# Receiver
class Light:
    def __init__(self):
        self.is_on = False
    
    def turn_on(self):
        self.is_on = True
        print("Light is ON")
    
    def turn_off(self):
        self.is_on = False
        print("Light is OFF")

# Concrete Commands
class LightOnCommand(Command):
    def __init__(self, light):
        self.light = light
    
    def execute(self):
        self.light.turn_on()
    
    def undo(self):
        self.light.turn_off()

class LightOffCommand(Command):
    def __init__(self, light):
        self.light = light
    
    def execute(self):
        self.light.turn_off()
    
    def undo(self):
        self.light.turn_on()

# Invoker
class RemoteControl:
    def __init__(self):
        self.command = None
        self.history = []
    
    def set_command(self, command):
        self.command = command
    
    def press_button(self):
        if self.command:
            self.command.execute()
            self.history.append(self.command)
    
    def press_undo(self):
        if self.history:
            command = self.history.pop()
            command.undo()

# 사용
light = Light()
light_on = LightOnCommand(light)
light_off = LightOffCommand(light)

remote = RemoteControl()

remote.set_command(light_on)
remote.press_button()  # Light is ON

remote.set_command(light_off)
remote.press_button()  # Light is OFF

remote.press_undo()  # Light is ON (undo)
```

### Macro Command

```python
class MacroCommand(Command):
    """여러 command 실행"""
    def __init__(self, commands):
        self.commands = commands
    
    def execute(self):
        for command in self.commands:
            command.execute()
    
    def undo(self):
        for command in reversed(self.commands):
            command.undo()

# 사용
tv = TV()
stereo = Stereo()
lights = Light()

party_on = MacroCommand([
    LightOnCommand(lights),
    TVOnCommand(tv),
    StereoOnCommand(stereo)
])

remote.set_command(party_on)
remote.press_button()  # 모든 장치 켜짐!
```

### 실전 예제: Text Editor

```python
class TextEditor:
    def __init__(self):
        self.text = ""
    
    def insert(self, position, text):
        self.text = self.text[:position] + text + self.text[position:]
    
    def delete(self, position, length):
        removed = self.text[position:position+length]
        self.text = self.text[:position] + self.text[position+length:]
        return removed

class InsertCommand(Command):
    def __init__(self, editor, position, text):
        self.editor = editor
        self.position = position
        self.text = text
    
    def execute(self):
        self.editor.insert(self.position, self.text)
    
    def undo(self):
        self.editor.delete(self.position, len(self.text))

class DeleteCommand(Command):
    def __init__(self, editor, position, length):
        self.editor = editor
        self.position = position
        self.length = length
        self.deleted_text = None
    
    def execute(self):
        self.deleted_text = self.editor.delete(self.position, self.length)
    
    def undo(self):
        self.editor.insert(self.position, self.deleted_text)

class CommandHistory:
    def __init__(self):
        self.history = []
        self.current = -1
    
    def execute(self, command):
        # Redo 후 새 command 실행 시 이후 히스토리 삭제
        self.history = self.history[:self.current + 1]
        
        command.execute()
        self.history.append(command)
        self.current += 1
    
    def undo(self):
        if self.current >= 0:
            self.history[self.current].undo()
            self.current -= 1
    
    def redo(self):
        if self.current < len(self.history) - 1:
            self.current += 1
            self.history[self.current].execute()

# 사용
editor = TextEditor()
history = CommandHistory()

# Type "Hello"
history.execute(InsertCommand(editor, 0, "Hello"))
print(editor.text)  # "Hello"

# Type " World"
history.execute(InsertCommand(editor, 5, " World"))
print(editor.text)  # "Hello World"

# Undo
history.undo()
print(editor.text)  # "Hello"

# Redo
history.redo()
print(editor.text)  # "Hello World"
```

---

## Template Method Pattern

### 문제

**알고리즘 구조 공유:**

```python
# Bad: 중복 코드
class PDFParser:
    def parse(self, file):
        self.open_file(file)
        data = self.extract_pdf_data()
        self.close_file()
        return data

class CSVParser:
    def parse(self, file):
        self.open_file(file)  # 중복
        data = self.extract_csv_data()
        self.close_file()  # 중복
        return data
```

### 해결

**Template method:**

```python
from abc import ABC, abstractmethod

class DataParser(ABC):
    def parse(self, file):
        """Template method"""
        self.open_file(file)
        data = self.extract_data()  # Hook
        self.process_data(data)  # Hook
        self.close_file()
        return data
    
    def open_file(self, file):
        print(f"Opening {file}")
        self.file = open(file, 'r')
    
    def close_file(self):
        print("Closing file")
        self.file.close()
    
    @abstractmethod
    def extract_data(self):
        """Subclass가 구현"""
        pass
    
    def process_data(self, data):
        """Optional hook"""
        pass

class CSVParser(DataParser):
    def extract_data(self):
        # CSV-specific
        return self.file.read().split(',')

class JSONParser(DataParser):
    def extract_data(self):
        import json
        return json.load(self.file)
    
    def process_data(self, data):
        # JSON-specific processing
        print("Validating JSON schema")

# 사용
csv_parser = CSVParser()
csv_data = csv_parser.parse("data.csv")

json_parser = JSONParser()
json_data = json_parser.parse("data.json")
```

---

## Iterator Pattern

### 문제

**컬렉션 순회:**

```python
# Bad: 내부 구조 노출
for i in range(len(collection._items)):
    item = collection._items[i]
```

### 해결

**Iterator 제공:**

```python
class BookShelf:
    def __init__(self):
        self._books = []
    
    def add_book(self, book):
        self._books.append(book)
    
    def __iter__(self):
        """Python의 iterator protocol"""
        return BookIterator(self._books)

class BookIterator:
    def __init__(self, books):
        self._books = books
        self._index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._index < len(self._books):
            book = self._books[self._index]
            self._index += 1
            return book
        raise StopIteration

# 사용
shelf = BookShelf()
shelf.add_book("Book 1")
shelf.add_book("Book 2")
shelf.add_book("Book 3")

for book in shelf:  # Pythonic!
    print(book)
```

### Generator (Python)

```python
class BookShelf:
    def __init__(self):
        self._books = []
    
    def add_book(self, book):
        self._books.append(book)
    
    def __iter__(self):
        """Generator로 간단히"""
        for book in self._books:
            yield book
    
    def reverse_iter(self):
        """역순"""
        for book in reversed(self._books):
            yield book

# 사용
for book in shelf:
    print(book)

for book in shelf.reverse_iter():
    print(book)
```

---

## State Pattern

### 문제

**상태별 동작:**

```python
# Bad: if-else
class Document:
    def publish(self):
        if self.state == "draft":
            self.state = "moderation"
        elif self.state == "moderation":
            self.state = "published"
        elif self.state == "published":
            print("Already published")
```

### 해결

**State 객체:**

```python
from abc import ABC, abstractmethod

# State interface
class State(ABC):
    @abstractmethod
    def publish(self, document):
        pass

# Concrete States
class DraftState(State):
    def publish(self, document):
        print("Moving to moderation")
        document.set_state(ModerationState())

class ModerationState(State):
    def publish(self, document):
        print("Publishing document")
        document.set_state(PublishedState())

class PublishedState(State):
    def publish(self, document):
        print("Already published")

# Context
class Document:
    def __init__(self):
        self._state = DraftState()
    
    def set_state(self, state):
        self._state = state
    
    def publish(self):
        self._state.publish(self)

# 사용
doc = Document()
doc.publish()  # Moving to moderation
doc.publish()  # Publishing document
doc.publish()  # Already published
```

---

## Chain of Responsibility

### 문제

**요청 처리자 선택:**

```python
# Bad: 모든 if-else
if user.role == "admin":
    handle_admin(request)
elif user.role == "manager":
    handle_manager(request)
elif user.role == "user":
    handle_user(request)
```

### 해결

**Handler chain:**

```python
from abc import ABC, abstractmethod

class Handler(ABC):
    def __init__(self):
        self._next_handler = None
    
    def set_next(self, handler):
        self._next_handler = handler
        return handler
    
    def handle(self, request):
        if self._next_handler:
            return self._next_handler.handle(request)
        return None

class AuthenticationHandler(Handler):
    def handle(self, request):
        if not request.get("token"):
            return {"error": "Unauthorized"}
        print("Authentication passed")
        return super().handle(request)

class AuthorizationHandler(Handler):
    def handle(self, request):
        if request.get("role") != "admin":
            return {"error": "Forbidden"}
        print("Authorization passed")
        return super().handle(request)

class ValidationHandler(Handler):
    def handle(self, request):
        if not request.get("data"):
            return {"error": "Invalid data"}
        print("Validation passed")
        return super().handle(request)

class BusinessLogicHandler(Handler):
    def handle(self, request):
        print("Processing business logic")
        return {"success": True}

# Chain 구성
auth = AuthenticationHandler()
authz = AuthorizationHandler()
validation = ValidationHandler()
business = BusinessLogicHandler()

auth.set_next(authz).set_next(validation).set_next(business)

# 사용
request = {
    "token": "abc",
    "role": "admin",
    "data": {"key": "value"}
}

result = auth.handle(request)
# Authentication passed
# Authorization passed
# Validation passed
# Processing business logic
```

---

## 요약

**행위 패턴:**

1. **Observer**: 상태 변화 알림
2. **Strategy**: 알고리즘 교체
3. **Command**: 요청 객체화 (undo/redo)
4. **Template Method**: 알고리즘 구조
5. **Iterator**: 컬렉션 순회
6. **State**: 상태별 동작
7. **Chain of Responsibility**: 처리자 체인

**언제 사용?**

```
이벤트 시스템 → Observer
알고리즘 선택 → Strategy
Undo/Redo → Command
공통 구조 → Template Method
순회 → Iterator
상태 기계 → State
Middleware → Chain
```

---

## 디자인 패턴 시리즈 완결! 🎉

**전체 23개 패턴:**

**생성 (1편):**
- Singleton, Factory, Builder, Prototype

**구조 (2편):**
- Adapter, Decorator, Proxy, Facade, Composite, Bridge

**행위 (3편):**
- Observer, Strategy, Command, Template Method, Iterator, State, Chain

**핵심 원칙:**
- SOLID
- DRY (Don't Repeat Yourself)
- KISS (Keep It Simple)
- YAGNI (You Aren't Gonna Need It)

이제 여러분은 디자인 패턴 마스터! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
