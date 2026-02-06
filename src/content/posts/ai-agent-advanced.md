---
title: "AI Agent #2: Advanced Agents - AutoGPT, BabyAGI, AgentGPT"
description: "자율적으로 목표를 달성하는 고급 AI Agent 시스템의 구조와 구현을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["ai-agent", "autogpt", "babyagi", "autonomous-agents", "llm"]
draft: false
---

# AI Agent #2: Advanced Agents

**"스스로 생각하고 행동하는 AI"**

기본 Agent:
```
User: "날씨 알려줘"
Agent: Weather API → "15도입니다"
```

Advanced Agent:
```
User: "AI 스타트업 창업 계획서 만들어줘"
Agent: 
1. 시장 조사 (웹 검색)
2. 경쟁사 분석
3. 비즈니스 모델 설계
4. 재무 계획
5. 문서 작성
→ 30페이지 계획서 완성!
```

---

## AutoGPT

### 개념

> **목표 달성까지 스스로 계획하고 실행하는 Agent**

**핵심:**
```
1. 목표 설정
2. 작업 분해
3. 실행
4. 결과 평가
5. 다음 작업 결정
→ 반복 (목표 달성까지)
```

### 아키텍처

```
┌──────────────────────────────┐
│        AutoGPT Loop          │
│                              │
│  ┌────────────────────────┐ │
│  │ 1. Think               │ │
│  │   "What should I do?"  │ │
│  └───────┬────────────────┘ │
│          │                   │
│  ┌───────▼────────────────┐ │
│  │ 2. Plan                │ │
│  │   Break down task      │ │
│  └───────┬────────────────┘ │
│          │                   │
│  ┌───────▼────────────────┐ │
│  │ 3. Execute             │ │
│  │   Use tools            │ │
│  └───────┬────────────────┘ │
│          │                   │
│  ┌───────▼────────────────┐ │
│  │ 4. Review              │ │
│  │   Evaluate result      │ │
│  └───────┬────────────────┘ │
│          │                   │
│  ┌───────▼────────────────┐ │
│  │ 5. Decide              │ │
│  │   Continue or finish?  │ │
│  └────────────────────────┘ │
└──────────────────────────────┘
```

### 구현

```python
from typing import List, Dict
import json

class AutoGPT:
    def __init__(self, llm, tools, max_iterations=25):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.memory = []  # Long-term memory
        self.goals = []
    
    def run(self, objective: str) -> str:
        """Run AutoGPT to achieve objective"""
        # 1. Initial planning
        self.goals = self.create_goals(objective)
        
        print(f"Objective: {objective}")
        print(f"Goals: {self.goals}")
        
        # 2. Execution loop
        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Think
            thoughts = self.think()
            print(f"Thoughts: {thoughts['reasoning']}")
            
            # Plan
            plan = thoughts['plan']
            print(f"Plan: {plan}")
            
            # Execute
            if thoughts['command']['name'] == 'finish':
                print("Task completed!")
                return thoughts['command']['args']['response']
            
            result = self.execute_command(thoughts['command'])
            print(f"Result: {result}")
            
            # Update memory
            self.memory.append({
                'iteration': iteration + 1,
                'thoughts': thoughts,
                'result': result
            })
            
            # Self-evaluate
            progress = self.evaluate_progress()
            print(f"Progress: {progress}")
        
        return "Max iterations reached"
    
    def create_goals(self, objective: str) -> List[str]:
        """Break objective into goals"""
        prompt = f"""Given this objective, create 3-5 specific goals:

Objective: {objective}

Goals (numbered list):"""
        
        response = self.llm.generate(prompt)
        goals = [line.strip() for line in response.split('\n') if line.strip()]
        return goals
    
    def think(self) -> Dict:
        """Generate thoughts and next action"""
        context = self.build_context()
        
        prompt = f"""You are an autonomous AI agent. Analyze the situation and decide next action.

Context:
{context}

Respond in JSON format:
{{
  "thoughts": {{
    "text": "current thoughts",
    "reasoning": "why this action",
    "plan": "short-term plan",
    "criticism": "self-critique"
  }},
  "command": {{
    "name": "command_name",
    "args": {{"arg": "value"}}
  }}
}}

Available commands: {list(self.tools.keys())}, finish
"""
        
        response = self.llm.generate(prompt)
        return json.loads(response)
    
    def build_context(self) -> str:
        """Build context from memory and goals"""
        context = f"Goals:\n"
        for i, goal in enumerate(self.goals, 1):
            context += f"{i}. {goal}\n"
        
        context += f"\nRecent actions:\n"
        for mem in self.memory[-5:]:  # Last 5
            context += f"- Iteration {mem['iteration']}: {mem['thoughts']['reasoning']}\n"
            context += f"  Result: {mem['result']}\n"
        
        return context
    
    def execute_command(self, command: Dict) -> str:
        """Execute a command"""
        name = command['name']
        args = command['args']
        
        if name not in self.tools:
            return f"Error: Unknown command {name}"
        
        tool = self.tools[name]
        return tool(**args)
    
    def evaluate_progress(self) -> str:
        """Self-evaluate progress"""
        prompt = f"""Evaluate progress toward goals:

Goals:
{self.goals}

Actions taken:
{self.memory[-3:]}  # Last 3 actions

Progress assessment:"""
        
        return self.llm.generate(prompt)

# Tools
tools = {
    'web_search': lambda query: web_search(query),
    'write_file': lambda filename, content: write_file(filename, content),
    'read_file': lambda filename: read_file(filename),
    'execute_python': lambda code: exec(code),
    'send_email': lambda to, subject, body: send_email(to, subject, body)
}

# Run
agent = AutoGPT(llm=OpenAI(), tools=tools)
result = agent.run("Create a business plan for an AI startup")
```

---

## BabyAGI

### 개념

> **Task 관리 시스템 + GPT**

**특징:**
```
1. Task Queue (우선순위)
2. Execution Agent
3. Task Creation Agent
4. Prioritization Agent
```

### 구조

```
┌─────────────────┐
│   Task Queue    │ (우선순위 큐)
│  1. Research    │
│  2. Analyze     │
│  3. Write       │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Execute │ → Result
    └────┬────┘
         │
    ┌────▼────────┐
    │ Create New  │ → New Tasks
    │   Tasks     │
    └────┬────────┘
         │
    ┌────▼────────┐
    │ Prioritize  │ → Reorder Queue
    └─────────────┘
```

### 구현

```python
from collections import deque
import heapq

class Task:
    def __init__(self, id, name, priority=0):
        self.id = id
        self.name = name
        self.priority = priority
    
    def __lt__(self, other):
        return self.priority < other.priority

class BabyAGI:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.task_queue = []  # Priority queue
        self.results = {}  # Task results
        self.task_id_counter = 0
    
    def run(self, objective: str, initial_task: str):
        """Run BabyAGI"""
        # Add initial task
        self.add_task(initial_task, priority=1)
        
        while self.task_queue:
            # 1. Get highest priority task
            task = heapq.heappop(self.task_queue)
            print(f"\n=== Executing Task {task.id}: {task.name} ===")
            
            # 2. Execute task
            result = self.execute_task(task, objective)
            print(f"Result: {result}")
            
            # Store result
            self.results[task.id] = result
            
            # 3. Create new tasks based on result
            new_tasks = self.create_new_tasks(task, result, objective)
            print(f"New tasks created: {[t.name for t in new_tasks]}")
            
            # Add to queue
            for new_task in new_tasks:
                self.add_task(new_task.name, new_task.priority)
            
            # 4. Prioritize tasks
            self.prioritize_tasks(objective)
    
    def add_task(self, task_name: str, priority: int = 0):
        """Add task to queue"""
        task = Task(
            id=self.task_id_counter,
            name=task_name,
            priority=priority
        )
        heapq.heappush(self.task_queue, task)
        self.task_id_counter += 1
    
    def execute_task(self, task: Task, objective: str) -> str:
        """Execute a single task"""
        context = self.build_context(task, objective)
        
        prompt = f"""Execute this task:

Objective: {objective}
Task: {task.name}

Context:
{context}

Execute the task and provide the result:"""
        
        response = self.llm.generate(prompt)
        return response
    
    def build_context(self, task: Task, objective: str) -> str:
        """Build context from previous results"""
        context = "Previous results:\n"
        for task_id, result in self.results.items():
            context += f"- Task {task_id}: {result[:100]}...\n"
        return context
    
    def create_new_tasks(
        self,
        task: Task,
        result: str,
        objective: str
    ) -> List[Task]:
        """Create new tasks based on result"""
        prompt = f"""Based on the result, create new tasks to achieve the objective.

Objective: {objective}
Completed Task: {task.name}
Result: {result}

Incomplete tasks:
{[t.name for t in self.task_queue]}

Create new tasks (JSON array):
[{{"name": "task name", "priority": 1}}]
"""
        
        response = self.llm.generate(prompt)
        task_dicts = json.loads(response)
        
        new_tasks = []
        for task_dict in task_dicts:
            new_task = Task(
                id=self.task_id_counter,
                name=task_dict['name'],
                priority=task_dict.get('priority', 0)
            )
            self.task_id_counter += 1
            new_tasks.append(new_task)
        
        return new_tasks
    
    def prioritize_tasks(self, objective: str):
        """Re-prioritize task queue"""
        if not self.task_queue:
            return
        
        # Get all tasks
        tasks = []
        while self.task_queue:
            tasks.append(heapq.heappop(self.task_queue))
        
        # Ask LLM to prioritize
        prompt = f"""Prioritize these tasks for the objective.

Objective: {objective}

Tasks:
{[f"{i}. {t.name}" for i, t in enumerate(tasks, 1)]}

Return prioritized order (comma-separated indices, highest priority first):"""
        
        response = self.llm.generate(prompt)
        order = [int(x.strip()) - 1 for x in response.split(',')]
        
        # Reorder and add back
        for i, idx in enumerate(order):
            tasks[idx].priority = len(order) - i
            heapq.heappush(self.task_queue, tasks[idx])

# Run
agent = BabyAGI(llm=OpenAI(), tools=tools)
agent.run(
    objective="Create a marketing strategy",
    initial_task="Research target market"
)
```

---

## LangChain Agents

### Plan-and-Execute

```python
from langchain.agents import PlanAndExecute, load_tools
from langchain.llms import OpenAI
from langchain.chains import LLMChain

class PlanAndExecuteAgent:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        self.tools = load_tools(['serpapi', 'python_repl', 'requests'])
        
        # Planner
        self.planner = LLMChain(
            llm=self.llm,
            prompt=self.create_planner_prompt()
        )
        
        # Executor
        self.executor = LLMChain(
            llm=self.llm,
            prompt=self.create_executor_prompt()
        )
    
    def run(self, objective: str) -> str:
        # 1. Create plan
        plan = self.planner.run(objective=objective)
        steps = self.parse_plan(plan)
        
        print(f"Plan: {steps}")
        
        # 2. Execute steps
        results = []
        for i, step in enumerate(steps):
            print(f"\nStep {i+1}: {step}")
            
            result = self.executor.run(
                step=step,
                previous_results=results,
                tools=self.tools
            )
            
            results.append({
                'step': step,
                'result': result
            })
            
            print(f"Result: {result}")
        
        # 3. Synthesize final answer
        return self.synthesize(objective, results)
```

---

## Multi-Agent Systems

### Hierarchical Agents

```python
class HierarchicalAgentSystem:
    def __init__(self, manager_llm, worker_llm, tools):
        self.manager = ManagerAgent(manager_llm)
        self.workers = {
            'researcher': ResearchAgent(worker_llm, tools),
            'writer': WriterAgent(worker_llm, tools),
            'coder': CoderAgent(worker_llm, tools)
        }
    
    def run(self, task: str) -> str:
        """Manager delegates to workers"""
        # 1. Manager analyzes task
        plan = self.manager.create_plan(task)
        
        # 2. Assign subtasks to workers
        results = {}
        for subtask in plan['subtasks']:
            worker_name = plan['assignments'][subtask]
            worker = self.workers[worker_name]
            
            print(f"Assigning '{subtask}' to {worker_name}")
            
            result = worker.execute(subtask)
            results[subtask] = result
        
        # 3. Manager synthesizes results
        final_result = self.manager.synthesize(task, results)
        
        return final_result

class ManagerAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def create_plan(self, task: str) -> Dict:
        """Break down task and assign to workers"""
        prompt = f"""Break down this task into subtasks and assign to workers.

Task: {task}

Available workers:
- researcher: Web search, data gathering
- writer: Content creation, documentation
- coder: Programming, scripting

Respond in JSON:
{{
  "subtasks": ["subtask1", "subtask2"],
  "assignments": {{"subtask1": "researcher", "subtask2": "writer"}}
}}
"""
        response = self.llm.generate(prompt)
        return json.loads(response)
    
    def synthesize(self, task: str, results: Dict) -> str:
        """Combine worker results"""
        prompt = f"""Synthesize results into final answer.

Original task: {task}

Results:
{json.dumps(results, indent=2)}

Final answer:"""
        
        return self.llm.generate(prompt)
```

### Collaborative Agents

```python
class CollaborativeAgents:
    """Agents that discuss and iterate"""
    def __init__(self, agents: List['Agent']):
        self.agents = agents
    
    def solve(self, problem: str, max_rounds=5) -> str:
        """Collaborative problem solving"""
        discussion = [{'role': 'user', 'content': problem}]
        
        for round in range(max_rounds):
            print(f"\n=== Round {round + 1} ===")
            
            for agent in self.agents:
                # Agent speaks
                response = agent.respond(discussion)
                
                discussion.append({
                    'role': agent.name,
                    'content': response
                })
                
                print(f"{agent.name}: {response}")
                
                # Check if consensus reached
                if self.check_consensus(discussion):
                    return self.extract_solution(discussion)
        
        return self.extract_solution(discussion)
    
    def check_consensus(self, discussion: List[Dict]) -> bool:
        """Check if agents agree"""
        # Simple: check if "agree" appears in recent messages
        recent = discussion[-len(self.agents):]
        return all('agree' in msg['content'].lower() for msg in recent)

# Example: Math problem solving
agents = [
    Agent("Alice", "mathematician"),
    Agent("Bob", "physicist"),
    Agent("Charlie", "engineer")
]

system = CollaborativeAgents(agents)
solution = system.solve("How to optimize a rocket trajectory?")
```

---

## Agent Evaluation

### Metrics

```python
class AgentEvaluator:
    def evaluate(self, agent, test_cases):
        """Evaluate agent performance"""
        results = {
            'success_rate': 0,
            'avg_steps': 0,
            'avg_cost': 0,
            'avg_time': 0
        }
        
        for test_case in test_cases:
            start_time = time.time()
            
            # Run agent
            output = agent.run(test_case['input'])
            
            # Evaluate
            success = self.check_success(
                output,
                test_case['expected']
            )
            
            results['success_rate'] += int(success)
            results['avg_steps'] += agent.step_count
            results['avg_cost'] += agent.total_cost
            results['avg_time'] += time.time() - start_time
        
        # Average
        n = len(test_cases)
        results['success_rate'] /= n
        results['avg_steps'] /= n
        results['avg_cost'] /= n
        results['avg_time'] /= n
        
        return results
    
    def check_success(self, output, expected):
        """Check if output matches expected"""
        # Use LLM to judge
        prompt = f"""Did the agent succeed?

Expected: {expected}
Actual: {output}

Success? (yes/no)"""
        
        response = llm.generate(prompt).lower()
        return 'yes' in response
```

---

## 실전 Tips

### 1. Cost Control

```python
class CostAwareAgent:
    def __init__(self, max_cost=1.0):
        self.max_cost = max_cost
        self.current_cost = 0
    
    def run(self, task):
        while not self.is_complete():
            # Check budget
            if self.current_cost >= self.max_cost:
                print("Budget exceeded!")
                break
            
            # Execute with cost tracking
            action, cost = self.execute_with_cost()
            self.current_cost += cost
```

### 2. Safety

```python
class SafeAgent:
    def __init__(self, dangerous_actions):
        self.dangerous_actions = dangerous_actions
    
    def execute(self, action):
        # Check if dangerous
        if self.is_dangerous(action):
            # Ask for confirmation
            if not self.get_user_approval(action):
                return "Action blocked by safety check"
        
        return self.run_action(action)
    
    def is_dangerous(self, action):
        return any(
            danger in action['name']
            for danger in self.dangerous_actions
        )
```

### 3. Human-in-the-Loop

```python
class HITLAgent:
    """Human-In-The-Loop Agent"""
    def run(self, task):
        plan = self.create_plan(task)
        
        # Show plan to human
        approved = input(f"Execute this plan? {plan}\n(y/n): ")
        
        if approved.lower() != 'y':
            # Revise plan
            feedback = input("Feedback: ")
            plan = self.revise_plan(plan, feedback)
        
        # Execute with checkpoints
        for step in plan:
            result = self.execute(step)
            
            # Critical step - ask human
            if step.is_critical:
                cont = input(f"Continue? Result: {result}\n(y/n): ")
                if cont.lower() != 'y':
                    break
```

---

## 요약

**Advanced Agents:**

1. **AutoGPT**: 자율적 목표 달성
2. **BabyAGI**: Task 관리 시스템
3. **Multi-Agent**: 협업, 계층 구조

**핵심 기능:**
- Long-term memory
- Self-evaluation
- Dynamic planning
- Tool use

**실전 고려사항:**
- Cost control
- Safety checks
- Human oversight

**다음 글:**
- **Multi-Agent Collaboration**: 협업 패턴
- **Agent Memory**: 장기 기억 시스템
- **Agent Safety**: 안전한 Agent 설계

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*
