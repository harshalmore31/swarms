# InteractiveGroupChat Documentation

The InteractiveGroupChat is a sophisticated multi-agent system that enables interactive conversations between users and AI agents using @mentions. This system allows users to direct tasks to specific agents and facilitates collaborative responses when multiple agents are mentioned.

## Features

| Feature | Description |
|---------|-------------|
| **@mentions Support** | Direct tasks to specific agents using @agent_name syntax |
| **Multi-Agent Collaboration** | Multiple mentioned agents can see and respond to each other's tasks |
| **Enhanced Collaborative Prompts** | Agents are trained to acknowledge, build upon, and synthesize each other's responses |
| **Speaker Functions** | Control the order in which agents respond (round robin, random, priority, custom) |
| **Dynamic Speaker Management** | Change speaker functions and priorities during runtime |
| **Random Dynamic Speaker** | Advanced speaker function that follows @mentions in agent responses |
| **Parallel and Sequential Strategies** | Support for both parallel and sequential agent execution |
| **Callable Function Support** | Supports both Agent instances and callable functions as chat participants |
| **Comprehensive Error Handling** | Custom error classes for different scenarios |
| **Conversation History** | Maintains a complete history of the conversation |
| **Flexible Output Formatting** | Configurable output format for conversation history |
| **Interactive Terminal Mode** | Full REPL interface for real-time chat with agents |

## Installation

```bash
pip install swarms
```

## Methods Reference

### Constructor (`__init__`)

**Description:**  
Initializes a new InteractiveGroupChat instance with the specified configuration.

**Arguments:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `id` | str | Unique identifier for the chat | auto-generated key |
| `name` | str | Name of the group chat | "InteractiveGroupChat" |
| `description` | str | Description of the chat's purpose | generic description |
| `agents` | List[Union[Agent, Callable]] | List of participating agents | empty list |
| `max_loops` | int | Maximum conversation turns | 1 |
| `output_type` | str | Type of output format | "string" |
| `interactive` | bool | Whether to enable interactive mode | False |
| `speaker_function` | Union[str, Callable] | Function to determine speaking order | round_robin_speaker |
| `speaker_state` | dict | Initial state for speaker function | {"current_index": 0} |

**Example:**

```python
from swarms import Agent, InteractiveGroupChat

# Create agents
financial_advisor = Agent(
    agent_name="FinancialAdvisor",
    system_prompt="You are a financial advisor specializing in investment strategies.",
    model_name="gpt-4"
)

tax_expert = Agent(
    agent_name="TaxExpert",
    system_prompt="You are a tax expert providing tax-related guidance.",
    model_name="gpt-4"
)

# Initialize group chat with speaker function
from swarms.structs.interactive_groupchat import round_robin_speaker

chat = InteractiveGroupChat(
    id="finance-chat-001",
    name="Financial Advisory Team",
    description="Expert financial guidance team",
    agents=[financial_advisor, tax_expert],
    max_loops=3,
    output_type="string",
    interactive=True,
    speaker_function=round_robin_speaker
)
```

### Run Method (`run`)

**Description:**  
Processes a task and gets responses from mentioned agents. This is the main method for sending tasks in non-interactive mode.

**Arguments:**

- `task` (str): The input task containing @mentions to agents
- `img` (Optional[str]): Optional image for the task
- `imgs` (Optional[List[str]]): Optional list of images for the task

**Returns:**

- str: Formatted conversation history including agent responses

**Example:**
```python
# Single agent interaction
response = chat.run("@FinancialAdvisor what are the best ETFs for 2024?")
print(response)

# Multiple agent collaboration
response = chat.run("@FinancialAdvisor and @TaxExpert, how can I minimize taxes on my investments?")
print(response)

# With image input
response = chat.run("@FinancialAdvisor analyze this chart", img="chart.png")
print(response)
```

### Start Interactive Session (`start_interactive_session`)

**Description:**  
Starts an interactive terminal session for real-time chat with agents. This creates a REPL (Read-Eval-Print Loop) interface.

**Arguments:**
None

**Features:**
- Real-time chat with agents using @mentions
- View available agents and their descriptions
- Change speaker functions during the session
- Built-in help system
- Graceful exit with 'exit' or 'quit' commands

**Example:**

```python
# Initialize chat with interactive mode
chat = InteractiveGroupChat(
    agents=[financial_advisor, tax_expert],
    interactive=True
)

# Start the interactive session
chat.start_interactive_session()
```

**Interactive Session Commands:**
- `@agent_name message` - Mention specific agents
- `help` or `?` - Show help information
- `speaker` - Change speaker function
- `exit` or `quit` - End the session

### Set Speaker Function (`set_speaker_function`)

**Description:**  

Dynamically changes the speaker function and optional state during runtime.

**Arguments:**

- `speaker_function` (Union[str, Callable]): Function that determines speaking order
  - String options: "round-robin-speaker", "random-speaker", "priority-speaker", "random-dynamic-speaker"
  - Callable: Custom function that takes (agents: List[str], **kwargs) -> str
- `speaker_state` (dict, optional): State for the speaker function

**Example:**
```python
from swarms.structs.interactive_groupchat import random_speaker, priority_speaker

# Change to random speaker function
chat.set_speaker_function(random_speaker)

# Change to priority speaker with custom priorities
chat.set_speaker_function(priority_speaker, {"financial_advisor": 3, "tax_expert": 2})

# Change to random dynamic speaker
chat.set_speaker_function("random-dynamic-speaker")
```

### Get Available Speaker Functions (`get_available_speaker_functions`)

**Description:**  

Returns a list of all available built-in speaker function names.

**Arguments:**
None

**Returns:**

- List[str]: List of available speaker function names

**Example:**
```python
available_functions = chat.get_available_speaker_functions()
print(available_functions)
# Output: ['round-robin-speaker', 'random-speaker', 'priority-speaker', 'random-dynamic-speaker']
```

### Get Current Speaker Function (`get_current_speaker_function`)

**Description:**  

Returns the name of the currently active speaker function.

**Arguments:**
None

**Returns:**

- str: Name of the current speaker function, or "custom" if it's a custom function

**Example:**
```python
current_function = chat.get_current_speaker_function()
print(current_function)  # Output: "round-robin-speaker"
```

### Set Priorities (`set_priorities`)

**Description:**  

Sets agent priorities for priority-based speaking order.

**Arguments:**

- `priorities` (dict): Dictionary mapping agent names to priority weights

**Example:**
```python
# Set agent priorities (higher numbers = higher priority)
chat.set_priorities({
    "financial_advisor": 5,
    "tax_expert": 3,
    "investment_analyst": 1
})
```

### Set Dynamic Strategy (`set_dynamic_strategy`)

**Description:**  

Sets the strategy for the random-dynamic-speaker function.

**Arguments:**

- `strategy` (str): Either "sequential" or "parallel"
  - "sequential": Process one agent at a time based on @mentions
  - "parallel": Process all mentioned agents simultaneously

**Example:**
```python
# Set to sequential strategy (one agent at a time)
chat.set_dynamic_strategy("sequential")

# Set to parallel strategy (all mentioned agents respond simultaneously)
chat.set_dynamic_strategy("parallel")
```

### Extract Mentions (`_extract_mentions`)

**Description:**  

Internal method that extracts @mentions from a task. Used by the run method to identify which agents should respond.

**Arguments:**

- `task` (str): The input task to extract mentions from

**Returns:**

- List[str]: List of mentioned agent names

**Example:**
```python
# Internal usage example (not typically called directly)
chat = InteractiveGroupChat(agents=[financial_advisor, tax_expert])
mentions = chat._extract_mentions("@FinancialAdvisor and @TaxExpert, please help")
print(mentions)  # ['FinancialAdvisor', 'TaxExpert']
```

### Validate Initialization (`_validate_initialization`)

**Description:**  

Internal method that validates the group chat configuration during initialization.

**Arguments:**
None

**Example:**

```python
# Internal validation happens automatically during initialization
chat = InteractiveGroupChat(
    agents=[financial_advisor],  # Valid: at least one agent
    max_loops=5  # Valid: positive number
)
```

### Setup Conversation Context (`_setup_conversation_context`)

**Description:**  

Internal method that sets up the initial conversation context with group chat information.

**Arguments:**

None

**Example:**

```python
# Context is automatically set up during initialization
chat = InteractiveGroupChat(
    name="Investment Team",
    description="Expert investment advice",
    agents=[financial_advisor, tax_expert]
)
# The conversation context now includes chat name, description, and agent info
```

### Update Agent Prompts (`_update_agent_prompts`)

**Description:**  

Internal method that updates each agent's system prompt with information about other agents and the group chat. This includes enhanced collaborative instructions that teach agents how to acknowledge, build upon, and synthesize each other's responses.

**Arguments:**

None

**Example:**
```python
# Agent prompts are automatically updated during initialization
chat = InteractiveGroupChat(agents=[financial_advisor, tax_expert])
# Each agent now knows about the other participants and how to collaborate effectively
```

### Get Speaking Order (`_get_speaking_order`)

**Description:**  

Internal method that determines the speaking order using the configured speaker function.

**Arguments:**

- `mentioned_agents` (List[str]): List of agent names that were mentioned

**Returns:**

- List[str]: List of agent names in the order they should speak

**Example:**
```python
# Internal usage (not typically called directly)
mentioned = ["financial_advisor", "tax_expert"]
order = chat._get_speaking_order(mentioned)
print(order)  # Order determined by speaker function
```

## Speaker Functions

InteractiveGroupChat supports various speaker functions that control the order in which agents respond when multiple agents are mentioned.

### Built-in Speaker Functions

#### Round Robin Speaker (`round_robin_speaker`)

Agents speak in a fixed order, cycling through the list in sequence.

```python
from swarms.structs.interactive_groupchat import InteractiveGroupChat, round_robin_speaker

chat = InteractiveGroupChat(
    agents=agents,
    speaker_function=round_robin_speaker,
    interactive=False,
)
```

**Behavior:**

- Agents speak in the order they were mentioned

- Maintains state between calls to continue the cycle

- Predictable and fair distribution of speaking turns

#### Random Speaker (`random_speaker`)

Agents speak in random order each time.

```python
from swarms.structs.interactive_groupchat import InteractiveGroupChat, random_speaker

chat = InteractiveGroupChat(
    agents=agents,
    speaker_function=random_speaker,
    interactive=False,
)
```

**Behavior:**

- Speaking order is randomized for each task

- Provides variety and prevents bias toward first-mentioned agents

- Good for brainstorming sessions

#### Priority Speaker (`priority_speaker`)

Agents speak based on priority weights assigned to each agent.

```python
from swarms.structs.interactive_groupchat import InteractiveGroupChat, priority_speaker

chat = InteractiveGroupChat(
    agents=agents,
    speaker_function=priority_speaker,
    speaker_state={"priorities": {"financial_advisor": 3, "tax_expert": 2, "analyst": 1}},
    interactive=False,
)
```

**Behavior:**

- Higher priority agents speak first

- Uses weighted probability for selection

- Good for hierarchical teams or expert-led discussions

#### Random Dynamic Speaker (`random_dynamic_speaker`)

Advanced speaker function that follows @mentions in agent responses, enabling dynamic conversation flow.

```python
from swarms.structs.interactive_groupchat import InteractiveGroupChat, random_dynamic_speaker

chat = InteractiveGroupChat(
    agents=agents,
    speaker_function=random_dynamic_speaker,
    speaker_state={"strategy": "parallel"},  # or "sequential"
    interactive=False,
)
```

**Behavior:**

- **First Call**: Randomly selects an agent to start the conversation
- **Subsequent Calls**: Extracts @mentions from the previous agent's response and selects the next speaker(s)
- **Two Strategies**:
  - **Sequential**: Processes one agent at a time based on @mentions
  - **Parallel**: Processes all mentioned agents simultaneously

**Example Dynamic Flow:**
```python
# Agent A responds: "I think @AgentB should analyze this data and @AgentC should review the methodology"
# With sequential strategy: Agent B speaks next
# With parallel strategy: Both Agent B and Agent C speak simultaneously
```

**Use Cases:**
- Complex problem-solving where agents need to delegate to specific experts
- Dynamic workflows where the conversation flow depends on agent responses
- Collaborative decision-making processes

### Custom Speaker Functions

You can create your own speaker functions to implement custom logic:

```python
def custom_speaker(agents: List[str], **kwargs) -> str:
    """
    Custom speaker function that selects agents based on specific criteria.
    
    Args:
        agents: List of agent names
        **kwargs: Additional arguments (context, time, etc.)
        
    Returns:
        Selected agent name
    """
    # Your custom logic here
    if "urgent" in kwargs.get("context", ""):
        return "emergency_agent" if "emergency_agent" in agents else agents[0]
    
    # Default to first agent
    return agents[0]

# Use custom speaker function
chat = InteractiveGroupChat(
    agents=agents,
    speaker_function=custom_speaker,
    interactive=False,
)
```

### Dynamic Speaker Function Changes

You can change the speaker function during runtime:

```python
# Start with round robin
chat = InteractiveGroupChat(
    agents=agents,
    speaker_function=round_robin_speaker,
    interactive=False,
)

# Change to random
chat.set_speaker_function(random_speaker)

# Change to priority with custom priorities
chat.set_priorities({"financial_advisor": 5, "tax_expert": 3, "analyst": 1})
chat.set_speaker_function(priority_speaker)

# Change to dynamic speaker with parallel strategy
chat.set_speaker_function("random-dynamic-speaker")
chat.set_dynamic_strategy("parallel")
```

## Enhanced Collaborative Behavior

The InteractiveGroupChat now includes enhanced collaborative prompts that ensure agents work together effectively.

### Collaborative Response Protocol

Every agent receives instructions to:

1. **Read and understand** all previous responses from other agents
2. **Acknowledge** what other agents have said
3. **Build upon** previous insights rather than repeating information
4. **Synthesize** multiple perspectives when possible
5. **Delegate** appropriately using @mentions

### Response Structure

Agents are guided to structure their responses as:

1. **ACKNOWLEDGE**: "I've reviewed the responses from @agent1 and @agent2..."
2. **BUILD**: "Building on @agent1's analysis of the data..."
3. **CONTRIBUTE**: "From my perspective, I would add..."
4. **COLLABORATE**: "To get a complete picture, let me ask @agent3 to..."
5. **SYNTHESIZE**: "Combining our insights, the key findings are..."

### Example Collaborative Response

```python
task = "Analyze our Q3 performance. @analyst @researcher @strategist"

# Expected collaborative behavior:
# Analyst: "Based on the data analysis, I can see clear growth trends in Q3..."
# Researcher: "Building on @analyst's data insights, I can add that market research shows..."
# Strategist: "Synthesizing @analyst's data and @researcher's market insights, I recommend..."
```

## Error Classes

### InteractiveGroupChatError

**Description:**  

Base exception class for InteractiveGroupChat errors.

**Example:**
```python
try:
    # Some operation that might fail
    chat.run("@InvalidAgent hello")
except InteractiveGroupChatError as e:
    print(f"Chat error occurred: {e}")
```

### AgentNotFoundError

**Description:**  

Raised when a mentioned agent is not found in the group.

**Example:**
```python
try:
    chat.run("@NonExistentAgent hello!")
except AgentNotFoundError as e:
    print(f"Agent not found: {e}")
```

### NoMentionedAgentsError

**Description:**  

Raised when no agents are mentioned in the task.

**Example:**

```python
try:
    chat.run("Hello everyone!")  # No @mentions
except NoMentionedAgentsError as e:
    print(f"No agents mentioned: {e}")
```

### InvalidTaskFormatError

**Description:**  

Raised when the task format is invalid.

**Example:**
```python
try:
    chat.run("@Invalid@Format")
except InvalidTaskFormatError as e:
    print(f"Invalid task format: {e}")
```

### InvalidSpeakerFunctionError

**Description:**  

Raised when an invalid speaker function is provided.

**Example:**
```python
def invalid_speaker(agents, **kwargs):
    return 123  # Should return string, not int

try:
    chat = InteractiveGroupChat(
        agents=agents,
        speaker_function=invalid_speaker,
    )
except InvalidSpeakerFunctionError as e:
    print(f"Invalid speaker function: {e}")
```

## Best Practices

| Best Practice | Description | Example |
|--------------|-------------|---------|
| Agent Naming | Use clear, unique names for agents to avoid confusion | `financial_advisor`, `tax_expert` |
| Task Format | Always use @mentions to direct tasks to specific agents | `@financial_advisor What's your investment advice?` |
| Speaker Functions | Choose appropriate speaker functions for your use case | Round robin for fairness, priority for expert-led discussions |
| Dynamic Speaker | Use random-dynamic-speaker for complex workflows with delegation | When agents need to call on specific experts |
| Strategy Selection | Choose sequential for focused discussions, parallel for brainstorming | Sequential for analysis, parallel for idea generation |
| Collaborative Design | Design agents with complementary expertise for better collaboration | Analyst + Researcher + Strategist |
| Error Handling | Implement proper error handling for various scenarios | `try/except` blocks for `AgentNotFoundError` |
| Context Management | Be aware that agents can see the full conversation history | Monitor conversation length and relevance |
| Resource Management | Consider the number of agents and task length to optimize performance | Limit max_loops and task size |
| Dynamic Adaptation | Change speaker functions based on different phases of work | Round robin for brainstorming, priority for decision-making |

## Usage Examples

### Basic Multi-Agent Collaboration

```python
from swarms import Agent
from swarms.structs.interactive_groupchat import InteractiveGroupChat, round_robin_speaker

# Create specialized agents
analyst = Agent(
    agent_name="analyst",
    system_prompt="You are a data analyst specializing in business intelligence.",
    llm="gpt-3.5-turbo",
)

researcher = Agent(
    agent_name="researcher", 
    system_prompt="You are a market researcher with expertise in consumer behavior.",
    llm="gpt-3.5-turbo",
)

strategist = Agent(
    agent_name="strategist",
    system_prompt="You are a strategic consultant who synthesizes insights into actionable recommendations.",
    llm="gpt-3.5-turbo",
)

# Create collaborative group chat
chat = InteractiveGroupChat(
    name="Business Analysis Team",
    description="A collaborative team for comprehensive business analysis",
    agents=[analyst, researcher, strategist],
    speaker_function=round_robin_speaker,
    interactive=False,
)

# Collaborative analysis task
task = """Analyze our company's Q3 performance. We have the following data:
- Revenue: $2.5M (up 15% from Q2)
- Customer acquisition cost: $45 (down 8% from Q2)
- Market share: 3.2% (up 0.5% from Q2)

@analyst @researcher @strategist please provide a comprehensive analysis."""

response = chat.run(task)
print(response)
```

### Priority-Based Expert Consultation

```python
from swarms.structs.interactive_groupchat import InteractiveGroupChat, priority_speaker

# Create expert agents with different priority levels
senior_expert = Agent(
    agent_name="senior_expert",
    system_prompt="You are a senior consultant with 15+ years of experience.",
    llm="gpt-4",
)

junior_expert = Agent(
    agent_name="junior_expert",
    system_prompt="You are a junior consultant with 3 years of experience.",
    llm="gpt-3.5-turbo",
)

assistant = Agent(
    agent_name="assistant",
    system_prompt="You are a research assistant who gathers supporting information.",
    llm="gpt-3.5-turbo",
)

# Create priority-based group chat
chat = InteractiveGroupChat(
    name="Expert Consultation Team",
    description="Expert-led consultation with collaborative input",
    agents=[senior_expert, junior_expert, assistant],
    speaker_function=priority_speaker,
    speaker_state={"priorities": {"senior_expert": 5, "junior_expert": 3, "assistant": 1}},
    interactive=False,
)

# Expert consultation task
task = """We need strategic advice on entering a new market. 
@senior_expert @junior_expert @assistant please provide your insights."""

response = chat.run(task)
print(response)
```

### Dynamic Speaker Function with Delegation

```python
from swarms.structs.interactive_groupchat import InteractiveGroupChat, random_dynamic_speaker

# Create specialized medical agents
cardiologist = Agent(
    agent_name="cardiologist",
    system_prompt="You are a cardiologist specializing in heart conditions.",
    llm="gpt-4",
)

oncologist = Agent(
    agent_name="oncologist",
    system_prompt="You are an oncologist specializing in cancer treatment.",
    llm="gpt-4",
)

endocrinologist = Agent(
    agent_name="endocrinologist",
    system_prompt="You are an endocrinologist specializing in hormone disorders.",
    llm="gpt-4",
)

# Create dynamic group chat
chat = InteractiveGroupChat(
    name="Medical Panel Discussion",
    description="A collaborative panel of medical specialists",
    agents=[cardiologist, oncologist, endocrinologist],
    speaker_function=random_dynamic_speaker,
    speaker_state={"strategy": "sequential"},
    interactive=False,
)

# Complex medical case with dynamic delegation
case = """CASE PRESENTATION:
A 65-year-old male with Type 2 diabetes, hypertension, and recent diagnosis of 
stage 3 colon cancer presents with chest pain and shortness of breath. 
ECG shows ST-segment elevation. Recent blood work shows elevated blood glucose (280 mg/dL) 
and signs of infection (WBC 15,000, CRP elevated).

@cardiologist @oncologist @endocrinologist please provide your assessment and treatment recommendations."""

response = chat.run(case)
print(response)
```

### Dynamic Speaker Function Changes

```python
from swarms.structs.interactive_groupchat import (
    InteractiveGroupChat, 
    round_robin_speaker, 
    random_speaker, 
    priority_speaker,
    random_dynamic_speaker
)

# Create brainstorming agents
creative_agent = Agent(agent_name="creative", system_prompt="You are a creative thinker.")
analytical_agent = Agent(agent_name="analytical", system_prompt="You are an analytical thinker.")
practical_agent = Agent(agent_name="practical", system_prompt="You are a practical implementer.")

chat = InteractiveGroupChat(
    name="Dynamic Team",
    agents=[creative_agent, analytical_agent, practical_agent],
    speaker_function=round_robin_speaker,
    interactive=False,
)

# Phase 1: Brainstorming (random order)
chat.set_speaker_function(random_speaker)
task1 = "Let's brainstorm new product ideas. @creative @analytical @practical"
response1 = chat.run(task1)

# Phase 2: Analysis (priority order)
chat.set_priorities({"analytical": 3, "creative": 2, "practical": 1})
chat.set_speaker_function(priority_speaker)
task2 = "Now let's analyze the feasibility of these ideas. @creative @analytical @practical"
response2 = chat.run(task2)

# Phase 3: Dynamic delegation (agents mention each other)
chat.set_speaker_function(random_dynamic_speaker)
chat.set_dynamic_strategy("sequential")
task3 = "Let's plan implementation with dynamic delegation. @creative @analytical @practical"
response3 = chat.run(task3)

# Phase 4: Final synthesis (round robin for equal input)
chat.set_speaker_function(round_robin_speaker)
task4 = "Finally, let's synthesize our findings. @creative @analytical @practical"
response4 = chat.run(task4)
```

### Custom Speaker Function

```python
def context_aware_speaker(agents: List[str], **kwargs) -> str:
    """Custom speaker function that selects agents based on context."""
    context = kwargs.get("context", "").lower()
    
    if "data" in context or "analysis" in context:
        return "analyst" if "analyst" in agents else agents[0]
    elif "market" in context or "research" in context:
        return "researcher" if "researcher" in agents else agents[0]
    elif "strategy" in context or "planning" in context:
        return "strategist" if "strategist" in agents else agents[0]
    else:
        return agents[0]

# Use custom speaker function
chat = InteractiveGroupChat(
    name="Context-Aware Team",
    agents=[analyst, researcher, strategist],
    speaker_function=context_aware_speaker,
    interactive=False,
)

# The speaker function will automatically select the most appropriate agent
task = "We need to analyze our market position and develop a strategy."
response = chat.run(task)
```

### Interactive Session with Enhanced Collaboration

```python
# Create agents designed for collaboration
data_scientist = Agent(
    agent_name="data_scientist",
    system_prompt="You are a data scientist. When collaborating, always reference specific data points and build upon others' insights with quantitative support.",
    llm="gpt-4",
)

business_analyst = Agent(
    agent_name="business_analyst",
    system_prompt="You are a business analyst. When collaborating, always connect business insights to practical implications and build upon data analysis with business context.",
    llm="gpt-3.5-turbo",
)

product_manager = Agent(
    agent_name="product_manager",
    system_prompt="You are a product manager. When collaborating, always synthesize insights from all team members and provide actionable product recommendations.",
    llm="gpt-3.5-turbo",
)

# Start interactive session
chat = InteractiveGroupChat(
    name="Product Development Team",
    description="A collaborative team for product development decisions",
    agents=[data_scientist, business_analyst, product_manager],
    speaker_function=round_robin_speaker,
    interactive=True,
)

# Start the interactive session
chat.start_interactive_session()
```

## Benefits and Use Cases

### Benefits of Enhanced Collaboration

1. **Reduced Redundancy**: Agents don't repeat what others have already said
2. **Improved Synthesis**: Multiple perspectives are integrated into coherent conclusions
3. **Better Delegation**: Agents naturally delegate to appropriate experts
4. **Enhanced Problem Solving**: Complex problems are addressed systematically
5. **More Natural Interactions**: Agents respond like real team members
6. **Dynamic Workflows**: Conversation flow adapts based on agent responses
7. **Flexible Execution**: Support for both sequential and parallel processing

### Use Cases

| Use Case Category | Specific Use Case | Agent Team Composition | Recommended Speaker Function |
|------------------|-------------------|----------------------|------------------------------|
| **Business Analysis and Strategy** | Data Analysis | Analyst + Researcher + Strategist | Round Robin |
| | Market Research | Multiple experts analyzing different aspects | Random Dynamic |
| | Strategic Planning | Expert-led discussions with collaborative input | Priority |
| **Product Development** | Requirements Gathering | Product Manager + Developer + Designer | Round Robin |
| | Technical Architecture | Senior + Junior developers with different expertise | Priority |
| | User Experience | UX Designer + Product Manager + Developer | Random Dynamic |
| **Research and Development** | Scientific Research | Multiple researchers with different specializations | Random Dynamic |
| | Literature Review | Different experts reviewing various aspects | Round Robin |
| | Experimental Design | Statistician + Domain Expert + Methodologist | Priority |
| **Creative Projects** | Content Creation | Writer + Editor + Designer | Random |
| | Marketing Campaigns | Creative + Analyst + Strategist | Random Dynamic |
| | Design Projects | Designer + Developer + Product Manager | Round Robin |
| **Problem Solving** | Troubleshooting | Technical + Business + User perspective experts | Priority |
| | Crisis Management | Emergency + Communication + Technical teams | Priority |
| | Decision Making | Executive + Analyst + Specialist | Priority |
| **Medical Consultation** | Complex Cases | Multiple specialists (Cardiologist + Oncologist + Endocrinologist) | Random Dynamic |
| | Treatment Planning | Senior + Junior doctors with different expertise | Priority |
| | Research Review | Multiple researchers reviewing different aspects | Round Robin |

### Speaker Function Selection Guide

| Use Case | Recommended Speaker Function | Strategy | Reasoning |
|----------|------------------------------|----------|-----------|
| Team Meetings | Round Robin | N/A | Ensures equal participation |
| Brainstorming | Random | N/A | Prevents bias and encourages creativity |
| Expert Consultation | Priority | N/A | Senior experts speak first |
| Problem Solving | Priority | N/A | Most relevant experts prioritize |
| Creative Sessions | Random | N/A | Encourages diverse perspectives |
| Decision Making | Priority | N/A | Decision makers speak first |
| Research Review | Round Robin | N/A | Equal contribution from all reviewers |
| Complex Workflows | Random Dynamic | Sequential | Follows natural conversation flow |
| Parallel Analysis | Random Dynamic | Parallel | Multiple agents work simultaneously |
| Medical Panels | Random Dynamic | Sequential | Specialists delegate to relevant experts |
| Technical Architecture | Random Dynamic | Sequential | Senior architects guide the discussion |

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our GitHub repository.

## License

This project is licensed under the Apache License - see the LICENSE file for details. 