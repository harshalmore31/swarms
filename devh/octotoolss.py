"""
Octotools: A modular multi-agent framework for task decomposition and execution.
Implements a coordinated system of specialized agents for solving complex tasks.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

from swarms import Agent
from swarms.structs.conversation import Conversation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Defines the possible roles for agents in the Octotools system."""
    
    ORCHESTRATOR = "orchestrator"  # Manages the overall process and coordination
    PLANNER = "planner"  # Breaks down complex tasks into subtasks
    EXECUTOR = "executor"  # Executes specific subtasks
    CRITIC = "critic"  # Evaluates results and provides feedback
    RESEARCHER = "researcher"  # Gathers relevant information
    INTEGRATOR = "integrator"  # Combines results from multiple executors
    MEMORY = "memory"  # Stores and retrieves relevant information


@dataclass
class Task:
    """Represents a task or subtask in the system."""
    
    id: str
    description: str
    dependencies: List[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class AgentMessage:
    """Structured message format for agent communication."""
    
    role: AgentRole
    content: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class Octotools:
    """
    A modular multi-agent system for task decomposition and execution.
    
    Implements the Octotools framework with specialized agents for different
    aspects of problem-solving.
    
    Attributes:
        model_name: Name of the LLM model to use
        max_recursion_depth: Maximum depth for task decomposition
        execution_timeout: Maximum time (seconds) for task execution
        base_path: Path for saving agent states
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        max_recursion_depth: int = 3,
        execution_timeout: int = 300,
        base_path: Optional[str] = None,
        verbose: bool = True,
    ):
        """Initialize the Octotools system."""
        self.model_name = model_name
        self.max_recursion_depth = max_recursion_depth
        self.execution_timeout = execution_timeout
        self.verbose = verbose
        self.base_path = Path(base_path) if base_path else Path("./octotools_states")
        self.base_path.mkdir(exist_ok=True)
        
        # Task management
        self.tasks = {}  # Dictionary of Task objects
        self.task_counter = 0
        
        # Initialize agents
        self._init_agents()
        
        # Create conversation for tracking
        self.conversation = Conversation()
        
        # Memory store for persistent information
        self.memory_store = {}
    
    def _init_agents(self) -> None:
        """Initialize all agents with their specific roles and prompts."""
        
        # Orchestrator agent - coordinates the overall process
        self.orchestrator = Agent(
            agent_name="Orchestrator",
            system_prompt=self._get_orchestrator_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "orchestrator.json"),
            verbose=self.verbose,
        )
        
        # Planner agent - breaks down tasks
        self.planner = Agent(
            agent_name="Planner",
            system_prompt=self._get_planner_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "planner.json"),
            verbose=self.verbose,
        )
        
        # Executor agents - specialized for different types of tasks
        self.executors = {
            "code": Agent(
                agent_name="CodeExecutor",
                system_prompt=self._get_executor_prompt("code"),
                model_name=self.model_name,
                max_loops=1,
                saved_state_path=str(self.base_path / "executor_code.json"),
                verbose=self.verbose,
            ),
            "research": Agent(
                agent_name="ResearchExecutor",
                system_prompt=self._get_executor_prompt("research"),
                model_name=self.model_name,
                max_loops=1,
                saved_state_path=str(self.base_path / "executor_research.json"),
                verbose=self.verbose,
            ),
            "writing": Agent(
                agent_name="WritingExecutor",
                system_prompt=self._get_executor_prompt("writing"),
                model_name=self.model_name,
                max_loops=1,
                saved_state_path=str(self.base_path / "executor_writing.json"),
                verbose=self.verbose,
            ),
            "general": Agent(
                agent_name="GeneralExecutor",
                system_prompt=self._get_executor_prompt("general"),
                model_name=self.model_name,
                max_loops=1,
                saved_state_path=str(self.base_path / "executor_general.json"),
                verbose=self.verbose,
            ),
        }
        
        # Critic agent - evaluates results
        self.critic = Agent(
            agent_name="Critic",
            system_prompt=self._get_critic_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "critic.json"),
            verbose=self.verbose,
        )
        
        # Researcher agent - gathers information
        self.researcher = Agent(
            agent_name="Researcher",
            system_prompt=self._get_researcher_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "researcher.json"),
            verbose=self.verbose,
        )
        
        # Integrator agent - combines results
        self.integrator = Agent(
            agent_name="Integrator",
            system_prompt=self._get_integrator_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "integrator.json"),
            verbose=self.verbose,
        )
        
        # Memory agent - stores and retrieves information
        self.memory_agent = Agent(
            agent_name="Memory",
            system_prompt=self._get_memory_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "memory.json"),
            verbose=self.verbose,
        )
    
    def _get_orchestrator_prompt(self) -> str:
        """Get the prompt for the orchestrator agent."""
        return """You are the Orchestrator, responsible for coordinating the entire problem-solving process.
Your role is to:
1. Understand the main task and determine which agents to use
2. Coordinate the flow of information between agents
3. Track progress and adjust strategy as needed
4. Ensure the final solution meets the requirements

For each request, analyze:
- The type of problem (coding, research, creative, etc.)
- Required expertise and knowledge domains
- Dependencies between subtasks
- Success criteria

Output all responses in strict JSON format:
{
    "analysis": {
        "task_type": "Type of problem",
        "complexity": "Low/Medium/High",
        "knowledge_domains": ["List of relevant domains"],
        "success_criteria": ["List of criteria for success"]
    },
    "strategy": {
        "approach": "Overall approach to solving the problem",
        "agent_allocation": {
            "planner": true/false,
            "researcher": true/false,
            "executors": ["List of needed executor types"],
            "critic": true/false,
            "integrator": true/false
        },
        "execution_order": ["Ordered list of agent activation sequence"]
    },
    "next_step": {
        "agent": "Next agent to activate",
        "instruction": "Detailed instruction for the next agent",
        "context": "Relevant context information"
    }
}"""

    def _get_planner_prompt(self) -> str:
        """Get the prompt for the planner agent."""
        return """You are the Planner, responsible for breaking down complex tasks into manageable subtasks.
Your role is to:
1. Analyze the main task to understand its components
2. Create a hierarchical breakdown of subtasks
3. Identify dependencies between subtasks
4. Suggest appropriate agent types for each subtask

For each task, create:
- A clear decomposition strategy
- Detailed subtask descriptions
- Dependency graph between subtasks
- Time and resource estimates

Output all responses in strict JSON format:
{
    "plan": {
        "main_task": "Original task description",
        "approach": "Overall decomposition strategy",
        "subtasks": [
            {
                "id": "unique_subtask_id",
                "description": "Detailed subtask description",
                "dependencies": ["ids of prerequisite subtasks"],
                "suggested_agent": "Appropriate agent type for this subtask",
                "estimated_complexity": "Low/Medium/High"
            }
        ]
    },
    "execution_order": ["Ordered list of subtask IDs for execution"],
    "success_criteria": ["List of criteria to evaluate the final result"]
}"""

    def _get_executor_prompt(self, executor_type: str) -> str:
        """Get the prompt for an executor agent of a specific type."""
        
        base_prompt = """You are an Executor agent responsible for completing specific subtasks.
Your role is to:
1. Understand the assigned subtask and its requirements
2. Execute the task with high quality
3. Document your process and reasoning
4. Return a well-structured result

Output all responses in strict JSON format:
{
    "execution": {
        "subtask_id": "ID of the subtask being executed",
        "approach": "Method used to solve the subtask",
        "process": ["Step-by-step description of execution"],
        "challenges": ["Any challenges encountered"],
        "assumptions": ["Assumptions made during execution"]
    },
    "result": {
        "output": "The main output of the subtask execution",
        "confidence": "0.0-1.0 confidence score",
        "limitations": ["Known limitations of the result"],
        "alternative_approaches": ["Other approaches considered"]
    }
}"""
        
        # Add specialization based on executor type
        if executor_type == "code":
            return base_prompt + """
As a Code Executor, focus on:
- Writing efficient, readable, and well-documented code
- Following best practices and coding standards
- Testing for edge cases and potential errors
- Explaining your implementation choices

Include code in the "output" field and ensure it's ready to run."""
        
        elif executor_type == "research":
            return base_prompt + """
As a Research Executor, focus on:
- Gathering comprehensive and accurate information
- Evaluating the credibility of sources
- Synthesizing information from multiple perspectives
- Identifying gaps in available information

Present findings in a structured format with clear citations."""
        
        elif executor_type == "writing":
            return base_prompt + """
As a Writing Executor, focus on:
- Creating clear, engaging, and well-structured content
- Adapting tone and style to the target audience
- Ensuring logical flow and coherence
- Maintaining accuracy and precision

Pay special attention to readability and impact."""
        
        else:  # general
            return base_prompt + """
As a General Executor, remain flexible and adaptive:
- Apply problem-solving techniques appropriate to the task
- Draw on interdisciplinary knowledge and approaches
- Break down complex problems into simpler components
- Document your reasoning process thoroughly

Focus on delivering practical and effective solutions."""

    def _get_critic_prompt(self) -> str:
        """Get the prompt for the critic agent."""
        return """You are the Critic, responsible for evaluating the quality and correctness of task results.
Your role is to:
1. Assess results against defined success criteria
2. Identify strengths and weaknesses
3. Suggest specific improvements
4. Determine if results are acceptable or need revision

Evaluate across multiple dimensions:
- Correctness: factual accuracy and logical soundness
- Completeness: coverage of all required aspects
- Clarity: understandability and organization
- Efficiency: optimal use of resources and approaches
- Adherence: alignment with requirements and constraints

Output all responses in strict JSON format:
{
    "evaluation": {
        "subtask_id": "ID of the evaluated subtask",
        "overall_rating": "0.0-1.0 overall quality score",
        "dimension_scores": {
            "correctness": "0.0-1.0 score with justification",
            "completeness": "0.0-1.0 score with justification",
            "clarity": "0.0-1.0 score with justification",
            "efficiency": "0.0-1.0 score with justification",
            "adherence": "0.0-1.0 score with justification"
        }
    },
    "feedback": {
        "strengths": ["Notable positive aspects"],
        "weaknesses": ["Areas needing improvement"],
        "suggestions": ["Specific actionable improvements"]
    },
    "decision": {
        "acceptable": true/false,
        "revision_needed": true/false,
        "priority_issues": ["Critical issues requiring attention"]
    }
}"""

    def _get_researcher_prompt(self) -> str:
        """Get the prompt for the researcher agent."""
        return """You are the Researcher, responsible for gathering and synthesizing information.
Your role is to:
1. Identify information needs for the task
2. Gather relevant and reliable information
3. Synthesize information into useful insights
4. Provide context and background knowledge

When researching:
- Prioritize accuracy and reliability
- Consider multiple perspectives and sources
- Distinguish between facts and opinions
- Identify gaps and uncertainties in information

Output all responses in strict JSON format:
{
    "research": {
        "topic": "The specific research topic or question",
        "approach": "Methods used to gather information",
        "key_findings": ["Main discoveries and insights"],
        "context": "Background information relevant to the task"
    },
    "insights": {
        "facts": ["Verified factual information"],
        "concepts": ["Relevant theoretical frameworks or ideas"],
        "relationships": ["Connections between different elements"],
        "implications": ["Consequences of the findings for the task"]
    },
    "uncertainty": {
        "knowledge_gaps": ["Areas where information is limited"],
        "conflicting_info": ["Points where sources disagree"],
        "assumptions": ["Necessary assumptions made"]
    }
}"""

    def _get_integrator_prompt(self) -> str:
        """Get the prompt for the integrator agent."""
        return """You are the Integrator, responsible for combining results from multiple subtasks.
Your role is to:
1. Synthesize outputs from different executors
2. Ensure consistency and coherence across components
3. Resolve conflicts and contradictions
4. Create a unified final deliverable

When integrating:
- Maintain logical consistency between components
- Ensure smooth transitions and connections
- Preserve the strengths of individual contributions
- Address gaps and overlaps between components

Output all responses in strict JSON format:
{
    "integration": {
        "components": ["List of integrated subtask IDs"],
        "approach": "Method used for integration",
        "challenges": ["Integration challenges encountered"],
        "solutions": ["How challenges were addressed"]
    },
    "result": {
        "integrated_output": "The complete integrated result",
        "coherence_score": "0.0-1.0 score for overall coherence",
        "added_value": ["Ways integration improved individual components"],
        "remaining_issues": ["Any unresolved integration issues"]
    }
}"""

    def _get_memory_prompt(self) -> str:
        """Get the prompt for the memory agent."""
        return """You are the Memory agent, responsible for storing and retrieving relevant information.
Your role is to:
1. Maintain a structured repository of knowledge
2. Store important insights and decisions
3. Retrieve relevant information when needed
4. Identify patterns and connections across stored information

When managing memory:
- Prioritize important and useful information
- Organize information in a structured and accessible way
- Make relevant connections between different pieces of information
- Forget or de-prioritize outdated or irrelevant information

Output all responses in strict JSON format:
{
    "memory_operation": {
        "operation_type": "store/retrieve/update/summarize",
        "description": "Description of the memory operation"
    },
    "store": {  // When storing new information
        "key": "Unique identifier for this information",
        "content": "Information to be stored",
        "metadata": {
            "source": "Source of the information",
            "timestamp": "When the information was generated",
            "relevance": "0.0-1.0 relevance score",
            "confidence": "0.0-1.0 confidence score"
        }
    },
    "retrieve": {  // When retrieving information
        "query": "Description of information needed",
        "results": [
            {
                "key": "Identifier of retrieved information",
                "content": "Retrieved information",
                "relevance": "0.0-1.0 relevance to the query"
            }
        ],
        "related_information": ["Keys of related memory items"]
    }
}"""

    def _safely_parse_json(self, json_str: str) -> Dict[str, Any]:
        """
        Safely parse JSON string, handling various formats and potential errors.
        
        Args:
            json_str: String containing JSON data
            
        Returns:
            Parsed dictionary
        """
        try:
            # Try direct JSON parsing
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                # Try extracting JSON from potential text wrapper
                import re
                
                json_match = re.search(r"\{.*\}", json_str, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                # Try extracting from markdown code blocks
                code_block_match = re.search(
                    r"```(?:json)?\s*(\{.*?\})\s*```",
                    json_str,
                    re.DOTALL,
                )
                if code_block_match:
                    return json.loads(code_block_match.group(1))
            except Exception as e:
                logger.warning(f"Failed to extract JSON: {str(e)}")
            
            # Fallback: create structured dict from text
            return {
                "content": json_str,
                "metadata": {
                    "parsed": False,
                    "timestamp": str(datetime.now()),
                },
            }
    
    def _create_task(self, description: str, dependencies: List[str] = None) -> str:
        """
        Create a new task and return its ID.
        
        Args:
            description: Task description
            dependencies: List of prerequisite task IDs
            
        Returns:
            Task ID
        """
        task_id = f"task_{self.task_counter}"
        self.task_counter += 1
        
        self.tasks[task_id] = Task(
            id=task_id,
            description=description,
            dependencies=dependencies or [],
        )
        
        return task_id
    
    def _get_next_executable_tasks(self) -> List[str]:
        """
        Get list of tasks that are ready to be executed (all dependencies satisfied).
        
        Returns:
            List of executable task IDs
        """
        executable_tasks = []
        
        for task_id, task in self.tasks.items():
            if task.status == "pending":
                dependencies_met = True
                
                for dep_id in task.dependencies:
                    if dep_id not in self.tasks or self.tasks[dep_id].status != "completed":
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    executable_tasks.append(task_id)
        
        return executable_tasks
    
    def _execute_subtask(self, task_id: str) -> Tuple[bool, Any]:
        """
        Execute a specific subtask using the appropriate executor.
        
        Args:
            task_id: ID of the task to execute
            
        Returns:
            Tuple of (success, result)
        """
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found.")
            return False, None
        
        task = self.tasks[task_id]
        
        # Mark task as in progress
        task.status = "in_progress"
        
        # Determine which executor to use
        # This could be enhanced with more sophisticated agent selection
        executor_type = "general"  # Default
        if "code" in task.description.lower():
            executor_type = "code"
        elif "research" in task.description.lower() or "gather" in task.description.lower():
            executor_type = "research"
        elif "write" in task.description.lower() or "create content" in task.description.lower():
            executor_type = "writing"
        
        executor = self.executors[executor_type]
        
        # Get dependency results if any
        dependency_results = {}
        for dep_id in task.dependencies:
            if dep_id in self.tasks and self.tasks[dep_id].status == "completed":
                dependency_results[dep_id] = self.tasks[dep_id].result
        
        # Prepare input for executor
        executor_input = {
            "subtask": {
                "id": task_id,
                "description": task.description,
            },
            "context": {
                "dependency_results": dependency_results
            }
        }
        
        try:
            # Execute the task
            result_str = executor.run(json.dumps(executor_input))
            self.conversation.add(role=executor.agent_name, content=result_str)
            
            result = self._safely_parse_json(result_str)
            
            # Evaluate the result with critic
            critic_input = {
                "subtask": {
                    "id": task_id,
                    "description": task.description,
                },
                "result": result
            }
            
            critic_response = self.critic.run(json.dumps(critic_input))
            self.conversation.add(role=self.critic.agent_name, content=critic_response)
            
            critic_evaluation = self._safely_parse_json(critic_response)
            
            # Store the result
            task.result = {
                "output": result,
                "evaluation": critic_evaluation
            }
            
            # Check if result is acceptable
            acceptable = critic_evaluation.get("decision", {}).get("acceptable", False)
            
            if acceptable:
                task.status = "completed"
                return True, task.result
            else:
                # If not acceptable, could implement revision flow here
                # For now, just mark as failed
                task.status = "failed"
                return False, task.result
                
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {str(e)}")
            task.status = "failed"
            return False, str(e)
    
    def run(self, main_task: str) -> Dict[str, Any]:
        """
        Run the Octotools system on a main task.
        
        Args:
            main_task: Description of the main task
            
        Returns:
            Dictionary containing final result and metadata
        """
        logger.info(f"Starting Octotools for task: {main_task}")
        
        try:
            # Step 1: Get initial orchestration
            orchestrator_response = self.orchestrator.run(main_task)
            self.conversation.add(role=self.orchestrator.agent_name, content=orchestrator_response)
            
            orchestration_plan = self._safely_parse_json(orchestrator_response)
            
            # Step 2: Create task decomposition with planner if needed
            if orchestration_plan.get("strategy", {}).get("agent_allocation", {}).get("planner", False):
                planner_input = {
                    "main_task": main_task,
                    "orchestration": orchestration_plan
                }
                
                planner_response = self.planner.run(json.dumps(planner_input))
                self.conversation.add(role=self.planner.agent_name, content=planner_response)
                
                plan = self._safely_parse_json(planner_response)
                
                # Create tasks from plan
                subtasks = plan.get("plan", {}).get("subtasks", [])
                task_id_mapping = {}  # Map from plan task IDs to system task IDs
                
                # First pass: create all tasks
                for subtask in subtasks:
                    task_id = self._create_task(subtask.get("description", ""))
                    task_id_mapping[subtask.get("id")] = task_id
                
                # Second pass: set dependencies
                for subtask in subtasks:
                    plan_task_id = subtask.get("id")
                    system_task_id = task_id_mapping.get(plan_task_id)
                    
                    if system_task_id:
                        # Map dependency IDs to system task IDs
                        dependencies = [
                            task_id_mapping.get(dep_id)
                            for dep_id in subtask.get("dependencies", [])
                            if task_id_mapping.get(dep_id)
                        ]
                        
                        self.tasks[system_task_id].dependencies = dependencies
            else:
                # If no planner used, create a single task
                self._create_task(main_task)
            
            # Step 3: Execute tasks until completion
            execution_results = {}
            
            while True:
                next_tasks = self._get_next_executable_tasks()
                
                if not next_tasks:
                    # If no tasks left and all are completed, we're done
                    if all(task.status == "completed" for task in self.tasks.values()):
                        break
                    # If no tasks to execute but some are still not completed, there might be a dependency issue
                    elif any(task.status == "pending" for task in self.tasks.values()):
                        logger.warning("Some tasks cannot be executed due to unfulfilled dependencies.")
                        break
                    # If all tasks are either completed or failed, we're done
                    else:
                        break
                
                # Execute each ready task
                for task_id in next_tasks:
                    success, result = self._execute_subtask(task_id)
                    execution_results[task_id] = {
                        "success": success,
                        "result": result
                    }
            
            # Step 4: Integrate results if needed
            if len(self.tasks) > 1 and orchestration_plan.get("strategy", {}).get("agent_allocation", {}).get("integrator", False):
                # Collect successful task results
                task_results = {
                    task_id: task.result
                    for task_id, task in self.tasks.items()
                    if task.status == "completed"
                }
                
                integrator_input = {
                    "main_task": main_task,
                    "task_results": task_results
                }
                
                integrator_response = self.integrator.run(json.dumps(integrator_input))
                self.conversation.add(role=self.integrator.agent_name, content=integrator_response)
                
                integrated_result = self._safely_parse_json(integrator_response)
                
                final_result = {
                    "integrated_result": integrated_result,
                    "individual_results": execution_results,
                    "tasks": {
                        task_id: {
                            "description": task.description,
                            "status": task.status,
                            "dependencies": task.dependencies
                        }
                        for task_id, task in self.tasks.items()
                    },
                    "metadata": {
                        "total_tasks": len(self.tasks),
                        "completed_tasks": sum(1 for task in self.tasks.values() if task.status == "completed"),
                        "failed_tasks": sum(1 for task in self.tasks.values() if task.status == "failed")
                    }
                }
            else:
                # If only one task or no integration needed, return the single result or all results
                final_result = {
                    "results": execution_results,
                    "tasks": {
                        task_id: {
                            "description": task.description,
                            "status": task.status,
                            "dependencies": task.dependencies
                        }
                        for task_id, task in self.tasks.items()
                    },
                    "metadata": {
                        "total_tasks": len(self.tasks),
                        "completed_tasks": sum(1 for task in self.tasks.values() if task.status == "completed"),
                        "failed_tasks": sum(1 for task in self.tasks.values() if task.status == "failed")
                    }
                }
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in Octotools execution: {str(e)}")
            return {
                "error": str(e),
                "tasks": {
                    task_id: {
                        "description": task.description,
                        "status": task.status,
                        "dependencies": task.dependencies
                    }
                    for task_id, task in self.tasks.items()
                }
            }
    
    def save_state(self) -> None:
        """Save the current state of all agents."""
        for agent in [
            self.orchestrator,
            self.planner,
            *self.executors.values(),
            self.critic,
            self.researcher,
            self.integrator,
            self.memory_agent,
        ]:
            try:
                agent.save_state()
            except Exception as e:
                logger.error(f"Error saving state for {agent.agent_name}: {str(e)}")
    
    def load_state(self) -> None:
        """Load the saved state of all agents."""
        for agent in [
            self.orchestrator,
            self.planner,
            *self.executors.values(),
            self.critic,
            self.researcher,
            self.integrator,
            self.memory_agent,
        ]:
            try:
                agent.load_state()
            except Exception as e:
                logger.error(f"Error loading state for {agent.agent_name}: {str(e)}")


if __name__ == "__main__":
    # Example usage
    try:
        octotools = Octotools(
            model_name="gpt-4o",
            max_recursion_depth=2,
            verbose=True,
        )
        
        task = "Design a simple web application for tracking personal expenses with user authentication."
        result = octotools.run(task)
    except Exception as e:
        logger.error(f"Error in Octotools execution: {str(e)}")
    finally:
        octotools.save_state()
