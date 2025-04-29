"""
OctoToolsSwarm: A multi-agent system for complex reasoning with extensible tools.
Implements the OctoTools framework using the swarms library and OpenAI API.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from swarms import Agent
from swarms.structs.conversation import Conversation

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get OpenAI API Key (replace with your key)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")


class ToolType(Enum):
    """Defines the types of tools available."""

    IMAGE_CAPTIONER = "image_captioner"
    OBJECT_DETECTOR = "object_detector"
    WEB_SEARCH = "web_search"
    PYTHON_CALCULATOR = "python_calculator"
    # Add more tool types as needed


@dataclass
class Tool:
    """
    Represents an external tool.

    Attributes:
        name: Unique name of the tool.
        description: Description of the tool's function.
        metadata: Dictionary containing tool metadata (input types, output type, etc.).
        execute_func: Callable function that executes the tool's logic.
    """

    name: str
    description: str
    metadata: Dict[str, Any]
    execute_func: Callable

    def execute(self, **kwargs):
        """Executes the tool's logic."""
        try:
            return self.execute_func(**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {str(e)}")
            return {"error": str(e)}


class AgentRole(Enum):
    """Defines the roles for agents in the OctoTools system."""

    PLANNER = "planner"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    SUMMARIZER = "summarizer"


class OctoToolsSwarm:
    """
    A multi-agent system implementing the OctoTools framework.

    Attributes:
        model_name: Name of the LLM model to use.
        max_iterations: Maximum number of action-execution iterations.
        base_path: Path for saving agent states.
        tools: List of available Tool objects.
    """

    def __init__(
        self,
        tools: List[Tool],
        model_name: str = "gpt-4",
        max_iterations: int = 10,
        base_path: Optional[str] = None,
    ):
        """Initialize the OctoToolsSwarm system."""
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.base_path = Path(base_path) if base_path else Path("./octotools_states")
        self.base_path.mkdir(exist_ok=True)
        self.tools = {tool.name: tool for tool in tools}  # Store tools in a dictionary

        # Initialize agents
        self._init_agents()

        # Create conversation tracker
        self.conversation = Conversation()
        self.memory = []  # Store the trajectory

    def _init_agents(self) -> None:
        """Initialize all agents with their specific roles and prompts."""
        # Planner agent
        self.planner = Agent(
            agent_name="OctoTools-Planner",
            system_prompt=self._get_planner_prompt(),
            model_name=self.model_name,
            max_loops=3,  # Increased max_loops for the planner
            saved_state_path=str(self.base_path / "planner.json"),
            verbose=True,
            openai_api_key=OPENAI_API_KEY,  # Pass API key
        )

        # Executor agent -- REMOVED, replaced by _execute_tool

        # Verifier agent
        self.verifier = Agent(
            agent_name="OctoTools-Verifier",
            system_prompt=self._get_verifier_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "verifier.json"),
            verbose=True,
            openai_api_key=OPENAI_API_KEY,  # Pass API key
        )

        # Summarizer agent
        self.summarizer = Agent(
            agent_name="OctoTools-Summarizer",
            system_prompt=self._get_summarizer_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "summarizer.json"),
            verbose=True,
            openai_api_key=OPENAI_API_KEY,  # Pass API key
        )

    def _get_planner_prompt(self) -> str:
        """Get the prompt for the planner agent (Improved with few-shot examples)."""
        return f"""You are the Planner in the OctoTools framework. Your role is to analyze the user's query,
        identify required skills, suggest relevant tools, and plan the steps to solve the problem.

        1. **Analyze the user's query:** Understand the requirements and identify the necessary skills and potentially relevant tools.
        2. **Perform high-level planning:**  Create a rough outline of how tools might be used to solve the problem.
        3. **Perform low-level planning (action prediction):**  At each step, select the best tool to use and formulate a specific sub-goal for that tool, considering the current context.

        Available Tools: {', '.join(self.tools.keys())}

        Output your response in JSON format.  Here are examples for different stages:

        **Query Analysis (High-Level Planning):**
        Example Input:
        Query: "What is the capital of France?"

        Example Output:
        ```json
        {{
            "summary": "The user is asking for the capital of France.",
            "required_skills": ["knowledge retrieval"],
            "relevant_tools": ["Web_Search_Tool"]
        }}
        ```

        **Action Prediction (Low-Level Planning):**
        Example Input:
        Context: {{ "query": "What is the capital of France?", "available_tools": ["Web_Search_Tool"] }}

        Example Output:
        ```json
        {{
            "justification": "The Web_Search_Tool can be used to directly find the capital of France.",
            "context": {{}},
            "sub_goal": "Search the web for 'capital of France'.",
            "tool_name": "Web_Search_Tool"
        }}
        ```
        Another Example:
        Context: {{"query": "How many objects are in the image?", "available_tools": ["Image_Captioner_Tool", "Object_Detector_Tool"], "image": "objects.png"}}
        
        Example Output:
        ```json
        {{
            "justification": "First, get a general description of the image to understand the context.",
            "context": {{ "image": "objects.png" }},
            "sub_goal": "Generate a description of the image.",
            "tool_name": "Image_Captioner_Tool"
        }}
        ```
        """

    def _get_verifier_prompt(self) -> str:
        """Get the prompt for the verifier agent (Improved with few-shot examples)."""
        return """You are the Context Verifier in the OctoTools framework. Your role is to analyze the current context
        and memory to determine if the problem is solved, if there are any inconsistencies, or if further steps are needed.

        Output your response in JSON format:

        Example Input:
        Context: { "last_result": { "result": "Caption: The image shows a cat." } }
        Memory: [ { "component": "Action Predictor", "result": { "tool_name": "Image_Captioner_Tool" } } ]

        Example Output:
        ```json
        {
            "completeness": "partial",
            "inconsistencies": [],
            "verification_needs": ["Object detection to confirm the presence of a cat."],
            "ambiguities": [],
            "stop_signal": false
        }
        ```

        Another Example:
        Context: { "last_result": { "result": ["Detected object: cat"] } }
        Memory:  [ { "component": "Action Predictor", "result": { "tool_name": "Object_Detector_Tool" } } ]
        
        Example Output:
        ```json
        {
            "completeness": "yes",
            "inconsistencies": [],
            "verification_needs": [],
            "ambiguities": [],
            "stop_signal": true
        }
        ```
        """

    def _get_summarizer_prompt(self) -> str:
        """Get the prompt for the summarizer agent (Improved with few-shot examples)."""
        return """You are the Solution Summarizer in the OctoTools framework.  Your role is to synthesize the final
        answer to the user's query based on the complete trajectory of actions and results.

        Output your response in JSON format:
        Example Input:
        Memory: [
            {"component": "Query Analyzer", "result": {"summary": "Find the capital of France."}},
            {"component": "Action Predictor", "result": {"tool_name": "Web_Search_Tool"}},
            {"component": "Tool Execution", "result": {"result": "The capital of France is Paris."}}
        ]

        Example Output:
        ```json
        {
            "final_answer": "The capital of France is Paris."
        }
        ```
        """

    def _safely_parse_json(self, json_str: str) -> Dict[str, Any]:
        """Safely parse JSON string, handling various formats."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                # Extract JSON from potential text wrapper
                json_match = re.search(r"\{.*\}", json_str, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except Exception as e:
                logger.warning(f"Failed to extract JSON: {str(e)}")

            # Fallback: create basic dict from text
            return {"content": json_str, "error": "Failed to parse JSON"}

    def _execute_tool(self, tool_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a tool based on its name and provided context.
        This method replaces the direct `exec()` call with a safer approach.
        """
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found."}

        tool = self.tools[tool_name]
        try:
            # Filter context to only include keys expected by the tool
            valid_inputs = {
                k: v for k, v in context.items() if k in tool.metadata["input_types"]
            }
            result = tool.execute(**valid_inputs)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    def run(self, query: str, image: Optional[str] = None) -> Dict[str, Any]:
        """Execute the task through the multi-agent workflow."""
        logger.info(f"Starting task: {query}")

        try:
            # Step 1: Query Analysis (High-Level Planning)
            query_analysis_response = self.planner.run(
                f"Analyze the following query and determine the necessary skills and"
                f" relevant tools: {query}"
            )
            self.conversation.add(
                role=self.planner.agent_name, content=query_analysis_response
            )
            query_analysis = self._safely_parse_json(query_analysis_response)
            self.memory.append(
                {"step": 0, "component": "Query Analyzer", "result": query_analysis}
            )

            context = {"image": image, "query": query}
            if "relevant_tools" in query_analysis:
                context["available_tools"] = query_analysis["relevant_tools"]

            step_count = 1

            # Step 2: Iterative Action-Execution Loop
            while step_count <= self.max_iterations:
                # Step 2a: Action Prediction (Low-Level Planning)
                action_response = self.planner.run(
                    f"Current Context: {json.dumps(context)}\n"
                    f"Available Tools:"
                    f" {', '.join(context.get('available_tools', []))}\nPlan the"
                    " next step."
                )
                self.conversation.add(
                    role=self.planner.agent_name, content=action_response
                )
                action = self._safely_parse_json(action_response)
                self.memory.append(
                    {"step": step_count, "component": "Action Predictor", "result": action}
                )

                # Check if action contains the necessary information
                if "tool_name" not in action or "sub_goal" not in action:
                    error_msg = (
                        "Action prediction did not return required fields"
                        " (tool_name, sub_goal)."
                    )
                    logger.error(error_msg)
                    self.memory.append(
                        {
                            "step": step_count,
                            "component": "Error",
                            "result": error_msg,
                        }
                    )
                    break  # Stop if the action prediction is invalid

                # Step 2b & 2c: Execute Tool (Combined, replacing Executor Agent)
                tool_result = self._execute_tool(
                    action["tool_name"],
                    {
                        **context,
                        **action.get("context", {}),
                        "sub_goal": action["sub_goal"],
                    },
                )
                self.memory.append(
                    {
                        "step": step_count,
                        "component": "Tool Execution",
                        "result": tool_result,
                    }
                )

                # Step 2d: Context Update
                context.update(
                    {
                        "last_action": action,
                        "last_result": tool_result.get("result"),
                        "last_error": tool_result.get("error"),
                    }
                )

                # Step 2e: Context Verification
                verification_response = self.verifier.run(
                    f"Current Context: {json.dumps(context)}\nMemory:"
                    f" {json.dumps(self.memory)}"
                )
                self.conversation.add(
                    role=self.verifier.agent_name, content=verification_response
                )
                verification = self._safely_parse_json(verification_response)
                self.memory.append(
                    {
                        "step": step_count,
                        "component": "Context Verifier",
                        "result": verification,
                    }
                )

                if verification.get("stop_signal"):
                    break

                step_count += 1

            # Step 3: Solution Summarization
            summarization_response = self.summarizer.run(
                f"Complete Trajectory: {json.dumps(self.memory)}"
            )
            self.conversation.add(
                role=self.summarizer.agent_name, content=summarization_response
            )
            summarization = self._safely_parse_json(summarization_response)

            return {
                "final_answer": summarization.get("final_answer", "No answer found."),
                "trajectory": self.memory,
                "conversation": self.conversation.return_history_as_string(),
            }

        except Exception as e:
            logger.error(f"Error executing task: {str(e)}")
            return {
                "error": str(e),
                "trajectory": self.memory,
                "conversation": self.conversation.return_history_as_string(),
            }

    def save_state(self) -> None:
        """Save the current state of all agents."""
        for agent in [self.planner, self.verifier, self.summarizer]:
            try:
                agent.save_state()
            except Exception as e:
                logger.error(f"Error saving state for {agent.agent_name}: {str(e)}")

    def load_state(self) -> None:
        """Load the saved state of all agents."""
        for agent in [self.planner, self.verifier, self.summarizer]:
            try:
                agent.load_state()
            except Exception as e:
                logger.error(f"Error loading state for {agent.agent_name}: {str(e)}")


# --- Example Usage ---


# Define dummy tool functions (replace with actual implementations)
def image_captioner_execute(image: str, prompt: str, **kwargs) -> str:
    """Dummy image captioner."""
    print(f"image_captioner_execute called with image: {image}, prompt: {prompt}")
    return f"Caption for {image}: A descriptive caption."  # Simplified


def object_detector_execute(image: str, labels: List[str], **kwargs) -> List[str]:
    """Dummy object detector."""
    print(f"object_detector_execute called with image: {image}, labels: {labels}")
    return [f"Detected {label}" for label in labels]  # Simplified


def web_search_execute(query: str, **kwargs) -> str:
    """Dummy web search."""
    print(f"web_search_execute called with query: {query}")
    return f"Search results for '{query}'..."  # Simplified


def python_calculator_execute(expression: str, **kwargs) -> str:
    """Dummy python calculator."""
    print(f"python_calculator_execute called with: {expression}")
    try:
        result = str(eval(expression))  # VERY UNSAFE, use AST in production
        return f"Result of {expression} is {result}"
    except Exception as e:
        return f"Error: {e}"


# Create Tool instances
image_captioner = Tool(
    name="Image_Captioner_Tool",
    description="Generates a caption for an image.",
    metadata={
        "input_types": {"image": "str", "prompt": "str"},
        "output_type": "str",
        "limitations": "May struggle with complex scenes or ambiguous objects.",
        "best_practices": "Use with clear, well-lit images. Provide specific prompts for better results.",
    },
    execute_func=image_captioner_execute,
)

object_detector = Tool(
    name="Object_Detector_Tool",
    description="Detects objects in an image.",
    metadata={
        "input_types": {"image": "str", "labels": "list"},
        "output_type": "list",
        "limitations": "Accuracy depends on the quality of the image and the clarity of the objects.",
        "best_practices": "Provide a list of specific object labels to detect.  Use high-resolution images.",
    },
    execute_func=object_detector_execute,
)

web_search = Tool(
    name="Web_Search_Tool",
    description="Performs a web search.",
    metadata={"input_types": {"query": "str"}, "output_type": "str"},
    execute_func=web_search_execute,
)

calculator = Tool(
    name="Python_Calculator_Tool",
    description="Evaluates a Python expression.",
    metadata={"input_types": {"expression": "str"}, "output_type": "str"},
    execute_func=python_calculator_execute,
)

# Create an OctoToolsSwarm agent
agent = OctoToolsSwarm(tools=[image_captioner, object_detector, web_search, calculator])

# Run the agent with a query
query = "What is the square root of the number of distinct objects in the image 'example.png'?"
# Create a dummy image file for testing
with open("example.png", "w") as f:
    f.write("Dummy image content")

image_path = "example.png"
result = agent.run(query, image=image_path)

print(result["final_answer"])
# print(result["trajectory"])  # Uncomment to see the full trajectory
# print(result["conversation"]) # Uncomment to see agent conversation