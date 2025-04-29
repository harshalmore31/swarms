"""
PCAgent: A hierarchical multi-agent framework for complex PC task automation.
Implements the PC-Agent architecture for GUI automation on PCs using openai, pyautogui, pywinauto, pytesseract, Pillow, and mss.
"""
import os
import sys
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import openai
import pyautogui
from pywinauto import Application, findwindows
from pywinauto.keyboard import send_keys
from PIL import Image
import pytesseract
import mss
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from swarms import Agent as SwarmsAgent
from swarms.structs.conversation import Conversation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up OpenAI API key (replace with your actual key)
openai.api_key = "OPENAI_API_KEY"

class AgentRole(Enum):
    """Defines the possible roles for agents in the PC-Agent system."""
    MANAGER = "manager"
    PROGRESS = "progress"
    DECISION = "decision"
    REFLECTION = "reflection"

@dataclass
class CommunicationEvent:
    """Represents a structured communication event between agents."""
    message: str
    background: Optional[str] = None
    intermediate_output: Optional[Dict[str, Any]] = None
    sender: str = ""
    receiver: str = ""
    timestamp: str = str(datetime.now())

@dataclass
class Action:
    """Represents an action to be executed on the PC."""
    type: str
    x: Optional[int] = None
    y: Optional[int] = None
    text: Optional[str] = None
    target: Optional[str] = None

class PCAgent:
    """
    A hierarchical multi-agent system for complex PC task automation.

    Implements the PC-Agent framework with structured communication protocols,
    hierarchical task decomposition, and reflection-based dynamic decision-making.
    """

    def __init__(
        self,
        max_iterations: int = 5,
        model_name: str = "gpt-4",  # Changed from gpt-4o to gpt-4
        base_path: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the PC-Agent system."""
        self.max_iterations = max_iterations
        self.model_name = model_name
        self.base_path = Path(base_path) if base_path else Path("./pc_agent_states")
        self.base_path.mkdir(exist_ok=True)

        # Set up API key
        if api_key:
            openai.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key must be provided either through api_key parameter or OPENAI_API_KEY environment variable")

        # Initialize screenshot tool
        self.sct = mss.mss()

        # Initialize tesseract path if needed
        if sys.platform == "win32":
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Initialize agents
        self._init_agents()

        # Create conversation for tracking
        self.conversation = Conversation()

    def _safely_parse_json(self, json_str: str) -> Dict[str, Any]:
        """
        Safely parse JSON string, handling various formats and potential errors.
        """
        try:
            # First try direct parsing
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                # Try to extract JSON from markdown-style code blocks
                import re
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', json_str)
                if json_match:
                    return json.loads(json_match.group(1))
                # Try to extract anything that looks like JSON
                json_match = re.search(r'\{[\s\S]*\}', json_str)
                if json_match:
                    return json.loads(json_match.group(0))
            except (json.JSONDecodeError, AttributeError):
                pass
            logger.warning("Failed to parse JSON, returning default structure")
            return {"content": json_str, "parsed": False}

    def _init_agents(self) -> None:
        """Initialize all agents with their specific roles and prompts."""
        # Manager Agent (Instruction-level): Decomposes instructions into subtasks
        self.manager = SwarmsAgent(
            agent_name="Manager-Agent",
            system_prompt=self._get_manager_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "manager.json"),
            verbose=True,
        )

        # Progress Agent (Subtask-level): Tracks and summarizes subtask progress
        self.progress = SwarmsAgent(
            agent_name="Progress-Agent",
            system_prompt=self._get_progress_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "progress.json"),
            verbose=True,
        )

        # Decision Agent (Action-level): Makes step-by-step decisions for subtasks
        self.decision = SwarmsAgent(
            agent_name="Decision-Agent",
            system_prompt=self._get_decision_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "decision.json"),
            verbose=True,
        )

        # Reflection Agent (Action-level): Provides feedback on action outcomes
        self.reflection = SwarmsAgent(
            agent_name="Reflection-Agent",
            system_prompt=self._get_reflection_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "reflection.json"),
            verbose=True,
        )

    def _get_manager_prompt(self) -> str:
        """Get the prompt for the Manager Agent."""
        return """You are a Manager Agent responsible for decomposing complex user instructions into parameterized subtasks for PC automation.

Given a user instruction, decompose it into subtasks with parameters, considering:
- Interdependencies between subtasks
- Application-specific requirements (e.g., Chrome, GitHub)
- Output format for communication hub

Output all responses in strict JSON format:
{
    "subtasks": [
        {
            "id": "Unique subtask ID",
            "description": "Detailed description of the subtask",
            "parameters": {
                "app": "Target application (e.g., Chrome, GitHub)",
                "input": "Required input parameters",
                "output_var": "Variable name for output (if applicable)"
            },
            "dependencies": ["List of preceding subtask IDs (if any)"]
        }
    ],
    "communication_hub": {
        "variables": {
            "var_name": "Description of variable value"
        }
    }
}"""

    def _get_progress_prompt(self) -> str:
        """Get the prompt for the Progress Agent."""
        return """You are a Progress Agent responsible for tracking and summarizing the progress of subtasks in PC automation.

Given a subtask, current progress, action, and reflection feedback:
- Update task progress
- Identify completed/remaining steps
- Provide summary for decision-making

Output all responses in strict JSON format:
{
    "task_progress": {
        "subtask_id": "ID of current subtask",
        "completed_steps": ["List of completed action IDs"],
        "pending_steps": ["List of remaining action IDs"],
        "status": "Current status (e.g., 'in_progress', 'completed', 'failed')"
    },
    "summary": "Brief summary of progress for decision-making"
}"""

    def _get_decision_prompt(self) -> str:
        """Get the prompt for the Decision Agent."""
        return """You are a Decision Agent responsible for making step-by-step action decisions for PC automation subtasks.

Given a subtask, current observation (screenshot), progress, and reflection feedback:
- Use Chain-of-Thought reasoning
- Generate specific actions (e.g., click, type, scroll) using pyautogui and pywinauto
- Consider the PC environment and interdependencies

Output all responses in strict JSON format:
{
    "thought": "Chain-of-Thought reasoning for the decision",
    "action": {
        "type": "Action type (e.g., 'click', 'type', 'scroll')",
        "parameters": {
            "x": "X-coordinate (if applicable)",
            "y": "Y-coordinate (if applicable)",
            "text": "Text to type (if applicable)",
            "target": "Target element description (if applicable)"
        }
    },
    "summary": "Summary of the action decision"
}"""

    def _get_reflection_prompt(self) -> str:
        """Get the prompt for the Reflection Agent."""
        return """You are a Reflection Agent responsible for evaluating action outcomes in PC automation.

Given a subtask, action, and before/after observations (screenshots):
- Assess if the action produced the expected outcome
- Identify errors or ineffective actions
- Provide feedback for correction or adjustment

Output all responses in strict JSON format:
{
    "evaluation": {
        "success": "Boolean (True/False) indicating if action was successful",
        "expected_outcome": "Description of expected outcome",
        "actual_outcome": "Description of actual outcome",
        "feedback": "Actionable suggestions for correction"
    }
}"""

    def _active_perception_module(self, screenshot: Image.Image) -> Dict[str, Any]:
        """
        Implement the Active Perception Module (APM) for perceiving screen elements.
        Uses pywinauto for accessibility and pytesseract for OCR.
        """
        try:
            # Convert screenshot to text using OCR
            text = pytesseract.image_to_string(screenshot)
            boxes = pytesseract.image_to_boxes(screenshot)  # Get bounding box data for text

            # Use pywinauto to get interactive elements (Windows accessibility tree)
            app = Application().top_window()
            elements = []
            for ctrl in app.controls():
                if ctrl.is_visible():
                    elements.append({
                        "type": ctrl.control_type,
                        "location": ctrl.rectangle(),
                        "description": ctrl.window_text()
                    })

            return {
                "interactive_elements": elements,
                "text": {
                    "content": text,
                    "boxes": [{k: v for k, v in box.items()} for box in boxes]
                }
            }
        except Exception as e:
            logger.error(f"Error in Active Perception Module: {str(e)}")
            return {"interactive_elements": [], "text": {"content": "", "boxes": []}}

    def _execute_action(self, action: Action) -> None:
        """Execute the action using pyautogui and pywinauto on the PC environment."""
        try:
            if action.type == "click":
                pyautogui.click(x=action.x, y=action.y)
            elif action.type == "type":
                if action.target:  # Use pywinauto for targeting specific elements if available
                    app = Application().connect(title=action.target)
                    app.type_keys(action.text)
                else:
                    pyautogui.typewrite(action.text)
            elif action.type == "scroll":
                pyautogui.scroll(action.y if action.y else 0)  # Scroll vertically
            elif action.type == "open_app":
                if action.target == "Chrome":
                    pyautogui.hotkey("win", "r")  # Open Run dialog
                    pyautogui.typewrite("chrome")
                    send_keys("{ENTER}")
            else:
                logger.warning(f"Unsupported action type: {action.type}")
        except Exception as e:
            logger.error(f"Action execution error for {action.type}: {str(e)}")
            raise

    def _get_screenshot(self) -> Image.Image:
        """Capture and return a screenshot using mss."""
        try:
            screenshot = self.sct.grab(self.sct.monitors[1])  # Grab primary monitor
            return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        except Exception as e:
            logger.error(f"Error capturing screenshot: {str(e)}")
            return Image.new("RGB", (1, 1))  # Return empty image as fallback

    def _decompose_instruction(self, instruction: str) -> Dict[str, Any]:
        """Decompose user instruction into subtasks using Manager Agent."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_manager_prompt()},
                    {"role": "user", "content": instruction}
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            return self._safely_parse_json(response.choices[0].message["content"])
        except Exception as e:
            logger.error(f"Error in instruction decomposition: {str(e)}")
            raise

    def _track_progress(self, subtask: Dict, progress: Dict, action: Dict, reflection: Dict) -> Dict[str, Any]:
        """Track subtask progress using Progress Agent."""
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": self._get_progress_prompt() + "\n" + json.dumps({
                "subtask": subtask,
                "previous_progress": progress,
                "action": action,
                "reflection": reflection
            })}]
        )
        return self._safely_parse_json(response.choices[0].message.content)

    def _make_decision(self, subtask: Dict, observation: Dict, progress: Dict, reflection: Dict) -> Dict[str, Any]:
        """Make action decision using Decision Agent."""
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": self._get_decision_prompt() + "\n" + json.dumps({
                "subtask": subtask,
                "observation": json.dumps(observation),
                "progress": progress,
                "reflection": reflection
            })}]
        )
        return self._safely_parse_json(response.choices[0].message.content)

    def _reflect_action(self, subtask: Dict, action: Dict, before: Image.Image, after: Image.Image) -> Dict[str, Any]:
        """Reflect on action outcome using Reflection Agent."""
        before_text = pytesseract.image_to_string(before)
        after_text = pytesseract.image_to_string(after)

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": self._get_reflection_prompt() + "\n" + json.dumps({
                "subtask": subtask,
                "action": asdict(action),
                "before_observation": before_text,
                "after_observation": after_text
            })}]
        )
        reflection_data = self._safely_parse_json(response.choices[0].message.content)

        # Add basic logic to determine success based on text changes
        if before_text != after_text:
            reflection_data["evaluation"]["success"] = True
            reflection_data["evaluation"]["actual_outcome"] = "Screen content changed as expected"
        else:
            reflection_data["evaluation"]["success"] = False
            reflection_data["evaluation"]["actual_outcome"] = "No change or unexpected change in screen content"

        return reflection_data

    def run(self, instruction: str) -> Dict[str, Any]:
        """
        Execute complex PC task automation based on the user instruction.

        Args:
            instruction: User instruction (e.g., "Open Chrome and search for https://github.com/browser-use/browser-use.")

        Returns:
            Dictionary containing task outcome and metadata
        """
        logger.info(f"Starting PC task automation for instruction: {instruction}")

        try:
            # Step 1: Manager Agent decomposes instruction into subtasks
            subtasks = self._decompose_instruction(instruction)

            # Communication hub for inter-subtask dependencies
            communication_hub = subtasks.get("communication_hub", {"variables": {}})
            completed_subtasks = set()

            for subtask in subtasks["subtasks"]:
                subtask_id = subtask["id"]
                logger.info(f"Processing subtask {subtask_id}: {subtask['description']}")

                # Check dependencies before processing
                if subtask["dependencies"]:
                    for dep_id in subtask["dependencies"]:
                        if dep_id not in completed_subtasks:
                            logger.warning(f"Skipping subtask {subtask_id} due to unmet dependency {dep_id}")
                            continue

                # Initialize progress tracking
                task_progress = {
                    "subtask_id": subtask_id,
                    "completed_steps": [],
                    "pending_steps": [],
                    "status": "in_progress"
                }
                observation = self._active_perception_module(self._get_screenshot())
                iteration = 0

                while task_progress["status"] == "in_progress" and iteration < self.max_iterations:
                    # Step 2: Decision Agent makes action decision
                    decision = self._make_decision(subtask, observation, task_progress, {})
                    action_data = decision["action"]
                    action = Action(**action_data)

                    # Step 3: Execute action and capture before/after screenshots
                    before_screenshot = self._get_screenshot()
                    self._execute_action(action)
                    after_screenshot = self._get_screenshot()

                    # Step 4: Reflection Agent evaluates action outcome
                    reflection = self._reflect_action(subtask, action, before_screenshot, after_screenshot)

                    # Step 5: Progress Agent updates progress
                    task_progress = self._track_progress(subtask, task_progress, asdict(action), reflection)

                    # Update observation for next iteration
                    observation = self._active_perception_module(after_screenshot)
                    iteration += 1

                    # Check if reflection indicates failure or correction needed
                    if not reflection["evaluation"]["success"]:
                        logger.warning(f"Action failed for subtask {subtask_id}. Attempting correction...")
                        # Optionally retry with adjusted action based on reflection feedback
                        if iteration >= self.max_iterations:
                            task_progress["status"] = "failed"
                            break

                # Update communication hub if subtask produces output
                if "output_var" in subtask["parameters"] and reflection["evaluation"]["success"]:
                    communication_hub["variables"][subtask["parameters"]["output_var"]] = reflection["evaluation"]["actual_outcome"]

                # Mark subtask as completed if successful
                if task_progress["status"] == "completed":
                    completed_subtasks.add(subtask_id)
                elif task_progress["status"] == "failed":
                    logger.warning(f"Subtask {subtask_id} failed after {iteration} iterations")
                    return {
                        "status": "failed",
                        "subtask": subtask,
                        "progress": task_progress,
                        "error": "Max iterations reached without completion"
                    }

            return {
                "status": "completed",
                "subtasks": subtasks["subtasks"],
                "communication_hub": communication_hub,
                "metadata": {
                    "iterations": iteration,
                    "timestamp": str(datetime.now())
                }
            }

        except Exception as e:
            logger.error(f"Error in PC task automation: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "metadata": {"timestamp": str(datetime.now())}
            }

    def cleanup(self) -> None:
        """Clean up resources."""
        self.sct.close()

    def save_state(self) -> None:
        """Save the current state of all agents."""
        for agent in [self.manager, self.progress, self.decision, self.reflection]:
            try:
                agent.save_state()
            except Exception as e:
                logger.error(f"Error saving state for {agent.agent_name}: {str(e)}")

    def load_state(self) -> None:
        """Load the saved state of all agents."""
        for agent in [self.manager, self.progress, self.decision, self.reflection]:
            try:
                agent.load_state()
            except Exception as e:
                logger.error(f"Error loading state for {agent.agent_name}: {str(e)}")

if __name__ == "__main__":
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Initialize PCAgent with environment variable for API key
        pc_agent = PCAgent(
            model_name="gpt-4",
            base_path="./pc_agent_states"
        )

        # Example user instruction
        instruction = "Open Chrome and search for https://github.com/kyegomez/swarms"
        
        try:
            result = pc_agent.run(instruction)
            logger.info(f"Task completed with result: {json.dumps(result, indent=2)}")
        except Exception as e:
            logger.error(f"Task failed: {str(e)}")
        finally:
            pc_agent.cleanup()
            pc_agent.save_state()

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)