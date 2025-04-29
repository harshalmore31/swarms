"""
PC-Agent Framework - Improved Implementation
A hierarchical multi-agent system for PC task automation.
Enhanced with proper agent interactions and workflow improvements.
"""

import json
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import sys
import traceback
from pathlib import Path
from swarms.structs.dev.agent_adapter import extract_json_from_response
from action_utils import execute_action_with_retry, validate_action_parameters

# Import statements for external dependencies
from swarms import Agent
from swarms.structs.conversation import Conversation

# --- Enums for System State Management ---
class SubtaskStatus(Enum):
    """Enum for tracking subtask execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYING = "retrying"

class ActionType(Enum):
    """Enum for categorizing types of actions."""
    CLICK = "click"
    TYPE = "type"
    DRAG = "drag"
    WAIT = "wait"
    SCROLL = "scroll"
    KEYBOARD_SHORTCUT = "keyboard_shortcut"
    CUSTOM = "custom"

# --- Logging Configuration ---
def setup_logging(level=logging.INFO, log_file=None):
    """Sets up basic logging configuration with optional file output."""
    handlers = []
    if log_file: # Fix: Corrected condition to check if log_file is truthy
        handlers.append(logging.FileHandler(log_file))
    handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=handlers
    )

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# --- Utility Functions ---
def safely_parse_json(json_str: str) -> Dict[str, Any]:
    """Safely parses JSON string, handling potential errors."""
    try:
        # First attempt standard parsing
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSONDecodeError: {e}. Attempting to fix and parse.")
        try:
            # Extract JSON from potential text wrapping
            import re
            json_pattern = r'```(?:json)?(.*?)```'
            matches = re.findall(json_pattern, json_str, re.DOTALL)
            if matches:
                json_str = matches[0].strip()

            # Additional cleanup
            json_str = json_str.strip().rstrip(',')  # Remove trailing comma if present

            # Try parsing again
            return json.loads(json_str)
        except json.JSONDecodeError as e2:
            logger.error(f"Failed to parse JSON even after cleanup: {e2}")
            logger.debug(f"Problematic JSON string: {json_str}")
            return {"error": "JSON Parse Error", "details": str(e2), "raw_content": json_str}
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
        return {"error": "JSON Parse Error", "details": str(e), "raw_content": json_str}

def create_screenshot(filename: str = None) -> str:
    """
    Takes a screenshot and saves it to the specified filename.
    Returns the path to the saved screenshot.
    """
    try:
        import pyautogui
        if not filename:
            filename = f"screenshot_{int(time.time())}.png"
        screenshot_path = Path(filename)
        pyautogui.screenshot(str(screenshot_path))
        logger.info(f"Screenshot saved to {screenshot_path}")
        return str(screenshot_path)
    except ImportError:
        logger.warning("pyautogui not installed, using placeholder for screenshot. Install pyautogui for real screenshot functionality: pip install pyautogui") # Improved warning message
        return "placeholder_screenshot.png"
    except Exception as e:
        logger.error(f"Error taking screenshot: {e}")
        return "error_screenshot.png"

# --- Communication Hub ---
class CommunicationHub:
    """
    Communication Hub for PC-Agent framework.
    Facilitates data sharing and communication between agents.
    Maintains task state and history.
    """
    def __init__(self):
        """Initializes the Communication Hub with separate namespaces for different types of data."""
        self._task_data = {}     # For task/subtask related data
        self._perception_data = {} # For APM perception results
        self._action_history = [] # History of actions taken
        self._reflection_history = [] # History of reflections on actions
        self._temporary_data = {} # For transient data exchanges

        logger.info("Initializing Communication Hub with enhanced data management")

    def update_task_data(self, key: str, value: Any) -> None:
        """Updates task-related data in the communication hub."""
        if not isinstance(key, str): # Added type check for key
            logger.error(f"Key must be a string, but got {type(key)}: {key}")
            return
        self._task_data[key] = value
        logger.debug(f"Task data updated: {key}")

    def get_task_data(self, key: str, default=None) -> Any:
        """Retrieves task-related data from the communication hub."""
        if not isinstance(key, str): # Added type check for key
            logger.error(f"Key must be a string, but got {type(key)}: {key}")
            return default
        value = self._task_data.get(key, default)
        logger.debug(f"Task data retrieved: {key}")
        return value

    def update_perception_data(self, key: str, value: Any) -> None:
        """Updates perception data in the communication hub."""
        if not isinstance(key, str): # Added type check for key
            logger.error(f"Key must be a string, but got {type(key)}: {key}")
            return
        self._perception_data[key] = value
        logger.debug(f"Perception data updated: {key}")

    def get_perception_data(self, key: str, default=None) -> Any:
        """Retrieves perception data from the communication hub."""
        if not isinstance(key, str): # Added type check for key
            logger.error(f"Key must be a string, but got {type(key)}: {key}")
            return default
        value = self._perception_data.get(key, default)
        logger.debug(f"Perception data retrieved: {key}")
        return value

    def add_action(self, action_data: dict) -> None:
        """Adds an action to the action history."""
        if not isinstance(action_data, dict): # Added type check for action_data
            logger.error(f"action_data must be a dict, but got {type(action_data)}: {action_data}")
            return
        self._action_history.append({
            "timestamp": time.time(),
            **action_data
        })
        logger.debug(f"Action added to history: {action_data.get('action_description', 'Unknown action')}")

    def add_reflection(self, reflection_data: dict) -> None:
        """Adds a reflection to the reflection history."""
        if not isinstance(reflection_data, dict): # Added type check for reflection_data
            logger.error(f"reflection_data must be a dict, but got {type(reflection_data)}: {reflection_data}")
            return
        self._reflection_history.append({
            "timestamp": time.time(),
            **reflection_data
        })
        logger.debug(f"Reflection added to history: {reflection_data.get('reflection_result', 'Unknown reflection')}")

    def get_latest_actions(self, count: int = 5) -> List[dict]:
        """Retrieves the latest actions from the action history."""
        if not isinstance(count, int) or count < 0: # Added type and value check for count
            logger.error(f"count must be a non-negative integer, but got {type(count)}: {count}")
            return []
        return self._action_history[-count:] if self._action_history else []

    def get_latest_reflections(self, count: int = 5) -> List[dict]:
        """Retrieves the latest reflections from the reflection history."""
        if not isinstance(count, int) or count < 0: # Added type and value check for count
            logger.error(f"count must be a non-negative integer, but got {type(count)}: {count}")
            return []
        return self._reflection_history[-count:] if self._reflection_history else []

    def get_actions_for_subtask(self, subtask_id: str) -> List[dict]:
        """Retrieves all actions for a specific subtask."""
        if not isinstance(subtask_id, str): # Added type check for subtask_id
            logger.error(f"subtask_id must be a string, but got {type(subtask_id)}: {subtask_id}")
            return []
        return [action for action in self._action_history if action.get("subtask_id") == subtask_id]

    def get_reflections_for_subtask(self, subtask_id: str) -> List[dict]:
        """Retrieves all reflections for a specific subtask."""
        if not isinstance(subtask_id, str): # Added type check for subtask_id
            logger.error(f"subtask_id must be a string, but got {type(subtask_id)}: {subtask_id}")
            return []
        return [reflection for reflection in self._reflection_history if reflection.get("subtask_id") == subtask_id]

    def update_temp_data(self, key: str, value: Any) -> None:
        """Updates temporary data in the communication hub."""
        if not isinstance(key, str): # Added type check for key
            logger.error(f"Key must be a string, but got {type(key)}: {key}")
            return
        self._temporary_data[key] = value
        logger.debug(f"Temporary data updated: {key}")

    def get_temp_data(self, key: str, default=None) -> Any:
        """Retrieves temporary data from the communication hub."""
        if not isinstance(key, str): # Added type check for key
            logger.error(f"Key must be a string, but got {type(key)}: {key}")
            return default
        value = self._temporary_data.get(key, default)
        logger.debug(f"Temporary data retrieved: {key}")
        return value

    def clear_temp_data(self) -> None:
        """Clears all temporary data from the communication hub."""
        self._temporary_data.clear()
        logger.debug("Temporary data cleared")

    def get_task_summary(self) -> dict:
        """Returns a summary of the current task state."""
        return {
            "task_data": dict(self._task_data),
            "action_count": len(self._action_history),
            "reflection_count": len(self._reflection_history),
            "latest_action": self._action_history[-1] if self._action_history else None,
            "latest_reflection": self._reflection_history[-1] if self._reflection_history else None
        }

    def clear_all(self) -> None:
        """Clears all data from the communication hub."""
        self._task_data.clear()
        self._perception_data.clear()
        self._action_history.clear()
        self._reflection_history.clear()
        self._temporary_data.clear()
        logger.info("Communication Hub cleared completely.")


# --- Active Perception Module (APM) ---
class ActivePerceptionModule:
    """
    Active Perception Module (APM) for PC-Agent.
    Provides perception capabilities for interacting with the PC environment.
    Combines accessibility tree information with MLLM-driven OCR for comprehensive perception.
    """
    def __init__(self, enable_ocr=True, enable_accessibility_tree=True, model_name="gemini/gemini-2.0-flash"):
        """
        Initializes the APM with optional OCR and accessibility tree capabilities.

        Args:
            enable_ocr: Whether to enable OCR capabilities
            enable_accessibility_tree: Whether to enable accessibility tree capabilities
            model_name: Model to use for MLLM-based perception
        """
        self.enable_ocr = enable_ocr
        self.enable_accessibility_tree = enable_accessibility_tree
        self.model_name = model_name
        self.ocr_agent = None

        logger.info(f"Initializing Active Perception Module with OCR: {enable_ocr}, Accessibility: {enable_accessibility_tree}")

        # Initialize MLLM agent for visual perception if OCR is enabled
        if self.enable_ocr:
            self._initialize_ocr_agent()

        # In a full implementation, initialize accessibility tree tools like pywinauto
        self._initialize_accessibility_tools()

    def _initialize_ocr_agent(self):
        """Initializes the OCR agent using an MLLM for visual perception."""
        # In a real implementation, this would initialize a model with vision capabilities
        try: # Added try-except block for agent initialization
            self.ocr_agent = Agent(
                agent_name="OCRAgent",
                system_prompt="You are a specialized OCR agent that extracts text and visual information from screenshots. " +
                             "Describe all visible text, buttons, input fields, and interactive elements you can see.",
                model_name=self.model_name,
                max_loops=1
            )
            logger.info("OCR Agent initialized for visual perception")
        except Exception as e:
            logger.error(f"Failed to initialize OCR Agent: {e}")
            self.ocr_agent = None # Ensure ocr_agent is None in case of failure
            self.enable_ocr = False # Disable OCR if agent initialization fails
            logger.warning("Disabling OCR functionality due to initialization failure.")

    def _initialize_accessibility_tools(self):
        """Initializes accessibility tree tools for element detection."""
        # In a real implementation, this would initialize pywinauto or similar
        try:
            # import pywinauto
            # self.desktop = pywinauto.Desktop(backend="uia")
            logger.info("Accessibility tools initialized (placeholder)")
        except ImportError:
            logger.warning("Accessibility tools import failed, using placeholder functionality")
            self.enable_accessibility_tree = False # Disable accessibility tree if import fails
            pass

    def get_accessibility_tree(self, window_title=None) -> dict:
        """
        Retrieves accessibility tree information for the current screen or specific window.

        Args:
            window_title: Optional title of window to focus on

        Returns:
            Dictionary containing accessibility tree information
        """
        logger.info(f"Getting accessibility tree information for window: {window_title or 'current screen'}")

        if not self.enable_accessibility_tree: # Check if accessibility tree is enabled
            logger.warning("Accessibility tree is disabled. Returning placeholder accessibility tree.")
            return {
                "elements": [],
                "window_title": window_title or "Current Screen",
                "timestamp": time.time(),
                "is_placeholder": True # Indicate placeholder data
            }

        # In a real implementation, use pywinauto to get actual accessibility tree
        # For now, return placeholder data
        return {
            "elements": [
                {"id": "element-1", "type": "button", "text": "Example Button", "bounding_box": [100, 100, 200, 150], "properties": {"can_click": True}},
                {"id": "element-2", "type": "input", "text": "", "bounding_box": [300, 100, 500, 130], "properties": {"can_input": True}},
                {"id": "element-3", "type": "menu", "text": "File", "bounding_box": [10, 10, 50, 30], "properties": {"can_click": True}}
                # ... more elements ...
            ],
            "window_title": window_title or "Current Screen",
            "timestamp": time.time()
        }

    def get_ocr_information(self, screenshot_path: str) -> dict:
        """
        Extracts text information from a screenshot using OCR.

        Args:
            screenshot_path: Path to the screenshot image

        Returns:
            Dictionary containing OCR results
        """
        logger.info(f"Getting OCR information from screenshot: {screenshot_path}")

        if not self.enable_ocr:
            logger.warning("OCR is disabled. Returning empty OCR results.")
            return {"ocr_results": [], "error": "OCR is disabled", "is_placeholder": True} # Indicate placeholder

        if not self.ocr_agent:
            logger.warning("OCR agent not initialized. Returning placeholder OCR results.")
            return {
                "ocr_results": [
                    {"text": "Example Text 1", "bounding_box": [100, 200, 300, 230], "confidence": 0.95},
                    {"text": "Example Text 2", "bounding_box": [400, 300, 600, 330], "confidence": 0.88}
                ],
                "is_placeholder": True
            }

        # In a real implementation, this would process the screenshot with the MLLM
        # For now, return placeholder OCR results
        return {
            "ocr_results": [
                {"text": "Search", "bounding_box": [120, 220, 180, 250], "confidence": 0.92, "element_type": "button"},
                {"text": "Type here to search", "bounding_box": [200, 220, 400, 250], "confidence": 0.88, "element_type": "input"},
                {"text": "Settings", "bounding_box": [50, 300, 120, 330], "confidence": 0.95, "element_type": "menu_item"}
            ],
            "timestamp": time.time(),
            "screenshot_path": screenshot_path
        }

    def find_element_by_text(self, screenshot_path: str, target_text: str, exact_match=False) -> dict:
        """
        Finds an element on the screen that contains the specified text.

        Args:
            screenshot_path: Path to the screenshot image
            target_text: Text to search for
            exact_match: Whether to require an exact match (vs substring)

        Returns:
            Dictionary containing information about the found element(s)
        """
        logger.info(f"Finding element with text '{target_text}' (exact match: {exact_match})")

        # Get OCR information
        ocr_results = self.get_ocr_information(screenshot_path)

        # Find matching text elements
        matches = []
        for result in ocr_results.get("ocr_results", []):
            text = result.get("text", "")
            if (exact_match and text == target_text) or (not exact_match and target_text.lower() in text.lower()):
                matches.append(result)

        # Also search accessibility tree
        if self.enable_accessibility_tree:
            acc_tree = self.get_accessibility_tree()
            for element in acc_tree.get("elements", []):
                text = element.get("text", "")
                if (exact_match and text == target_text) or (not exact_match and target_text.lower() in text.lower()):
                    matches.append(element)

        return {
            "query": target_text,
            "exact_match": exact_match,
            "matches": matches,
            "match_count": len(matches),
            "screenshot_path": screenshot_path,
            "timestamp": time.time()
        }

    def analyze_screen(self, screenshot_path: str = None, query: str = None) -> dict:
        """
        Performs a comprehensive analysis of the current screen.

        Args:
            screenshot_path: Path to screenshot or None to take a new screenshot
            query: Optional specific query to focus the analysis

        Returns:
            Dictionary containing comprehensive perception results
        """
        # Take a new screenshot if none provided
        if not screenshot_path:
            screenshot_path = create_screenshot()

        logger.info(f"Analyzing screen with screenshot: {screenshot_path}, query: {query or 'general analysis'}")

        # Collect perception data from multiple sources
        perception_data = {
            "timestamp": time.time(),
            "screenshot_path": screenshot_path,
            "query": query
        }

        # Get accessibility tree information if enabled
        if self.enable_accessibility_tree:
            perception_data["accessibility_tree"] = self.get_accessibility_tree()

        # Get OCR information if enabled
        if self.enable_ocr:
            perception_data["ocr_information"] = self.get_ocr_information(screenshot_path)

        # Combine and analyze the results
        perception_data["combined_elements"] = self._combine_perception_sources(
            perception_data.get("accessibility_tree", {}).get("elements", []),
            perception_data.get("ocr_information", {}).get("ocr_results", [])
        )

        # Add specific query results if a query was provided
        if query:
            perception_data["query_results"] = self.find_element_by_text(screenshot_path, query, exact_match=False)

        return perception_data

    def _combine_perception_sources(self, accessibility_elements, ocr_elements) -> List[dict]:
        """
        Combines elements from accessibility tree and OCR results,
        attempting to resolve duplicates.

        Returns:
            List of unique elements with combined information
        """
        # This would be more sophisticated in a real implementation
        # For now, just combine the lists with simple deduplication based on position overlap

        combined = []

        # Add all accessibility elements
        for element in accessibility_elements:
            combined.append({
                "source": "accessibility_tree",
                **element
            })

        # Add OCR elements, checking for potential duplicates
        for ocr_element in ocr_elements:
            # Check if this element overlaps with an existing one
            is_duplicate = False
            for existing in combined:
                if self._check_bounding_box_overlap(
                    existing.get("bounding_box", [0, 0, 0, 0]),
                    ocr_element.get("bounding_box", [0, 0, 0, 0])
                ):
                    # Enhance the existing element with OCR data
                    existing["ocr_text"] = ocr_element.get("text")
                    existing["source"] = "combined"
                    is_duplicate = True
                    break

            if not is_duplicate:
                combined.append({
                    "source": "ocr",
                    **ocr_element
                })

        return combined

    def _check_bounding_box_overlap(self, box1, box2, threshold=0.5) -> bool:
        """
        Checks if two bounding boxes overlap significantly.

        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            threshold: Minimum overlap ratio to consider boxes as overlapping

        Returns:
            Boolean indicating significant overlap
        """
        # Simple placeholder implementation
        # A real implementation would calculate actual intersection over union
        if not box1 or not box2: # Handle cases where bounding boxes are missing or None
            return False
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection coordinates
        x_intersection_start = max(x1_1, x1_2)
        y_intersection_start = max(y1_1, y1_2)
        x_intersection_end = min(x2_1, x2_2)
        y_intersection_end = min(y2_1, y2_2)

        # Calculate intersection area
        intersection_area = max(0, x_intersection_end - x_intersection_start) * max(0, y_intersection_end - y_intersection_start)

        # Calculate area of box1 and box2
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Calculate overlap ratio (using area1 as base, can also use area2 or min/max/avg of both)
        overlap_ratio = intersection_area / area1 if area1 > 0 else 0 # Avoid division by zero

        return overlap_ratio >= threshold


    def execute_action(self, action_type: str, parameters: dict) -> bool:
        """
        Executes an action on the screen using pyautogui or similar.

        Args:
            action_type: Type of action to perform (click, type, etc.)
            parameters: Parameters for the action

        Returns:
            Boolean indicating success or failure
        """
        logger.info(f"Executing action: {action_type} with parameters: {parameters}")

        # Use the improved action execution utility
        result = execute_action_with_retry(action_type, parameters)
        return result["success"]

# --- Agent Implementations ---
class ManagerAgent(Agent):
    """
    Manager Agent for PC-Agent framework.
    Decomposes user instructions into subtasks and manages workflow.
    """
    def __init__(self, communication_hub: CommunicationHub, model_name="gemini/gemini-2.0-flash", **kwargs):
        """
        Initializes the Manager Agent.

        Args:
            communication_hub: Reference to the communication hub
            model_name: Model to use for agent
        """
        super().__init__(
            agent_name="ManagerAgent",
            system_prompt=self._get_system_prompt(),
            model_name=model_name,
            **kwargs
        )
        self.communication_hub = communication_hub
        logger.info(f"Initialized Manager Agent with model {model_name}")

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for the Manager Agent."""
        return """You are the Manager Agent in the PC-Agent framework. Your role is to decompose complex user instructions for PC automation into a sequence of well-defined, parameterized subtasks.

        Analyze the user instruction and break it down into a series of actionable subtasks. Each subtask should be a clear, concise step that can be executed by other agents. Consider potential dependencies between subtasks.

        Example Instruction: "Open Chrome, go to google.com, and search for 'weather in London'."

        Decomposed Subtasks (Example JSON Output):
        ```json
        {
            "task_analysis": "The user wants to get the weather in London using Google Chrome.",
            "subtasks": [
                {
                    "id": "subtask-1",
                    "description": "Open Google Chrome browser.",
                    "parameters": {},
                    "expected_outcome": "Google Chrome browser is open and ready."
                },
                {
                    "id": "subtask-2",
                    "description": "Navigate to google.com in Chrome.",
                    "parameters": {"url": "google.com"},
                    "dependency": "subtask-1",
                    "expected_outcome": "google.com webpage is loaded in Chrome."
                },
                {
                    "id": "subtask-3",
                    "description": "Search for 'weather in London' in the Google search bar.",
                    "parameters": {"query": "weather in London"},
                    "dependency": "subtask-2",
                    "expected_outcome": "Google search results page for 'weather in London' is displayed."
                }
            ],
            "overall_approach": "Execute subtasks sequentially, ensuring each subtask's expected outcome is achieved before proceeding to the next."
        }
        ```

        Your response should be in JSON format, containing:
        - "task_analysis": A brief summary of the overall user instruction.
        - "subtasks": A list of subtask objects, each with "id", "description", "parameters" (if any), "dependency" (if any, referring to subtask id), and "expected_outcome".
        - "overall_approach": A high-level strategy for executing the decomposed task.

        Consider potential error states and how to handle them. Think about the sequence of interactions needed and break complex actions into simpler steps.
        """

    def decompose_instruction(self, instruction: str) -> dict:
        """
        Decomposes the user instruction into subtasks.

        Args:
            instruction: User instruction to decompose

        Returns:
            Dictionary containing decomposed subtasks
        """
        logger.info(f"Decomposing instruction: {instruction}")
        try: # Added try-except for decompose_instruction
            response = self.run(instruction) # Use swarms.Agent.run method
            subtasks_data = safely_parse_json(response)

            # Update communication hub with task analysis
            if "task_analysis" in subtasks_data:
                self.communication_hub.update_task_data("task_analysis", subtasks_data["task_analysis"])

            # Initialize subtask statuses
            if "subtasks" in subtasks_data:
                for subtask in subtasks_data["subtasks"]:
                    subtask_id = subtask.get("id")
                    if subtask_id:
                        # Add status to each subtask
                        subtask["status"] = SubtaskStatus.PENDING.value
                        # Store subtask in communication hub
                        self.communication_hub.update_task_data(f"subtask_{subtask_id}", subtask)

                # Store the list of subtask IDs in order
                subtask_ids = [subtask.get("id") for subtask in subtasks_data["subtasks"] if subtask.get("id")]
                self.communication_hub.update_task_data("subtask_order", subtask_ids)

            return subtasks_data
        except Exception as e:
            logger.error(f"Error decomposing instruction: {e}")
            return {
                "error": str(e),
                "task_analysis": "Failed to decompose instruction due to an error.",
                "subtasks": []
            }


    def get_next_subtask(self) -> Optional[dict]:
        """
        Determines the next subtask to execute based on dependencies and status.

        Returns:
            Next subtask to execute or None if no subtasks are ready
        """
        subtask_order = self.communication_hub.get_task_data("subtask_order", [])

        for subtask_id in subtask_order:
            subtask = self.communication_hub.get_task_data(f"subtask_{subtask_id}")

            if not subtask:
                continue

            if subtask.get("status") == SubtaskStatus.PENDING.value:
                # Check if dependency is satisfied
                dependency_id = subtask.get("dependency")
                if not dependency_id:
                    # No dependency, can execute
                    return subtask
                else:
                    # Check dependency status
                    dependency = self.communication_hub.get_task_data(f"subtask_{dependency_id}")
                    if dependency and dependency.get("status") == SubtaskStatus.SUCCEEDED.value:
                        # Dependency satisfied
                        return subtask

        # No subtasks ready to execute
        return None

    def update_subtask_status(self, subtask_id: str, new_status: SubtaskStatus) -> None:
        """
        Updates the status of a subtask.

        Args:
            subtask_id: ID of the subtask to update
            new_status: New status to set
        """
        if not isinstance(subtask_id, str): # Added type check for subtask_id
            logger.error(f"subtask_id must be a string, but got {type(subtask_id)}: {subtask_id}")
            return
        if not isinstance(new_status, SubtaskStatus): # Added type check for new_status
            logger.error(f"new_status must be a SubtaskStatus enum, but got {type(new_status)}: {new_status}")
            return

        subtask = self.communication_hub.get_task_data(f"subtask_{subtask_id}")
        if subtask:
            subtask["status"] = new_status.value
            self.communication_hub.update_task_data(f"subtask_{subtask_id}", subtask)
            logger.info(f"Updated subtask {subtask_id} status to {new_status.value}")
        else:
            logger.warning(f"Subtask {subtask_id} not found in communication hub.")


    def get_task_progress(self) -> dict:
        """
        Gets the overall progress of the task.

        Returns:
            Dictionary containing progress information
        """
        subtask_order = self.communication_hub.get_task_data("subtask_order", [])

        total_subtasks = len(subtask_order)
        completed_subtasks = 0
        failed_subtasks = 0
        pending_subtasks = 0
        in_progress_subtasks = 0

        subtasks_status = []

        for subtask_id in subtask_order:
            subtask = self.communication_hub.get_task_data(f"subtask_{subtask_id}")

            if not subtask:
                continue

            status = subtask.get("status")
            subtasks_status.append({
                "id": subtask_id,
                "description": subtask.get("description"),
                "status": status
            })

            if status == SubtaskStatus.SUCCEEDED.value:
                completed_subtasks += 1
            elif status == SubtaskStatus.FAILED.value:
                failed_subtasks += 1
            elif status == SubtaskStatus.PENDING.value:
                pending_subtasks += 1
            elif status == SubtaskStatus.IN_PROGRESS.value:
                in_progress_subtasks += 1

        progress_percentage = (completed_subtasks / total_subtasks * 100) if total_subtasks > 0 else 0

        return {
            "total_subtasks": total_subtasks,
            "completed_subtasks": completed_subtasks,
            "failed_subtasks": failed_subtasks,
            "pending_subtasks": pending_subtasks,
            "in_progress_subtasks": in_progress_subtasks,
            "progress_percentage": progress_percentage,
            "subtasks_status": subtasks_status,
            "task_analysis": self.communication_hub.get_task_data("task_analysis")
        }

    def run(self, instruction: str) -> str:
        """
        Entry point for the Manager Agent. Decomposes the instruction.

        Args:
            instruction: User instruction to process

        Returns:
            JSON string containing decomposed subtasks
        """
        try:
            response = super().run(instruction) # Call the base Agent's run method
            return response
        except Exception as e:
            logger.error(f"Error in Manager Agent execution: {e}")
            return json.dumps({
                "error": str(e),
                "task_analysis": "Failed to process instruction",
                "subtasks": []
            })


class NavigatorAgent(Agent):
    """
    Navigator Agent for PC-Agent framework.
    Plans detailed steps for completing subtasks, using perception data.
    """
    def __init__(self, communication_hub: CommunicationHub, active_perception_module: ActivePerceptionModule,
                 model_name="gemini/gemini-2.0-flash", **kwargs):
        """
        Initializes the Navigator Agent.

        Args:
            communication_hub: Reference to the communication hub
            active_perception_module: Reference to the Active Perception Module
            model_name: Model to use for agent
        """
        super().__init__(
            agent_name="NavigatorAgent",
            system_prompt=self._get_system_prompt(),
            model_name=model_name,
            **kwargs
        )
        self.communication_hub = communication_hub
        self.apm = active_perception_module
        logger.info(f"Initialized Navigator Agent with model {model_name}")

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for the Navigator Agent."""
        return """You are the Navigator Agent in the PC-Agent framework. Your role is to plan the detailed steps needed to achieve a given subtask.

        You will be provided with:
        1. A subtask description and parameters
        2. Current screen information (OCR text, UI elements)

        Your job is to analyze the screen information, determine where the required UI elements are located, and plan the precise actions needed to complete the subtask.

        Output a detailed navigation plan in JSON format:
        ```json
        {
            "subtask_analysis": "Brief analysis of what needs to be done",
            "target_elements": [
                {
                    "element_type": "button/input/link/etc",
                    "identifier": "text or description to identify the element",
                    "confidence": 0.9,
                    "location": [x, y, width, height] or "unknown",
                    "action": "click/type/scroll/etc"
                }
            ],
            "action_plan": [
                {
                    "action_type": "click",
                    "parameters": {"coordinates": [x, y]},
                    "reason": "Explanation of why this action is needed"
                },
                {
                    "action_type": "type",
                    "parameters": {"text": "example search text"},
                    "reason": "Explanation of why this action is needed"
                }
            ],
            "contingency_plans": [
                {
                    "trigger": "If [some condition]",
                    "actions": ["Alternative action 1", "Alternative action 2"]
                }
            ],
            "completion_criteria": "How to determine if subtask is complete"
        }
        ```

        Always be precise in your navigation plan. If you cannot confidently identify a UI element from the screen information provided, indicate that additional perception is needed.
        """

    def plan_subtask(self, subtask: dict, screen_info: dict) -> dict:
        """
        Plans the detailed steps to execute a subtask using screen information.

        Args:
            subtask: Subtask to plan for
            screen_info: Information about the current screen state

        Returns:
            Dictionary containing the detailed navigation plan
        """
        logger.info(f"Planning navigation for subtask: {subtask.get('id')}")
        try: # Added try-except for plan_subtask
            # Prepare input for the agent
            prompt = self._create_planning_prompt(subtask, screen_info)

            # Get navigation plan from the agent
            response = self.run(prompt)
            navigation_plan = safely_parse_json(response)

            # Store navigation plan in communication hub
            subtask_id = subtask.get("id")
            if subtask_id:
                self.communication_hub.update_task_data(f"navigation_plan_{subtask_id}", navigation_plan)

            return navigation_plan
        except Exception as e:
            logger.error(f"Error planning subtask: {e}")
            return {
                "error": str(e),
                "subtask_analysis": "Failed to create navigation plan due to an error.",
                "target_elements": [],
                "action_plan": [],
                "contingency_plans": [],
                "completion_criteria": "Error during planning."
            }


    def _create_planning_prompt(self, subtask: dict, screen_info: dict) -> str:
        """
        Creates the prompt for planning a subtask.

        Args:
            subtask: Subtask to plan for
            screen_info: Information about the current screen state

        Returns:
            Prompt for the agent
        """
        # Extract relevant information from screen_info
        accessible_elements = []
        ocr_text = []

        if "accessibility_tree" in screen_info:
            accessible_elements = screen_info["accessibility_tree"].get("elements", [])

        if "ocr_information" in screen_info:
            ocr_text = screen_info["ocr_information"].get("ocr_results", [])

        # Create prompt with more structure for better agent understanding
        prompt = f"""
        # Navigation Planning Task
        
        ## Subtask Information
        Description: {subtask.get("description", "Unknown subtask")}
        Parameters: {json.dumps(subtask.get("parameters", {}), indent=2)}
        Expected outcome: {subtask.get("expected_outcome", "No expected outcome provided")}

        ## Current Screen Information

        ### Accessibility Elements:
        ```json
        {json.dumps(accessible_elements, indent=2)}
        ```

        ### OCR Text Elements:
        ```json
        {json.dumps(ocr_text, indent=2)}
        ```

        ## Instructions
        Based on this information, create a detailed navigation plan to complete this subtask.
        Include specific UI elements to interact with and precise actions (click, type, etc.).
        
        Your plan should include:
        1. Analysis of what needs to be done
        2. Target UI elements with their locations
        3. Step-by-step action plan with specific parameters
        4. Alternative approaches if the primary plan fails
        5. How to verify the subtask is completed successfully
        
        Format your response as a single JSON object.
        """

        return prompt

    def analyze_navigation_failure(self, subtask: dict, navigation_plan: dict, execution_result: dict) -> dict:
        """
        Analyzes a navigation failure and suggests adjustments.

        Args:
            subtask: Failed subtask
            navigation_plan: Original navigation plan
            execution_result: Results from execution attempt

        Returns:
            Dictionary containing analysis and adjustments
        """
        logger.info(f"Analyzing navigation failure for subtask: {subtask.get('id')}")
        try: # Added try-except for analyze_navigation_failure
            # Prepare input for the agent
            prompt = f"""
            A navigation plan has failed to execute successfully.

            Subtask: {json.dumps(subtask, indent=2)}

            Original Navigation Plan: {json.dumps(navigation_plan, indent=2)}

            Execution Result: {json.dumps(execution_result, indent=2)}

            Please analyze the failure and suggest adjustments to the navigation plan.
            """

            # Get analysis from the agent
            response = self.run(prompt)
            analysis = safely_parse_json(response)

            # Store analysis in communication hub
            subtask_id = subtask.get("id")
            if subtask_id:
                self.communication_hub.update_task_data(f"navigation_failure_analysis_{subtask_id}", analysis)

            return analysis
        except Exception as e:
            logger.error(f"Error analyzing navigation failure: {e}")
            return {
                "error": str(e),
                "reflection_summary": "Failed to analyze navigation failure.",
                "strengths": [],
                "issues_identified": [],
                "pattern_recognition": [],
                "optimization_suggestions": [],
                "learning_opportunities": []
            }


class ExecutorAgent(Agent):
    """
    Executor Agent for PC-Agent framework.
    Executes the planned actions using the active perception module.
    """
    def __init__(self, communication_hub: CommunicationHub, active_perception_module: ActivePerceptionModule,
                 model_name="gemini/gemini-2.0-flash", **kwargs):
        """
        Initializes the Executor Agent.

        Args:
            communication_hub: Reference to the communication hub
            active_perception_module: Reference to the Active Perception Module
            model_name: Model to use for agent
        """
        super().__init__(
            agent_name="ExecutorAgent",
            system_prompt=self._get_system_prompt(),
            model_name=model_name,
            **kwargs
        )
        self.communication_hub = communication_hub
        self.apm = active_perception_module
        logger.info(f"Initialized Executor Agent with model {model_name}")

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for the Executor Agent."""
        return """You are the Executor Agent in the PC-Agent framework. Your role is to execute the planned actions for a subtask by interacting with the computer interface.

        You will be provided with:
        1. A navigation plan with detailed actions
        2. Current screen information

        Your job is to determine the exact parameters needed for each action (e.g., precise coordinates for clicks), execute them, and verify the results.

        For each action, you should:
        1. Analyze the current screen to find the target UI element
        2. Determine the precise parameters for the action
        3. Execute the action
        4. Verify the result

        Output your execution report in JSON format:
        ```json
        {
            "execution_summary": "Brief summary of what was executed",
            "actions_executed": [
                {
                    "action_type": "click",
                    "parameters": {"coordinates": [x, y]},
                    "status": "success/failure",
                    "verification": "Description of what happened after execution"
                }
            ],
            "subtask_status": "completed/failed/partial",
            "issues_encountered": ["Issue 1", "Issue 2"],
            "next_steps": "Recommendation for next steps if needed"
        }
        ```

        Be precise and careful in your execution. If an action fails, try alternative approaches from the contingency plans before reporting failure.
        """

    def execute_navigation_plan(self, subtask: dict, navigation_plan: dict) -> dict:
        """
        Executes a navigation plan for a subtask.

        Args:
            subtask: Subtask to execute
            navigation_plan: Navigation plan to execute

        Returns:
            Dictionary containing execution results
        """
        logger.info(f"Executing navigation plan for subtask: {subtask.get('id')}")
        execution_results = {
            "execution_summary": "Execution started",
            "actions_executed": [],
            "subtask_status": "in_progress",
            "issues_encountered": [],
            "next_steps": "",
            "subtask_id": subtask.get("id")
        }
        try: # Added try-except for execute_navigation_plan
            # Take screenshot for current state
            screenshot_path = create_screenshot()

            # Get current screen information
            screen_info = self.apm.analyze_screen(screenshot_path)
            self.communication_hub.update_perception_data(f"screen_info_{subtask.get('id')}", screen_info) # Store screen info in hub

            # Prepare input for the agent
            prompt = self._create_execution_prompt(subtask, navigation_plan, screen_info)

            # Get execution steps from the agent
            response = self.run(prompt)
            execution_steps = safely_parse_json(response)

            execution_results["execution_summary"] = execution_steps.get("execution_summary", "No execution summary provided")
            execution_results["next_steps"] = execution_steps.get("next_steps", "")


            for step in execution_steps.get("actions_executed", []):
                action_type = step.get("action_type")
                parameters = step.get("parameters", {})

                # Execute the action
                success = self.apm.execute_action(action_type, parameters)

                # Record the result
                step["status"] = "success" if success else "failure"
                execution_results["actions_executed"].append(step)

                # Add issue if action failed
                if not success:
                    execution_results["issues_encountered"].append(f"Failed to execute {action_type} with parameters {parameters}")

            # Determine overall status
            failed_actions = [step for step in execution_results["actions_executed"] if step["status"] == "failure"]
            if not failed_actions and execution_results["actions_executed"]: # Check if actions were actually executed
                execution_results["subtask_status"] = "completed"
            elif len(failed_actions) == len(execution_results["actions_executed"]) and execution_results["actions_executed"]: # Check if actions were actually executed
                execution_results["subtask_status"] = "failed"
            elif execution_results["actions_executed"]: # Check if actions were actually executed
                execution_results["subtask_status"] = "partial"
            else: # No actions were executed, consider it failed or incomplete depending on context
                execution_results["subtask_status"] = "failed" # Or "incomplete" based on desired behavior


            # Store execution results in communication hub
            subtask_id = subtask.get("id")
            if subtask_id:
                self.communication_hub.update_task_data(f"execution_results_{subtask_id}", execution_results)

                # Add to action history
                for step in execution_results["actions_executed"]:
                    self.communication_hub.add_action({
                        "action_type": step.get("action_type"),
                        "parameters": step.get("parameters"),
                        "status": step.get("status"),
                        "subtask_id": subtask_id
                    })

            return execution_results

        except Exception as e:
            logger.error(f"Error executing navigation plan: {e}")
            execution_results["subtask_status"] = "failed"
            execution_results["issues_encountered"].append(f"Exception during execution: {e}")
            return execution_results


    def _create_execution_prompt(self, subtask: dict, navigation_plan: dict, screen_info: dict) -> str:
        """
        Creates the prompt for executing a navigation plan.

        Args:
            subtask: Subtask to execute
            navigation_plan: Navigation plan to execute
            screen_info: Information about the current screen state

        Returns:
            Prompt for the agent
        """
        # Create prompt
        prompt = f"""
        Subtask: {subtask.get("description", "Unknown subtask")}

        Navigation Plan:
        {json.dumps(navigation_plan, indent=2)}

        Current Screen Information:
        {json.dumps(screen_info, indent=2)}

        Based on this information, determine the exact parameters for each action in the navigation plan and execute them.
        Report the results of each action and the overall execution status.
        """

        return prompt


class ReflectorAgent(Agent):
    """
    Reflector Agent for PC-Agent framework.
    Analyzes execution results and provides insights and improvements.
    """
    def __init__(self, communication_hub: CommunicationHub, model_name="gemini/gemini-2.0-flash", **kwargs):
        """
        Initializes the Reflector Agent.

        Args:
            communication_hub: Reference to the communication hub
            model_name: Model to use for agent
        """
        super().__init__(
            agent_name="ReflectorAgent",
            system_prompt=self._get_system_prompt(),
            model_name=model_name,
            **kwargs
        )
        self.communication_hub = communication_hub
        logger.info(f"Initialized Reflector Agent with model {model_name}")

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for the Reflector Agent."""
        return """You are the Reflector Agent in the PC-Agent framework. Your role is to analyze the execution of subtasks, identify patterns, and provide insights and improvements.

        You will be provided with:
        1. A subtask description and parameters
        2. The navigation plan used for the subtask
        3. The execution results
        4. Previous reflection history (if available)

        Your job is to critically analyze what happened, identify any issues or inefficiencies, and suggest improvements.

        Output your reflection in JSON format:
        ```json
        {
            "reflection_summary": "Brief summary of your reflection",
            "strengths": ["Strength 1", "Strength 2"],
            "issues_identified": [
                {
                    "issue": "Description of an issue",
                    "impact": "Impact of the issue",
                    "improvement": "Suggested improvement"
                }
            ],
            "pattern_recognition": ["Pattern 1", "Pattern 2"],
            "optimization_suggestions": ["Suggestion 1", "Suggestion 2"],
            "learning_opportunities": ["Learning 1", "Learning 2"]
        }
        ```

        Be insightful and constructive in your reflection. Focus not just on issues but also on what worked well and patterns that could help improve future subtasks.
        """

    def reflect_on_execution(self, subtask: dict, navigation_plan: dict, execution_results: dict) -> dict:
        """
        Reflects on the execution of a subtask.

        Args:
            subtask: Subtask that was executed
            navigation_plan: Navigation plan that was used
            execution_results: Results from execution

        Returns:
            Dictionary containing reflection
        """
        logger.info(f"Reflecting on execution of subtask: {subtask.get('id')}")
        try: # Added try-except for reflect_on_execution
            # Get previous reflections for context
            previous_reflections = self.communication_hub.get_latest_reflections()

            # Prepare input for the agent
            prompt = self._create_reflection_prompt(subtask, navigation_plan, execution_results, previous_reflections)

            # Get reflection from the agent
            response = self.run(prompt)
            reflection = safely_parse_json(response)

            # Store reflection in communication hub
            subtask_id = subtask.get("id")
            if subtask_id:
                self.communication_hub.update_task_data(f"reflection_{subtask_id}", reflection)

                # Add to reflection history
                self.communication_hub.add_reflection({
                    "subtask_id": subtask_id,
                    "reflection_result": reflection,
                    "subtask_status": execution_results.get("subtask_status")
                })

            return reflection
        except Exception as e:
            logger.error(f"Error reflecting on execution: {e}")
            return {
                "error": str(e),
                "reflection_summary": "Reflection failed due to an error.",
                "strengths": [],
                "issues_identified": [],
                "pattern_recognition": [],
                "optimization_suggestions": [],
                "learning_opportunities": []
            }


    def _create_reflection_prompt(self, subtask: dict, navigation_plan: dict, execution_results: dict, previous_reflections: list) -> str:
        """
        Creates the prompt for reflecting on execution.

        Args:
            subtask: Subtask that was executed
            navigation_plan: Navigation plan that was used
            execution_results: Results from execution
            previous_reflections: Previous reflections for context

        Returns:
            Prompt for the agent
        """
        # Create prompt
        prompt = f"""
        Subtask: {json.dumps(subtask, indent=2)}

        Navigation Plan: {json.dumps(navigation_plan, indent=2)}

        Execution Results: {json.dumps(execution_results, indent=2)}

        Previous Reflections: {json.dumps(previous_reflections[-3:] if previous_reflections else [], indent=2)}

        Based on this information, provide a detailed reflection on the execution of this subtask.
        Identify what went well, what issues occurred, and suggest improvements for future subtasks.
        """

        return prompt


class PCAgent:
    """
    Main PC-Agent class that orchestrates the multi-agent workflow.
    Provides a unified interface for PC task automation.
    """
    def __init__(self, model_name="gemini/gemini-2.0-flash"):
        """
        Initializes the PC-Agent with its component agents and modules.

        Args:
            model_name: Model to use for agents
        """
        # Initialize core components
        self.communication_hub = CommunicationHub()
        self.active_perception_module = ActivePerceptionModule(model_name=model_name)

        # Initialize agents
        self.manager_agent = ManagerAgent(self.communication_hub, model_name=model_name)
        self.navigator_agent = NavigatorAgent(self.communication_hub, self.active_perception_module, model_name=model_name)
        self.executor_agent = ExecutorAgent(self.communication_hub, self.active_perception_module, model_name=model_name)
        self.reflector_agent = ReflectorAgent(self.communication_hub, model_name=model_name)

        # Set up state tracking
        self._task_in_progress = False
        self._current_subtask = None
        self._conversation_history = Conversation()

        logger.info(f"Initialized PC-Agent with model {model_name}")

    def execute_task(self, instruction: str) -> dict:
        """
        Main entry point for executing a PC task.

        Args:
            instruction: User instruction to execute

        Returns:
            Dictionary containing task results
        """
        logger.info(f"Starting execution of task: {instruction}")

        # Reset state for new task
        self.communication_hub.clear_all()
        self._task_in_progress = True

        try:
            # Step 1: Manager decomposes instruction into subtasks
            subtasks_data = self.manager_agent.decompose_instruction(instruction)

            # Record initial task data
            self._conversation_history.add_message("user", instruction)
            self._conversation_history.add_message("system", f"Task decomposed into {len(subtasks_data.get('subtasks', []))} subtasks.")

            # Step 2: Process each subtask sequentially
            while self._task_in_progress:
                # Get next subtask to process
                next_subtask = self.manager_agent.get_next_subtask()

                if not next_subtask:
                    # No more subtasks to process
                    self._task_in_progress = False
                    break

                # Update current subtask
                self._current_subtask = next_subtask
                subtask_id = next_subtask.get("id")

                # Update subtask status
                self.manager_agent.update_subtask_status(subtask_id, SubtaskStatus.IN_PROGRESS)

                try:
                    # Step 3: Take screenshot and analyze screen
                    screenshot_path = create_screenshot()
                    screen_info = self.active_perception_module.analyze_screen(screenshot_path)
                    self.communication_hub.update_perception_data(f"screen_info_{subtask_id}", screen_info)

                    # Step 4: Navigator plans detailed steps
                    navigation_plan = self.navigator_agent.plan_subtask(next_subtask, screen_info)

                    # Step 5: Executor executes the plan
                    execution_results = self.executor_agent.execute_navigation_plan(next_subtask, navigation_plan)

                    # Step 6: Update subtask status based on execution results
                    if execution_results.get("subtask_status") == "completed":
                        self.manager_agent.update_subtask_status(subtask_id, SubtaskStatus.SUCCEEDED)
                    elif execution_results.get("subtask_status") == "failed":
                        self.manager_agent.update_subtask_status(subtask_id, SubtaskStatus.FAILED)
                    else:
                        # Partial completion or other state
                        self.manager_agent.update_subtask_status(subtask_id, SubtaskStatus.FAILED) # Modified to FAILED for partial/unknown

                    # Step 7: Reflector analyzes execution
                    reflection = self.reflector_agent.reflect_on_execution(next_subtask, navigation_plan, execution_results)

                    # Record subtask execution details
                    self._conversation_history.add_message("system", f"Subtask '{next_subtask.get('description')}' execution status: {execution_results.get('subtask_status')}")

                except Exception as e:
                    logger.error(f"Error processing subtask {subtask_id}: {e}")
                    self.manager_agent.update_subtask_status(subtask_id, SubtaskStatus.FAILED)
                    self._conversation_history.add_message("system", f"Error processing subtask '{next_subtask.get('description')}': {str(e)}")
                    continue # Continue to next subtask even if one fails in try block

            # Compile final results
            task_progress = self.manager_agent.get_task_progress()
            final_result = {
                "task_instruction": instruction,
                "task_progress": task_progress, # Ensure task_progress is always included
                "task_completed": task_progress["progress_percentage"] == 100.0,
                "action_history": self.communication_hub.get_latest_actions(100),  # Get up to 100 actions
                "reflection_summary": self._compile_reflection_summary()
            }

            logger.info(f"Task execution completed with progress: {task_progress['progress_percentage']}%")
            return final_result

        except Exception as e:
            logger.error(f"Error executing task: {e}")
            self._task_in_progress = False
            return {
                "task_instruction": instruction,
                "error": str(e),
                "task_completed": False,
                "task_progress": { # Modified to include all keys with default 0 values
                    "progress_percentage": 0,
                    "completed_subtasks": 0,
                    "failed_subtasks": 0,
                    "pending_subtasks": 0,
                    "in_progress_subtasks": 0,
                    "total_subtasks": 0,
                    "subtasks_status": [],
                    "task_analysis": "Task execution failed"
                }
            }

    def _compile_reflection_summary(self) -> dict:
        """
        Compiles a summary of all reflections from the task.

        Returns:
            Dictionary containing reflection summary
        """
        all_reflections = self.communication_hub.get_latest_reflections(100)  # Get up to 100 reflections

        # If no reflections available, create a simple default summary
        if not all_reflections:
            logger.warning("No reflections available for summary. Creating default reflection summary.")
            return {
                "reflection_summary": "No detailed reflections available for this task execution.",
                "strengths": ["Task decomposition was performed successfully."],
                "issues_identified": [],
                "pattern_recognition": [],
                "optimization_suggestions": ["Consider running a complete task execution to generate detailed reflections."],
                "learning_opportunities": []
            }

        # Prepare input for the reflector agent
        prompt = f"""
        # Task Reflection Summary
        
        Task execution has completed. Please analyze the following reflections and provide a comprehensive summary.

        ## Task Analysis
        {self.communication_hub.get_task_data("task_analysis", "No task analysis available")}
        
        ## Individual Reflections
        ```json
        {json.dumps(all_reflections, indent=2)}
        ```
        
        ## Task Progress
        ```json
        {json.dumps(self.manager_agent.get_task_progress(), indent=2)}
        ```

        Provide a summary that highlights:
        - Overall success or failure of the task
        - Common patterns across subtasks
        - Key strengths in the execution
        - Major issues encountered
        - Optimization opportunities
        - Learning points for future tasks
        
        Format your response as a single JSON object with these sections.   
        """

        try:
            # Get summary from the reflector agent
            response = self.reflector_agent.run(prompt)
            return extract_json_from_response(response)
        except Exception as e:
            logger.error(f"Error compiling reflection summary: {e}")
            tb_str = traceback.format_exception(*sys.exc_info())
            logger.debug(f"Traceback: {''.join(tb_str)}")
            return {
                "error": str(e),
                "reflection_summary": "Failed to compile reflection summary due to an error.",
                "strengths": [],
                "issues_identified": [{"issue": "Error during reflection compilation", "impact": "Missing insights", "improvement": "Check logs for details"}],
                "pattern_recognition": [],
                "optimization_suggestions": [],
                "learning_opportunities": []
            }

    def get_task_progress(self) -> dict:
        """
        Gets the current progress of the task.

        Returns:
            Dictionary containing progress information
        """
        return self.manager_agent.get_task_progress() if self._task_in_progress else {"progress_percentage": 0}

    def get_conversation_history(self) -> Conversation:
        """
        Gets the conversation history for the task.

        Returns:
            Conversation object containing the history
        """
        return self._conversation_history

    def cancel_task(self) -> None:
        """Cancels the current task."""
        self._task_in_progress = False
        logger.info("Task execution cancelled.")

    def retry_subtask(self, subtask_id: str) -> dict:
        """
        Retries a failed subtask.

        Args:
            subtask_id: ID of the subtask to retry

        Returns:
            Dictionary containing retry results
        """
        logger.info(f"Retrying subtask: {subtask_id}")
        if not isinstance(subtask_id, str): # Added type check for subtask_id
            logger.error(f"subtask_id must be a string, but got {type(subtask_id)}: {subtask_id}")
            return {"error": "Invalid subtask_id", "success": False}

        # Get the subtask
        subtask = self.communication_hub.get_task_data(f"subtask_{subtask_id}")
        if not subtask:
            logger.error(f"Subtask {subtask_id} not found")
            return {"error": f"Subtask {subtask_id} not found", "success": False}

        # Update subtask status
        self.manager_agent.update_subtask_status(subtask_id, SubtaskStatus.RETRYING)

        try:
            # Take a new screenshot and analyze screen
            screenshot_path = create_screenshot()
            screen_info = self.active_perception_module.analyze_screen(screenshot_path)
            self.communication_hub.update_perception_data(f"screen_info_{subtask_id}", screen_info)

            # Get previous navigation plan for context
            old_navigation_plan = self.communication_hub.get_task_data(f"navigation_plan_{subtask_id}", {})

            # Create a new navigation plan with awareness of the previous failure
            prompt = f"""
            This is a retry of a previously failed subtask.

            Subtask: {json.dumps(subtask, indent=2)}

            Previous navigation plan that failed: {json.dumps(old_navigation_plan, indent=2)}

            Previous execution results: {json.dumps(self.communication_hub.get_task_data(f"execution_results_{subtask_id}", {}), indent=2)}

            Current screen information: {json.dumps(screen_info, indent=2)}

            Please create an improved navigation plan that addresses the previous failure points.
            """

            response = self.navigator_agent.run(prompt)
            new_navigation_plan = safely_parse_json(response)

            # Update navigation plan
            self.communication_hub.update_task_data(f"navigation_plan_{subtask_id}", new_navigation_plan)

            # Execute the new plan
            execution_results = self.executor_agent.execute_navigation_plan(subtask, new_navigation_plan)

            # Update subtask status
            if execution_results.get("subtask_status") == "completed":
                self.manager_agent.update_subtask_status(subtask_id, SubtaskStatus.SUCCEEDED)
                success = True
            else:
                self.manager_agent.update_subtask_status(subtask_id, SubtaskStatus.FAILED)
                success = False

            # Reflect on retry
            self.reflector_agent.reflect_on_execution(subtask, new_navigation_plan, execution_results)

            return {
                "subtask_id": subtask_id,
                "success": success,
                "execution_results": execution_results
            }

        except Exception as e:
            logger.error(f"Error retrying subtask {subtask_id}: {e}")
            self.manager_agent.update_subtask_status(subtask_id, SubtaskStatus.FAILED)
            return {"error": str(e), "success": False}

    def generate_task_report(self) -> str:
        """
        Generates a comprehensive report of the task execution.

        Returns:
            String containing formatted report
        """
        task_progress = self.manager_agent.get_task_progress()
        task_analysis = task_progress.get("task_analysis", "No task analysis available")
        reflections = self.communication_hub.get_latest_reflections(10)
        actions = self.communication_hub.get_latest_actions(20)

        prompt = f"""
        Please generate a comprehensive task report based on the following information:

        Task Analysis: {task_analysis}

        Task Progress: {json.dumps(task_progress, indent=2)}

        Recent Actions: {json.dumps(actions, indent=2)}

        Reflections: {json.dumps(reflections, indent=2)}

        The report should be formatted in Markdown and include:
        1. Executive Summary
        2. Task Breakdown and Progress
        3. Key Actions Performed
        4. Challenges Encountered
        5. Learnings and Insights
        6. Recommendations for Future Tasks
        """
        try: # Added try-except for generate_task_report
            report = self.reflector_agent.run(prompt)
            return report
        except Exception as e:
            logger.error(f"Error generating task report: {e}")
            return "Error generating task report. Please check logs for details."

    def save_task_history(self, filepath: str) -> bool:
        """
        Saves the complete task history to a file.

        Args:
            filepath: Path to save the history

        Returns:
            Boolean indicating success or failure
        """
        if not isinstance(filepath, str): # Added type check for filepath
            logger.error(f"filepath must be a string, but got {type(filepath)}: {filepath}")
            return False
        try:
            task_data = {
                "task_analysis": self.communication_hub.get_task_data("task_analysis"),
                "subtasks": [],
                "actions": self.communication_hub.get_latest_actions(1000),
                "reflections": self.communication_hub.get_latest_reflections(1000),
                "task_progress": self.manager_agent.get_task_progress()
            }

            # Get all subtasks
            subtask_order = self.communication_hub.get_task_data("subtask_order", [])
            for subtask_id in subtask_order:
                subtask = self.communication_hub.get_task_data(f"subtask_{subtask_id}")
                if subtask:
                    navigation_plan = self.communication_hub.get_task_data(f"navigation_plan_{subtask_id}", {})
                    execution_results = self.communication_hub.get_task_data(f"execution_results_{subtask_id}", {})
                    reflection = self.communication_hub.get_task_data(f"reflection_{subtask_id}", {})
                    task_data["subtasks"].append({
                        "subtask": subtask,
                        "navigation_plan": navigation_plan,
                        "execution_results": execution_results,
                        "reflection": reflection
                    })

            # Save to file
            with open(filepath, 'w') as f:
                json.dump(task_data, f, indent=2)

            logger.info(f"Task history saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving task history: {e}")
            return False

    def load_task_history(self, filepath: str) -> bool:
        """
        Loads task history from a file.

        Args:
            filepath: Path to load the history from

        Returns:
            Boolean indicating success or failure
        """
        if not isinstance(filepath, str): # Added type check for filepath
            logger.error(f"filepath must be a string, but got {type(filepath)}: {filepath}")
            return False
        try:
            with open(filepath, 'r') as f:
                task_data = json.load(f)

            # Reset current state
            self.communication_hub.clear_all()

            # Load task analysis
            if "task_analysis" in task_data:
                self.communication_hub.update_task_data("task_analysis", task_data["task_analysis"])

            # Load subtasks
            subtask_order = []
            for subtask_data in task_data.get("subtasks", []):
                subtask = subtask_data.get("subtask")
                if subtask and "id" in subtask:
                    subtask_id = subtask["id"]
                    subtask_order.append(subtask_id)

                    # Store subtask and related data
                    self.communication_hub.update_task_data(f"subtask_{subtask_id}", subtask)

                    if "navigation_plan" in subtask_data:
                        self.communication_hub.update_task_data(f"navigation_plan_{subtask_id}", subtask_data["navigation_plan"])

                    if "execution_results" in subtask_data:
                        self.communication_hub.update_task_data(f"execution_results_{subtask_id}", subtask_data["execution_results"])

                    if "reflection" in subtask_data:
                        self.communication_hub.update_task_data(f"reflection_{subtask_id}", subtask_data["reflection"])

            # Store subtask order
            self.communication_hub.update_task_data("subtask_order", subtask_order)

            # Load actions and reflections
            for action in task_data.get("actions", []):
                self.communication_hub.add_action(action)

            for reflection in task_data.get("reflections", []):
                self.communication_hub.add_reflection(reflection)

            logger.info(f"Task history loaded from {filepath}")
            return True
        except FileNotFoundError: # Handle file not found specifically
            logger.error(f"Task history file not found: {filepath}")
            return False
        except json.JSONDecodeError: # Handle JSON decode errors
            logger.error(f"Error decoding JSON from file: {filepath}. File might be corrupted.")
            return False
        except Exception as e:
            logger.error(f"Error loading task history: {e}")
            return False


def demo_pc_agent(instruction: str = None):
    """
    Runs a demonstration of the PC-Agent with a sample instruction.

    Args:
        instruction: Optional instruction to execute
    """
    if not instruction:
        instruction = "Open Notepad, type 'Hello World from PC-Agent', save the file to the desktop as 'pc-agent-demo.txt', and close Notepad."

    print(f"🤖 PC-Agent Demo 🤖\n")
    print(f"Instruction: {instruction}\n")

    # Initialize PC-Agent
    agent = PCAgent()

    # Execute the task
    print("Executing task...\n")
    result = agent.execute_task(instruction)

    # Display results
    print("\n--- Task Execution Results ---\n")
    print(f"Task Completed: {result['task_completed']}")
    if 'task_progress' in result: # Check if task_progress key exists before accessing
        print(f"Progress: {result['task_progress']['progress_percentage']:.1f}%")
        if 'completed_subtasks' in result['task_progress'] and 'total_subtasks' in result['task_progress']: # Check for key existence before access
            print(f"Completed Subtasks: {result['task_progress']['completed_subtasks']}/{result['task_progress']['total_subtasks']}")
        else:
            print("Completed Subtasks information is not available in task progress.")


        print("\n--- Subtask Status ---\n")
        if 'subtasks_status' in result['task_progress']: # Check if subtasks_status key exists
            for subtask in result['task_progress']['subtasks_status']:
                status_icon = "✅" if subtask['status'] == SubtaskStatus.SUCCEEDED.value else "❌"
                print(f"{status_icon} {subtask['description']}")
        else:
            print("No subtask status information available.")
    else:
        print("Task progress information is not available due to an error.")

    # Generate and display report
    print("\n--- Task Report ---\n")
    report = agent.generate_task_report()
    print(report)

    # Save history
    history_file = f"pc-agent-history-{int(time.time())}.json"
    agent.save_task_history(history_file)
    print(f"\nTask history saved to {history_file}")

if __name__ == "__main__":
    import argparse

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="PC-Agent: Multi-agent system for PC task automation")
    parser.add_argument("--instruction", type=str, help="Instruction to execute")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file", type=str, help="Log file path")
    parser.add_argument("--load-history", type=str, help="Load task history from file")
    parser.add_argument("--save-history", type=str, help="Save task history to file")
    parser.add_argument("--model", type=str, default="gemini/gemini-2.0-flash", 
                        help="Model to use for agents")
    parser.add_argument("--debug-mode", action="store_true", help="Enable debug mode with additional logging")

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level, log_file=args.log_file)

    if args.debug_mode:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled with verbose logging")
    
    # Ensure dependencies are available
    try:
        import pyautogui
        logger.info("pyautogui is available for system automation")
    except ImportError:
        logger.warning("pyautogui not found. Install it with: pip install pyautogui")
        logger.warning("Running in simulation mode - actions will be placeholders")

    # Report agent initialization
    logger.info(f"Using model: {args.model}")

    if args.load_history:
        # Load and display history
        agent = PCAgent(model_name=args.model)
        if agent.load_task_history(args.load_history):
            progress = agent.get_task_progress()
            print(f"Loaded task history from {args.load_history}")
            print(f"Task progress: {progress.get('progress_percentage', 0):.1f}%")
            print(f"Subtasks: {progress.get('completed_subtasks', 0)}/{progress.get('total_subtasks', 0)} completed")
        else:
            print(f"Failed to load task history from {args.load_history}")
    elif args.instruction:
        # Execute specific instruction
        agent = PCAgent(model_name=args.model)
        result = agent.execute_task(args.instruction)

        # Save history if specified
        if args.save_history:
            agent.save_task_history(args.save_history)
            print(f"Task history saved to {args.save_history}")
    else:
        # Run demo
        demo_pc_agent()
