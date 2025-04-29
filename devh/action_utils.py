"""
Action Utilities for PC-Agent Framework

Provides helper functions for executing actions on the system
with robust error handling and validation.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

def validate_action_parameters(action_type: str, parameters: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validates parameters for an action based on its type.
    
    Args:
        action_type: Type of action to validate parameters for
        parameters: Dictionary of parameters to validate
        
    Returns:
        Tuple of (is_valid, validation_message)
    """
    if action_type == "click":
        if "coordinates" not in parameters:
            return False, "Missing 'coordinates' parameter for click action"
        coordinates = parameters.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) != 2:
            return False, f"Invalid coordinates format: {coordinates}. Expected [x, y]."
        return True, "Valid click parameters"
        
    elif action_type == "type":
        if "text" not in parameters:
            return False, "Missing 'text' parameter for type action"
        text = parameters.get("text")
        if not isinstance(text, str):
            return False, f"Invalid text format: {text}. Expected string."
        return True, "Valid type parameters"
        
    elif action_type == "wait":
        seconds = parameters.get("seconds", 1)
        if not isinstance(seconds, (int, float)) or seconds < 0:
            return False, f"Invalid wait time: {seconds}. Expected non-negative number."
        return True, "Valid wait parameters"
    
    elif action_type == "keyboard_shortcut":
        if "keys" not in parameters:
            return False, "Missing 'keys' parameter for keyboard shortcut action"
        keys = parameters.get("keys")
        if not isinstance(keys, (list, str)) or (isinstance(keys, list) and not keys):
            return False, f"Invalid keys format: {keys}. Expected non-empty string or list."
        return True, "Valid keyboard shortcut parameters"
    
    elif action_type == "scroll":
        amount = parameters.get("amount", 0)
        if not isinstance(amount, int):
            return False, f"Invalid scroll amount: {amount}. Expected integer."
        return True, "Valid scroll parameters"
    
    elif action_type == "drag":
        if "start_coordinates" not in parameters or "end_coordinates" not in parameters:
            return False, "Missing 'start_coordinates' or 'end_coordinates' parameter for drag action"
        start = parameters.get("start_coordinates")
        end = parameters.get("end_coordinates")
        if not isinstance(start, list) or len(start) != 2:
            return False, f"Invalid start coordinates format: {start}. Expected [x, y]."
        if not isinstance(end, list) or len(end) != 2:
            return False, f"Invalid end coordinates format: {end}. Expected [x, y]."
        return True, "Valid drag parameters"
        
    return True, f"No validation implemented for action type: {action_type}"

def execute_action_with_retry(action_type: str, parameters: Dict[str, Any], max_retries: int = 2) -> Dict[str, Any]:
    """
    Executes an action with retry logic and detailed error reporting.
    
    Args:
        action_type: Type of action to execute
        parameters: Parameters for the action
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary with execution results
    """
    result = {
        "action_type": action_type,
        "parameters": parameters,
        "success": False,
        "attempts": 0,
        "error": None
    }
    
    # Validate parameters first
    is_valid, validation_message = validate_action_parameters(action_type, parameters)
    if not is_valid:
        result["error"] = validation_message
        logger.warning(f"Action validation failed: {validation_message}")
        return result
    
    # Try to execute with retries
    attempts = 0
    while attempts <= max_retries:
        attempts += 1
        result["attempts"] = attempts
        
        try:
            # Import pyautogui only when needed
            try:
                import pyautogui
                pyautogui.PAUSE = 0.5  # Add pause between actions for stability
            except ImportError:
                logger.warning("pyautogui not installed, using placeholder actions")
                time.sleep(0.5)
                if attempts == max_retries:
                    result["success"] = True
                    result["note"] = "Using placeholder actions (pyautogui not installed)"
                    return result
                continue
                
            # Execute based on action type
            if action_type == "click":
                x, y = parameters["coordinates"]
                pyautogui.click(x, y)
                result["success"] = True
                logger.info(f"Clicked at coordinates {x}, {y}")
                
            elif action_type == "type":
                text = parameters["text"]
                pyautogui.typewrite(text)
                result["success"] = True
                logger.info(f"Typed text: {text}")
                
            elif action_type == "wait":
                seconds = parameters.get("seconds", 1)
                time.sleep(seconds)
                result["success"] = True
                logger.info(f"Waited for {seconds} seconds")
                
            elif action_type == "keyboard_shortcut":
                keys = parameters["keys"]
                if isinstance(keys, list):
                    pyautogui.hotkey(*keys)
                else:
                    pyautogui.press(keys)
                result["success"] = True
                logger.info(f"Pressed keyboard shortcut: {keys}")
                
            elif action_type == "scroll":
                amount = parameters.get("amount", 0)
                pyautogui.scroll(amount)
                result["success"] = True
                logger.info(f"Scrolled by amount: {amount}")
                
            elif action_type == "drag":
                start_x, start_y = parameters["start_coordinates"]
                end_x, end_y = parameters["end_coordinates"]
                pyautogui.moveTo(start_x, start_y)
                pyautogui.dragTo(end_x, end_y, button='left', duration=0.5)
                result["success"] = True
                logger.info(f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})")
                
            else:
                result["error"] = f"Unsupported action type: {action_type}"
                logger.warning(result["error"])
                return result
                
            # If we got here, the action succeeded
            break
                
        except Exception as e:
            logger.error(f"Action execution failed (attempt {attempts}/{max_retries+1}): {str(e)}")
            result["error"] = str(e)
            
            # Wait briefly before retrying
            time.sleep(0.5)
    
    return result
