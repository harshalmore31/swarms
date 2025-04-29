"""
PC-Agent Utilities
Helper functions and utilities for the PC-Agent framework.
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Screen capture and analysis utilities
def capture_region_screenshot(region=None, filename=None):
    """
    Captures a screenshot of a specific region instead of the full screen.
    
    Args:
        region: Tuple of (x, y, width, height) or None for full screen
        filename: Optional filename to save screenshot
        
    Returns:
        Path to the saved screenshot
    """
    try:
        import pyautogui
        if not filename:
            filename = f"region_screenshot_{int(time.time())}.png"
        
        screenshot_path = Path(filename)
        
        if region:
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()
            
        screenshot.save(str(screenshot_path))
        logger.info(f"Region screenshot saved to {screenshot_path}")
        return str(screenshot_path)
    except ImportError:
        logger.warning("pyautogui not installed, using placeholder for screenshot")
        return "placeholder_screenshot.png"
    except Exception as e:
        logger.error(f"Error taking region screenshot: {e}")
        return "error_screenshot.png"

def compare_screenshots(screenshot1: str, screenshot2: str) -> float:
    """
    Compares two screenshots and returns a similarity score.
    
    Args:
        screenshot1: Path to first screenshot
        screenshot2: Path to second screenshot
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    try:
        from PIL import Image
        import numpy as np
        
        # Open images
        img1 = Image.open(screenshot1)
        img2 = Image.open(screenshot2)
        
        # Resize to same dimensions if needed
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)
        
        # Convert to numpy arrays
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        
        # Calculate similarity (using mean absolute difference)
        max_diff = 255 * 3  # maximum possible difference per pixel (R,G,B)
        diff = np.mean(np.abs(arr1 - arr2)) / max_diff
        similarity = 1.0 - diff
        
        logger.info(f"Screenshot similarity: {similarity:.4f}")
        return similarity
    except ImportError:
        logger.warning("PIL or numpy not installed, cannot compare screenshots")
        return 0.5  # Return middle value
    except Exception as e:
        logger.error(f"Error comparing screenshots: {e}")
        return 0.0

# Element detection and interaction utilities
def find_element_by_image(template_path: str, screenshot_path: str = None, confidence: float = 0.8) -> Optional[dict]:
    """
    Finds an element on the screen by matching a template image.
    
    Args:
        template_path: Path to template image to match
        screenshot_path: Path to screenshot or None to take a new screenshot
        confidence: Minimum confidence threshold (0.0-1.0)
        
    Returns:
        Dictionary with coordinates or None if not found
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image
        
        # Take screenshot if not provided
        if not screenshot_path:
            from swarms.structs.pc_agent import create_screenshot
            screenshot_path = create_screenshot()
        
        # Read images
        screenshot = cv2.imread(screenshot_path)
        template = cv2.imread(template_path)
        
        if screenshot is None or template is None:
            logger.error(f"Failed to read images: {screenshot_path}, {template_path}")
            return None
        
        # Template matching
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val < confidence:
            logger.info(f"Template match not found (confidence: {max_val:.2f} < threshold: {confidence:.2f})")
            return None
        
        # Get coordinates
        h, w = template.shape[:2]
        x, y = max_loc
        center_x = x + w // 2
        center_y = y + h // 2
        
        logger.info(f"Template match found at ({center_x}, {center_y}) with confidence: {max_val:.2f}")
        
        return {
            "top_left": (x, y),
            "bottom_right": (x + w, y + h),
            "center": (center_x, center_y),
            "confidence": float(max_val),
            "dimensions": (w, h)
        }
    except ImportError:
        logger.warning("OpenCV not installed, cannot perform image matching")
        return None
    except Exception as e:
        logger.error(f"Error in template matching: {e}")
        return None

def wait_for_element(checker_func, timeout: int = 10, interval: float = 0.5) -> Optional[dict]:
    """
    Waits for an element to appear on screen using a checker function.
    
    Args:
        checker_func: Function that returns element data or None if not found
        timeout: Maximum time to wait in seconds
        interval: How often to check in seconds
        
    Returns:
        Element data or None if timeout
    """
    start_time = time.time()
    end_time = start_time + timeout
    
    while time.time() < end_time:
        # Call the checker function
        element_data = checker_func()
        
        # If element found, return it
        if element_data:
            logger.info(f"Element found after {time.time() - start_time:.2f} seconds")
            return element_data
        
        # Sleep for the interval
        time.sleep(interval)
    
    logger.warning(f"Element not found within timeout period ({timeout} seconds)")
    return None

def find_text_on_screen(text: str, screenshot_path: str = None, exact_match: bool = False) -> Optional[dict]:
    """
    Finds text on the screen using OCR.
    
    Args:
        text: Text to find
        screenshot_path: Path to screenshot or None to take a new screenshot
        exact_match: Whether to require an exact match
        
    Returns:
        Dictionary with text location or None if not found
    """
    try:
        from swarms.structs.pc_agent import create_screenshot, ActivePerceptionModule
        
        # Take screenshot if not provided
        if not screenshot_path:
            screenshot_path = create_screenshot()
        
        # Initialize perception module
        apm = ActivePerceptionModule(enable_ocr=True, enable_accessibility_tree=False)
        
        # Find text element
        results = apm.find_element_by_text(screenshot_path, text, exact_match=exact_match)
        
        # Return first match if any
        if results.get("match_count", 0) > 0:
            logger.info(f"Found text '{text}' on screen")
            return results["matches"][0]
        
        logger.info(f"Text '{text}' not found on screen")
        return None
    except Exception as e:
        logger.error(f"Error finding text on screen: {e}")
        return None

def safe_click(x: int, y: int) -> bool:
    """
    Safely clicks at specified coordinates with error handling.
    
    Args:
        x: X coordinate
        y: Y coordinate
    
    Returns:
        Boolean indicating success
    """
    try:
        import pyautogui
        pyautogui.click(x=x, y=y)
        logger.info(f"Clicked at ({x}, {y})")
        return True
    except ImportError:
        logger.warning("pyautogui not installed, cannot click")
        return False
    except Exception as e:
        logger.error(f"Error clicking at ({x}, {y}): {e}")
        return False

def safe_type(text: str, interval: float = 0.05) -> bool:
    """
    Safely types text with error handling.
    
    Args:
        text: Text to type
        interval: Time between keystrokes
        
    Returns:
        Boolean indicating success
    """
    try:
        import pyautogui
        pyautogui.write(text, interval=interval)
        logger.info(f"Typed text: '{text}'")
        return True
    except ImportError:
        logger.warning("pyautogui not installed, cannot type text")
        return False
    except Exception as e:
        logger.error(f"Error typing text: {e}")
        return False

def safe_hotkey(*keys) -> bool:
    """
    Safely executes keyboard shortcut with error handling.
    
    Args:
        *keys: Keys for the hotkey (e.g., 'ctrl', 'c')
        
    Returns:
        Boolean indicating success
    """
    try:
        import pyautogui
        pyautogui.hotkey(*keys)
        logger.info(f"Executed hotkey: {keys}")
        return True
    except ImportError:
        logger.warning("pyautogui not installed, cannot execute hotkey")
        return False
    except Exception as e:
        logger.error(f"Error executing hotkey: {e}")
        return False

# Task execution monitoring and helpers
def retry_with_backoff(func, max_attempts=3, initial_delay=1.0):
    """
    Retries a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay between retries in seconds
        
    Returns: