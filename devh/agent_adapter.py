"""
Agent Adapter for PC-Agent Framework

Provides utility functions for connecting with different agent frameworks
and handling properly formatted outputs.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extracts JSON data from an agent response that may contain formatting or other text.
    
    Args:
        response: String response from agent that may contain JSON data
        
    Returns:
        Dictionary parsed from JSON in the response
    """
    try:
        # Try direct parsing first
        return json.loads(response)
    except json.JSONDecodeError:
        # If direct parsing fails, look for JSON in code blocks or other formats
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_pattern, response)
        
        # Try each match until we find valid JSON
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
                
        # Try finding JSON objects without code blocks
        json_pattern = r'{[\s\S]*}'
        matches = re.findall(json_pattern, response)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
                
        # If we couldn't extract valid JSON, return the response as a string
        logger.warning(f"Couldn't extract JSON from response: {response[:100]}...")
        return {"raw_response": response}
    
def format_agent_prompt(prompt_type: str, data: Dict[str, Any]) -> str:
    """
    Formats data into a prompt suitable for specific agent types.
    
    Args:
        prompt_type: Type of prompt to format (e.g., "navigation", "execution")
        data: Data to include in the prompt
        
    Returns:
        Formatted prompt string
    """
    if prompt_type == "navigation":
        template = """
        # Navigation Planning
        
        ## Subtask
        {subtask_description}
        
        ## Parameters
        {subtask_parameters}
        
        ## Expected Outcome
        {expected_outcome}
        
        ## Current Screen Information
        {screen_info}
        
        Plan detailed steps to accomplish this subtask based on the current screen state.
        Focus on identifying UI elements and specifying precise actions (click, type, etc.).
        """
        
        return template.format(
            subtask_description=data.get("description", "No description provided"),
            subtask_parameters=json.dumps(data.get("parameters", {}), indent=2),
            expected_outcome=data.get("expected_outcome", "No expected outcome provided"),
            screen_info=json.dumps(data.get("screen_info", {}), indent=2)
        )
    
    elif prompt_type == "execution":
        template = """
        # Action Execution
        
        ## Subtask
        {subtask_description}
        
        ## Navigation Plan
        {navigation_plan}
        
        ## Current Screen Information
        {screen_info}
        
        Execute each action in the navigation plan with precise parameters.
        Report the outcome of each action and verify the final result.
        """
        
        return template.format(
            subtask_description=data.get("description", "No description provided"),
            navigation_plan=json.dumps(data.get("navigation_plan", {}), indent=2),
            screen_info=json.dumps(data.get("screen_info", {}), indent=2)
        )
    
    else:
        return json.dumps(data, indent=2)
