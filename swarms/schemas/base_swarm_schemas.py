from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
import uuid
import time

class AgentInputConfig(BaseModel):
    """
    Configuration for an agent. This can be further customized
    per agent type if needed.
    """
    agent_name: str = Field(..., description="Name of the agent")
    # ... other common agent settings like model_name, temperature etc...

class BaseSwarmSchema(BaseModel):
    """
    Base schema for all swarm types.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    agents: List[AgentInputConfig]  # Using AgentInputConfig
    max_loops: int = 1
    swarm_type: str  # e.g., "SequentialWorkflow", "ConcurrentWorkflow", etc.
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    config: Dict[str, Any] = Field(default_factory=dict)  # Flexible config

    @validator("config")
    def validate_config(cls, v, values):
        """
        Validates the 'config' dictionary based on the 'swarm_type'.
        """
        swarm_type = values.get("swarm_type")
        if swarm_type == "SequentialWorkflow":
            # Validate required config for SequentialWorkflow
            if "flow" not in v:
                raise ValueError("SequentialWorkflow requires a 'flow' configuration.")
        elif swarm_type == "ConcurrentWorkflow":
            # Validate required config for ConcurrentWorkflow
            if "max_workers" not in v:
                raise ValueError("ConcurrentWorkflow requires a 'max_workers' configuration.")
        elif swarm_type == "AgentRearrange":
            if "flow" not in v:
                raise ValueError("AgentRearrange requires a 'flow' configuration.")
        elif swarm_type == "MixtureOfAgents":
            if "aggregator_agent" not in v:
                raise ValueError(
                    "MixtureOfAgents requires an"
                    " 'aggregator_agent' configuration."
                )
        elif swarm_type == "SpreadSheetSwarm":
            if "save_file_path" not in v:
                raise ValueError("SpreadSheetSwarm requires a 'save_file_path' configuration.")
        # ... (Add validation for other swarm types) ...
        return v