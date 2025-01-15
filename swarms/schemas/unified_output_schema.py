# swarms/schemas/unified_output_schema.py

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from swarms.schemas.agent_step_schemas import ManySteps
from swarms.structs.output_types import OutputType

def swarm_id():
    return uuid.uuid4().hex


class UnifiedOutputSchemaInput(BaseModel):
    swarm_id: Optional[str] = Field(default_factory=swarm_id)
    name: Optional[str] = None
    description: Optional[str] = None
    flow: Optional[str] = None
    max_loops: Optional[int] = None
    time: str = Field(
        default_factory=lambda: datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        description="The time the swarm started execution",
    )
    output_type: OutputType = Field(default="final")

class UnifiedOutputSchema(BaseModel):
    output_id: str = Field(
        default_factory=swarm_id, description="Output-UUID"
    )
    swarm_type: str
    input: Optional[UnifiedOutputSchemaInput] = None
    outputs: Optional[List[ManySteps]] = None
    aggregator_agent_summary: Optional[str] = None
    time_completed: Optional[str] = Field(
         default_factory=lambda: datetime.now().strftime(
             "%Y-%m-%d %H:%M:%S"
         ),
         description="The time the agent completed execution",
     )
    time: Optional[str] = Field(
        default_factory=lambda: datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        description="The time the agent started execution",
    )