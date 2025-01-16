import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator
from swarms.structs.output_types import OutputType

def generate_id() -> str:
    """Generate a unique identifier"""
    return uuid.uuid4().hex

class BaseSwarmComponent(BaseModel):
    """Base class for shared attributes across components"""
    id: str = Field(default_factory=generate_id)
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Step(BaseSwarmComponent):
    """Enhanced step class with standardized timing and role information"""
    role: str
    content: str
    step_type: Optional[str] = None
    duration: Optional[float] = None
    status: str = "completed"
    error: Optional[Dict[str, Any]] = None

class AgentConfig(BaseSwarmComponent):
    """Flexible agent configuration supporting various agent types"""
    agent_type: str
    system_prompt: Optional[str] = None
    llm_config: Dict[str, Any] = Field(default_factory=dict)
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    workflow_config: Dict[str, Any] = Field(default_factory=dict)
    custom_config: Dict[str, Any] = Field(default_factory=dict)

class AgentOutput(BaseSwarmComponent):
    """Universal agent output format supporting all agent types"""
    agent_name: str
    config: AgentConfig
    steps: List[Step] = Field(default_factory=list)
    result: Optional[Any] = None
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    state: Dict[str, Any] = Field(default_factory=dict)
    
    def add_step(self, role: str, content: str, **kwargs) -> None:
        """Add a step with timing information"""
        step = Step(
            role=role,
            content=content,
            **kwargs
        )
        self.steps.append(step)

class SwarmMetrics(BaseModel):
    """Comprehensive swarm performance metrics"""
    total_agents: int = 0
    active_agents: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_steps: int = 0
    execution_time: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    resource_usage: Dict[str, Any] = Field(default_factory=dict)
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)

class SwarmState(BaseModel):
    """Swarm state management"""
    status: str = "initialized"
    current_phase: Optional[str] = None
    active_tasks: List[str] = Field(default_factory=list)
    completed_tasks: List[str] = Field(default_factory=list)
    failed_tasks: List[str] = Field(default_factory=list)
    task_dependencies: Dict[str, List[str]] = Field(default_factory=dict)

class UnifiedOutputSchemaInput(BaseSwarmComponent):
    """Enhanced input schema with workflow management"""
    name: Optional[str] = None
    description: Optional[str] = None
    swarm_type: str
    workflow_type: Optional[str] = None
    agents: List[AgentConfig] = Field(default_factory=list)
    max_loops: Optional[int] = None
    timeout: Optional[float] = None
    output_type: OutputType = Field(default="final")
    workflow_config: Dict[str, Any] = Field(default_factory=dict)

class UnifiedOutputSchema(BaseSwarmComponent):
    """Flexible unified output schema supporting dynamic swarm types"""
    input: UnifiedOutputSchemaInput
    outputs: List[AgentOutput] = Field(default_factory=list)
    intermediate_results: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: SwarmMetrics = Field(default_factory=SwarmMetrics)
    state: SwarmState = Field(default_factory=SwarmState)
    
    def add_agent_output(self, output: AgentOutput) -> None:
        """Add agent output with automatic metrics update"""
        self.outputs.append(output)
        self._update_metrics(output)
    
    def add_intermediate_result(self, result: Dict[str, Any], phase: Optional[str] = None) -> None:
        """Add an intermediate result with phase tracking"""
        self.intermediate_results.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "phase": phase,
            "data": result
        })
    
    def update_state(self, status: Optional[str] = None, phase: Optional[str] = None, **kwargs) -> None:
        """Update swarm state with custom attributes"""
        if status:
            self.state.status = status
        if phase:
            self.state.current_phase = phase
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
    
    def _update_metrics(self, output: AgentOutput) -> None:
        """Update metrics based on agent output"""
        self.metrics.total_agents += 1
        self.metrics.total_steps += len(output.steps)
        if output.performance_metrics:
            for key, value in output.performance_metrics.items():
                if key not in self.metrics.custom_metrics:
                    self.metrics.custom_metrics[key] = []
                self.metrics.custom_metrics[key].append(value)
    
    @model_validator(mode='after')
    def validate_workflow(self) -> 'UnifiedOutputSchema':
        """Validate workflow configuration and dependencies"""
        if self.input.workflow_type and not self.input.workflow_config:
            raise ValueError(f"Workflow configuration required for type: {self.input.workflow_type}")
        return self