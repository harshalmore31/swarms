from typing import Any, Callable, Dict, List, Union, Optional
import functools
import datetime
from pydantic import BaseModel

from swarms.schemas.unified_output_schema import (
    UnifiedOutputSchema,
    UnifiedOutputSchemaInput,
    AgentOutput,
    AgentConfig,
    SwarmMetrics,
    SwarmState,
    Step
)
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="output_formatter")

class SwarmOutputFormatter:
    """
    Enhanced formatter class for handling complex swarm outputs with comprehensive
    metrics tracking and state management.
    """
    
    def format(
        self,
        swarm_type: str,
        raw_output: Any,
        name: str = None,
        description: str = None,
        workflow_type: str = None,
        max_loops: int = 1,
        output_type: str = "final",
        agent_configs: List[AgentConfig] = None,
        agent_outputs: List[AgentOutput] = None,
        workflow_config: Dict[str, Any] = None,
        timeout: Optional[float] = None,
    ) -> UnifiedOutputSchema:
        """
        Formats raw swarm output into a structured schema with enhanced tracking.
        
        Args:
            swarm_type (str): Type of swarm
            raw_output (Any): Raw output data
            name (str, optional): Swarm name
            description (str, optional): Swarm description
            workflow_type (str, optional): Type of workflow
            max_loops (int, optional): Maximum loops
            output_type (str, optional): Output type
            agent_configs (List[AgentConfig], optional): Agent configurations
            agent_outputs (List[AgentOutput], optional): Agent outputs
            workflow_config (Dict[str, Any], optional): Workflow configuration
            timeout (float, optional): Execution timeout
        
        Returns:
            UnifiedOutputSchema: Formatted output schema
        """
        try:
            logger.info(f"Formatting output for swarm type: {swarm_type}")
            
            # Create input schema
            input_data = UnifiedOutputSchemaInput(
                name=name,
                description=description,
                swarm_type=swarm_type,
                workflow_type=workflow_type,
                max_loops=max_loops,
                output_type=output_type,
                agents=agent_configs or [],
                workflow_config=workflow_config or {},
                timeout=timeout
            )

            # Initialize output schema
            output = UnifiedOutputSchema(
                input=input_data,
                outputs=[],
                metrics=SwarmMetrics(),
                state=SwarmState()
            )

            # Process and add agent outputs
            if agent_outputs:
                for agent_output in agent_outputs:
                    output.add_agent_output(agent_output)

            # Process raw output
            processed_output = self._process_raw_output(raw_output)
            if processed_output:
                output.add_intermediate_result({
                    "type": "raw_output",
                    "data": processed_output
                })

            # Update final state and metrics
            self._update_final_metrics(output)
            output.update_state(status="completed")

            return output

        except Exception as e:
            logger.error(f"Error formatting output: {e}")
            self._handle_formatting_error(e)
            raise

    def _process_raw_output(self, output: Any) -> Any:
        """
        Process raw output with enhanced type handling and validation.
        """
        if isinstance(output, BaseModel):
            return output.model_dump()
        elif isinstance(output, dict):
            return {k: self._process_raw_output(v) for k, v in output.items()}
        elif isinstance(output, list):
            return [self._process_raw_output(item) for item in output]
        elif isinstance(output, (datetime.datetime, datetime.date)):
            return output.isoformat()
        elif isinstance(output, (int, float, str, bool, type(None))):
            return output
        else:
            try:
                return str(output)
            except Exception as e:
                logger.warning(f"Could not process output of type {type(output)}: {e}")
                return None

    def _update_final_metrics(self, output: UnifiedOutputSchema) -> None:
        """
        Update final metrics based on overall execution.
        """
        metrics = output.metrics
        
        # Calculate success rate
        if metrics.total_agents > 0:
            metrics.success_rate = (metrics.completed_tasks / metrics.total_agents) * 100
            
        # Update execution time
        try:
            start_time = datetime.datetime.strptime(output.timestamp, "%Y-%m-%d %H:%M:%S")
            end_time = datetime.datetime.now()
            metrics.execution_time = (end_time - start_time).total_seconds()
        except Exception as e:
            logger.warning(f"Error calculating execution time: {e}")

    def _handle_formatting_error(self, error: Exception) -> None:
        """
        Handle formatting errors with detailed logging.
        """
        logger.error(f"Formatting error: {str(error)}")
        logger.exception(error)

def output_schema(func: Callable) -> Callable:
    """
    Enhanced decorator for formatting swarm outputs with error handling
    and comprehensive metrics tracking.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Dict:
        try:
            formatter = SwarmOutputFormatter()
            instance = args[0]
            
            # Start timing
            start_time = datetime.datetime.now()
            
            # Get instance attributes
            name = getattr(instance, "name", None)
            description = getattr(instance, "description", None)
            workflow_type = getattr(instance, "workflow_type", None)
            max_loops = getattr(instance, "max_loops", 1)
            output_type = getattr(instance, "output_type", "final")
            swarm_type = getattr(instance, "swarm_type", func.__name__)
            workflow_config = getattr(instance, "workflow_config", {})
            timeout = getattr(instance, "timeout", None)
            
            # Execute the run function
            raw_output = func(*args, **kwargs)
            
            # Get agent configurations and outputs
            agent_configs = getattr(instance, "agent_configs", None)
            agent_outputs = getattr(instance, "agent_outputs", None)
            
            # Format the output
            formatted_output = formatter.format(
                swarm_type=swarm_type,
                raw_output=raw_output,
                name=name,
                description=description,
                workflow_type=workflow_type,
                max_loops=max_loops,
                output_type=output_type,
                agent_configs=agent_configs,
                agent_outputs=agent_outputs,
                workflow_config=workflow_config,
                timeout=timeout
            )
            
            # Save metadata if method exists
            if hasattr(instance, "save_metadata") and callable(
                getattr(instance, "save_metadata")
            ):
                instance.save_metadata(formatted_output.model_dump())
            
            return formatted_output.model_dump()
            
        except Exception as e:
            logger.error(f"An error occurred in output formatting: {e}")
            raise
            
    return wrapper