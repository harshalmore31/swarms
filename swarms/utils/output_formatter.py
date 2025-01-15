from typing import Any, Callable, Dict, List
import functools
import datetime

from swarms.schemas.agent_step_schemas import ManySteps
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="output_formatter")

class SwarmOutputFormatter:
    """
    A class that formats raw output from swarms to the unified JSON schema.
    """

    def format(
        self,
        swarm_type: str,
        raw_output: Any,
        name: str = None,
        description: str = None,
        flow: str = None,
        max_loops: int = 1,
        output_type: str = None,
        agent_outputs: List[ManySteps] = None,
    ) -> dict:
        """
        Formats the raw output data from a swarm into a structured JSON response.

        Args:
            swarm_type (str): The type of swarm.
            raw_output (Any): The raw output data from the swarm.
            name (str, optional): Name of the swarm.
            description (str, optional): Description of the swarm.
            flow (str, optional): Flow type.
            max_loops (int, optional): Maximum number of loops. Defaults to 1.
            output_type (str, optional): Type of output.
            agent_outputs (List[ManySteps], optional): List of agent step outputs.

        Returns:
            dict: The structured JSON output.
        """
        try:
            logger.info(f"Formatting output for swarm type: {swarm_type}")

            input_data = {
                "name": name,
                "description": description,
                "flow": flow,
                "max_loops": max_loops,
                "output_type": output_type,
            }

            output = {
                "swarm_type": swarm_type,
                "input": input_data,
                "outputs": self._serialize_output(raw_output),
            }

            return output
        except Exception as e:
            logger.error(f"Error formatting output: {e}")
            raise

    def _serialize_output(self, output: Any) -> Any:
        """
        Recursively process the output data to serialize datetime objects and convert ManySteps to dict.

        Args:
            output (Any): The output data to process.

        Returns:
            Any: The serialized output data.
        """
        if isinstance(output, dict):
            return {k: self._serialize_output(v) for k, v in output.items()}
        elif isinstance(output, list):
            return [self._serialize_output(item) for item in output]
        elif isinstance(output, datetime.datetime):
            return output.isoformat()
        elif isinstance(output, ManySteps):
            # Convert ManySteps to a dictionary using its model_dump() method
            # This is the Pydantic v2 way to convert to dict
            return output.model_dump()
        else:
            return output

def output_schema(func: Callable) -> Callable:
    """
    A decorator to format the output of a swarm using the SwarmOutputFormatter.

    Args:
        func (Callable): The swarm's run method to be wrapped.

    Returns:
        Callable: The wrapped function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> dict:
        """
        Wraps the run function and passes the output to the formatter.

        Returns:
            dict: The formatted output
        """
        try:
            formatter = SwarmOutputFormatter()
            instance = args[0]
            
            # Get attributes with getattr and default values
            name = getattr(instance, "name", None)
            description = getattr(instance, "description", None)
            flow = getattr(instance, "flow", None)
            max_loops = getattr(instance, "max_loops", 1)
            output_type = getattr(instance, "output_type", "final")
            swarm_type = getattr(instance, "name", func.__name__)

            # Execute the run function
            raw_output = func(*args, **kwargs)

            # Format the output
            formatted_output = formatter.format(
                swarm_type=swarm_type,
                raw_output=raw_output,
                name=name,
                description=description,
                flow=flow,
                max_loops=max_loops,
                output_type=output_type,
            )

            # Save metadata if the method exists
            if hasattr(instance, "save_metadata") and callable(
                getattr(instance, "save_metadata")
            ):
                instance.save_metadata(formatted_output)

            return formatted_output

        except Exception as e:
            logger.error(f"An error occurred in output formatting: {e}")
            raise

    return wrapper