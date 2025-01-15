# sequential_workflow.py
import concurrent
from typing import List, Optional

from swarms.structs.agent import Agent
from swarms.structs.output_types import OutputType
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_formatter import output_schema
from swarms.schemas.agent_step_schemas import ManySteps
from swarms.utils.file_processing import create_file_in_folder
import os
import json

logger = initialize_logger(log_folder="sequential_workflow")

class SequentialWorkflow:
    """
    Initializes a SequentialWorkflow object, which orchestrates the execution of a sequence of agents.

    Args:
        name (str, optional): The name of the workflow. Defaults to "SequentialWorkflow".
        description (str, optional): A description of the workflow. Defaults to "Sequential Workflow, where agents are executed in a sequence."
        agents (List[Agent], optional): The list of agents in the workflow. Defaults to None.
        max_loops (int, optional): The maximum number of loops to execute the workflow. Defaults to 1.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Raises:
        ValueError: If agents list is None or empty, or if max_loops is 0
    """

    def __init__(
        self,
        name: str = "SequentialWorkflow",
        description: str = "Sequential Workflow, where agents are executed in a sequence.",
        agents: List[Agent] = [],
        max_loops: int = 1,
        output_type: OutputType = "all",
        return_json: bool = False,
        shared_memory_system: callable = None,
        save_to_file: bool = True,
        agent_save_path: str = "sequential_workflow_agent_out",
        *args,
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.agents = agents
        self.max_loops = max_loops
        self.output_type = output_type
        self.return_json = return_json
        self.shared_memory_system = shared_memory_system
        self.save_to_file = save_to_file
        self.agent_save_path = agent_save_path

        self.reliability_check()
        self.flow = self.sequential_flow()

        # Create the output path directory if it doesn't exist
        if not os.path.exists(self.agent_save_path):
            os.makedirs(self.agent_save_path)

    def sequential_flow(self):
        # Only create flow if agents exist
        if self.agents:
            # Create flow by joining agent names with arrows
            agent_names = []
            for agent in self.agents:
                try:
                    # Try to get agent_name, fallback to name if not available
                    agent_name = (
                        getattr(agent, "agent_name", None)
                        or agent.name
                    )
                    agent_names.append(agent_name)
                except AttributeError:
                    logger.warning(
                        f"Could not get name for agent {agent}"
                    )
                    continue

            if agent_names:
                flow = " -> ".join(agent_names)
            else:
                flow = ""
                logger.warning(
                    "No valid agent names found to create flow"
                )
        else:
            flow = ""
            logger.warning("No agents provided to create flow")

        return flow

    def reliability_check(self):
        if self.agents is None or len(self.agents) == 0:
            raise ValueError("Agents list cannot be None or empty")

        if self.max_loops == 0:
            raise ValueError("max_loops cannot be 0")

        logger.info("Checks completed your swarm is ready.")

    @output_schema
    def run(
        self,
        task: str,
        img: Optional[str] = None,
        *args,
        **kwargs,
    ) -> List[ManySteps]:
        """
        Executes a task through the agents in the dynamically constructed flow.

        Args:
            task (str): The task for the agents to execute.
            img (str): The image for the agents to execute.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[ManySteps]: The list of steps executed by the agents.

        Raises:
            ValueError: If task is None or empty
            Exception: If any error occurs during task execution
        """
        try:
            if not task:
                raise ValueError("Task cannot be None or empty")

            logger.info(f"Running task '{task}' through workflow")

            agent_outputs = []
            for agent in self.agents:
                try:
                    logger.info(
                        f"Running agent {agent.agent_name} in sequential"
                        " workflow"
                    )
                    agent.run(task, img=img, *args, **kwargs)

                    # Save the output of the agent to the agent_outputs list
                    agent_outputs.append(agent.agent_output)

                    # Save the output to a file if save_to_file is True
                    if self.save_to_file:
                        try:
                            create_file_in_folder(
                                self.agent_save_path,
                                f"{agent.agent_name}_{agent.agent_output.run_id}.json",
                                agent.agent_output.model_dump_json(indent=4),
                            )
                            logger.info(
                                f"Saved output of agent {agent.agent_name} to file"
                            )
                        except Exception as error:
                            logger.error(
                                f"Error saving output of agent {agent.agent_name} to file: {error}"
                            )
                            raise error

                except Exception as error:
                    logger.error(
                        f"Error running agent {agent.agent_name}: {error}"
                    )
                    raise error

            return agent_outputs

        except Exception as e:
            logger.error(
                f"An error occurred while executing the task: {e}"
            )
            raise e

    def __call__(self, task: str, *args, **kwargs) -> str:
        return self.run(task, *args, **kwargs)

    def run_batched(self, tasks: List[str]) -> List[str]:
        """
        Executes a batch of tasks through the agents in the dynamically constructed flow.

        Args:
            tasks (List[str]): The tasks for the agents to execute.

        Returns:
            List[str]: The final results after processing through all agents.

        Raises:
            ValueError: If tasks is None or empty
            Exception: If any error occurs during task execution
        """
        if not tasks or not all(
            isinstance(task, str) for task in tasks
        ):
            raise ValueError(
                "Tasks must be a non-empty list of strings"
            )

        try:
            return [self.run(task) for task in tasks]
        except Exception as e:
            logger.error(
                f"An error occurred while executing the batch of tasks: {e}"
            )
            raise

    async def run_async(self, task: str) -> str:
        """
        Executes the task through the agents in the dynamically constructed flow asynchronously.

        Args:
            task (str): The task for the agents to execute.

        Returns:
            str: The final result after processing through all agents.

        Raises:
            ValueError: If task is None or empty
            Exception: If any error occurs during task execution
        """
        if not task or not isinstance(task, str):
            raise ValueError("Task must be a non-empty string")

        try:
            return await self.run(task)
        except Exception as e:
            logger.error(
                f"An error occurred while executing the task asynchronously: {e}"
            )
            raise

    async def run_concurrent(self, tasks: List[str]) -> List[str]:
        """
        Executes a batch of tasks through the agents in the dynamically constructed flow concurrently.

        Args:
            tasks (List[str]): The tasks for the agents to execute.

        Returns:
            List[str]: The final results after processing through all agents.

        Raises:
            ValueError: If tasks is None or empty
            Exception: If any error occurs during task execution
        """
        if not tasks or not all(
            isinstance(task, str) for task in tasks
        ):
            raise ValueError(
                "Tasks must be a non-empty list of strings"
            )

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.run, task) for task in tasks
                ]
                return [
                    future.result()
                    for future in concurrent.futures.as_completed(
                        futures
                    )
                ]
        except Exception as e:
            logger.error(
                f"An error occurred while executing the batch of tasks concurrently: {e}"
            )
            raise

    def save_metadata(self, formatted_output: str):
        """
        Saves the metadata to a JSON file based on the auto_save flag.

        Example:
            >>> workflow.save_metadata()
            >>> # Metadata will be saved to the specified path if auto_save is True.
        """
        # Save metadata to a JSON file
        if self.save_to_file:
            logger.info(
                f"Saving metadata to {self.agent_save_path}"
            )
            create_file_in_folder(
                os.getenv("WORKSPACE_DIR"),
                f"{self.agent_save_path}/{self.name}_metadata.json",
                json.dumps(formatted_output, indent=4),
            )