import asyncio
import csv
import datetime
import json
import os
import uuid
from typing import Dict, List, Union

import aiofiles
from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.telemetry.capture_sys_data import log_agent_data
from swarms.utils.file_processing import create_file_in_folder
from swarms.utils.loguru_logger import initialize_logger
from swarms.schemas.output_schemas import (
    SwarmOutputFormatter,
    AgentTaskOutput,
    Step,
)

logger = initialize_logger(log_folder="spreadsheet_swarm")

time = datetime.datetime.now().isoformat()
uuid_hex = uuid.uuid4().hex

# --------------- NEW CHANGE START ---------------
# Format time variable to be compatible across operating systems
formatted_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
# --------------- NEW CHANGE END ---------------

class AgentConfig(BaseModel):
    """Configuration for an agent loaded from CSV"""

    agent_name: str
    description: str
    system_prompt: str
    task: str

class AgentOutput(BaseModel):
    agent_name: str
    task: str
    result: str
    timestamp: str

class SwarmRunMetadata(BaseModel):
    run_id: str = Field(
        default_factory=lambda: f"spreadsheet_swarm_run_{uuid_hex}"
    )
    name: str
    description: str
    agents: List[str]
    start_time: str = Field(
        default_factory=lambda: time,
        description="The start time of the swarm run.",
    )
    end_time: str
    tasks_completed: int
    outputs: List[AgentOutput]
    number_of_agents: int = Field(
        ...,
        description="The number of agents participating in the swarm.",
    )

class SpreadSheetSwarm(BaseSwarm):
    """
    A swarm that processes tasks concurrently using multiple agents.

    Args:
        name (str, optional): The name of the swarm. Defaults to "Spreadsheet-Swarm".
        description (str, optional): The description of the swarm. Defaults to "A swarm that processes tasks concurrently using multiple agents.".
        agents (Union[Agent, List[Agent]], optional): The agents participating in the swarm. Defaults to an empty list.
        autosave_on (bool, optional): Whether to enable autosave of swarm metadata. Defaults to True.
        save_file_path (str, optional): The file path to save the swarm metadata as a CSV file. Defaults to "spreedsheet_swarm.csv".
        max_loops (int, optional): The number of times to repeat the swarm tasks. Defaults to 1.
        workspace_dir (str, optional): The directory path of the workspace. Defaults to the value of the "WORKSPACE_DIR" environment variable.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        name: str = "Spreadsheet-Swarm",
        description: str = "A swarm that that processes tasks concurrently using multiple agents and saves the metadata to a CSV file.",
        agents: Union[Agent, List[Agent]] = [],
        autosave_on: bool = True,
        save_file_path: str = None,
        max_loops: int = 1,
        workspace_dir: str = os.getenv("WORKSPACE_DIR"),
        load_path: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            agents=agents if isinstance(agents, list) else [agents],
            *args,
            **kwargs,
        )
        self.name = name
        self.description = description
        self.save_file_path = save_file_path
        self.autosave_on = autosave_on
        self.max_loops = max_loops
        self.workspace_dir = workspace_dir
        self.load_path = load_path
        self.agent_configs: Dict[str, AgentConfig] = {}

        # --------------- NEW CHANGE START ---------------
        # The save_file_path now uses the formatted_time and uuid_hex
        self.save_file_path = (
            f"spreadsheet_swarm_run_id_{formatted_time}.csv"
        )
        # --------------- NEW CHANGE END ---------------
        self.output_formatter = SwarmOutputFormatter()
        self.reliability_check()

    def reliability_check(self):
        """
        Check the reliability of the swarm.

        Raises:
            ValueError: If no agents are provided or no save file path is provided.
        """
        logger.info("Checking the reliability of the swarm...")

        # if not self.agents:
        #     raise ValueError("No agents are provided.")
        # if not self.save_file_path:
        #     raise ValueError("No save file path is provided.")
        if not self.max_loops:
            raise ValueError("No max loops are provided.")

        logger.info("Swarm reliability check passed.")
        logger.info("Swarm is ready to run.")

    async def _load_from_csv(self):
        """
        Load agent configurations from a CSV file.
        Expected CSV format: agent_name,description,system_prompt,task

        Args:
            csv_path (str): Path to the CSV file containing agent configurations
        """
        try:
            csv_path = self.load_path
            logger.info(
                f"Loading agent configurations from {csv_path}"
            )

            async with aiofiles.open(csv_path, mode="r") as file:
                content = await file.read()
                csv_reader = csv.DictReader(content.splitlines())

                for row in csv_reader:
                    config = AgentConfig(
                        agent_name=row["agent_name"],
                        description=row["description"],
                        system_prompt=row["system_prompt"],
                        task=row["task"],
                    )

                    # Create new agent with configuration
                    new_agent = Agent(
                        agent_name=config.agent_name,
                        system_prompt=config.system_prompt,
                        description=config.description,
                        model_name=(
                            row["model_name"]
                            if "model_name" in row
                            else "openai/gpt-4o"
                        ),
                        docs=[row["docs"]] if "docs" in row else "",
                        dynamic_temperature_enabled=True,
                        max_loops=(
                            row["max_loops"]
                            if "max_loops" in row
                            else 1
                        ),
                        user_name=(
                            row["user_name"]
                            if "user_name" in row
                            else "user"
                        ),
                        # output_type="str",
                        stopping_token=(
                            row["stopping_token"]
                            if "stopping_token" in row
                            else None
                        ),
                    )

                    # Add agent to swarm
                    self.agents.append(new_agent)
                    self.agent_configs[config.agent_name] = config

            # Update metadata with new agents
            self.metadata.agents = [
                agent.agent_name for agent in self.agents
            ]
            self.metadata.number_of_agents = len(self.agents)
            logger.info(
                f"Loaded {len(self.agent_configs)} agent configurations"
            )
        except Exception as e:
            logger.error(f"Error loading agent configurations: {e}")

    def load_from_csv(self):
        asyncio.run(self._load_from_csv())

    async def _run_agent(self, agent: Agent, task: str, *args, **kwargs) -> AgentTaskOutput:
        """
        Runs a single agent with tracking and output formatting.
        """
        
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: agent.run(task, *args, **kwargs))
            
            end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            steps = []
            if agent.agent_output and agent.agent_output.steps:
                for step_data in agent.agent_output.steps:
                    step = Step(
                        id=str(uuid.uuid4()),  # Generate new UUID for each step
                        name=agent.agent_name,
                        task=task,
                        input=step_data.get("role"),
                        output=step_data.get("content"),
                        error=None,
                        start_time=step_data.get("timestamp"),
                        end_time=None,  # You might need to add end_time to agent steps
                        runtime=None,  # Calculate runtime if needed
                        tokens_used=None,  # Extract token usage if available
                        cost=None,  # Calculate cost if available
                        metadata={},
                    )
                    steps.append(step)
            else:
                # If agent.agent_output.steps is None or empty, create a single step with the agent's output
                step = Step(
                    id=str(uuid.uuid4()),
                    name=agent.agent_name,
                    task=task,
                    input=task,  # Input is the task itself if no steps are available
                    output=response,  # Output from agent.run()
                    error=None,
                    start_time=start_time,
                    end_time=end_time,
                    runtime=None,  # Calculate runtime if needed
                    tokens_used=None,  # Extract token usage if available
                    cost=None,  # Calculate cost if available
                    metadata={},
                )
                steps = [step]

            return AgentTaskOutput(
                id=str(uuid.uuid4()),  # Generate new UUID for each agent task output
                agent_name=agent.agent_name,
                task=task,
                steps=steps,
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as e:
            logger.error(f"Error running agent {agent.agent_name}: {e}")
            return AgentTaskOutput(
                id=str(uuid.uuid4()),
                agent_name=agent.agent_name,
                task=task,
                steps=[Step(id=str(uuid.uuid4()), name=agent.agent_name, task=task, error=str(e), start_time=start_time, end_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))],
                start_time=start_time,
                end_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

    async def _run_concurrently(self, task: str, *args, **kwargs):
        """Runs agents concurrently."""
        tasks = [
            self._run_agent(agent, task, *args, **kwargs)
            for _ in range(self.max_loops)
            for agent in self.agents
        ]
        outputs = await asyncio.gather(*tasks)
        return outputs

    async def _run(self, task: str = None, *args, **kwargs):
        """
        Run the swarm with the specified task.

        Args:
            task (str): The task to be executed by the swarm.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The JSON representation of the swarm metadata.

        """
        
        if task is None and self.agent_configs:
            outputs = await self.run_from_config()
        else:
            outputs = await self._run_concurrently(task, *args, **kwargs)
            
        
        
        swarm_output = self.output_formatter.format_output(
            swarm_id=str(uuid.uuid4()),
            swarm_type=self.name,
            task=task,
            agent_outputs=outputs,
            swarm_specific_output={},
        )
        
        if self.autosave_on:
            await self._save_to_csv(swarm_output)
            
        return swarm_output

    def run(self, task: str = None, *args, **kwargs):
        """
        Run the swarm with the specified task.

        Args:
            task (str): The task to be executed by the swarm.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The JSON representation of the swarm metadata.

        """
        return asyncio.run(self._run(task, *args, **kwargs))
    
    async def run_from_config(self):
        """
        Run all agents with their configured tasks concurrently
        """
        logger.info("Running agents from configuration")

        tasks = []
        for agent in self.agents:
            config = self.agent_configs.get(agent.agent_name)
            if config:
                for _ in range(self.max_loops):
                    tasks.append(
                        self._run_agent(agent, config.task)
                    )

        # Run all tasks concurrently using asyncio.gather
        agent_outputs = await asyncio.gather(*tasks)

        swarm_output = self.output_formatter.format_output(
            swarm_id=str(uuid.uuid4()),
            swarm_type=self.name,
            task="Multiple tasks from config",
            agent_outputs=agent_outputs,
            swarm_specific_output={},
        )

        if self.autosave_on:
            await self._save_to_csv(swarm_output)

        return swarm_output

    async def _save_to_csv(self, formatted_output: str):
        """
        Save the swarm metadata to a CSV file.
        """
        if not self.autosave_on:
            return

        try:
            # Parse the JSON output
            data = json.loads(formatted_output)

            # Extract the agent outputs
            agent_outputs = data.get("agent_outputs", [])

            # Prepare the CSV data
            csv_data = []
            headers = ["swarm_id", "swarm_type", "agent_id", "agent_name", "task", "step_id", "step_input", "step_output", "step_error", "step_start_time", "step_end_time", "step_runtime", "step_tokens_used", "step_cost", "step_metadata", "agent_start_time", "agent_end_time", "agent_total_tokens", "agent_cost"]

            for output in agent_outputs:
                for step in output.get("steps", []):
                    csv_data.append([
                        data.get("swarm_id"),
                        data.get("swarm_type"),
                        output.get("id"),
                        output.get("agent_name"),
                        output.get("task"),
                        step.get("id"),
                        step.get("input"),
                        step.get("output"),
                        step.get("error"),
                        step.get("start_time"),
                        step.get("end_time"),
                        step.get("runtime"),
                        step.get("tokens_used"),
                        step.get("cost"),
                        json.dumps(step.get("metadata")),
                        output.get("start_time"),
                        output.get("end_time"),
                        output.get("total_tokens"),
                        output.get("cost")
                    ])

            # Write the data to CSV
            async with aiofiles.open(self.save_file_path, mode="w", newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                await writer.writerow(headers)  # Fix: Await the coroutine
                await writer.writerows(csv_data)  # Fix: Await the coroutine

            logger.info(f"Swarm metadata saved to {self.save_file_path}")

        except Exception as e:
            logger.error(f"Error saving swarm metadata to CSV: {e}")