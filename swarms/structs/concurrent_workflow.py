import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.utils.file_processing import create_file_in_folder
import concurrent
from clusterops import (
    execute_on_gpu,
    execute_with_cpu_cores,
    execute_on_multiple_gpus,
    list_available_gpus,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.schemas.unified_output_schema import (
    UnifiedOutputSchema,
    UnifiedOutputSchemaInput,
    AgentOutput,
    AgentConfig,
    SwarmMetrics,
    SwarmState,
    Step
)
from swarms.utils.output_formatter import SwarmOutputFormatter, output_schema

logger = initialize_logger(log_folder="concurrent_workflow")

class ConcurrentWorkflow(BaseSwarm):
    """
    Represents a concurrent workflow that executes multiple agents concurrently in a production-grade manner.
    """

    def __init__(
        self,
        name: str = "ConcurrentWorkflow",
        description: str = "Execution of multiple agents concurrently",
        agents: List[Agent] = [],
        metadata_output_path: str = "agent_metadata.json",
        auto_save: bool = True,
        max_loops: int = 1,
        return_str_on: bool = False,
        agent_responses: list = [],
        auto_generate_prompts: bool = False,
        max_workers: int = None,
        workflow_type: str = "concurrent",
        workflow_config: Dict[str, Any] = None,
        timeout: Optional[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            agents=agents,
            *args,
            **kwargs,
        )
        self.name = name
        self.description = description
        self.agents = agents
        self.metadata_output_path = metadata_output_path
        self.auto_save = auto_save
        self.max_loops = max_loops
        self.return_str_on = return_str_on
        self.agent_responses = agent_responses
        self.auto_generate_prompts = auto_generate_prompts
        self.max_workers = max_workers or os.cpu_count()
        self.workflow_type = workflow_type
        self.workflow_config = workflow_config or {}
        self.timeout = timeout
        self.tasks = []
        self.formatter = SwarmOutputFormatter()
        
        # Initialize SwarmState
        self.state = SwarmState()
        self.metrics = SwarmMetrics()
        
        self.reliability_check()

    def _create_agent_config(self, agent: Agent) -> AgentConfig:
        """Create AgentConfig from Agent instance"""
        return AgentConfig(
            agent_type=agent.__class__.__name__,
            system_prompt=getattr(agent, 'system_prompt', None),
            llm_config=getattr(agent, 'llm_config', {}),
            tools=getattr(agent, 'tools', []),
            workflow_config=self.workflow_config
        )

    @retry(wait=wait_exponential(min=2), stop=stop_after_attempt(3))
    async def _run_agent(
        self,
        agent: Agent,
        task: str,
        img: str,
        executor: ThreadPoolExecutor,
        *args,
        **kwargs,
    ) -> AgentOutput:
        """Run a single agent with enhanced output tracking"""
        start_time = datetime.now()
        agent_config = self._create_agent_config(agent)
        
        try:
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(
                executor, agent.run, task, img, *args, **kwargs
            )
            
            agent_output = AgentOutput(
                agent_name=agent.agent_name,
                config=agent_config,
                result=output,
                performance_metrics={
                    "duration": (datetime.now() - start_time).total_seconds()
                }
            )
            
            agent_output.add_step(
                role="agent",
                content=str(output),
                step_type="execution",
                duration=(datetime.now() - start_time).total_seconds()
            )
            
            return agent_output

        except Exception as e:
            logger.error(f"Error running agent {agent.agent_name}: {e}")
            agent_output = AgentOutput(
                agent_name=agent.agent_name,
                config=agent_config,
                error={"message": str(e), "type": type(e).__name__}
            )
            agent_output.add_step(
                role="error",
                content=str(e),
                step_type="error",
                status="failed"
            )
            raise

    @output_schema
    async def _execute_agents_concurrently(
        self, task: str, img: str, *args, **kwargs
    ) -> UnifiedOutputSchema:
        """Execute multiple agents concurrently with unified output schema"""
        self.state.status = "running"
        self.state.current_phase = "agent_execution"
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            agent_configs = [self._create_agent_config(agent) for agent in self.agents]
            tasks_to_run = [
                self._run_agent(agent, task, img, executor, *args, **kwargs)
                for agent in self.agents
            ]
            
            agent_outputs = await asyncio.gather(*tasks_to_run, return_exceptions=True)
            
            # Update metrics
            self.metrics.total_agents = len(self.agents)
            self.metrics.completed_tasks = sum(
                1 for output in agent_outputs 
                if isinstance(output, AgentOutput) and not output.error
            )
            self.metrics.failed_tasks = len(agent_outputs) - self.metrics.completed_tasks
            
            # Create unified output schema
            input_schema = UnifiedOutputSchemaInput(
                name=self.name,
                description=self.description,
                swarm_type="concurrent",
                workflow_type=self.workflow_type,
                agents=agent_configs,
                max_loops=self.max_loops,
                timeout=self.timeout,
                workflow_config=self.workflow_config
            )
            
            return UnifiedOutputSchema(
                input=input_schema,
                outputs=[out for out in agent_outputs if isinstance(out, AgentOutput)],
                metrics=self.metrics,
                state=self.state
            )

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save metadata with enhanced error handling"""
        if self.auto_save:
            try:
                logger.info(f"Saving metadata to {self.metadata_output_path}")
                create_file_in_folder(
                    os.getenv("WORKSPACE_DIR"),
                    self.metadata_output_path,
                    metadata
                )
            except Exception as e:
                logger.error(f"Error saving metadata: {e}")
                raise

    @output_schema
    def _run(
        self, task: str, img: str, *args, **kwargs
    ) -> UnifiedOutputSchema:
        """Run the workflow with unified output schema"""
        logger.info(f"Running concurrent workflow with {len(self.agents)} agents.")
        
        try:
            output_schema = asyncio.run(
                self._execute_agents_concurrently(task, img, *args, **kwargs)
            )
            
            if self.return_str_on:
                return "\n".join(
                    f"Agent: {output.agent_name}\nResult: {output.result}\n"
                    for output in output_schema.outputs
                )
            
            return output_schema
            
        except Exception as e:
            logger.error(f"Error in workflow execution: {e}")
            raise

    # Rest of the class implementation remains the same...
    # (run, run_batched, run_async, run_batched_async, run_parallel, run_parallel_async)

    def run(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        is_last: bool = False,
        device: str = "cpu",  # gpu
        device_id: int = 0,
        all_cores: bool = True,  # Defaults to using all available cores
        all_gpus: bool = False,
        *args,
        **kwargs,
    ) -> Any:
        """
        Executes the agent's run method on a specified device.

        This method attempts to execute the agent's run method on a specified device, either CPU or GPU. It logs the device selection and the number of cores or GPU ID used. If the device is set to CPU, it can use all available cores or a specific core specified by `device_id`. If the device is set to GPU, it uses the GPU specified by `device_id`.

        Args:
            task (Optional[str], optional): The task to be executed. Defaults to None.
            img (Optional[str], optional): The image to be processed. Defaults to None.
            is_last (bool, optional): Indicates if this is the last task. Defaults to False.
            device (str, optional): The device to use for execution. Defaults to "cpu".
            device_id (int, optional): The ID of the GPU to use if device is set to "gpu". Defaults to 0.
            all_cores (bool, optional): If True, uses all available CPU cores. Defaults to True.
            all_gpus (bool, optional): If True, uses all available GPUS. Defaults to True.
            *args: Additional positional arguments to be passed to the execution method.
            **kwargs: Additional keyword arguments to be passed to the execution method.

        Returns:
            Any: The result of the execution.

        Raises:
            ValueError: If an invalid device is specified.
            Exception: If any other error occurs during execution.
        """
        if task is not None:
            self.tasks.append(task)

        try:
            logger.info(f"Attempting to run on device: {device}")
            if device == "cpu":
                logger.info("Device set to CPU")
                if all_cores is True:
                    count = os.cpu_count()
                    logger.info(
                        f"Using all available CPU cores: {count}"
                    )
                else:
                    count = device_id
                    logger.info(f"Using specific CPU core: {count}")

                return execute_with_cpu_cores(
                    count, self._run, task, img, *args, **kwargs
                )

            elif device == "gpu":
                logger.info("Device set to GPU")
                return execute_on_gpu(
                    device_id, self._run, task, img, *args, **kwargs
                )

            elif all_gpus is True:
                return execute_on_multiple_gpus(
                    [int(gpu) for gpu in list_available_gpus()],
                    self._run,
                    task,
                    img,
                    *args,
                    **kwargs,
                )
            else:
                raise ValueError(
                    f"Invalid device specified: {device}. Supported devices are 'cpu' and 'gpu'."
                )
        except ValueError as e:
            logger.error(f"Invalid device specified: {e}")
            raise e
        except Exception as e:
            logger.error(f"An error occurred during execution: {e}")
            raise e

    def run_batched(
        self, tasks: List[str]
    ) -> List[Union[Dict[str, Any], str]]:
        """
        Runs the workflow for a batch of tasks, executes agents concurrently for each task, and saves metadata in a production-grade manner.

        Args:
            tasks (List[str]): A list of tasks or queries to give to all agents.

        Returns:
            List[Union[Dict[str, Any], str]]: A list of final metadata for each task, either as a dictionary or a string.

        Example:
            >>> tasks = ["Task 1", "Task 2"]
            >>> results = workflow.run_batched(tasks)
            >>> print(results)
        """
        results = []
        for task in tasks:
            result = self.run(task)
            results.append(result)
        return results

    def run_async(self, task: str) -> asyncio.Future:
        """
        Runs the workflow asynchronously for the given task, executes agents concurrently, and saves metadata in a production-grade manner.

        Args:
            task (str): The task or query to give to all agents.

        Returns:
            asyncio.Future: A future object representing the asynchronous operation.

        Example:
            >>> async def run_async_example():
            >>>     future = workflow.run_async(task="Example task")
            >>>     result = await future
            >>>     print(result)
        """
        logger.info(
            f"Running concurrent workflow asynchronously with {len(self.agents)} agents."
        )
        return asyncio.ensure_future(self.run(task))

    def run_batched_async(
        self, tasks: List[str]
    ) -> List[asyncio.Future]:
        """
        Runs the workflow asynchronously for a batch of tasks, executes agents concurrently for each task, and saves metadata in a production-grade manner.

        Args:
            tasks (List[str]): A list of tasks or queries to give to all agents.

        Returns:
            List[asyncio.Future]: A list of future objects representing the asynchronous operations for each task.

        Example:
            >>> tasks = ["Task 1", "Task 2"]
            >>> futures = workflow.run_batched_async(tasks)
            >>> results = await asyncio.gather(*futures)
            >>> print(results)
        """
        futures = []
        for task in tasks:
            future = self.run_async(task)
            futures.append(future)
        return futures

    def run_parallel(
        self, tasks: List[str]
    ) -> List[Union[Dict[str, Any], str]]:
        """
        Runs the workflow in parallel for a batch of tasks, executes agents concurrently for each task, and saves metadata in a production-grade manner.

        Args:
            tasks (List[str]): A list of tasks or queries to give to all agents.

        Returns:
            List[Union[Dict[str, Any], str]]: A list of final metadata for each task, either as a dictionary or a string.

        Example:
            >>> tasks = ["Task 1", "Task 2"]
            >>> results = workflow.run_parallel(tasks)
            >>> print(results)
        """
        with ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            futures = {
                executor.submit(self.run, task): task
                for task in tasks
            }
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
        return results

    def run_parallel_async(
        self, tasks: List[str]
    ) -> List[asyncio.Future]:
        """
        Runs the workflow in parallel asynchronously for a batch of tasks, executes agents concurrently for each task, and saves metadata in a production-grade manner.

        Args:
            tasks (List[str]): A list of tasks or queries to give to all agents.

        Returns:
            List[asyncio.Future]: A list of future objects representing the asynchronous operations for each task.

        Example:
            >>> tasks = ["Task 1", "Task 2"]
            >>> futures = workflow.run_parallel_async(tasks)
            >>> results = await asyncio.gather(*futures)
            >>> print(results)
        """
        futures = []
        for task in tasks:
            future = self.run_async(task)
            futures.append(future)
        return futures


# if __name__ == "__main__":
#     # Assuming you've already initialized some agents outside of this class
#     model = OpenAIChat(
#         api_key=os.getenv("OPENAI_API_KEY"),
#         model_name="gpt-4o-mini",
#         temperature=0.1,
#     )
#     agents = [
#         Agent(
#             agent_name=f"Financial-Analysis-Agent-{i}",
#             system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
#             llm=model,
#             max_loops=1,
#             autosave=True,
#             dashboard=False,
#             verbose=True,
#             dynamic_temperature_enabled=True,
#             saved_state_path=f"finance_agent_{i}.json",
#             user_name="swarms_corp",
#             retry_attempts=1,
#             context_length=200000,
#             return_step_meta=False,
#         )
#         for i in range(3)  # Adjust number of agents as needed
#     ]

#     # Initialize the workflow with the list of agents
#     workflow = ConcurrentWorkflow(
#         agents=agents,
#         metadata_output_path="agent_metadata_4.json",
#         return_str_on=True,
#     )

#     # Define the task for all agents
#     task = "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria?"

#     # Run the workflow and save metadata
#     metadata = workflow.run(task)
#     print(metadata)
