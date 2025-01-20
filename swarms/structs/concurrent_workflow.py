import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

from swarms.schemas.base_swarm_schemas import BaseSwarmSchema, AgentInputConfig
from swarms.schemas.output_schemas import OutputSchema, AgentTaskOutput, Step, SwarmOutputFormatter
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.agent import Agent
from swarms.utils.loguru_logger import initialize_logger
from clusterops import (
    execute_on_gpu,
    execute_with_cpu_cores,
    execute_on_multiple_gpus,
    list_available_gpus,
)

logger = initialize_logger(log_folder="concurrent_workflow")

class ConcurrentWorkflow(BaseSwarm):
    """
    Enhanced concurrent workflow that executes multiple agents concurrently with unified schema support.
    """
    
    def __init__(
        self,
        name: str = "ConcurrentWorkflow",
        description: str = "Execution of multiple agents concurrently",
        agents: List[Agent] = [],
        max_loops: int = 1,
        auto_save: bool = True,
        metadata_output_path: str = "agent_metadata.json",
        max_workers: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(name=name, description=description, agents=agents, *args, **kwargs)
        
        # Initialize base configuration
        self.base_config = BaseSwarmSchema(
            name=name,
            description=description,
            agents=[
                AgentInputConfig(agent_name=agent.agent_name)
                for agent in agents
            ],
            max_loops=max_loops,
            swarm_type="ConcurrentWorkflow",
            config={"max_workers": max_workers or os.cpu_count()}
        )
        
        self.max_workers = max_workers or os.cpu_count()
        self.auto_save = auto_save
        self.metadata_output_path = metadata_output_path
        self.output_formatter = SwarmOutputFormatter()
        
    async def _run_agent(
        self,
        agent: Agent,
        task: str,
        executor: ThreadPoolExecutor,
        *args,
        **kwargs
    ) -> AgentTaskOutput:
        """Runs a single agent with tracking and output formatting."""
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        step = Step(
            name=agent.agent_name,
            task=task,
            start_time=start_time
        )
        
        try:
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(
                executor, agent.run, task, *args, **kwargs
            )
            step.output = output
            
        except Exception as e:
            logger.error(f"Error running agent {agent.agent_name}: {e}")
            step.error = str(e)
            raise
        finally:
            step.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
            step.runtime = time.time() - time.mktime(time.strptime(start_time, "%Y-%m-%d %H:%M:%S"))
        
        return AgentTaskOutput(
            agent_name=agent.agent_name,
            task=task,
            steps=[step],
            end_time=step.end_time
        )

    async def _execute_agents_concurrently(
        self, task: str, *args, **kwargs
    ) -> OutputSchema:
        """Executes multiple agents concurrently with unified output schema."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks_to_run = [
                self._run_agent(agent, task, executor, *args, **kwargs)
                for agent in self.agents
            ]
            agent_outputs = await asyncio.gather(*tasks_to_run)
            
        return OutputSchema(
            swarm_id=self.base_config.id,
            swarm_type=self.base_config.swarm_type,
            task=task,
            agent_outputs=agent_outputs,
            swarm_specific_output={
                "max_workers": self.max_workers,
                "total_agents": len(self.agents)
            }
        )

    def _run(
        self, task: str, *args, **kwargs
    ) -> Union[Dict[str, Any], str]:
        """Internal run method with output formatting."""
        logger.info(f"Running concurrent workflow with {len(self.agents)} agents.")
        
        output_schema = asyncio.run(
            self._execute_agents_concurrently(task, *args, **kwargs)
        )
        
        formatted_output = self.output_formatter.format_output(
            swarm_id=output_schema.swarm_id,
            swarm_type=output_schema.swarm_type,
            task=output_schema.task,
            agent_outputs=output_schema.agent_outputs,
            swarm_specific_output=output_schema.swarm_specific_output
        )
        
        if self.auto_save:
            with open(self.metadata_output_path, 'w') as f:
                f.write(formatted_output)
                
        return formatted_output

    def run(
        self,
        task: str,
        device: str = "cpu",
        device_id: int = 0,
        all_cores: bool = True,
        all_gpus: bool = False,
        no_use_clusterops: bool = False,
        *args,
        **kwargs
    ) -> Any:
        """
        Main execution method with device management.
        
        Args:
            task: The task to execute
            device: Device to run on ('cpu' or 'gpu')
            device_id: Specific device ID to use
            all_cores: Whether to use all CPU cores
            all_gpus: Whether to use all available GPUs
            no_use_clusterops: If True, bypasses clusterops and runs directly
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        try:
            # Direct execution without clusterops if specified
            if no_use_clusterops:
                logger.info("Bypassing clusterops and executing directly")
                return self._run(task, *args, **kwargs)
            
            logger.info(f"Attempting to run on device: {device}")
            
            if device == "cpu":
                if all_cores:
                    # Use available cores count instead of all cores
                    available_cores = len([i for i in range(os.cpu_count())])
                    count = min(available_cores, self.max_workers)
                else:
                    # For single core, use device_id directly
                    count = min(device_id, os.cpu_count() - 1) if device_id > 0 else 1
                    
                logger.info(f"Using CPU cores: {count}")
                return execute_with_cpu_cores(count, self._run, task, *args, **kwargs)
                
            elif device == "gpu":
                logger.info(f"Using GPU device {device_id}")
                return execute_on_gpu(device_id, self._run, task, *args, **kwargs)
                
            elif all_gpus:
                available_gpus = [int(gpu) for gpu in list_available_gpus()]
                if not available_gpus:
                    logger.warning("No GPUs available, falling back to CPU")
                    return self.run(task, device="cpu", all_cores=True, *args, **kwargs)
                    
                logger.info(f"Using all available GPUs: {available_gpus}")
                return execute_on_multiple_gpus(available_gpus, self._run, task, *args, **kwargs)
                
            else:
                raise ValueError(f"Invalid device configuration")
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
            if "Invalid core count" in str(e):
                logger.info("Retrying with single core execution")
                return self.run(task, device="cpu", all_cores=False, device_id=1, *args, **kwargs)
            raise

    def run_batched(
        self, tasks: List[str], *args, **kwargs
    ) -> List[str]:
        """Executes multiple tasks in batch."""
        return [self.run(task, *args, **kwargs) for task in tasks]

    def run_parallel(
        self, tasks: List[str], *args, **kwargs
    ) -> List[str]:
        """Executes multiple tasks in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.run, task, *args, **kwargs)
                for task in tasks
            ]
            return [future.result() for future in futures]

    async def run_async(
        self, task: str, *args, **kwargs
    ) -> str:
        """Executes a single task asynchronously."""
        return await asyncio.to_thread(self.run, task, *args, **kwargs)

    async def run_batched_async(
        self, tasks: List[str], *args, **kwargs
    ) -> List[str]:
        """Executes multiple tasks asynchronously."""
        return await asyncio.gather(
            *[self.run_async(task, *args, **kwargs) for task in tasks]
        )