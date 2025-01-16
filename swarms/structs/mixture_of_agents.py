import asyncio
import time
from typing import Any, Dict, List, Optional
from datetime import datetime

from swarms.structs.agent import Agent
from swarms.telemetry.capture_sys_data import log_agent_data
from swarms.prompts.ag_prompt import aggregator_system_prompt
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

logger = initialize_logger(log_folder="mixture_of_agents")

class MixtureOfAgents:
    """
    Enhanced MixtureOfAgents class with unified output schema integration.
    """

    def __init__(
        self,
        name: str = "MixtureOfAgents",
        description: str = "A class to run a mixture of agents and aggregate their responses.",
        agents: List[Agent] = [],
        aggregator_agent: Agent = None,
        aggregator_system_prompt: str = "",
        layers: int = 3,
        max_loops: int = 1,
        timeout: Optional[float] = None
    ) -> None:
        """
        Initialize the MixtureOfAgents with unified schema support.
        """
        self.name = name
        self.description = description
        self.agents = agents
        self.aggregator_agent = aggregator_agent
        self.aggregator_system_prompt = aggregator_system_prompt
        self.layers = layers
        self.max_loops = max_loops
        self.timeout = timeout
        
        # Convert agents to unified schema format
        self.agent_configs = [
            AgentConfig(
                agent_type="reference",
                system_prompt=agent.system_prompt,
                llm_config=agent.llm_config if hasattr(agent, 'llm_config') else {},
                workflow_config={"layer": 0}
            ) for agent in self.agents
        ]
        
        # Add aggregator config
        if aggregator_agent:
            self.agent_configs.append(
                AgentConfig(
                    agent_type="aggregator",
                    system_prompt=aggregator_system_prompt,
                    llm_config=aggregator_agent.llm_config if hasattr(aggregator_agent, 'llm_config') else {},
                    workflow_config={"layer": layers}
                )
            )

        # Initialize unified output schema
        self.output_schema = UnifiedOutputSchema(
            input=UnifiedOutputSchemaInput(
                name=name,
                description=description,
                swarm_type="MixtureOfAgents",
                workflow_type="layered",
                max_loops=max_loops,
                agents=self.agent_configs,
                workflow_config={
                    "layers": layers,
                    "aggregator_system_prompt": aggregator_system_prompt
                },
                timeout=timeout
            ),
            metrics=SwarmMetrics(),
            state=SwarmState(status="initialized")
        )

        self.reliability_check()

    def reliability_check(self) -> None:
        """
        Enhanced reliability check with state updates.
        """
        logger.info("Checking reliability of MixtureOfAgents configuration")

        try:
            if not self.agents:
                raise ValueError("No reference agents provided")
            if not self.aggregator_agent:
                raise ValueError("No aggregator agent provided")
            if not self.aggregator_system_prompt:
                raise ValueError("No aggregator system prompt provided")
            if self.layers < 1:
                raise ValueError("Layers must be greater than 0")

            self.output_schema.update_state(status="validated")
            logger.info("Reliability check passed")
            
        except Exception as e:
            self.output_schema.update_state(status="error")
            logger.error(f"Reliability check failed: {str(e)}")
            raise

    def _get_final_system_prompt(self, system_prompt: str, results: List[str]) -> str:
        """
        Construct system prompt with previous responses.
        """
        return system_prompt + "\n" + "\n".join(
            [f"{i+1}. {str(element)}" for i, element in enumerate(results)]
        )

    async def _run_agent_async(
        self,
        agent: Agent,
        task: str,
        layer: int = 0,
        prev_responses: Optional[List[str]] = None,
    ) -> str:
        """
        Enhanced async agent execution with unified output tracking.
        """
        try:
            # Create agent output entry
            agent_output = AgentOutput(
                agent_name=agent.agent_name,
                config=self.agent_configs[layer],
                state={"layer": layer}
            )

            # Update system prompt if needed
            if prev_responses:
                system_prompt = self._get_final_system_prompt(
                    self.aggregator_system_prompt, prev_responses
                )
                agent.system_prompt = system_prompt
                agent_output.config.system_prompt = system_prompt

            # Record start time
            start_time = datetime.now()

            # Run agent
            response = await asyncio.to_thread(agent.run, task)

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()

            # Add step to agent output
            agent_output.add_step(
                role=agent.agent_name,
                content=response,
                step_type="inference",
                duration=duration
            )

            # Add performance metrics
            agent_output.performance_metrics.update({
                "duration": duration,
                "tokens_used": getattr(agent, "total_tokens", 0),
                "layer": layer
            })

            # Add to unified output schema
            self.output_schema.add_agent_output(agent_output)

            return response

        except Exception as e:
            logger.error(f"Error running agent {agent.agent_name}: {str(e)}")
            self.output_schema.metrics.error_count += 1
            raise

    async def _run_async(self, task: str) -> None:
        """
        Enhanced async execution with comprehensive state tracking.
        """
        try:
            self.output_schema.update_state(status="running", current_phase="initial_layer")

            # Initial layer
            results = await asyncio.gather(
                *[self._run_agent_async(agent, task, i) 
                  for i, agent in enumerate(self.agents)]
            )

            # Intermediate layers
            for layer in range(1, self.layers - 1):
                self.output_schema.update_state(
                    current_phase=f"intermediate_layer_{layer}"
                )
                results = await asyncio.gather(
                    *[self._run_agent_async(
                        agent, task, layer, prev_responses=results
                    ) for agent in self.agents]
                )

            # Final aggregation
            self.output_schema.update_state(current_phase="aggregation")
            final_result = await self._run_agent_async(
                self.aggregator_agent,
                task,
                self.layers - 1,
                prev_responses=results
            )

            # Add final result
            self.output_schema.add_intermediate_result(
                {"final_aggregation": final_result},
                phase="completion"
            )

            # Update final state
            self.output_schema.update_state(
                status="completed",
                current_phase="finished"
            )

        except Exception as e:
            self.output_schema.update_state(status="error")
            logger.error(f"Error in async execution: {str(e)}")
            raise

    def run(self, task: str) -> Dict[str, Any]:
        """
        Enhanced synchronous wrapper with unified output.
        """
        try:
            # Execute async workflow
            asyncio.run(self._run_async(task))

            # Log telemetry data
            log_agent_data(self.output_schema.model_dump())

            return self.output_schema.model_dump()

        except Exception as e:
            logger.error(f"Error in mixture execution: {str(e)}")
            self.output_schema.update_state(status="error")
            raise