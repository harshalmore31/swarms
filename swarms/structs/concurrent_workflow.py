import concurrent.futures
import asyncio
import os
import time
import threading
from typing import Callable, List, Optional, Union

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.formatter import formatter

logger = initialize_logger(log_folder="concurrent_workflow")


class ConcurrentWorkflow(BaseSwarm):
    """
    Represents a concurrent workflow that executes multiple agents concurrently in a production-grade manner.
    Features include:
    - Real-time streaming dashboard with per-agent streaming output
    - Async streaming support for concurrent agents
    - Enhanced error handling and retries
    - Input validation

    Args:
        name (str): The name of the workflow. Defaults to "ConcurrentWorkflow".
        description (str): The description of the workflow. Defaults to "Execution of multiple agents concurrently".
        agents (List[Agent]): The list of agents to be executed concurrently. Defaults to an empty list.
        metadata_output_path (str): The path to save the metadata output. Defaults to "agent_metadata.json".
        auto_save (bool): Flag indicating whether to automatically save the metadata. Defaults to False.
        output_type (str): The type of output format. Defaults to "dict".
        max_loops (int): The maximum number of loops for each agent. Defaults to 1.
        return_str_on (bool): Flag indicating whether to return the output as a string. Defaults to False.
        auto_generate_prompts (bool): Flag indicating whether to auto-generate prompts for agents. Defaults to False.
        return_entire_history (bool): Flag indicating whether to return the entire conversation history. Defaults to False.
        show_dashboard (bool): Flag indicating whether to show a real-time dashboard. Defaults to True.
        streaming_dashboard (bool): Flag indicating whether to enable streaming dashboard. Defaults to True.
        dashboard_refresh_rate (float): Refresh rate for the dashboard in seconds. Defaults to 0.1.

    Raises:
        ValueError: If the list of agents is empty or if the description is empty.

    Attributes:
        name (str): The name of the workflow.
        description (str): The description of the workflow.
        agents (List[Agent]): The list of agents to be executed concurrently.
        metadata_output_path (str): The path to save the metadata output.
        auto_save (bool): Flag indicating whether to automatically save the metadata.
        output_type (str): The type of output format.
        max_loops (int): The maximum number of loops for each agent.
        auto_generate_prompts (bool): Flag indicating whether to auto-generate prompts for agents.
        show_dashboard (bool): Flag indicating whether to show a real-time dashboard.
        streaming_dashboard (bool): Flag indicating whether to enable streaming dashboard.
        dashboard_refresh_rate (float): Refresh rate for the dashboard in seconds.
        agent_statuses (dict): Dictionary to track agent statuses.
        agent_streaming_content (dict): Dictionary to track real-time streaming content for each agent.
    """

    def __init__(
        self,
        name: str = "ConcurrentWorkflow",
        description: str = "Execution of multiple agents concurrently",
        agents: List[Union[Agent, Callable]] = [],
        metadata_output_path: str = "agent_metadata.json",
        auto_save: bool = True,
        output_type: str = "dict-all-except-first",
        max_loops: int = 1,
        auto_generate_prompts: bool = False,
        show_dashboard: bool = True,
        streaming_dashboard: bool = True,
        dashboard_refresh_rate: float = 0.1,
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
        self.auto_generate_prompts = auto_generate_prompts
        self.output_type = output_type
        self.show_dashboard = show_dashboard
        self.streaming_dashboard = streaming_dashboard
        self.dashboard_refresh_rate = dashboard_refresh_rate
        self.agent_statuses = {
            agent.agent_name: {"status": "pending", "output": ""}
            for agent in agents
        }
        
        # New: Track streaming content for each agent
        self.agent_streaming_content = {
            agent.agent_name: {"current_stream": "", "final_output": ""}
            for agent in agents
        }
        
        # Thread-safe lock for updating agent data
        self._update_lock = threading.Lock()
        
        # Dashboard control
        self._dashboard_task = None
        self._dashboard_running = False

        self.reliability_check()
        self.conversation = Conversation()

        if self.show_dashboard is True:
            self.agents = self.fix_agents()

    def fix_agents(self):
        """Configure agents for dashboard display"""
        if self.show_dashboard is True:
            for agent in self.agents:
                agent.print_on = False  # Disable individual agent printing
                if self.streaming_dashboard:
                    agent.streaming_on = True  # Enable streaming for dashboard
        return self.agents

    def reliability_check(self):
        try:
            if self.agents is None:
                raise ValueError(
                    "ConcurrentWorkflow: No agents provided"
                )

            if len(self.agents) == 0:
                raise ValueError(
                    "ConcurrentWorkflow: No agents provided"
                )

            if len(self.agents) == 1:
                logger.warning(
                    "ConcurrentWorkflow: Only one agent provided. With ConcurrentWorkflow, you should use at least 2+ agents."
                )
        except Exception as e:
            logger.error(
                f"ConcurrentWorkflow: Reliability check failed: {e}"
            )
            raise

    def activate_auto_prompt_engineering(self):
        """
        Activates the auto-generate prompts feature for all agents in the workflow.

        Example:
            >>> workflow = ConcurrentWorkflow(agents=[Agent()])
            >>> workflow.activate_auto_prompt_engineering()
            >>> # All agents in the workflow will now auto-generate prompts.
        """
        if self.auto_generate_prompts is True:
            for agent in self.agents:
                agent.auto_generate_prompt = True

    def _update_agent_status(self, agent_name: str, status: str, output: str = None):
        """Thread-safe method to update agent status"""
        with self._update_lock:
            self.agent_statuses[agent_name]["status"] = status
            if output is not None:
                self.agent_statuses[agent_name]["output"] = output

    def _update_agent_stream(self, agent_name: str, stream_content: str, is_final: bool = False):
        """Thread-safe method to update agent streaming content"""
        with self._update_lock:
            if is_final:
                self.agent_streaming_content[agent_name]["final_output"] = stream_content
                self.agent_streaming_content[agent_name]["current_stream"] = ""
            else:
                self.agent_streaming_content[agent_name]["current_stream"] = stream_content

    def _get_dashboard_data(self):
        """Get current dashboard data in a thread-safe way"""
        with self._update_lock:
            agents_data = []
            for agent in self.agents:
                agent_name = agent.agent_name
                status = self.agent_statuses[agent_name]["status"]
                
                # Determine output to display
                if status == "running" and self.streaming_dashboard:
                    # Show streaming content with live tokens
                    stream_content = self.agent_streaming_content[agent_name]["current_stream"]
                    display_output = stream_content if stream_content else "🔄 Initializing..."
                else:
                    # Show final output or status message
                    final_output = self.agent_streaming_content[agent_name]["final_output"]
                    status_output = self.agent_statuses[agent_name]["output"]
                    display_output = final_output or status_output
                
                agents_data.append({
                    "name": agent_name,
                    "status": status,
                    "output": display_output
                })
            
            return agents_data

    async def _dashboard_updater(self, title: str = "🤖 Agent Dashboard"):
        """Async task to continuously update the dashboard"""
        self._dashboard_running = True
        
        try:
            while self._dashboard_running:
                if self.show_dashboard:
                    agents_data = self._get_dashboard_data()
                    
                    # Check if all agents are done
                    all_done = all(
                        data["status"] in ["completed", "error"] 
                        for data in agents_data
                    )
                    
                    formatter.print_agent_dashboard(
                        agents_data, 
                        title, 
                        is_final=all_done
                    )
                    
                    if all_done:
                        break
                
                await asyncio.sleep(self.dashboard_refresh_rate)
                
        except Exception as e:
            logger.error(f"Dashboard updater error: {e}")
        finally:
            self._dashboard_running = False

    async def _run_agent_async(self, agent: Agent, task: str, img: Optional[str] = None, imgs: Optional[List[str]] = None):
        """Run a single agent asynchronously with streaming support"""
        agent_name = agent.agent_name
        
        try:
            # Update status to running
            self._update_agent_status(agent_name, "running")
            
            # Create a custom streaming callback for this agent
            original_print_on = agent.print_on
            agent.print_on = False  # Ensure agent doesn't print directly
            
            # If streaming is enabled, we need to capture the streaming output
            if self.streaming_dashboard and agent.streaming_on:
                # Override the agent's streaming behavior to capture output
                accumulated_output = ""
                
                # Store original LLM streaming setting
                original_stream = getattr(agent.llm, 'stream', False) if agent.llm else False
                
                # Enable streaming in the LLM
                if agent.llm:
                    agent.llm.stream = True
                
                # Add direct LLM response interception
                original_llm_run = agent.llm.run if agent.llm else None
                
                # Create wrapper for LLM run to capture streaming
                def llm_run_wrapper(*args, **kwargs):
                    nonlocal accumulated_output
                    
                    # Call original LLM run
                    response = original_llm_run(*args, **kwargs)
                    
                    # If response is a generator (streaming), wrap it
                    if hasattr(response, '__iter__') and not isinstance(response, str):
                        def stream_wrapper():
                            nonlocal accumulated_output
                            for chunk in response:
                                # Extract content from chunk
                                if hasattr(chunk, 'choices') and chunk.choices and chunk.choices[0].delta.content:
                                    content = chunk.choices[0].delta.content
                                    accumulated_output += content
                                    # Update dashboard with streaming content
                                    self._update_agent_stream(agent_name, accumulated_output)
                                yield chunk
                        return stream_wrapper()
                    else:
                        # Non-streaming response
                        return response
                
                # Run the agent in a thread to avoid blocking
                def run_with_stream_capture():
                    nonlocal accumulated_output
                    
                    # Replace LLM run method temporarily
                    if agent.llm and original_llm_run:
                        agent.llm.run = llm_run_wrapper
                    
                    # Create a custom chunk callback for real-time updates
                    def stream_chunk_callback(chunk):
                        nonlocal accumulated_output
                        accumulated_output += chunk
                        # Update the dashboard immediately with new content
                        self._update_agent_stream(agent_name, accumulated_output)
                    
                    # Monkey patch the streaming panel to capture content
                    original_print_streaming = formatter.print_streaming_panel
                    
                    def capture_streaming_panel(streaming_response, title="", style=None, collect_chunks=False, on_chunk_callback=None):
                        # Always enable chunk collection and add our callback
                        return original_print_streaming(
                            streaming_response, 
                            title, 
                            style, 
                            collect_chunks=True, 
                            on_chunk_callback=stream_chunk_callback
                        )
                    
                    # Temporarily replace the streaming function
                    formatter.print_streaming_panel = capture_streaming_panel
                    
                    try:
                        # Run the agent
                        output = agent.run(task=task, img=img, imgs=imgs)
                        return output
                    finally:
                        # Restore original functions
                        formatter.print_streaming_panel = original_print_streaming
                        if agent.llm and original_llm_run:
                            agent.llm.run = original_llm_run
                
                # Run in thread and await completion
                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(None, run_with_stream_capture)
                
                # Restore original LLM streaming setting
                if agent.llm:
                    agent.llm.stream = original_stream
                    
            else:
                # Non-streaming execution
                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(
                    None, 
                    lambda: agent.run(task=task, img=img, imgs=imgs)
                )
            
            # Restore original print setting
            agent.print_on = original_print_on
            
            # Update status to completed with final output
            self._update_agent_status(agent_name, "completed", str(output))
            self._update_agent_stream(agent_name, str(output), is_final=True)
            
            return output
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self._update_agent_status(agent_name, "error", error_msg)
            self._update_agent_stream(agent_name, error_msg, is_final=True)
            logger.error(f"Agent {agent_name} failed: {str(e)}")
            return error_msg

    async def run_async_with_streaming_dashboard(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
    ):
        """
        Executes all agents concurrently with real-time streaming dashboard.
        """
        try:
            self.conversation.add(role="User", content=task)

            # Reset agent statuses and streaming content
            for agent in self.agents:
                self._update_agent_status(agent.agent_name, "pending", "")
                self._update_agent_stream(agent.agent_name, "", is_final=False)

            # Start dashboard updater task
            dashboard_task = None
            if self.show_dashboard:
                dashboard_task = asyncio.create_task(
                    self._dashboard_updater("🤖 Concurrent Agent Streaming Dashboard")
                )

            # Create agent tasks
            agent_tasks = [
                self._run_agent_async(agent, task, img, imgs)
                for agent in self.agents
            ]

            # Wait for all agents to complete
            results = await asyncio.gather(*agent_tasks, return_exceptions=True)

            # Stop dashboard updater
            if dashboard_task:
                self._dashboard_running = False
                await dashboard_task

            # Process results
            final_results = []
            for i, (agent, result) in enumerate(zip(self.agents, results)):
                if isinstance(result, Exception):
                    error_msg = f"Error: {str(result)}"
                    final_results.append((agent.agent_name, error_msg))
                    logger.error(f"Agent {agent.agent_name} failed: {str(result)}")
                else:
                    final_results.append((agent.agent_name, result))

            # Add all results to conversation
            for agent_name, output in final_results:
                self.conversation.add(role=agent_name, content=str(output))

            # Show final dashboard
            if self.show_dashboard:
                agents_data = self._get_dashboard_data()
                formatter.print_agent_dashboard(
                    agents_data, 
                    "🎉 Final Concurrent Streaming Dashboard", 
                    is_final=True
                )

            return history_output_formatter(
                conversation=self.conversation,
                type=self.output_type,
            )
            
        except Exception as e:
            logger.error(f"Error in async streaming workflow: {e}")
            raise
        finally:
            # Clean up dashboard display
            if self.show_dashboard:
                formatter.stop_dashboard()

    def display_agent_dashboard(
        self,
        title: str = "🤖 Agent Dashboard",
        is_final: bool = False,
    ) -> None:
        """
        Displays the current status of all agents in a beautiful dashboard format.

        Args:
            title (str): The title of the dashboard.
            is_final (bool): Flag indicating whether this is the final dashboard.
        """
        agents_data = self._get_dashboard_data()
        formatter.print_agent_dashboard(agents_data, title, is_final)

    def run_with_dashboard(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
    ):
        """
        Executes all agents in the workflow concurrently on the given task.
        Now includes real-time dashboard updates.
        """
        try:
            self.conversation.add(role="User", content=task)

            # Reset agent statuses
            for agent in self.agents:
                self.agent_statuses[agent.agent_name] = {
                    "status": "pending",
                    "output": "",
                }

            # Display initial dashboard if enabled
            if self.show_dashboard:
                self.display_agent_dashboard()

            # Use 95% of available CPU cores for optimal performance
            max_workers = int(os.cpu_count() * 0.95)

            # Create a list to store all futures and their results
            futures = []
            results = []

            def run_agent_with_status(agent, task, img, imgs):
                try:
                    # Update status to running
                    self.agent_statuses[agent.agent_name][
                        "status"
                    ] = "running"
                    if self.show_dashboard:
                        self.display_agent_dashboard()

                    # Run the agent
                    output = agent.run(task=task, img=img, imgs=imgs)

                    # Update status to completed
                    self.agent_statuses[agent.agent_name][
                        "status"
                    ] = "completed"
                    self.agent_statuses[agent.agent_name][
                        "output"
                    ] = output
                    if self.show_dashboard:
                        self.display_agent_dashboard()

                    return output
                except Exception as e:
                    # Update status to error
                    self.agent_statuses[agent.agent_name][
                        "status"
                    ] = "error"
                    self.agent_statuses[agent.agent_name][
                        "output"
                    ] = f"Error: {str(e)}"
                    if self.show_dashboard:
                        self.display_agent_dashboard()
                    raise

            # Run agents concurrently using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # Submit all agent tasks
                futures = [
                    executor.submit(
                        run_agent_with_status, agent, task, img, imgs
                    )
                    for agent in self.agents
                ]

                # Wait for all futures to complete
                concurrent.futures.wait(futures)

                # Process results in order of completion
                for future, agent in zip(futures, self.agents):
                    try:
                        output = future.result()
                        results.append((agent.agent_name, output))
                    except Exception as e:
                        logger.error(
                            f"Agent {agent.agent_name} failed: {str(e)}"
                        )
                        results.append(
                            (agent.agent_name, f"Error: {str(e)}")
                        )

            # Add all results to conversation
            for agent_name, output in results:
                self.conversation.add(role=agent_name, content=output)

            # Display final dashboard if enabled
            if self.show_dashboard:
                self.display_agent_dashboard(
                    "🎉 Final Agent Dashboard", is_final=True
                )

            return history_output_formatter(
                conversation=self.conversation,
                type=self.output_type,
            )
        finally:
            # Always clean up the dashboard display
            if self.show_dashboard:
                formatter.stop_dashboard()

    def _run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
    ):
        """
        Executes all agents in the workflow concurrently on the given task.

        Args:
            task (str): The task to be executed by all agents.
            img (Optional[str]): Optional image path for agents that support image input.
            imgs (Optional[List[str]]): Optional list of image paths for agents that support multiple image inputs.

        Returns:
            The formatted output based on the configured output_type.

        Example:
            >>> workflow = ConcurrentWorkflow(agents=[agent1, agent2])
            >>> result = workflow.run("Analyze this financial data")
            >>> print(result)
        """
        self.conversation.add(role="User", content=task)

        # Use 95% of available CPU cores for optimal performance
        max_workers = int(os.cpu_count() * 0.95)

        # Run agents concurrently using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Submit all agent tasks and store with their index
            future_to_agent = {
                executor.submit(
                    agent.run, task=task, img=img, imgs=imgs
                ): agent
                for agent in self.agents
            }

            # Collect results and add to conversation in completion order
            for future in concurrent.futures.as_completed(
                future_to_agent
            ):
                agent = future_to_agent[future]
                output = future.result()
                self.conversation.add(role=agent.name, content=output)

        return history_output_formatter(
            conversation=self.conversation,
            type=self.output_type,
        )

    def run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
    ):
        """
        Executes all agents in the workflow concurrently on the given task.
        
        If streaming_dashboard is enabled, uses async execution with real-time streaming.
        Otherwise, falls back to standard execution.
        """
        if self.streaming_dashboard and self.show_dashboard:
            # Use async streaming execution
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            try:
                return loop.run_until_complete(
                    self.run_async_with_streaming_dashboard(task, img, imgs)
                )
            except Exception as e:
                logger.error(f"Async streaming failed, falling back to standard execution: {e}")
                return self.run_with_dashboard(task, img, imgs)
        elif self.show_dashboard:
            return self.run_with_dashboard(task, img, imgs)
        else:
            return self._run(task, img, imgs)

    def batch_run(
        self,
        tasks: List[str],
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
    ):
        """
        Executes the workflow on multiple tasks sequentially.

        Args:
            tasks (List[str]): List of tasks to be executed by all agents.
            img (Optional[str]): Optional image path for agents that support image input.
            imgs (Optional[List[str]]): Optional list of image paths for agents that support multiple image inputs.

        Returns:
            List of results, one for each task.

        Example:
            >>> workflow = ConcurrentWorkflow(agents=[agent1, agent2])
            >>> tasks = ["Task 1", "Task 2", "Task 3"]
            >>> results = workflow.batch_run(tasks)
            >>> print(len(results))  # 3
        """
        return [
            self.run(task=task, img=img, imgs=imgs) for task in tasks
        ]


# if __name__ == "__main__":
#     # Assuming you've already initialized some agents outside of this class
#     agents = [
#         Agent(
#             agent_name=f"Financial-Analysis-Agent-{i}",
#             system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
#             model_name="gpt-4o",
#             max_loops=1,
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
