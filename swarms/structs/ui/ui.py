import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional
import json
import time
import asyncio
import gradio as gr
import re

# Import necessary classes and functions from swarms library
from swarms.structs.agent import Agent
from swarms.structs.concurrent_workflow import ConcurrentWorkflow
from swarms.structs.mixture_of_agents import MixtureOfAgents
from swarms.structs.rearrange import AgentRearrange
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm
from swarms.structs.swarm_matcher import swarm_matcher, SwarmMatcher, SwarmMatcherConfig, initialize_swarm_types
from swarms.structs.swarm_router import SwarmRouter
from swarms.utils.loguru_logger import initialize_logger
from groq_model import OpenAIChat # Import OpenAIChat from the correct location
from swarms.utils.file_processing import create_file_in_folder
from doc_master import doc_master



# Initialize logger
logger = initialize_logger(log_folder="swarm_ui")


# Load environment variables
load_dotenv()

# Get the OpenAI API key from the environment variable
api_key = os.getenv("GROQ_API_KEY")

# Model initialization
model = OpenAIChat(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=api_key,
    model_name="llama-3.1-70b-versatile",
    temperature=0.1,
)

# Define the path to agent_prompts.json
PROMPT_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_prompts.json")
logger.info(f"Loading prompts from: {PROMPT_JSON_PATH}")

# Global log storage
execution_logs = []

def log_event(level: str, message: str, metadata: Optional[Dict] = None):
    """
    Log an event and store it in the execution logs.

    Args:
        level: Log level (info, warning, error, etc.)
        message: Log message
        metadata: Optional metadata dictionary
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "level": level,
        "message": message,
        "metadata": metadata or {}
    }
    execution_logs.append(log_entry)

    # Also log to the logger
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message)


def get_logs(router: Optional['SwarmRouter'] = None) -> List[str]:
    """
    Get formatted logs from both the execution logs and router logs if available.

    Args:
        router: Optional SwarmRouter instance to get additional logs from
    
    Returns:
        List of formatted log strings
    """
    formatted_logs = []

    # Add execution logs
    for log in execution_logs:
        metadata_str = ""
        if log["metadata"]:
            metadata_str = f" | Metadata: {json.dumps(log['metadata'])}"
        formatted_logs.append(
            f"[{log['timestamp']}] {log['level'].upper()}: {log['message']}{metadata_str}"
        )

    # Add router logs if available
    if router and hasattr(router, 'get_logs'):
        try:
            router_logs = router.get_logs()
            formatted_logs.extend([
                f"[{log.timestamp}] ROUTER - {log.level}: {log.message}"
                for log in router_logs
            ])
        except Exception as e:
            formatted_logs.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Failed to get router logs: {str(e)}")

    return formatted_logs


def clear_logs():
    """Clear the execution logs."""
    execution_logs.clear()


def load_prompts_from_json() -> Dict[str, str]:
    """Robust prompt loading with comprehensive error handling."""
    try:
        if not os.path.exists(PROMPT_JSON_PATH):
            error_msg = f"Prompts file not found at: {PROMPT_JSON_PATH}"
            log_event("error", error_msg)
            # Load default prompts
            return {
                "agent.data_extractor": "You are a data extraction agent...",
                "agent.summarizer": "You are a summarization agent...",
                "agent.onboarding_agent": "You are an onboarding agent..."
            }

        with open(PROMPT_JSON_PATH, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON in prompts file: {str(e)}"
                log_event("error", error_msg)
                raise
                
            if not isinstance(data, dict):
                error_msg = "Prompts file must contain a JSON object"
                log_event("error", error_msg)
                raise ValueError(error_msg)
                
            prompts = {}
            for agent_name, details in data.items():
                if not isinstance(details, dict) or "system_prompt" not in details:
                    log_event("warning", f"Skipping invalid agent config: {agent_name}")
                    continue
                    
                prompts[f"agent.{agent_name}"] = details["system_prompt"]
            
            if not prompts:
                error_msg = "No valid prompts found in prompts file"
                log_event("error", error_msg)
                # Load default prompts
                return {
                    "agent.data_extractor": "You are a data extraction agent...",
                    "agent.summarizer": "You are a summarization agent...",
                    "agent.onboarding_agent": "You are an onboarding agent..."
                }
                
            log_event("info", f"Successfully loaded {len(prompts)} prompts from JSON")
            return prompts
            
    except Exception as e:
        error_msg = f"Error loading prompts: {str(e)}"
        log_event("error", error_msg)
        # Load default prompts
        return {
            "agent.data_extractor": "You are a data extraction agent...",
            "agent.summarizer": "You are a summarization agent...",
            "agent.onboarding_agent": "You are an onboarding agent..."
        }


# Load prompts
AGENT_PROMPTS = load_prompts_from_json()

def initialize_agents(
    data_temp: float,
    sum_temp: float,
    agent_keys: List[str]
) -> List[Agent]:
    """Enhanced agent initialization with more robust configuration."""
    agents = []
    seen_names = set()
    for agent_key in agent_keys:
        if agent_key not in AGENT_PROMPTS:
            raise ValueError(f"Invalid agent key: {agent_key}")

        agent_prompt = AGENT_PROMPTS[agent_key]
        agent_name = agent_key.split('.')[-1]
        
        # Ensure unique agent names
        base_name = agent_name
        counter = 1
        while agent_name in seen_names:
            agent_name = f"{base_name}_{counter}"
            counter += 1
        seen_names.add(agent_name)
      
        agent = Agent(
            agent_name=f"Agent-{agent_name}",
            system_prompt=agent_prompt,
            llm=model,
            max_loops=1,
            autosave=True,
            verbose=True,
            dynamic_temperature_enabled=True,
            saved_state_path=f"agent_{agent_name}.json",
            user_name="pe_firm",
            retry_attempts=1,
            context_length=200000,
            output_type="string",
            temperature=data_temp,
        )
        agents.append(agent)

    return agents

async def execute_task(task: str, max_loops: int, data_temp: float, sum_temp: float,
                        swarm_type: str, agent_keys: List[str], flow: str = None) -> Tuple[Dict[str, str], 'SwarmRouter', str]:
    """
    Enhanced task execution with comprehensive error handling and result processing.
    """
    start_time = time.time()
    log_event("info", f"Starting task execution: {task}")

    try:
        # Initialize agents
        try:
            agents = initialize_agents(data_temp, sum_temp, agent_keys)
            log_event("info", f"Successfully initialized {len(agents)} agents")
        except Exception as e:
            error_msg = f"Agent initialization error: {str(e)}"
            log_event("error", error_msg)
            return {}, None, error_msg
        
        # Create a SwarmRouter to manage the different swarm types
        router_kwargs = {
            "name": "multi-agent-workflow",
            "description": f"Executing {swarm_type} workflow",
            "max_loops": max_loops,
            "agents": agents,
            "autosave": True,
            "return_json": True,
            "output_type": "string"
        }
        
        # Swarm-specific configurations
        if swarm_type == "SpreadSheetSwarm":
            output_dir = "swarm_outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a sanitized filename using only safe characters
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"swarm_output_{timestamp}.csv"
            output_path = os.path.join(output_dir, output_file)
            
            # Validate the output path
            try:
                # Ensure the path is valid and writable
                with open(output_path, 'w') as f:
                    pass
                os.remove(output_path)  # Clean up the test file
            except OSError as e:
                error_msg = f"Invalid output path: {str(e)}"
                log_event("error", error_msg)
                return {}, None, error_msg
            
            # Create and initialize SpreadSheetSwarm
            try:
                swarm = SpreadSheetSwarm(
                    agents=agents,
                    max_loops=max_loops,
                    name="spreadsheet-swarm",
                    description="SpreadSheet processing workflow",
                    save_file_path=output_path,
                    workspace_dir=output_dir,
                    llm=model,
                    autosave=True
                )
            except Exception as e:
                error_msg = f"Failed to initialize SpreadSheetSwarm: {str(e)}"
                log_event("error", error_msg)
                return {}, None, error_msg
            
            # Execute the swarm with proper error handling
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(lambda: swarm.run(task=task)),
                    timeout=900  # 15 minutes timeout
                )
                
                # Verify the output file was created
                if not os.path.exists(output_path):
                    error_msg = "Output file was not created"
                    log_event("error", error_msg)
                    return {}, None, error_msg
                
                return {
                    "CSV File Path": output_path,
                    "Status": "Success",
                    "Message": "Spreadsheet processing completed successfully",
                    "Result": str(result)
                }, swarm, ""
                
            except asyncio.TimeoutError:
                error_msg = "SpreadSheetSwarm execution timed out after 900 seconds"
                log_event("error", error_msg)
                return {}, None, error_msg
            except Exception as e:
                error_msg = f"SpreadSheetSwarm execution error: {str(e)}"
                log_event("error", error_msg)
                return {}, None, error_msg
                
        # Rest of the swarm type handling...
        elif swarm_type == "AgentRearrange":
            if not flow:
                return {}, None, "Flow configuration is required for AgentRearrange"
            router_kwargs["flow"] = flow
            
        elif swarm_type == "MixtureOfAgents":
            if len(agents) < 2:
                return {}, None, "MixtureOfAgents requires at least 2 agents"
            router_kwargs.update({
                "aggregator_agent": agents[-1],
                "layers": max_loops
            })

        # Create router and execute task for non-SpreadSheetSwarm types
        if swarm_type != "SpreadSheetSwarm":
            try:
                timeout = 450  # Default timeout
                await asyncio.sleep(0.5)
                
                router = SwarmRouter(**router_kwargs)
                router.swarm_type = swarm_type
                
                result = await asyncio.wait_for(
                    asyncio.to_thread(router.run, task=task),
                    timeout=timeout
                )
                
                # Process results
                def process_result(result):
                    """Process and standardize the result output."""
                    if isinstance(result, str):
                        try:
                            # Attempt to parse as JSON
                            parsed_result = json.loads(result)
                            return parsed_result
                        except json.JSONDecodeError:
                            # Fallback to string result
                            return {"Final Output": result}
                    
                    if isinstance(result, dict):
                        # Clean and standardize dictionary results
                        return {str(k): str(v) for k, v in result.items()}
                    
                    return {"Final Output": str(result)}

                responses = process_result(result)
                return responses, router, ""
                
            except asyncio.TimeoutError:
                error_msg = f"Task execution timed out after {timeout} seconds"
                log_event("error", error_msg)
                return {}, None, error_msg
            except Exception as e:
                error_msg = f"Task execution error: {str(e)}"
                log_event("error", error_msg)
                return {}, None, error_msg

    except Exception as e:
        error_msg = f"Unexpected error in task execution: {str(e)}"
        log_event("error", error_msg)
        return {}, None, error_msg

def _extract_concurrent_responses(result: str, agents: List[Agent]) -> Dict[str, str]:
    """
    Extract unique responses for each agent in a ConcurrentWorkflow.

    Args:
        result (str): Full output from SwarmRouter
        agents (List[Agent]): List of agents used in the task

    Returns:
        Dict[str, str]: Unique responses for each agent
    """
    agent_responses = {}
    for agent in agents:
            # Pattern to capture "Agent Name: ... Response: ... " format
            pattern = rf"Agent Name:\s*{re.escape(agent.agent_name)}\s*Response:\s*(.+?)(?=Agent Name:|$)"

            match = re.search(pattern, result, re.DOTALL | re.IGNORECASE | re.MULTILINE)
            if match:
                agent_responses[agent.agent_name] = match.group(1).strip()
            else:
                agent_responses[agent.agent_name] = "No response from the Agent"
    return agent_responses


class UI:
    def __init__(self, theme):
        self.theme = theme
        self.blocks = gr.Blocks(theme=self.theme)
        self.components = {}  # Dictionary to store UI components

    def create_markdown(self, text, is_header=False):
        if is_header:
            markdown = gr.Markdown(f"<h1 style='color: #ffffff; text-align: center;'>{text}</h1>")
        else:
            markdown = gr.Markdown(f"<p style='color: #cccccc; text-align: center;'>{text}</p>")
        self.components[f'markdown_{text}'] = markdown
        return markdown

    def create_text_input(self, label, lines=3, placeholder=""):
        text_input = gr.Textbox(
            label=label,
            lines=lines,
            placeholder=placeholder,
            elem_classes=["custom-input"],
        )
        self.components[f'text_input_{label}'] = text_input
        return text_input

    def create_slider(self, label, minimum=0, maximum=1, value=0.5, step=0.1):
        slider = gr.Slider(
            minimum=minimum,
            maximum=maximum,
            value=value,
            step=step,
            label=label,
            interactive=True,
        )
        self.components[f'slider_{label}']
        return slider

    def create_dropdown(self, label, choices, value=None, multiselect=False):
        if not choices:
            choices = ["No options available"]
        if value is None and choices:
            value = choices[0] if not multiselect else [choices[0]]
            
        dropdown = gr.Dropdown(
            label=label,
            choices=choices,
            value=value,
            interactive=True,
            multiselect=multiselect,
        )
        self.components[f'dropdown_{label}'] = dropdown
        return dropdown
    
    def create_button(self, text, variant="primary"):
        button = gr.Button(text, variant=variant)
        self.components[f'button_{text}'] = button
        return button

    def create_text_output(self, label, lines=10, placeholder=""):
        text_output = gr.Textbox(
            label=label,
            interactive=False,
            placeholder=placeholder,
            lines=lines,
            elem_classes=["custom-output"],
        )
        self.components[f'text_output_{label}'] = text_output
        return text_output

    def create_tab(self, label, content_function):
        with gr.Tab(label):
             content_function(self)

    def set_event_listener(self, button, function, inputs, outputs):
         button.click(function, inputs=inputs, outputs=outputs)

    def get_components(self, *keys):
         if not keys:
             return self.components # return all components
         return [self.components[key] for key in keys]
    
    def create_json_output(self, label, placeholder=""):
        json_output = gr.JSON(
            label=label,
            value = {},
            elem_classes=["custom-output"],
        )
        self.components[f'json_output_{label}'] = json_output
        return json_output

    def build(self):
        return self.blocks

    def create_conditional_input(self, component, visible_when, watch_component):
        """Create an input that's only visible under certain conditions"""
        watch_component.change(
            fn=lambda x: gr.update(visible=visible_when(x)),
            inputs=[watch_component],
            outputs=[component]
        )
    
    @staticmethod
    def create_ui_theme(primary_color="red"):
        return gr.themes.Soft(
         primary_hue=primary_color,
         secondary_hue="gray",
         neutral_hue="gray",
    ).set(
        body_background_fill="#20252c",
        body_text_color="#f0f0f0",
        button_primary_background_fill=primary_color,
        button_primary_text_color="#ffffff",
        button_secondary_background_fill=primary_color,
        button_secondary_text_color="#ffffff",
        shadow_drop="0px 2px 4px rgba(0, 0, 0, 0.3)",
    )

    def create_agent_details_tab(self):
        """Create the agent details tab content."""
        with gr.Column():
            gr.Markdown("### Agent Details")
            gr.Markdown("""
            **Available Agent Types:**
            - Data Extraction Agent: Specialized in extracting relevant information
            - Summary Agent: Creates concise summaries of information
            - Analysis Agent: Performs detailed analysis of data
            
            **Swarm Types:**
            - ConcurrentWorkflow: Agents work in parallel
            - SequentialWorkflow: Agents work in sequence
            - AgentRearrange: Custom agent execution flow
            - MixtureOfAgents: Combines multiple agents with an aggregator
            - SpreadSheetSwarm: Specialized for spreadsheet operations
            - Auto: Automatically determines optimal workflow
            """)
            return gr.Column()

    def create_logs_tab(self):
        """Create the logs tab content."""
        with gr.Column():
            gr.Markdown("### Execution Logs")
            logs_display = gr.Textbox(
                label="System Logs",
                placeholder="Execution logs will appear here...",
                interactive=False,
                lines=10
            )
            return logs_display

def create_app():
    # Initialize UI
    theme = UI.create_ui_theme(primary_color="red")
    ui = UI(theme=theme)

    with ui.blocks:
        with gr.Row():
            with gr.Column(scale=4):  # Left column (80% width)
                ui.create_markdown("Swarms", is_header=True)
                ui.create_markdown(
                    "<b>The Enterprise-Grade Production-Ready Multi-Agent Orchestration Framework</b>"
                )
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Row():
                            task_input = gr.Textbox(
                                label="Task Description", 
                                placeholder="Describe your task here...",
                                lines=3
                            )
                        with gr.Row():
                            with gr.Column(scale=1):
                                # Get available agent prompts
                                available_prompts = list(AGENT_PROMPTS.keys()) if AGENT_PROMPTS else ["No agents available"]
                                agent_prompt_selector = gr.Dropdown(
                                    label="Select Agent Prompts",
                                    choices=available_prompts,
                                    value=[available_prompts[0]] if available_prompts else None,
                                    multiselect=True,
                                    interactive=True
                                )
                            with gr.Column(scale=1):
                                # Get available swarm types
                                swarm_types = [
                                    "SequentialWorkflow", "ConcurrentWorkflow", "AgentRearrange",
                                    "MixtureOfAgents", "SpreadSheetSwarm", "auto"
                                ]
                                agent_selector = gr.Dropdown(
                                    label="Select Swarm",
                                    choices=swarm_types,
                                    value=swarm_types[0],
                                    multiselect=False,
                                    interactive=True
                                )
                                
                                # Flow configuration components for AgentRearrange
                                with gr.Column(visible=False) as flow_config:
                                    flow_text = gr.Textbox(
                                        label="Agent Flow Configuration",
                                        placeholder="Enter agent flow (e.g., Agent1 -> Agent2 -> Agent3)",
                                        lines=2
                                    )
                                    gr.Markdown(
                                        """
                                        **Flow Configuration Help:**
                                        - Enter agent names separated by ' -> '
                                        - Example: Agent1 -> Agent2 -> Agent3
                                        - Use exact agent names from the prompts above
                                        """
                                    )

                    with gr.Column(scale=2, min_width=200):
                        with gr.Row():
                            max_loops_slider = gr.Slider(
                                label="Max Loops",
                                minimum=1,
                                maximum=10,
                                value=1,
                                step=1
                            )

                        with gr.Row():
                            dynamic_slider = gr.Slider(
                                label="Dynamic Temp",
                                minimum=0,
                                maximum=1,
                                value=0.1,
                                step=0.01
                            )
                        with gr.Row():
                            loading_status = gr.Textbox(
                                label="Status",
                                value="Ready",
                                interactive=False
                            )

                with gr.Row():
                    run_button = gr.Button("Run Task", variant="primary")
                    cancel_button = gr.Button("Cancel", variant="secondary")

                # Add loading indicator and status
                with gr.Row():
                    agent_output_display = gr.Textbox(
                        label="Agent Responses",
                        placeholder="Responses will appear here...",
                        interactive=False,
                        lines=10
                    )
            
                def update_flow_agents(agent_keys):
                    """Update flow agents based on selected agent prompts."""
                    if not agent_keys:
                        log_event("warning", "No agents selected for flow configuration")
                        return [], "No agents selected"
                    agent_names = [key.split('.')[-1] for key in agent_keys]
                    log_event("info", f"Updated flow agents with {len(agent_names)} agents")
                    return agent_names, "Select agents in execution order"

                def update_flow_preview(selected_flow_agents):
                    """Update flow preview based on selected agents."""
                    if not selected_flow_agents:
                        return "Flow will be shown here..."
                    flow = " -> ".join(selected_flow_agents)
                    log_event("info", f"Updated flow preview: {flow}")
                    return flow

                def update_ui_for_swarm_type(swarm_type):
                    """Update UI components based on selected swarm type."""
                    is_agent_rearrange = swarm_type == "AgentRearrange"
                    is_mixture = swarm_type == "MixtureOfAgents"
                    is_spreadsheet = swarm_type == "SpreadSheetSwarm"
                    
                    max_loops = 5 if is_mixture or is_spreadsheet else 10
                    log_event("info", f"Swarm type changed to {swarm_type}, max loops set to {max_loops}")
                    
                    # Return visibility state for flow configuration and max loops update
                    return (
                        gr.update(visible=is_agent_rearrange),  # For flow_config
                        gr.update(maximum=max_loops),  # For max_loops_slider
                        f"Selected {swarm_type}"  # For loading_status
                    )

                async def run_task_wrapper(task, max_loops, data_temp, swarm_type, agent_prompt_selector, flow_text, sum_temp):
                    """Execute the task and update the UI with progress."""
                    try:
                        if not task:
                            yield "Please provide a task description.", "Error: Missing task"
                            return
                            
                        if not agent_prompt_selector or len(agent_prompt_selector) == 0:
                            yield "Please select at least one agent.", "Error: No agents selected"
                            return
                            
                        log_event("info", f"Starting task with agents: {agent_prompt_selector}")
                        
                        # Update status
                        yield "Processing...", "Running task..."
                        
                        # Prepare flow for AgentRearrange
                        flow = None
                        if swarm_type == "AgentRearrange":
                            if not flow_text:
                                yield "Please provide the agent flow configuration.", "Error: Flow not configured"
                                return
                            flow = flow_text
                        
                        # Execute task
                        responses, router, error = await execute_task(
                            task=task,
                            max_loops=max_loops,
                            data_temp=data_temp,
                            sum_temp=sum_temp,
                            swarm_type=swarm_type,
                            agent_keys=agent_prompt_selector,
                            flow=flow
                        )
                        
                        if error:
                            yield f"Error: {error}", "Error occurred"
                            return
                        
                        # Format output based on swarm type
                        output_lines = []
                        if router.swarm_type == "AgentRearrange":
                            for key, value in responses.items():
                                output_lines.append(f"### Step {key} ###\n{value}\n{'='*50}\n")
                        elif router.swarm_type == "MixtureOfAgents":
                            # Add individual agent outputs
                            for key, value in responses.items():
                                if key != "Aggregated Summary":
                                    output_lines.append(f"### {key} ###\n{value}\n")
                            # Add aggregated summary at the end
                            if "Aggregated Summary" in responses:
                                output_lines.append(f"\n### Aggregated Summary ###\n{responses['Aggregated Summary']}\n{'='*50}\n")
                        elif router.swarm_type == "SpreadSheetSwarm":
                            output_lines.append(f"### Spreadsheet Output ###\n{responses.get('CSV File Path', 'No file path provided')}\n{'='*50}\n")
                        elif router.swarm_type == "ConcurrentWorkflow":
                            for key, value in responses.items():
                                output_lines.append(f"### {key} ###\n{value}\n{'='*50}\n")
                        else:  # SequentialWorkflow, auto
                            if isinstance(responses, dict):
                                for key, value in responses.items():
                                    output_lines.append(f"### {key} ###\n{value}\n{'='*50}\n")
                            else:
                                output_lines.append(str(responses))
                        
                        yield "\n".join(output_lines), "Completed"
                        
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        log_event("error", error_msg)
                        yield error_msg, "Error occurred"

                # Connect the update functions
                agent_selector.change(
                    fn=update_ui_for_swarm_type,
                    inputs=[agent_selector],
                    outputs=[flow_config, max_loops_slider, loading_status]
                )

                # Create event trigger
                # Create event trigger for run button
                run_event = run_button.click(
                    fn=run_task_wrapper,
                    inputs=[
                        task_input,
                        max_loops_slider,
                        dynamic_slider,
                        agent_selector,
                        agent_prompt_selector,
                        flow_text
                    ],
                    outputs=[agent_output_display, loading_status]
                )

                # Connect cancel button to interrupt processing
                def cancel_task():
                    log_event("info", "Task cancelled by user")
                    return "Task cancelled.", "Cancelled"
                
                cancel_button.click(
                    fn=cancel_task,
                    inputs=None,
                    outputs=[agent_output_display, loading_status],
                    cancels=run_event
                )

            # with gr.Column(scale=1):  # Right column
            #     with gr.Tabs():
            #         with gr.Tab("Agent Details"):
            #             gr.Markdown("""
            #             ### Available Agent Types
            #             - **Data Extraction Agent**: Specialized in extracting relevant information
            #             - **Summary Agent**: Creates concise summaries of information
            #             - **Analysis Agent**: Performs detailed analysis of data
                        

            with gr.Column(scale=1):  # Right column
                with gr.Tabs():
                    with gr.Tab("Agent Details"):
                        ui.create_agent_details_tab()
                    
                    with gr.Tab("Logs"):
                        logs_display = ui.create_logs_tab()
                        def update_logs_display():
                            """Update logs display with current logs."""
                            logs = get_logs()
                            formatted_logs = "\n".join(logs)
                            return formatted_logs
                        
                        # Update logs when tab is selected
                        logs_tab = gr.Tab("Logs")
                        logs_tab.select(fn=update_logs_display, inputs=None, outputs=[logs_display])


    return ui.build()

# Launch the app
if __name__ == "__main__":
    app = create_app()
    app.launch()