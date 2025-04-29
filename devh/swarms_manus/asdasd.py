import gradio as gr
import pandas as pd
import json
import os
from pathlib import Path
import re
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

# Import the SwarmsTaskAgent class and related classes
from sass import (
    SwarmsTaskAgent, 
    Task, 
    TaskStatus,
    TaskAnalysis
)

# Define file paths
DEFAULT_TASKS_FILE = "swarms/structs/swarms_manus/todo.csv"
DEFAULT_OUTPUT_DIR = "output"

class GradioTaskInterface:
    """Gradio interface for the Swarms Task Agent"""
    
    def __init__(
        self,
        model_name: str = "gemini/gemini-2.0-flash",
        tasks_file: str = DEFAULT_TASKS_FILE,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ):
        self.model_name = model_name
        self.tasks_file = Path(tasks_file)
        self.output_dir = Path(output_dir)
        
        # Initialize the task agent
        self.task_agent = SwarmsTaskAgent(
            model_name=model_name,
            tasks_file=tasks_file,
            output_dir=output_dir
        )
        
        # Chat history for interactive mode
        self.chat_history = []
    
    def get_tasks_df(self) -> pd.DataFrame:
        """Get tasks as a DataFrame for display"""
        if not self.task_agent.tasks:
            return pd.DataFrame(columns=["id", "name", "description", "status"])
        
        task_dicts = [task.to_dict() for task in self.task_agent.tasks]
        return pd.DataFrame(task_dicts)
    
    def get_tasks_list(self) -> str:
        """Get tasks as a formatted string for display"""
        if not self.task_agent.tasks:
            return "No tasks available"
        
        tasks_text = ""
        for task in self.task_agent.tasks:
            checkbox = "☑" if task.status == TaskStatus.COMPLETED else "☐"
            tasks_text += f"{checkbox} **Task #{task.id}**: {task.name}\n"
        
        return tasks_text
    
    def add_task(self, name: str, description: str) -> Tuple[str, str]:
        """Add a new task"""
        if not name or not description:
            return "Please provide both a name and description for the task.", self.get_tasks_list()
        
        # Generate a new ID (max existing ID + 1)
        new_id = max([t.id for t in self.task_agent.tasks], default=0) + 1
        
        new_task = Task(
            id=new_id,
            name=name,
            description=description,
            status=TaskStatus.TODO
        )
        
        self.task_agent.tasks.append(new_task)
        self.task_agent._save_tasks()
        
        return f"Added task #{new_id}: {name}", self.get_tasks_list()
    
    def update_task_status(self, task_id: str, new_status: str) -> Tuple[str, str]:
        """Update the status of a task"""
        try:
            task_id = int(task_id)
            task = next((t for t in self.task_agent.tasks if t.id == task_id), None)
            
            if not task:
                return f"Task with ID {task_id} not found.", self.get_tasks_list()
            
            # Map the status string to TaskStatus enum
            status_map = {
                "to do": TaskStatus.TODO,
                "in progress": TaskStatus.IN_PROGRESS,
                "completed": TaskStatus.COMPLETED
            }
            
            if new_status not in status_map:
                return f"Invalid status: {new_status}", self.get_tasks_list()
            
            task.status = status_map[new_status]
            self.task_agent._save_tasks()
            
            return f"Updated task #{task_id} status to {new_status}", self.get_tasks_list()
            
        except ValueError:
            return "Invalid task ID. Please enter a number.", self.get_tasks_list()
    
    def run_autonomous_mode(self, api_key: str, model_name: str) -> Tuple[str, str]:
        """Run the autonomous mode and return results"""
        # Update the model if changed
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            self.task_agent = SwarmsTaskAgent(
                model_name=model_name,
                tasks_file=str(self.tasks_file),
                output_dir=str(self.output_dir)
            )
        
        # Set API key if provided
        if api_key:
            # This is a placeholder - implement according to your API key handling mechanism
            os.environ["API_KEY"] = api_key
        
        # Get all uncompleted tasks
        pending_tasks = [t for t in self.task_agent.tasks if t.status != TaskStatus.COMPLETED]
        
        if not pending_tasks:
            return "No pending tasks found.", self.get_tasks_list()
        
        output = f"Running in Autonomous Mode - Analyzing {len(pending_tasks)} tasks\n\n"
        
        # Analyze all tasks to find those that can be done autonomously
        autonomous_tasks = []
        for task in pending_tasks:
            output += f"Analyzing task {task.id}: {task.name}\n"
            analysis = self.task_agent._analyze_task(task)
            
            if analysis.autonomous:
                output += f"✅ Task '{task.name}' can be executed autonomously.\n"
                output += f"Reasoning: {analysis.reasoning}\n\n"
                autonomous_tasks.append((task, analysis))
            else:
                output += f"❌ Task '{task.name}' cannot be executed autonomously.\n"
                output += f"Reasoning: {analysis.reasoning}\n"
                if analysis.ambiguities:
                    for ambiguity in analysis.ambiguities:
                        output += f"  - {ambiguity}\n"
                output += "\n"
        
        if not autonomous_tasks:
            output += "No tasks can be executed autonomously. Try interactive mode."
            return output, self.get_tasks_list()
        
        # Process the autonomous tasks
        output += f"\nFound {len(autonomous_tasks)} tasks that can be executed autonomously\n\n"
        for task, analysis in autonomous_tasks:
            output += f"Processing task {task.id}: {task.name}\n"
            
            # Update status to in progress
            task.status = TaskStatus.IN_PROGRESS
            self.task_agent._save_tasks()
            
            # Show execution plan
            output += "Execution Plan:\n"
            for i, step in enumerate(analysis.execution_plan, 1):
                output += f"  {i}. {step}\n"
            
            # Execute the task
            output += f"\nExecuting task: {task.name}\n"
            result = self.task_agent._execute_task(task, analysis.execution_plan)
            
            # Save the result and update status
            self.task_agent._save_result(task, result)
            task.status = TaskStatus.COMPLETED
            self.task_agent._save_tasks()
            
            output += f"Task completed: {task.name}\nResult saved to: {self.output_dir}/{task.id}_{task.name.replace(' ', '_')}.md\n\n"
        
        return output, self.get_tasks_list()
    
    def chat(self, message: str, history=None) -> List[Tuple[str, str]]:
        """Process messages in interactive chat mode"""
        if history is None:
            history = []
            
        # Handle different types of requests
        input_lower = message.lower()
        response = ""
        
        # Task listing
        if any(phrase in input_lower for phrase in ["show", "list", "display", "all tasks"]):
            tasks_list = self.get_tasks_list()
            response = f"Here are your tasks:\n\n{tasks_list}"
        
        # Working on a task - look for references
        elif any(phrase in input_lower for phrase in ["work on", "do", "help with", "start", "process", "execute"]):
            # Try to extract task ID from message
            task_id_match = re.search(r'#?(\d+)', message)
            
            if task_id_match:
                task_id = int(task_id_match.group(1))
                task = next((t for t in self.task_agent.tasks if t.id == task_id), None)
                
                if task:
                    # Analyze the task
                    response = f"Analyzing task #{task_id}: {task.name}\n\n"
                    analysis = self.task_agent._analyze_task(task)
                    
                    # Update status to in progress
                    task.status = TaskStatus.IN_PROGRESS
                    self.task_agent._save_tasks()
                    
                    # Handle the task
                    result = self.task_agent._execute_task(task, analysis.execution_plan)
                    
                    # Save the result
                    self.task_agent._save_result(task, result)
                    
                    # Mark as completed
                    task.status = TaskStatus.COMPLETED
                    self.task_agent._save_tasks()
                    
                    response += f"Task execution complete!\n\nResults:\n\n{result}\n\n"
                    response += f"Task marked as completed and saved to output directory."
                else:
                    response = f"Task with ID {task_id} not found."
            else:
                # Use AI to interpret the request
                task_context = [t.to_dict() for t in self.task_agent.tasks]
                interpretation_prompt = f"""
                The user said: "{message}"
                
                Given the available tasks:
                {json.dumps(task_context, indent=2)}
                
                Which task do you think they're referring to? Respond with just the task ID number.
                If you can't determine a specific task, respond with "unknown".
                """
                
                ai_response = self.task_agent.agent.run(interpretation_prompt)
                
                # Try to extract a task ID from the response
                try:
                    task_id_match = re.search(r'\d+', ai_response)
                    if task_id_match:
                        task_id = int(task_id_match.group(0))
                        task = next((t for t in self.task_agent.tasks if t.id == task_id), None)
                        
                        if task:
                            response = f"I think you're referring to task #{task_id}: {task.name}\n\n"
                            response += "Would you like me to work on this task? If so, please say 'work on task #" + str(task_id) + "'"
                        else:
                            response = "I couldn't identify which task you're referring to. Please specify by task ID (e.g., 'work on task #3')."
                    else:
                        response = "I'm not sure which task you're referring to. Please specify by task ID (e.g., 'work on task #3')."
                except:
                    response = "I'm not sure which task you're referring to. Please specify by task ID (e.g., 'work on task #3')."
        
        # Default: use the agent to respond to the query
        else:
            task_context = [t.to_dict() for t in self.task_agent.tasks]
            
            conversation_prompt = f"""
            User said: "{message}"
            
            Context - All available tasks:
            {json.dumps(task_context, indent=2)}
            
            Please respond to the user's message in a helpful way. If they seem to be asking
            about tasks or wanting to do something with tasks, help guide them with suggestions
            on what they can do in this system.
            """
            
            response = self.task_agent.agent.run(conversation_prompt)
        
        # Update the conversation history
        history.append((message, response))
        return history

    def view_task_output(self, task_id: str) -> str:
        """View the output file for a completed task"""
        try:
            task_id = int(task_id)
            task = next((t for t in self.task_agent.tasks if t.id == task_id), None)
            
            if not task:
                return f"Task with ID {task_id} not found."
            
            file_name = f"{task.id}_{task.name.replace(' ', '_')}.md"
            file_path = self.output_dir / file_name
            
            if not file_path.exists():
                return f"No output file found for task #{task_id}."
            
            with open(file_path, "r") as f:
                content = f.read()
                
            return content
            
        except ValueError:
            return "Invalid task ID. Please enter a number."
        except Exception as e:
            return f"Error reading output file: {str(e)}"
            
    def add_new_task_ui(self, task_name, task_description) -> Tuple[str, str]:
        """Add a new task from the UI"""
        return self.add_task(task_name, task_description)

def create_gradio_interface():
    """Create and launch the Gradio interface based on the new design"""
    # Initialize the interface
    interface = GradioTaskInterface()
    
    # Custom CSS for better styling
    custom_css = """
    .container {
        max-width: 1200px;
        margin: 0 auto;
    }
    .header {
        text-align: center;
        margin-bottom: 20px;
    }
    .setting-panel {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .task-panel {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        min-height: 200px;
    }
    .chat-panel {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 15px;
    }
    .results-panel {
        background-color: #f0fff0;
        border-radius: 10px;
        padding: 15px;
        min-height: 200px;
    }
    """
    
    # Create the Gradio app with improved layout
    with gr.Blocks(title="Swarms Autonomous Agent", css=custom_css, theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🤖 Swarms Autonomous Agent", elem_classes=["header"])
        
        with gr.Row():
            # Left side - Settings and Chat
            with gr.Column(scale=3):
                # Settings Panel
                with gr.Column(elem_classes=["setting-panel"]):
                    gr.Markdown("### ⚙️ Agent Configuration")
                    
                    with gr.Row():
                        model_input = gr.Textbox(
                            label="Model",
                            value="gemini/gemini-2.0-flash",
                            info="Select AI model to use"
                        )
                        apikey_input = gr.Textbox(
                            label="API Key",
                            placeholder="Enter your API key",
                            type="password",
                            info="Your API key is securely stored"
                        )
                    
                    with gr.Row():
                        auto_toggle = gr.Checkbox(
                            label="Auto Mode",
                            info="Enable autonomous task processing"
                        )
                        auto_btn = gr.Button(
                            "Process Tasks", 
                            variant="primary",
                            size="sm"
                        )
                
                # Process Display
                with gr.Column():
                    process_output = gr.Textbox(
                        label="Process Output",
                        placeholder="Task processing results will appear here...",
                        lines=8,
                        interactive=False
                    )
                
                # Chat Section
                with gr.Column(elem_classes=["chat-panel"]):
                    gr.Markdown("### 💬 Task Assistant")
                    
                    chatbot = gr.Chatbot(
                        height=350,
                        value=[["How can I help with your tasks today?", "I can help you manage and execute tasks. Try asking me to list your tasks or work on a specific one."]],
                        elem_id="chatbot"
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            show_label=False,
                            placeholder="Ask me about your tasks or tell me what to do...",
                            scale=8
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
            
            # Right side - Tasks and Results
            with gr.Column(scale=2):
                # New Task Creation
                with gr.Column(elem_classes=["setting-panel"]):
                    gr.Markdown("### ➕ Add New Task")
                    new_task_name = gr.Textbox(label="Task Name")
                    new_task_desc = gr.Textbox(label="Task Description", lines=2)
                    add_task_btn = gr.Button("Add Task", variant="secondary")
                    add_task_result = gr.Markdown("")
                
                # Tasks List
                with gr.Column(elem_classes=["task-panel"]):
                    task_header = gr.Markdown("### 📋 Tasks")
                    tasks_list = gr.Markdown(interface.get_tasks_list())
                    
                    with gr.Row():
                        task_id_input = gr.Textbox(label="Task ID", placeholder="Enter task ID")
                        view_task_btn = gr.Button("View Results", variant="secondary", size="sm")
                
                # Results Area
                with gr.Column(elem_classes=["results-panel"]):
                    gr.Markdown("### 📊 Task Results")
                    results_area = gr.Markdown("Select a task to view its results...")
        
        # Set up event handlers
        auto_btn.click(
            interface.run_autonomous_mode,
            inputs=[apikey_input, model_input],
            outputs=[process_output, tasks_list]
        )
        
        msg.submit(
            interface.chat,
            inputs=[msg, chatbot],
            outputs=[chatbot],
        )
        
        send_btn.click(
            interface.chat,
            inputs=[msg, chatbot],
            outputs=[chatbot],
        )
        
        add_task_btn.click(
            interface.add_new_task_ui,
            inputs=[new_task_name, new_task_desc],
            outputs=[add_task_result, tasks_list]
        )
        
        view_task_btn.click(
            interface.view_task_output,
            inputs=[task_id_input],
            outputs=[results_area]
        )
    
    return app

if __name__ == "__main__":
    # Make sure output directory exists
    if not os.path.exists(DEFAULT_OUTPUT_DIR):
        os.makedirs(DEFAULT_OUTPUT_DIR)
    
    # Launch the app
    app = create_gradio_interface()
    app.launch(share=False)  # Set share=True if you want a public link