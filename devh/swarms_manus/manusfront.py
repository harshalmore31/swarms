import gradio as gr
import requests
import json
import os
import time
from datetime import datetime
import threading
import markdown

# API endpoint configuration
API_BASE_URL = "http://localhost:5000/api"

# Custom CSS to match the provided design
custom_css = """
:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --secondary: #10b981;
    --light: #f9fafb;
    --dark: #1f2937;
    --gray: #9ca3af;
    --light-gray: #e5e7eb;
    --danger: #ef4444;
}

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif !important;
}

.app-header {
    background-color: white;
    border-bottom: 1px solid var(--light-gray);
    padding: 1rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
}

.logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-weight: 600;
    font-size: 1.25rem;
    color: var(--primary);
}

.connection-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
}

.status-indicator {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}

.status-indicator.connected {
    background-color: var(--secondary);
}

.status-indicator.disconnected {
    background-color: var(--danger);
}

.chat-message {
    max-width: 85%;
    padding: 0.75rem 1rem;
    border-radius: 1rem;
    position: relative;
    line-height: 1.5;
    margin-bottom: 0.5rem;
}

.chat-message.user {
    align-self: flex-end;
    background-color: var(--primary);
    color: white;
    border-bottom-right-radius: 0.25rem;
    margin-left: auto;
}

.chat-message.agent {
    align-self: flex-start;
    background-color: var(--light-gray);
    border-bottom-left-radius: 0.25rem;
    margin-right: auto;
}

.task-item {
    background-color: white;
    border: 1px solid var(--light-gray);
    border-radius: 0.5rem;
    padding: 1rem;
    margin-bottom: 1rem;
    cursor: pointer;
    transition: border-color 0.2s, box-shadow 0.2s;
}

.task-item:hover {
    border-color: var(--primary);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.task-item.selected {
    border-color: var(--primary);
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
}

.panel-title {
    font-weight: 600;
    margin-bottom: 0.75rem;
    color: var(--dark);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.panel-content {
    background-color: #f9fafb;
    border-radius: 0.5rem;
    padding: 1rem;
    white-space: pre-wrap;
    font-size: 0.875rem;
    line-height: 1.6;
    overflow-x: auto;
}

.status-badge {
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    border-radius: 1rem;
    display: inline-block;
    margin-left: 0.5rem;
}

.status-todo {
    background-color: #fee2e2;
    color: #b91c1c;
}

.status-in-progress {
    background-color: #fef3c7;
    color: #92400e;
}

.status-done {
    background-color: #d1fae5;
    color: #065f46;
}

button.primary-button {
    background-color: var(--primary) !important;
}

button.primary-button:hover {
    background-color: var(--primary-dark) !important;
}
"""

# Utility functions to interact with the backend API
def get_all_tasks():
    """Fetch all tasks from the backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/tasks")
        return response.json().get('tasks', [])
    except Exception as e:
        return [{"name": f"Error loading tasks: {str(e)}", "status": "error"}]

def add_new_task(name, description, is_auto):
    """Add a new task to the backend"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/tasks",
            json={"name": name, "description": description, "is_auto": is_auto}
        )
        return response.json()
    except Exception as e:
        return {"success": False, "message": f"Error adding task: {str(e)}"}

def update_task_status(task_id, new_status):
    """Update a task's status"""
    try:
        response = requests.put(
            f"{API_BASE_URL}/tasks/{task_id}",
            json={"status": new_status}
        )
        return response.json()
    except Exception as e:
        return {"success": False, "message": f"Error updating task: {str(e)}"}

def delete_task_by_id(task_id):
    """Delete a task by ID"""
    try:
        response = requests.delete(f"{API_BASE_URL}/tasks/{task_id}")
        return response.json()
    except Exception as e:
        return {"success": False, "message": f"Error deleting task: {str(e)}"}

def analyze_task_by_id(task_id):
    """Analyze a task using the Swarms Agent"""
    try:
        response = requests.post(f"{API_BASE_URL}/tasks/{task_id}/analyze")
        return response.json()
    except Exception as e:
        return {"success": False, "message": f"Error analyzing task: {str(e)}"}

def execute_task_by_id(task_id):
    """Execute a task using the Swarms Agent"""
    try:
        response = requests.post(f"{API_BASE_URL}/tasks/{task_id}/execute")
        return response.json()
    except Exception as e:
        return {"success": False, "message": f"Error executing task: {str(e)}"}

def read_task_output(filepath):
    """Read the content of a task output file"""
    try:
        task_outputs_dir = os.path.join(os.getcwd(), 'task_outputs')
        full_path = os.path.join(task_outputs_dir, filepath)
        
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        else:
            return f"File not found: {filepath}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Check API connection status
def check_connection():
    """Check if the backend API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200
    except:
        return False

# Format task status with emoji and badge style
def format_task_status(status):
    if status == "to do":
        return "⏳ <span class='status-badge status-todo'>To Do</span>"
    elif status == "in progress":
        return "🔄 <span class='status-badge status-in-progress'>In Progress</span>"
    elif status == "done":
        return "✅ <span class='status-badge status-done'>Done</span>"
    else:
        return f"❓ <span class='status-badge'>{status}</span>"

# Gradio interface functions
def refresh_task_list():
    """Refresh the task list and format it for display"""
    tasks = get_all_tasks()
    
    # Format tasks for display
    task_display = []
    for task in tasks:
        status_emoji = "⏳" if task['status'] == 'to do' else "🔄" if task['status'] == 'in progress' else "✅" if task['status'] == 'done' else "❓"
        auto_flag = "🤖" if task.get('is_auto', False) else "👤"
        
        task_display.append({
            "id": task['id'],
            "display": f"{status_emoji} {auto_flag} {task['name']} ({task['status']})"
        })
    
    return task_display

def add_task(name, description, is_auto):
    """Add a new task via the UI"""
    if not name:
        return "Task name is required", refresh_task_list()
    
    result = add_new_task(name, description, is_auto)
    message = f"✅ Task added: {name}" if result.get('success', False) else f"❌ Failed: {result.get('message', 'Unknown error')}"
    
    return message, refresh_task_list()

def handle_task_action(task_display_list, action, process_output):
    """Handle various task actions (analyze, execute, delete)"""
    if not task_display_list:
        return "Please select a task first", None, process_output + "\n⚠️ No task selected"
    
    # Extract task ID from the selected task
    selected_task = task_display_list[0]
    task_id = selected_task.get('id')
    
    if action == "Analyze":
        process_output += f"\n🔍 Analyzing task {selected_task.get('display')}...\n"
        result = analyze_task_by_id(task_id)
        
        # Format the analysis result
        if result.get('analysis'):
            task_output = f"## Analysis for: {result.get('taskName')}\n\n{result.get('analysis')}"
            process_output += f"✅ Analysis completed for task {selected_task.get('display')}\n"
        else:
            task_output = f"Failed to analyze task: {result.get('message', 'Unknown error')}"
            process_output += f"❌ Analysis failed: {result.get('message', 'Unknown error')}\n"
        
        return "Task analyzed", task_output, process_output
    
    elif action == "Execute":
        process_output += f"\n⚙️ Executing task {selected_task.get('display')}...\n"
        result = execute_task_by_id(task_id)
        
        if result.get('success', False):
            # Read the output file
            file_path = result.get('documentPath')
            if file_path:
                content = read_task_output(file_path)
                task_output = f"## Task Output: {result.get('taskName')}\n\n{content}"
                process_output += f"✅ Execution completed. Output saved to {file_path}\n"
            else:
                task_output = f"Task executed but no output file was created."
                process_output += f"⚠️ Execution completed but no output file was created\n"
        else:
            task_output = f"Failed to execute task: {result.get('message', 'Unknown error')}"
            process_output += f"❌ Execution failed: {result.get('message', 'Unknown error')}\n"
        
        return "Task executed", task_output, process_output
    
    elif action == "Delete":
        process_output += f"\n🗑️ Deleting task {selected_task.get('display')}...\n"
        result = delete_task_by_id(task_id)
        
        message = f"✅ Task deleted" if result.get('success', False) else f"❌ Failed to delete task: {result.get('message', 'Unknown error')}"
        process_output += message + "\n"
        
        return message, None, process_output
    
    return "Unknown action", None, process_output + "\n⚠️ Unknown action requested"

def chat_with_agent(message, history, process_output):
    """Chat with the Swarms agent directly"""
    from swarms import Agent
    
    process_output += f"\n💬 Processing chat message: '{message}'\n"
    
    try:
        agent = Agent(
            agent_name="Swarms-Agent",
            system_prompt="""You are Swarms, an intelligent AI assistant.
            Provide helpful, accurate, and concise responses to the user's questions.
            If asked about tasks, guide the user on how to use the task management interface.""",
            model_name="gpt-4o-mini",
            max_loops=1
        )
        
        response = agent.run(message)
        process_output += f"✅ Chat response generated\n"
        
        # Return the proper format expected by the ChatInterface
        return [(message, response)], process_output
    except Exception as e:
        error_message = f"❌ Error in chat processing: {str(e)}\n"
        process_output += error_message
        return [(message, f"Error: {str(e)}")], process_output

# Format the chatbot messages to match the design
def format_chat_html(history):
    formatted_html = ""
    for message in history:
        user_msg, bot_msg = message
        formatted_html += f'<div class="chat-message user">{user_msg}</div>'
        formatted_html += f'<div class="chat-message agent">{bot_msg}</div>'
    return formatted_html

# Set up the Gradio interface
def create_interface():
    with gr.Blocks(title="Swarms Autonomous Agent", css=custom_css) as app:
        # Custom header with connection status
        is_connected = check_connection()
        status_class = "connected" if is_connected else "disconnected"
        status_text = "Connected" if is_connected else "Disconnected"
        
        gr.HTML(f"""
        <div class="app-header">
            <div class="logo">
                <i class="fas fa-robot"></i>
                <span>Swarms Autonomous Agent</span>
            </div>
            <div class="connection-status">
                <div class="status-indicator {status_class}"></div>
                <span>{status_text}</span>
            </div>
        </div>
        """)
        
        with gr.Row():
            # Left column: Task Management
            with gr.Column(scale=1):
                with gr.Box():
                    gr.Markdown("## Task Management")
                    
                    # Task creation section
                    with gr.Group():
                        task_name = gr.Textbox(label="Task Name")
                        task_description = gr.Textbox(label="Task Description", lines=3)
                        is_auto = gr.Checkbox(label="Autonomous Task", info="If checked, the agent will process this task automatically")
                        add_task_btn = gr.Button("Add Task", elem_classes=["primary-button"])
                        task_message = gr.Textbox(label="Status", interactive=False)
                    
                    # Task list section
                    task_list = gr.Dropdown(
                        label="Task List", 
                        multiselect=True,
                        info="Select a task to perform actions"
                    )
                    refresh_btn = gr.Button("Refresh Tasks")
                    
                    # Task actions
                    with gr.Row():
                        analyze_btn = gr.Button("Analyze", elem_classes=["primary-button"])
                        execute_btn = gr.Button("Execute", elem_classes=["primary-button"])
                        delete_btn = gr.Button("Delete")
            
            # Right column: Output and Chat
            with gr.Column(scale=2):
                # Task panels in tabs
                with gr.Tabs():
                    with gr.TabItem("Task Analysis"):
                        with gr.Box():
                            gr.HTML('<div class="panel-title"><i class="fas fa-search"></i> Task Analysis</div>')
                            task_analysis = gr.Markdown(elem_classes=["panel-content"])
                    
                    with gr.TabItem("Task Output"):
                        with gr.Box():
                            gr.HTML('<div class="panel-title"><i class="fas fa-file-alt"></i> Task Output</div>')
                            task_output = gr.Markdown(elem_classes=["panel-content"])
                    
                    with gr.TabItem("Chat with Agent"):
                        with gr.Box():
                            gr.HTML('<div class="panel-title"><i class="fas fa-comment"></i> Chat with Swarms</div>')
                            chatbot = gr.HTML(value='<div class="chat-message agent">Hello! I\'m Swarms, your autonomous AI agent. I\'m ready to help with your tasks. Type \'show tasks\' to see what I can work on.</div>')
                            
                            with gr.Row():
                                chat_msg = gr.Textbox(
                                    placeholder="Type your message here...",
                                    show_label=False
                                )
                                chat_btn = gr.Button("Send", elem_classes=["primary-button"])
                
                # Process log at bottom
                process_log = gr.Textbox(
                    label="Process Log", 
                    lines=8,
                    value="📋 System initialized. Ready to process tasks.",
                    interactive=False
                )
                
                # Define chat submit function
                def chat_submit(message, history_html, process_log):
                    if not message.strip():
                        return "", history_html, process_log
                    
                    # Extract history from HTML if needed
                    # For simplicity, we'll just add to existing HTML
                    new_html = history_html + f'<div class="chat-message user">{message}</div>'
                    
                    try:
                        # Process with agent
                        from swarms import Agent
                        process_log += f"\n💬 Processing: '{message}'\n"
                        
                        agent = Agent(
                            agent_name="Swarms-Agent",
                            system_prompt="""You are Swarms, an intelligent AI assistant.
                            Provide helpful, accurate, and concise responses to user questions.
                            If asked about tasks, guide the user on how to use the task management interface.""",
                            model_name="gpt-4o-mini",
                            max_loops=1
                        )
                        
                        response = agent.run(message)
                        process_log += f"✅ Response generated\n"
                        new_html += f'<div class="chat-message agent">{response}</div>'
                        
                    except Exception as e:
                        error_message = f"❌ Error: {str(e)}\n"
                        process_log += error_message
                        new_html += f'<div class="chat-message agent">Error: {str(e)}</div>'
                    
                    return "", new_html, process_log
                
                # Wire up chat functionality
                chat_btn.click(
                    chat_submit,
                    inputs=[chat_msg, chatbot, process_log],
                    outputs=[chat_msg, chatbot, process_log]
                )
                
                chat_msg.submit(
                    chat_submit,
                    inputs=[chat_msg, chatbot, process_log],
                    outputs=[chat_msg, chatbot, process_log]
                )
        
        # Event handlers for task management
        add_task_btn.click(
            add_task, 
            inputs=[task_name, task_description, is_auto], 
            outputs=[task_message, task_list]
        )
        
        refresh_btn.click(
            lambda: (refresh_task_list()),
            outputs=[task_list]
        )
        
        analyze_btn.click(
            lambda task_list, process_log: handle_task_action(task_list, "Analyze", process_log),
            inputs=[task_list, process_log],
            outputs=[task_message, task_analysis, process_log]
        )
        
        execute_btn.click(
            lambda task_list, process_log: handle_task_action(task_list, "Execute", process_log),
            inputs=[task_list, process_log],
            outputs=[task_message, task_output, process_log]
        )
        
        delete_btn.click(
            lambda task_list, process_log: handle_task_action(task_list, "Delete", process_log),
            inputs=[task_list, process_log],
            outputs=[task_message, task_output, process_log]
        )
        
        # Initialize the task list
        app.load(refresh_task_list, outputs=[task_list])
    
    return app

# Main function to launch the application
def main():
    # Create and launch the Gradio interface
    interface = create_interface()
    interface.queue()
    interface.launch(share=True, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()