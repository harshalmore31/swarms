import os
import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
import uuid
from typing import List, Dict, Any, Optional
import threading

# Rich library for UI elements
from rich.console import Console
from rich import print as rprint

# Swarms library for agent functionality
from swarms import Agent

# Flask for API
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize console
console = Console()

# Initialize Swarms model
DEFAULT_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    console.print("[bold red]OPENAI_API_KEY is not defined in environment variables.[/bold red]")
    exit(1)

# Task database (in-memory for simplicity)
# In a production environment, use a proper database like SQLite, PostgreSQL, etc.
TASKS_DB = []

# Create output directory if it doesn't exist
def ensure_output_directory():
    output_dir = Path.cwd() / 'task_outputs'
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir

# Database operations (CRUD)
def get_tasks_by_status(status: str) -> Dict[str, Any]:
    """Get tasks with a specific status."""
    filtered_tasks = [task for task in TASKS_DB if task.get("status") == status]
    return {
        "success": True,
        "count": len(filtered_tasks),
        "tasks": filtered_tasks
    }

def get_all_tasks() -> Dict[str, Any]:
    """Get all tasks."""
    return {
        "success": True,
        "count": len(TASKS_DB),
        "tasks": TASKS_DB
    }

def add_task(name: str, description: str, is_auto: bool = False) -> Dict[str, Any]:
    """Add a new task to the database."""
    task_id = str(uuid.uuid4())
    new_task = {
        "id": task_id,
        "name": name,
        "description": description,
        "status": "to do",
        "is_auto": is_auto,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    TASKS_DB.append(new_task)
    return {
        "success": True,
        "message": f"Task '{name}' created successfully",
        "task": new_task
    }

def update_task_status(task_id: str, new_status: str) -> Dict[str, Any]:
    """Update the status of a task."""
    for task in TASKS_DB:
        if task["id"] == task_id:
            task["status"] = new_status
            task["updated_at"] = datetime.now().isoformat()
            return {
                "success": True,
                "message": f"Task '{task['name']}' updated from '{task['status']}' to '{new_status}'",
                "task": task
            }
    return {
        "success": False,
        "message": f"Task with ID '{task_id}' not found"
    }

def delete_task(task_id: str) -> Dict[str, Any]:
    """Delete a task from the database."""
    for i, task in enumerate(TASKS_DB):
        if task["id"] == task_id:
            deleted_task = TASKS_DB.pop(i)
            return {
                "success": True,
                "message": f"Task '{deleted_task['name']}' deleted successfully",
                "task": deleted_task
            }
    return {
        "success": False,
        "message": f"Task with ID '{task_id}' not found"
    }

def find_task_by_id(task_id: str) -> Optional[Dict[str, Any]]:
    """Find a task by its ID."""
    for task in TASKS_DB:
        if task["id"] == task_id:
            return task
    return None

def find_task_by_name(task_name: str) -> Optional[Dict[str, Any]]:
    """Find a task by its name (case-insensitive)."""
    for task in TASKS_DB:
        if task["name"].lower() == task_name.lower():
            return task
    return None

# Create project document function
def create_project_document(task_name: str, content: str, document_type: str) -> Dict[str, Any]:
    """Create a document for a specific task."""
    try:
        output_dir = ensure_output_directory()
        sanitized_task_name = "".join(c if c.isalnum() else "_" for c in task_name.lower())
        filename = f"{sanitized_task_name}_{document_type}_{int(time.time())}.md"
        
        header = f"# {task_name} - {document_type.capitalize()}\n\nCreated: {datetime.now().isoformat()}\n\n"
        document_content = header + content
        
        file_path = output_dir / filename
        file_path.write_text(document_content, encoding='utf-8')
        
        return {
            "success": True,
            "message": f'Created {document_type} document for "{task_name}"',
            "filePath": str(filename)
        }
        
    except Exception as e:
        console.print(f"[bold red]Error creating {document_type} document:[/bold red] {str(e)}")
        return {
            "success": False, 
            "message": f'Failed to create {document_type} document: {str(e)}'
        }

# Swarms Agent Functions
def analyze_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a task using Swarms Agent."""
    console.print(f"[bold blue]Analyzing task:[/bold blue] {task['name']}")
    
    system_prompt = """
    You are Phoenix, an advanced autonomous AI agent.
    
    Analyze the given task and provide a comprehensive assessment that includes:
    1. Task Domain: Identify the primary subject area (software development, research, documentation, etc.)
    2. Requirements: Outline what seems to be required based on the task name and description
    3. Ambiguities: Identify any unclear aspects that might need clarification
    4. Task Complexity: Estimate whether this is a simple, moderate, or complex task
    5. Action Plan: Provide a detailed step-by-step plan for executing this task
    6. Required Resources: List any tools, information, or resources needed
    7. Estimated Time: Provide a rough estimate for task completion
    8. Autonomous Actions: Specify which parts you can complete autonomously vs. which require human input
    
    Structure your analysis clearly and be specific in your plan. Don't use generic approaches - tailor your response to this exact task.
    """
    
    try:
        agent = Agent(
            agent_name="Task-Analysis-Agent",
            system_prompt=system_prompt,
            model_name=DEFAULT_MODEL,
            max_loops=1
        )
        
        task_input = f"Task Name: {task['name']}\nTask Description: {task.get('description', 'No description provided')}"
        analysis = agent.run(task_input)
        
        # Determine if task can be executed autonomously
        can_execute = not (
            "clarification needed" in analysis.lower() or 
            "more information required" in analysis.lower() or
            "human input needed" in analysis.lower()
        )
        
        return {
            "taskName": task["name"],
            "taskId": task["id"],
            "analysis": analysis,
            "canExecuteAutonomously": can_execute and task.get("is_auto", False)
        }
        
    except Exception as e:
        console.print(f"[bold red]Error analyzing task:[/bold red] {str(e)}")
        return {
            "taskName": task["name"],
            "taskId": task["id"],
            "analysis": f"Error analyzing task: {str(e)}",
            "canExecuteAutonomously": False
        }

def execute_task(task: Dict[str, Any], analysis: str) -> Dict[str, Any]:
    """Execute a task autonomously using Swarms Agent."""
    console.print(f"[bold green]Executing task:[/bold green] {task['name']}")
    
    system_prompt = """
    You are Phoenix, an autonomous AI agent executing a specific task.
    
    Your goal is to generate the complete content needed to fulfill this task.
    Be comprehensive, specific, and provide implementation-ready output.
    
    If this is a:
    - Documentation task: Write the complete documentation
    - Planning task: Create the full, detailed plan
    - Code task: Write the complete code with comments
    - Research task: Produce comprehensive research findings
    
    Your output should be complete, high-quality, and ready for implementation with no placeholders.
    """
    
    try:
        agent = Agent(
            agent_name="Task-Execution-Agent",
            system_prompt=system_prompt,
            model_name=DEFAULT_MODEL,
            max_loops=1
        )
        
        task_input = f"""
        Task: {task['name']}
        Description: {task.get('description', 'No description provided')}
        
        Previous Analysis:
        {analysis}
        
        Please execute this task completely and provide the full output.
        """
        
        output_content = agent.run(task_input)
        
        # Determine document type based on task name and analysis
        document_type = "documentation"
        analysis_lower = analysis.lower()
        task_name_lower = task["name"].lower()
        
        if ("code" in analysis_lower or "code" in task_name_lower or 
            "develop" in task_name_lower or "program" in task_name_lower):
            document_type = "code"
        elif ("research" in analysis_lower or "research" in task_name_lower or
              "analyze" in task_name_lower or "study" in task_name_lower):
            document_type = "research"
        elif ("plan" in analysis_lower or "plan" in task_name_lower or
              "strategy" in task_name_lower or "roadmap" in task_name_lower):
            document_type = "plan"
        
        # Create document with the content
        document = create_project_document(task["name"], output_content, document_type)
        
        # Update task status to in progress
        update_task_status(task["id"], "done")  # Changed from "in progress" to "done" since task is completed
        
        return {
            "success": True,
            "taskName": task["name"],
            "taskId": task["id"],
            "documentType": document_type,
            "documentPath": document.get("filePath", ""),
            "message": f"Successfully executed task and created {document_type} document."
        }
        
    except Exception as e:
        console.print(f"[bold red]Error executing task {task['name']}:[/bold red] {str(e)}")
        return {
            "success": False,
            "taskName": task["name"],
            "taskId": task["id"],
            "message": f"Failed to execute task: {str(e)}"
        }

# Background task processor
def process_auto_tasks():
    """
    Background process to continuously check for and process
    autonomous tasks marked with 'auto'.
    """
    while True:
        try:
            # Get tasks with 'to do' status
            todo_tasks = get_tasks_by_status("to do")
            
            for task in todo_tasks["tasks"]:
                # Only process auto tasks
                if task.get("is_auto", False):
                    console.print(f"[bold blue]Found auto task:[/bold blue] {task['name']}")
                    
                    # Analyze the task
                    analysis_result = analyze_task(task)
                    
                    # If can be executed autonomously, execute it
                    if analysis_result.get("canExecuteAutonomously", False):
                        execution_result = execute_task(task, analysis_result["analysis"])
                        
                        if execution_result.get("success", False):
                            console.print(f"[bold green]✅ Completed:[/bold green] {task['name']}")
                            # Update status to done
                            update_task_status(task["id"], "done")
                        else:
                            console.print(f"[bold red]❌ Failed:[/bold red] {task['name']}")
            
            # Sleep to avoid high CPU usage
            time.sleep(10)
            
        except Exception as e:
            console.print(f"[bold red]Error in background task processor:[/bold red] {str(e)}")
            time.sleep(30)  # Longer sleep on error

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# API routes
@app.route('/api/tasks', methods=['GET'])
def api_get_tasks():
    """Get all tasks or filter by status."""
    status = request.args.get('status')
    if status:
        return jsonify(get_tasks_by_status(status))
    return jsonify(get_all_tasks())

@app.route('/api/tasks', methods=['POST'])
def api_add_task():
    """Add a new task."""
    data = request.json
    if not data or 'name' not in data:
        return jsonify({"success": False, "message": "Task name is required"}), 400
    
    name = data.get('name')
    description = data.get('description', '')
    is_auto = data.get('is_auto', False)
    
    result = add_task(name, description, is_auto)
    return jsonify(result), 201 if result["success"] else 400

@app.route('/api/tasks/<task_id>', methods=['PUT'])
def api_update_task(task_id):
    """Update a task's status."""
    data = request.json
    if not data or 'status' not in data:
        return jsonify({"success": False, "message": "Status is required"}), 400
    
    result = update_task_status(task_id, data['status'])
    return jsonify(result), 200 if result["success"] else 404

@app.route('/api/tasks/<task_id>', methods=['DELETE'])
def api_delete_task(task_id):
    """Delete a task."""
    result = delete_task(task_id)
    return jsonify(result), 200 if result["success"] else 404

@app.route('/api/tasks/<task_id>/analyze', methods=['POST'])
def api_analyze_task(task_id):
    """Analyze a specific task."""
    task = find_task_by_id(task_id)
    if not task:
        return jsonify({"success": False, "message": "Task not found"}), 404
    
    result = analyze_task(task)
    return jsonify(result)

@app.route('/api/tasks/<task_id>/execute', methods=['POST'])
def api_execute_task(task_id):
    """Execute a specific task."""
    task = find_task_by_id(task_id)
    if not task:
        return jsonify({"success": False, "message": "Task not found"}), 404
    
    # First analyze the task
    analysis_result = analyze_task(task)
    
    # Then execute it
    result = execute_task(task, analysis_result["analysis"])
    return jsonify(result)

@app.route('/api/task_output/<path:filename>', methods=['GET'])
def get_task_output(filename):
    """Get the content of a task output file."""
    try:
        output_dir = ensure_output_directory()
        file_path = output_dir / filename
        
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({"success": True, "content": content})
        else:
            return jsonify({"success": False, "message": f"File not found: {filename}"}), 404
    except Exception as e:
        return jsonify({"success": False, "message": f"Error reading file: {str(e)}"}), 500

# Main function to run the Flask app
def main():
    """Start the application."""
    console.print("[bold blue]Phoenix Autonomous Agent v3.0[/bold blue]")
    console.print("[cyan]Swarms-based AI Task Management System[/cyan]")
    
    # Add some initial tasks for testing
    add_task("Create a simple Python script for data analysis", "Write a script that can analyze CSV files and produce statistical summaries.", is_auto=True)
    add_task("Research the impact of AI on healthcare", "Provide a comprehensive report on how AI is transforming healthcare delivery and outcomes.", is_auto=True)
    add_task("Design a database schema for a blog system", "Create a normalized database design for a blogging platform with users, posts, comments, and categories.", is_auto=False)
    
    # Start background task processor in a separate thread
    bg_thread = threading.Thread(target=process_auto_tasks, daemon=True)
    bg_thread.start()
    
    # Start Flask app
    console.print("[green]Starting API server on http://localhost:5000[/green]")
    app.run(debug=False, host='0.0.0.0', port=5000)

# Entry point
if __name__ == "__main__":
    main()