"""
Swarms Autonomous Task Agent

An AI-driven task management tool that interacts with a todo.csv file.
Works in two modes:
- Autonomous: Automatically processes tasks marked with is_auto="yes"
- Interactive: Works with the user to decide which tasks to process

CSV Format: id,name,description,status,is_auto
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Confirm, Prompt
from rich.table import Table

from swarms import Agent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Rich console for better CLI output
console = Console()


class TaskStatus(Enum):
    """Task status options"""
    TODO = "to do"
    IN_PROGRESS = "in progress"
    COMPLETED = "completed"


class TaskExecutability(Enum):
    """Whether a task can be run autonomously"""
    AUTO = "yes"
    INTERACTIVE = "no"


@dataclass
class Task:
    """Represents a task from the todo.csv file"""
    id: int
    name: str
    description: str
    status: TaskStatus
    is_auto: TaskExecutability

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary format for saving to CSV"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "is_auto": self.is_auto.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create a Task instance from a dictionary"""
        return cls(
            id=int(data["id"]),
            name=data["name"],
            description=data["description"],
            status=TaskStatus(data["status"]),
            is_auto=TaskExecutability(data["is_auto"]),
        )


@dataclass
class TaskAnalysis:
    """Result of analyzing a task"""
    complexity: str
    ambiguities: List[str]
    autonomous: bool


class SwarmsTaskAgent:
    """Agent that processes tasks from a todo.csv file"""
    
    def __init__(
        self,
        model_name: str = "gemini/gemini-2.0-flash",
        tasks_file: str = r"swarms\structs\swarms_manus\todo.csv",
        output_dir: str = "output",
    ):
        self.model_name = model_name
        self.tasks_file = Path(tasks_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tasks = []
        
        # Initialize the agent and load tasks
        self._init_agent()
        self._load_tasks()

    def _init_agent(self) -> None:
        """Initialize the Swarms agent with appropriate prompts"""
        self.agent = Agent(
            agent_name="SwarmsAuto",
            system_prompt="""
            You are Swarms Auto, an autonomous task management AI.

            Your responsibilities:
            - Analyze tasks to determine if they can be done autonomously
            - Execute tasks based on their descriptions
            - Generate clear, well-structured outputs
            - Update task status when completed
            
            Always provide thoughtful, clear responses that directly address the task.
            """,
            model_name=self.model_name,
        )

    def _load_tasks(self) -> None:
        """Load tasks from the CSV file, creating if doesn't exist"""
        try:
            if not self.tasks_file.exists():
                # Create an empty CSV with the required headers
                pd.DataFrame(
                    columns=["id", "name", "description", "status", "is_auto"]
                ).to_csv(self.tasks_file, index=False)
                console.print(f"[yellow]Created new empty tasks file: {self.tasks_file}[/]")
                self.tasks = []
                return

            # Read existing CSV
            df = pd.read_csv(self.tasks_file)
            
            # Convert data to Task objects
            self.tasks = []
            for _, row in df.iterrows():
                task_dict = row.to_dict()
                # Handle potential missing or empty values
                if pd.isna(task_dict.get("status")) or task_dict.get("status") == "":
                    task_dict["status"] = TaskStatus.TODO.value
                if pd.isna(task_dict.get("is_auto")) or task_dict.get("is_auto") == "":
                    task_dict["is_auto"] = TaskExecutability.INTERACTIVE.value
                    
                self.tasks.append(Task.from_dict(task_dict))
                
            console.print(f"[green]Loaded {len(self.tasks)} tasks from {self.tasks_file}[/]")
            
        except Exception as e:
            console.print(f"[red]Error loading tasks: {e}[/]")
            self.tasks = []

    def _save_tasks(self) -> None:
        """Save tasks to the CSV file"""
        try:
            task_dicts = [task.to_dict() for task in self.tasks]
            df = pd.DataFrame(task_dicts)
            df.to_csv(self.tasks_file, index=False)
            console.print(f"[cyan]Tasks saved to {self.tasks_file}[/]")
        except Exception as e:
            console.print(f"[red]Error saving tasks: {e}[/]")

    def _display_tasks(self) -> None:
        """Display tasks in a formatted table"""
        if not self.tasks:
            console.print("[yellow]No tasks found.[/]")
            return
            
        table = Table(title="Tasks")
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Description", style="white")
        table.add_column("Status", justify="center")
        table.add_column("Auto", justify="center")
        
        for task in self.tasks:
            table.add_row(
                str(task.id),
                task.name,
                task.description[:40] + "..." if len(task.description) > 40 else task.description,
                task.status.value,
                "✓" if task.is_auto == TaskExecutability.AUTO else "✗"
            )
            
        console.print(table)

    def _analyze_task(self, task: Task) -> TaskAnalysis:
        """Analyze a task to determine if it can be done autonomously"""
        prompt = f"""
        Analyze this task: "{task.description}"
        Respond in JSON format with these fields:
        - complexity: "Easy", "Medium", or "Hard"
        - ambiguities: List of any unclear parts of the task
        - autonomous: Boolean whether this can be fully automated (true/false)
        """
        
        response = self.agent.run(prompt)
        
        # Extract JSON from the response
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                data = json.loads(response)
                
            return TaskAnalysis(
                complexity=data.get("complexity", "Unknown"),
                ambiguities=data.get("ambiguities", []),
                autonomous=data.get("autonomous", False)
            )
        except Exception as e:
            console.print(f"[red]Error parsing analysis response: {e}[/]")
            return TaskAnalysis(
                complexity="Unknown",
                ambiguities=["Failed to analyze task"],
                autonomous=False
            )

    def _execute_task(self, task: Task) -> str:
        """Execute a task and return the result"""
        prompt = f"""
        Task: {task.description}
        
        Please complete this task with detailed steps and explanations.
        Your response should be well-formatted and ready to save as a document.
        """
        
        return self.agent.run(prompt)

    def _handle_interactive_task(self, task: Task, ambiguities: List[str]) -> str:
        """Get user input on ambiguities and then execute the task"""
        console.print(Markdown(f"# Task: {task.name}"))
        console.print(Markdown(f"**Description:** {task.description}"))
        
        if ambiguities:
            console.print("\n[yellow]This task has ambiguities that need clarification:[/]")
            for i, ambiguity in enumerate(ambiguities, 1):
                console.print(f"  {i}. {ambiguity}")
                
            clarification = Prompt.ask("\nPlease provide clarification")
            
            prompt = f"""
            Task: {task.description}
            
            User clarification: {clarification}
            
            Please complete this task with the clarification in mind.
            Provide a detailed response that can be saved as a document.
            """
            
            return self.agent.run(prompt)
        else:
            return self._execute_task(task)

    def _save_result(self, task: Task, result: str) -> None:
        """Save the task result to a file"""
        file_name = f"{task.id}_{task.name.replace(' ', '_')}.md"
        file_path = self.output_dir / file_name
        
        try:
            with open(file_path, "w") as f:
                f.write(f"# {task.name}\n\n")
                f.write(f"**Description:** {task.description}\n\n")
                f.write("## Result\n\n")
                f.write(result)
                
            console.print(f"[green]Saved result to: {file_path}[/]")
        except Exception as e:
            console.print(f"[red]Error saving result: {e}[/]")

    def run_autonomous(self) -> None:
        """Process all tasks marked for autonomous execution"""
        auto_tasks = [t for t in self.tasks 
                     if t.is_auto == TaskExecutability.AUTO 
                     and t.status != TaskStatus.COMPLETED]
        
        if not auto_tasks:
            console.print("[yellow]No autonomous tasks found.[/]")
            return
            
        console.print(f"[blue]Running in Autonomous Mode - Found {len(auto_tasks)} tasks[/]")
        
        for task in auto_tasks:
            console.print(f"\n[cyan]Processing task {task.id}: {task.name}[/]")
            
            # First analyze if the task can actually be done autonomously
            analysis = self._analyze_task(task)
            
            if not analysis.autonomous:
                console.print(f"[yellow]Task '{task.name}' cannot be executed autonomously:[/]")
                for ambiguity in analysis.ambiguities:
                    console.print(f"  - {ambiguity}")
                console.print("[yellow]Skipping this task in autonomous mode.[/]")
                continue
                
            # Update status to in progress
            task.status = TaskStatus.IN_PROGRESS
            self._save_tasks()
            
            # Execute the task
            console.print(f"[green]Executing task: {task.name}[/]")
            result = self._execute_task(task)
            
            # Save the result and update status
            self._save_result(task, result)
            task.status = TaskStatus.COMPLETED
            self._save_tasks()
            
            console.print(f"[green]Completed task: {task.name}[/]")

    def run_interactive(self) -> None:
        """Interactively work with the user on tasks"""
        console.print("[blue]Running in Interactive Mode[/]")
        
        while True:
            self._display_tasks()
            
            console.print("\n[cyan]What would you like to do?[/]")
            console.print("1. Choose a task to work on")
            console.print("2. Add a new task")
            console.print("3. Exit interactive mode")
            
            choice = Prompt.ask("Enter your choice", choices=["1", "2", "3"])
            
            if choice == "3":
                break
                
            elif choice == "2":
                # Add a new task
                name = Prompt.ask("Task name")
                description = Prompt.ask("Task description")
                is_auto = Prompt.ask("Can this be run autonomously?", 
                                    choices=["yes", "no"], default="no")
                
                # Generate a new ID (max existing ID + 1)
                new_id = max([t.id for t in self.tasks], default=0) + 1
                
                new_task = Task(
                    id=new_id,
                    name=name,
                    description=description,
                    status=TaskStatus.TODO,
                    is_auto=TaskExecutability.AUTO if is_auto == "yes" else TaskExecutability.INTERACTIVE
                )
                
                self.tasks.append(new_task)
                self._save_tasks()
                console.print(f"[green]Added new task with ID {new_id}[/]")
                
            elif choice == "1":
                # Choose a task to work on
                if not self.tasks:
                    console.print("[yellow]No tasks available.[/]")
                    continue
                    
                task_id = Prompt.ask("Enter the ID of the task to work on")
                try:
                    task_id = int(task_id)
                    task = next((t for t in self.tasks if t.id == task_id), None)
                    
                    if not task:
                        console.print(f"[red]No task found with ID {task_id}[/]")
                        continue
                        
                    if task.status == TaskStatus.COMPLETED:
                        if Confirm.ask(f"Task '{task.name}' is already completed. Work on it again?"):
                            pass  # Continue working on it
                        else:
                            continue
                            
                    # Analyze the task
                    analysis = self._analyze_task(task)
                    
                    # Update status to in progress
                    task.status = TaskStatus.IN_PROGRESS
                    self._save_tasks()
                    
                    # Handle the task based on analysis
                    result = self._handle_interactive_task(task, analysis.ambiguities)
                    
                    # Save the result
                    self._save_result(task, result)
                    
                    # Mark as completed if confirmed
                    if Confirm.ask("Mark this task as completed?", default=True):
                        task.status = TaskStatus.COMPLETED
                        self._save_tasks()
                        console.print(f"[green]Marked task {task_id} as completed[/]")
                    
                except ValueError:
                    console.print("[red]Invalid task ID. Please enter a number.[/]")

    def run(self) -> None:
        """Main entry point: Choose mode and run"""
        console.print("[bold blue]Swarms Autonomous Task Agent[/]")
        console.print("This agent will help you manage and execute tasks from todo.csv\n")
        
        self._display_tasks()
        
        mode = Prompt.ask(
            "\nChoose operating mode",
            choices=["autonomous", "interactive"],
            default="interactive"
        )
        
        if mode == "autonomous":
            self.run_autonomous()
        else:
            self.run_interactive()
            
        # Show final task status
        console.print("\n[bold blue]Final Task Status:[/]")
        self._display_tasks()


if __name__ == "__main__":
    try:
        agent = SwarmsTaskAgent()
        agent.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user.[/]")
    except Exception as e:
        console.print(f"[red]An error occurred: {e}[/]")