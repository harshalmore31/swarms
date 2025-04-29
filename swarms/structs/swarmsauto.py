"""
Swarms Autonomous Task Agent

An AI-driven task management tool that interacts with a todo.csv file.
Works in two modes:
- Autonomous: AI evaluates and processes tasks it determines it can handle
- Interactive: Natural language chat interface to manage and process tasks
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


@dataclass
class Task:
    """Represents a task from the todo.csv file"""
    id: int
    name: str
    description: str
    status: TaskStatus

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary format for saving to CSV"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create a Task instance from a dictionary"""
        return cls(
            id=int(data["id"]),
            name=data["name"],
            description=data["description"],
            status=TaskStatus(data["status"]),
        )


@dataclass
class TaskAnalysis:
    """Result of analyzing a task"""
    complexity: str
    ambiguities: List[str]
    autonomous: bool
    reasoning: str
    execution_plan: List[str]


class SwarmsTaskAgent:
    """Agent that processes tasks from a todo.csv file"""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        tasks_file: str = r"todo.csv",
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
            You are Swarms Auto, an intelligent task management AI.

            Your responsibilities:
            - Analyze tasks to determine if you can execute them autonomously
            - Reason about task requirements and limitations
            - Plan detailed execution steps for each task
            - Execute tasks based on their descriptions
            - Generate clear, well-structured outputs
            - Update task status when completed
            
            When analyzing tasks, carefully consider:
            1. Is the task well-defined with clear objectives?
            2. Do you have all required information to complete it?
            3. Is the task within your capabilities (no external API access, physical actions, etc.)?
            4. Can you produce a complete solution with your knowledge?
            
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
                    columns=["id", "name", "description", "status"]
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
        
        for task in self.tasks:
            table.add_row(
                str(task.id),
                task.name,
                task.description[:40] + "..." if len(task.description) > 40 else task.description,
                task.status.value
            )
            
        console.print(table)

    def _analyze_task(self, task: Task) -> TaskAnalysis:
        """Analyze a task to determine if it can be done autonomously"""
        # Convert all tasks to a list of dictionaries to provide context
        task_context = [t.to_dict() for t in self.tasks]
        
        prompt = f"""
        Analyze this task: "{task.description}"
        
        Context - All available tasks:
        {json.dumps(task_context, indent=2)}
        
        Respond in JSON format with these fields:
        - complexity: "Low", "Medium", or "High"
        - ambiguities: List of any unclear parts of the task
        - autonomous: Boolean whether you can fully execute this task (true/false)
        - reasoning: Your reasoning for the autonomous decision
        - execution_plan: List of steps you would take to execute this task
        
        Be thorough in your analysis. Consider if you have all required information and abilities to complete the task.
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
                autonomous=data.get("autonomous", False),
                reasoning=data.get("reasoning", "No reasoning provided"),
                execution_plan=data.get("execution_plan", [])
            )
        except Exception as e:
            console.print(f"[red]Error parsing analysis response: {e}[/]")
            return TaskAnalysis(
                complexity="Unknown",
                ambiguities=["Failed to analyze task"],
                autonomous=False,
                reasoning="Analysis failed",
                execution_plan=[]
            )

    def _execute_task(self, task: Task, execution_plan: List[str]) -> str:
        """Execute a task and return the result"""
        # Convert all tasks to a list of dictionaries to provide context
        task_context = [t.to_dict() for t in self.tasks]
        
        prompt = f"""
        Task to execute: {task.description}
        
        Context - All available tasks:
        {json.dumps(task_context, indent=2)}
        
        Execution Plan:
        {chr(10).join([f"{i+1}. {step}" for i, step in enumerate(execution_plan)])}
        
        Please complete this task following the execution plan.
        Your response should be well-formatted and ready to save as a document.
        Include your thought process and explain any decisions made during execution.
        """
        
        return self.agent.run(prompt)

    def _handle_interactive_task(self, task: Task, analysis: TaskAnalysis) -> str:
        """Get user input on ambiguities and then execute the task"""
        console.print(Markdown(f"# Task: {task.name}"))
        console.print(Markdown(f"**Description:** {task.description}"))
        
        # Display analysis to user
        console.print("\n[cyan]Task Analysis:[/]")
        console.print(f"Complexity: [bold]{analysis.complexity}[/]")
        console.print(f"Can be executed autonomously: [bold]{'Yes' if analysis.autonomous else 'No'}[/]")
        console.print(f"Reasoning: {analysis.reasoning}")
        
        if analysis.ambiguities:
            console.print("\n[yellow]Ambiguities that need clarification:[/]")
            for i, ambiguity in enumerate(analysis.ambiguities, 1):
                console.print(f"  {i}. {ambiguity}")
                
            clarification = Prompt.ask("\nPlease provide clarification")
            
            # Convert all tasks to a list of dictionaries to provide context
            task_context = [t.to_dict() for t in self.tasks]
            
            # Update execution plan with clarification
            updated_prompt = f"""
            Task: {task.description}
            
            Context - All available tasks:
            {json.dumps(task_context, indent=2)}
            
            User clarification: {clarification}
            
            Based on this clarification, provide an updated execution plan as a JSON list of steps.
            """
            
            response = self.agent.run(updated_prompt)
            
            try:
                # Extract JSON list from response
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if match:
                    updated_plan = json.loads(match.group(0))
                else:
                    updated_plan = analysis.execution_plan
                    console.print("[yellow]Could not extract updated plan, using original.[/]")
            except Exception as e:
                updated_plan = analysis.execution_plan
                console.print(f"[yellow]Error parsing updated plan: {e}. Using original.[/]")
            
            # Execute with updated plan and clarification
            prompt = f"""
            Task: {task.description}
            
            Context - All available tasks:
            {json.dumps(task_context, indent=2)}
            
            User clarification: {clarification}
            
            Execution Plan:
            {chr(10).join([f"{i+1}. {step}" for i, step in enumerate(updated_plan)])}
            
            Please complete this task with the clarification in mind, following the execution plan.
            Provide a detailed response that can be saved as a document.
            """
            
            return self.agent.run(prompt)
        else:
            return self._execute_task(task, analysis.execution_plan)

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

    def _get_task_by_reference(self, user_input: str) -> Optional[Task]:
        """Try to identify a task from user input by ID or name"""
        # First try to parse as an ID
        try:
            task_id = int(user_input.strip())
            task = next((t for t in self.tasks if t.id == task_id), None)
            if task:
                return task
        except ValueError:
            pass
        
        # Try to find by name (partial match)
        lower_input = user_input.lower()
        matching_tasks = [t for t in self.tasks if lower_input in t.name.lower()]
        
        if len(matching_tasks) == 1:
            return matching_tasks[0]
        elif len(matching_tasks) > 1:
            # If multiple matches, ask for clarification
            console.print("[yellow]Multiple tasks match your input. Please specify which one:[/]")
            for t in matching_tasks:
                console.print(f"  {t.id}: {t.name}")
            
            task_id_str = Prompt.ask("Enter the ID of the task you want")
            try:
                task_id = int(task_id_str)
                return next((t for t in matching_tasks if t.id == task_id), None)
            except ValueError:
                console.print("[red]Invalid ID format.[/]")
                return None
                
        return None

    def run_autonomous(self) -> None:
        """Process tasks the AI determines it can handle autonomously"""
        # Get all uncompleted tasks
        pending_tasks = [t for t in self.tasks if t.status != TaskStatus.COMPLETED]
        
        if not pending_tasks:
            console.print("[yellow]No pending tasks found.[/]")
            return
            
        console.print(f"[blue]Running in Autonomous Mode - Analyzing {len(pending_tasks)} tasks[/]")
        
        # First analyze all tasks to find those that can be done autonomously
        autonomous_tasks = []
        for task in pending_tasks:
            console.print(f"\n[cyan]Analyzing task {task.id}: {task.name}[/]")
            analysis = self._analyze_task(task)
            
            if analysis.autonomous:
                console.print(f"[green]Task '{task.name}' can be executed autonomously.[/]")
                console.print(f"Reasoning: {analysis.reasoning}")
                autonomous_tasks.append((task, analysis))
            else:
                console.print(f"[yellow]Task '{task.name}' cannot be executed autonomously:[/]")
                console.print(f"Reasoning: {analysis.reasoning}")
                if analysis.ambiguities:
                    for ambiguity in analysis.ambiguities:
                        console.print(f"  - {ambiguity}")
        
        if not autonomous_tasks:
            console.print("[yellow]No tasks can be executed autonomously. Try interactive mode.[/]")
            return
            
        # Process the autonomous tasks
        console.print(f"\n[blue]Found {len(autonomous_tasks)} tasks that can be executed autonomously[/]")
        for task, analysis in autonomous_tasks:
            console.print(f"\n[cyan]Processing task {task.id}: {task.name}[/]")
            
            # Update status to in progress
            task.status = TaskStatus.IN_PROGRESS
            self._save_tasks()
            
            # Show execution plan
            console.print("\n[cyan]Execution Plan:[/]")
            for i, step in enumerate(analysis.execution_plan, 1):
                console.print(f"  {i}. {step}")
            
            # Execute the task
            console.print(f"\n[green]Executing task: {task.name}[/]")
            result = self._execute_task(task, analysis.execution_plan)
            
            # Save the result and update status
            self._save_result(task, result)
            task.status = TaskStatus.COMPLETED
            self._save_tasks()
            
            console.print(f"[green]Completed task: {task.name}[/]")

    def run_interactive(self) -> None:
        """Interactive chat interface for working with tasks"""
        console.print("[bold blue]Welcome to Interactive Mode[/]")
        console.print("You can chat with me about your tasks. Examples:")
        console.print("- 'Show me all tasks'")
        console.print("- 'Add a new task'")
        console.print("- 'Work on task #3' or 'Help me with the website task'")
        console.print("- 'Exit' or 'Quit' to leave interactive mode")
        
        while True:
            self._display_tasks()
            
            # Get natural language input from user
            user_input = Prompt.ask("\nWhat would you like to do?")
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                console.print("[blue]Exiting interactive mode. Goodbye![/]")
                break
                
            # Process the user input
            input_lower = user_input.lower()
            
            # Task listing
            if any(phrase in input_lower for phrase in ["show", "list", "display", "all tasks"]):
                # Already displayed above, just continue
                continue
                
            # Adding a new task
            elif any(phrase in input_lower for phrase in ["add", "create", "new task"]):
                name = Prompt.ask("Task name")
                description = Prompt.ask("Task description")
                
                # Generate a new ID (max existing ID + 1)
                new_id = max([t.id for t in self.tasks], default=0) + 1
                
                new_task = Task(
                    id=new_id,
                    name=name,
                    description=description,
                    status=TaskStatus.TODO
                )
                
                self.tasks.append(new_task)
                self._save_tasks()
                console.print(f"[green]Added new task #{new_id}: {name}[/]")
                
            # Working on a task - look for "work on", "do", "help with", etc.
            elif any(phrase in input_lower for phrase in ["work on", "do", "help with", "start", "process"]):
                # Extract potential task reference
                task = self._get_task_by_reference(input_lower)
                
                if not task:
                    # Pass user input to AI for interpretation
                    task_context = [t.to_dict() for t in self.tasks]
                    interpretation_prompt = f"""
                    The user said: "{user_input}"
                    
                    Given the available tasks:
                    {json.dumps(task_context, indent=2)}
                    
                    Which task do you think they're referring to? Respond with just the task ID number.
                    If you can't determine a specific task, respond with "unknown".
                    """
                    
                    response = self.agent.run(interpretation_prompt)
                    
                    # Try to extract a task ID from the response
                    try:
                        task_id_match = re.search(r'\d+', response)
                        if task_id_match:
                            task_id = int(task_id_match.group(0))
                            task = next((t for t in self.tasks if t.id == task_id), None)
                    except:
                        pass
                
                if not task:
                    console.print("[yellow]I'm not sure which task you're referring to. Please specify the task ID or name.[/]")
                    continue
                
                if task.status == TaskStatus.COMPLETED:
                    if Confirm.ask(f"Task '{task.name}' is already completed. Work on it again?"):
                        pass  # Continue working on it
                    else:
                        continue
                
                # Analyze the task
                console.print(f"[cyan]Analyzing task: {task.name}[/]")
                analysis = self._analyze_task(task)
                
                # Update status to in progress
                task.status = TaskStatus.IN_PROGRESS
                self._save_tasks()
                
                # Handle the task with the analysis
                result = self._handle_interactive_task(task, analysis)
                
                # Save the result
                self._save_result(task, result)
                
                # Mark as completed if confirmed
                if Confirm.ask("Mark this task as completed?", default=True):
                    task.status = TaskStatus.COMPLETED
                    self._save_tasks()
                    console.print(f"[green]Marked task {task.id} as completed[/]")
                
            # Handle direct AI conversation
            else:
                # Pass the entire conversation to the AI
                task_context = [t.to_dict() for t in self.tasks]
                
                conversation_prompt = f"""
                User said: "{user_input}"
                
                Context - All available tasks:
                {json.dumps(task_context, indent=2)}
                
                Please respond to the user's message in a helpful way. If they seem to be asking
                about tasks or wanting to do something with tasks, help guide them with suggestions
                on what they can do in this system.
                """
                
                response = self.agent.run(conversation_prompt)
                console.print(f"[cyan]{response}[/]")

    def run(self) -> None:
        """Main entry point: Choose mode and run"""
        console.print("[bold blue]Swarms Autonomous Task Agent[/]")
        console.print("This agent will help you manage and execute tasks from todo.csv\n")
        
        self._display_tasks()
        
        console.print("\n[cyan]How would you like to interact with your tasks?[/]")
        console.print("You can say 'auto' for AI to handle tasks automatically, or")
        console.print("'interactive' to chat and work together on tasks.")
        
        mode = Prompt.ask("Your choice", default="interactive")
        
        if "auto" in mode.lower():
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