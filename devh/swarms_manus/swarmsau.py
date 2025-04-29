"""
Swarms Autonomous Task Agent: An AI-driven task manager using the swarms library.

This agent fetches, analyzes, executes, and updates tasks, leveraging an LLM for autonomous operations.
It supports both fully automated task completion and interactive modes for tasks requiring human input.

Enhanced with Rich library for improved output and user interaction.
"""

import json
import logging
import os
import re  # Import the regular expression module
import shutil
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Confirm, Prompt
from rich.table import Table

from swarms import Agent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()

# Load environment variables (API keys, etc.)
load_dotenv()

# --- Global Settings ---
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini/gemini-2.0-flash")  # Or any default model
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"
TASKS_FILE = "tasks.csv"
OUTPUT_DIR = "output"
BACKUP_DIR = "backups"
MAX_ITERATIONS = 3

# --- Global Variables ---
terminating = False  # Global flag to trigger shutdown
tasks_db_lock = threading.RLock()
AUTO_MODE = False  # Global flag for autonomous mode
STATUS_CHANGE_EVENT = threading.Event()


class TaskStatus(Enum):
    """Define task status options."""

    TODO = "to do"
    IN_PROGRESS = "in progress"
    COMPLETED = "completed"


class TaskExecutability(Enum):
    """Classify task executability."""

    AUTO = "yes"
    INTERACTIVE = "no"


@dataclass
class Task:
    """Represents a single task."""

    id: int
    name: str
    description: str
    status: TaskStatus
    is_auto: TaskExecutability
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )  # Add created_at
    updated_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )  # Add updated_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert the task to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "is_auto": self.is_auto.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class TaskAnalysisResult:
    """Structured result of AI task analysis."""

    required_resources: List[str]
    complexity: str  # "Easy", "Medium", "Hard"
    dependencies: List[str]
    ambiguities: List[str]
    autonomous: bool


@dataclass
class AIResponse:
    """Generic structure for handling AI responses."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# --- Logger Class ---
class Logger:
    """Improved logger for console and file output."""

    def __init__(self, log_file_name: str = "swarms_agent.log"):
        self.logs_dir = Path(OUTPUT_DIR) / "logs"
        self.logs_dir.mkdir(exist_ok=True, parents=True)
        self.log_file = self.logs_dir / log_file_name
        self.log_lock = threading.Lock()
        self.console = Console()  # Use Rich Console

    def log(self, message: str, level: str = "INFO"):
        """Log a message to both the console and the log file."""
        timestamp = datetime.now().isoformat(timespec="seconds")
        entry = f"[{timestamp}] [{level}] {message}"
        with self.log_lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
            if level == "ERROR":
                self.console.print(entry, style="bold red")
            elif level == "WARNING":
                self.console.print(entry, style="bold yellow")
            elif level == "SUCCESS":
                self.console.print(entry, style="bold green")
            else:
                self.console.print(entry)


# --- Instantiate Logger ---
logger = Logger()


class TodoWatcher:
    """Monitors the CSV file and updates the in-memory task list."""

    def __init__(self, tasks_file: str = TASKS_FILE):
        self.tasks_file = Path(tasks_file)
        self.backup_dir = Path(BACKUP_DIR)
        self.backup_dir.mkdir(exist_ok=True)
        self.last_modified = None
        self.lock = threading.RLock()
        self.update_interval = 3  # seconds between checks
        self.retry_count = 3
        self.retry_delay = 2

    def create_template_if_not_exists(self):
        """Create a template CSV if it doesn't exist."""
        with self.lock:
            if not self.tasks_file.exists():
                self._create_template_file()
                self.last_modified = self.tasks_file.stat().st_mtime
                return True
            return True

    def _create_template_file(self):
        """Creates the initial CSV template."""
        with open(self.tasks_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["id", "name", "description", "status", "is_auto"])
            # Sample Tasks
            writer.writerow(
                ["1", "Research Swarms", "Research the Swarms library", "to do", "yes"]
            )
            writer.writerow(
                [
                    "2",
                    "Write Documentation",
                    "Write documentation for Project X",
                    "to do",
                    "no",
                ]
            )
        logger.log(f"Created CSV template: {self.tasks_file}")

    def create_backup(self):
        """Creates a backup of the CSV file."""
        if self.tasks_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"tasks_backup_{timestamp}.csv"
            try:
                shutil.copy2(self.tasks_file, backup_file)
                logger.log(f"Backup created: {backup_file.name}")
                # Keep only the last 10 backups
                backups = sorted(self.backup_dir.glob("tasks_backup_*.csv"))
                if len(backups) > 10:
                    for old_backup in backups[:-10]:
                        old_backup.unlink()
                        logger.log(f"Removed old backup: {old_backup.name}")
            except Exception as e:
                logger.log(f"Backup error: {e}", "ERROR")

    def parse_todo_file(self):
        """Loads tasks from the CSV and updates the global task list."""
        if not self.tasks_file.exists():
            self.create_template_if_not_exists()

        for attempt in range(self.retry_count):
            try:
                tasks = []
                with open(self.tasks_file, "r", encoding="utf-8") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if not all(
                            k in row
                            for k in ["id", "name", "description", "status", "is_auto"]
                        ):
                            logger.log(f"Skipping invalid row: {row}", "WARNING")
                            continue

                        # Convert 'status' and 'is_auto' to enums, handling case variations
                        try:
                            status = TaskStatus(row["status"].lower().strip())
                        except ValueError:
                            logger.log(
                                f"Invalid status '{row['status']}' in row: {row}.  Using"
                                " default 'to do'.",
                                "WARNING",
                            )
                            status = TaskStatus.TODO  # Default to TODO

                        try:
                            is_auto = TaskExecutability(row["is_auto"].lower().strip())
                        except ValueError:
                            logger.log(
                                f"Invalid is_auto '{row['is_auto']}' in row: {row}."
                                "  Using default 'no'.",
                                "WARNING",
                            )
                            is_auto = (
                                TaskExecutability.INTERACTIVE
                            )  # Default to INTERACTIVE

                        tasks.append(
                            Task(
                                id=int(row["id"]),
                                name=row["name"],
                                description=row["description"],
                                status=status,
                                is_auto=is_auto,
                            )
                        )

                with tasks_db_lock:
                    global tasks_list
                    tasks_list = tasks
                # NO CSV UPDATE HERE --  Moved to _update_task_status
                logger.log("Parsed CSV and updated in-memory tasks.")
                return tasks  # Return the tasks

            except Exception as e:
                logger.log(f"CSV parse error (attempt {attempt+1}): {e}", "ERROR")
                time.sleep(self.retry_delay)

        logger.log(
            "Failed to parse CSV after retries; recreating template.", "WARNING"
        )
        self.create_template_if_not_exists()
        return []

    def check_for_changes(self):
        """Checks if the CSV file has been modified."""
        try:
            current_mtime = self.tasks_file.stat().st_mtime
            if (self.last_modified is None) or (current_mtime > self.last_modified):
                self.last_modified = current_mtime
                return True
            return False
        except Exception as e:
            logger.log(f"Error checking CSV changes: {e}", "ERROR")
            return False

    def watch_and_update(self):
        """Watches the CSV file for changes and updates the task list."""
        global terminating
        while not terminating:
            try:
                if self.check_for_changes():
                    logger.log("Change detected in CSV; re-parsing tasks.")
                    self.parse_todo_file()
                    STATUS_CHANGE_EVENT.set()  # Signal task list change
                time.sleep(self.update_interval)
            except Exception as e:
                logger.log(f"Error in CSV watcher: {e}", "ERROR")
                time.sleep(self.update_interval * 2)

    def update_todo_file(self, force_write=False):
        """Updates the CSV file with the current task list."""
        with self.lock:
            if not self.tasks_file.exists():
                logger.log("CSV file not found for update.", "ERROR")
                return False

            for attempt in range(self.retry_count):
                try:
                    self.create_backup()
                    temp_file = self.tasks_file.with_suffix(".tmp")
                    with open(temp_file, "w", newline="", encoding="utf-8") as csvfile:
                        fieldnames = [
                            "id",
                            "name",
                            "description",
                            "status",
                            "is_auto",
                            "created_at",
                            "updated_at",
                        ]  # Include created_at, updated_at
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        with tasks_db_lock:
                            for task in tasks_list:
                                # Convert enums to strings before writing
                                task_dict = task.to_dict()
                                task_dict["status"] = task_dict["status"].value
                                task_dict["is_auto"] = task_dict["is_auto"].value
                                writer.writerow(task_dict)
                    shutil.move(str(temp_file), str(self.tasks_file))
                    self.last_modified = self.tasks_file.stat().st_mtime
                    logger.log("CSV updated successfully.")
                    return True
                except Exception as e:
                    logger.log(f"CSV update error (attempt {attempt+1}): {e}", "ERROR")
                    time.sleep(self.retry_delay)

            logger.log("All attempts to update CSV failed.", "ERROR")
            return False


class TaskProcessor:
    """Processes tasks, particularly those marked for autonomous execution."""

    def __init__(self, todo_watcher: TodoWatcher, agent: "SwarmsAutoAgent"):
        self.todo_watcher = todo_watcher
        self.agent = agent  # Reference to the SwarmsAutoAgent instance
        self.thread = None
        self.lock = threading.RLock()
        self.running = True

    def start(self):
        """Starts the task processing thread."""
        self.running = True
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._process_tasks, daemon=True)
            self.thread.start()
            logger.log("TaskProcessor thread started.")

    def stop(self):
        """Stops the task processing thread."""
        logger.log("Stopping TaskProcessor...")
        self.running = False
        STATUS_CHANGE_EVENT.set()  # Wake up the thread
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)  # Give it some time to finish
            logger.log("TaskProcessor stopped.")

    def _process_tasks(self):
        """Main loop for processing tasks."""
        global terminating, AUTO_MODE
        while not terminating and self.running:
            try:
                STATUS_CHANGE_EVENT.wait(timeout=5)  # Wait for a task list change
                STATUS_CHANGE_EVENT.clear()

                if not AUTO_MODE:
                    time.sleep(1)  # Check less frequently if not in auto mode
                    continue

                with tasks_db_lock:
                    # Process only auto tasks, and only TODO or IN_PROGRESS
                    tasks_to_process = [
                        task
                        for task in tasks_list
                        if task.is_auto == TaskExecutability.AUTO
                        and task.status in (TaskStatus.TODO, TaskStatus.IN_PROGRESS)
                    ]

                for task in tasks_to_process:
                    if not self.running or terminating:
                        break  # Exit if shutting down

                    if task.status == TaskStatus.TODO:
                        self.agent.process_single_task(task)  # Use agent's method
                    elif task.status == TaskStatus.IN_PROGRESS:
                        # For simplicity, just mark IN_PROGRESS as DONE.  You could add resume logic.
                        self.agent._update_task_status(
                            task, TaskStatus.COMPLETED
                        )  # Use agent

            except Exception as e:
                logger.log(f"Error in TaskProcessor: {e}", "ERROR")
                time.sleep(5)


class SwarmsAutoAgent:
    """
    An autonomous AI agent for managing and executing tasks.

    Attributes:
        model_name: Name of the LLM to use (default: "gpt-4")
        tasks_file: Path to the CSV file containing tasks.
        output_dir: Directory to save generated documents.
        max_iterations: Maximum number of interaction loops.
        interactive_mode: Whether to enable interactive user input (set during initialization).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        tasks_file: str = TASKS_FILE,
        output_dir: str = OUTPUT_DIR,
        max_iterations: int = MAX_ITERATIONS,
        interactive_mode: bool = True,  # Now explicitly set during initialization
    ):
        """Initialize the SwarmsAutoAgent."""
        self.model_name = model_name
        self.tasks_file = Path(tasks_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_iterations = max_iterations
        self.interactive_mode = interactive_mode  # Store the user's choice
        self._init_agent()

    def _init_agent(self) -> None:
        """Initialize the core Agent instance."""
        self.agent = Agent(
            agent_name="SwarmsAuto",
            system_prompt=self._get_system_prompt(),
            model_name=self.model_name,
            max_loops=self.max_iterations,
        )

    def _get_system_prompt(self) -> str:
        """Define the main system prompt for the agent."""
        return """
        You are Swarms Auto, an autonomous task management AI.

        Your responsibilities include:
        - Analyzing tasks from a to-do list
        - Determining the best approach for each task
        - Executing tasks autonomously when possible
        - Generating required outputs (documents, code, research)
        - Interacting with the user for clarification when needed
        - Updating task statuses

        Always follow best practices and provide clear, well-structured outputs.
        """

    def _get_task_analysis_prompt(self, task: Task) -> str:
        """Generate prompt for task analysis."""
        return f"""
        Analyze the following task: "{task.description}".
        Identify:
        - Required resources
        - Task complexity (Easy, Medium, Hard)
        - Dependencies (if any)
        - Potential ambiguities
        - Whether it can be autonomously executed or needs human input

        Output in JSON format:
        {{
            "required_resources": ["resource1", "resource2"],
            "complexity": "Easy",
            "dependencies": ["dependency1"],
            "ambiguities": ["ambiguity1"],
            "autonomous": true
        }}
        """

    def _get_autonomous_execution_prompt(self, task: Task) -> str:
        """Generate prompt for autonomous task execution."""
        return f"""
        Generate the required output for the task: "{task.description}".
        Ensure that:
        - It includes all necessary details
        - The content is formatted properly
        - Code snippets (if any) are correctly structured
        - It follows best practices
        """

    def _get_user_interaction_prompt(
        self, task: Task, ambiguities: List[str]
    ) -> str:
        """Generate prompt for user interaction."""
        ambiguities_str = "\n".join([f"- {ambiguity}" for ambiguity in ambiguities])
        return f"""
        The task "{task.description}" has some ambiguities.
        Please provide clarification on the following aspects:
        {ambiguities_str}
        """

    def _analyze_task(self, task: Task) -> TaskAnalysisResult:
        """Analyze a task using the AI agent."""

        if TEST_MODE:
            # Simulate analysis in test mode
            return TaskAnalysisResult(
                required_resources=["Simulated Resource"],
                complexity="Easy",
                dependencies=[],
                ambiguities=[],
                autonomous=True,
            )

        analysis_prompt = self._get_task_analysis_prompt(task)
        response = self.agent.run(analysis_prompt)

        # Improved JSON parsing with fallback using regex
        try:
            analysis_data = json.loads(response)
        except json.JSONDecodeError:
            try:
                # Extract JSON using regex
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    analysis_data = json.loads(match.group(0))
                else:
                    raise ValueError("No valid JSON found in response.")
            except Exception as e:
                logger.log(
                    f"Failed to parse AI response as JSON: {response}. Error: {e}",
                    "ERROR",
                )
                return TaskAnalysisResult(
                    required_resources=[],
                    complexity="Unknown",
                    dependencies=[],
                    ambiguities=["Could not parse AI analysis"],
                    autonomous=False,
                )

        return TaskAnalysisResult(
            required_resources=analysis_data.get("required_resources", []),
            complexity=analysis_data.get("complexity", "Unknown"),
            dependencies=analysis_data.get("dependencies", []),
            ambiguities=analysis_data.get("ambiguities", []),
            autonomous=analysis_data.get("autonomous", False),
        )

    def _execute_task_autonomously(self, task: Task) -> AIResponse:
        """Execute a task autonomously."""

        if TEST_MODE:
            # Simulate execution in test mode
            simulated_output = (
                "# Execution Output\nThis is a simulated, complete output for the"
                " task."
            )
            return AIResponse(content=simulated_output)

        execution_prompt = self._get_autonomous_execution_prompt(task)
        response_text = self.agent.run(execution_prompt)
        return AIResponse(content=response_text)

    def _handle_interactive_task(
        self, task: Task, ambiguities: List[str]
    ) -> AIResponse:
        """Handle tasks requiring user interaction."""
        if not self.interactive_mode:
            logger.log(
                "Task requires interaction, but interactive mode is disabled.",
                "WARNING",
            )
            return AIResponse(content="Interaction required, but interactive mode is off.")

        interaction_prompt = self._get_user_interaction_prompt(task, ambiguities)
        console.print(Markdown(f"**User Interaction Required:**\n{interaction_prompt}"))

        user_input = Prompt.ask("Your input")  # Get user input from CLI using Rich
        # Combine user input with the original prompt for context
        combined_input = f"{interaction_prompt}\n\nUser Input: {user_input}"
        # Get AI response to the combined input
        response_text = self.agent.run(combined_input)
        return AIResponse(content=response_text)

    def _determine_document_type(self, task: Task, analysis: str) -> str:
        """Determine the type of document to generate (from reference code)."""
        text = (
            task.name.lower()
            + " "
            + task.description.lower()
            + " "
            + analysis.lower()
        )
        candidates = {
            "code": ["code", "develop", "program", "script", "implement"],
            "research": ["research", "analyze", "study", "investigate"],
            "plan": ["plan", "roadmap", "strategy"],
            "documentation": ["document", "manual", "guide"],
        }
        scores = {
            k: sum(word in text for word in keywords)
            for k, keywords in candidates.items()
        }
        return (
            max(scores, key=scores.get) if sum(scores.values()) > 0 else "documentation"
        )

    def _generate_document(self, task: Task, ai_response: AIResponse) -> None:
        """Generate a markdown document for the completed task."""
        # Determine document type
        analysis_str = (
            ""
            if not isinstance(ai_response, AIResponse)
            else ai_response.content
            if hasattr(ai_response, "content")
            else str(ai_response)
        )
        document_type = self._determine_document_type(task, analysis_str)

        file_name = f"{task.id}_{task.name.replace(' ', '_')}_{document_type}.md"
        file_path = self.output_dir / file_name

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# {task.name} ({document_type})\n\n")
                f.write(f"**Description:** {task.description}\n\n")
                f.write(f"**Status:** {task.status.value}\n\n")
                f.write(f"**Created at:** {task.created_at}\n\n")  # created_at
                f.write(f"**Updated at:** {task.updated_at}\n\n")  # updated_at
                f.write("## AI-Generated Content:\n\n")
                f.write(ai_response.content)
            logger.log(f"Document generated: {file_path}", "SUCCESS")

        except Exception as e:
            logger.log(f"Error generating document for task {task.id}: {e}", "ERROR")

    def _update_task_status(self, task: Task, new_status: TaskStatus) -> None:
        """Update task status and timestamps, then save to CSV."""
        with tasks_db_lock:
            task.status = new_status
            task.updated_at = datetime.now().isoformat()  # Update the timestamp
            logger.log(
                f"Updated status for task '{task.name}' to {new_status.value}",
                "SUCCESS",
            )
            # Moved CSV update here
            todo_watcher = TodoWatcher()
            todo_watcher.update_todo_file()
        STATUS_CHANGE_EVENT.set()  # Signal for update


    def _display_task_table(self) -> None:
        """Display current tasks in a Rich table."""
        with tasks_db_lock:  # Acquire lock
            table = Table(
                "ID",
                "Name",
                "Status",
                "Autonomous",
                "Created",
                "Updated",
                title="Current Tasks",
                show_lines=True,
            )
            for task in tasks_list:
                table.add_row(
                    str(task.id),
                    task.name,
                    task.status.value,
                    "Yes" if task.is_auto == TaskExecutability.AUTO else "No",
                    task.created_at,
                    task.updated_at,
                )
            console.print(table)

    def process_single_task(self, task: Task) -> None:
        """Processes a single task, handling both auto and interactive."""
        logger.log(f"Processing task: {task.name}")

        if task.is_auto == TaskExecutability.AUTO:
            analysis_result = self._analyze_task(task)
            if analysis_result.autonomous:
                self._update_task_status(task, TaskStatus.IN_PROGRESS)
                ai_response = self._execute_task_autonomously(task)
                self._generate_document(task, ai_response)
                self._update_task_status(task, TaskStatus.COMPLETED)
            else:
                # Handle initially auto tasks later deemed non-autonomous
                logger.log(
                    f"Task '{task.name}' requires interaction based on analysis.",
                    "INFO",
                )
                if self.interactive_mode:
                    ai_response = self._handle_interactive_task(
                        task, analysis_result.ambiguities
                    )
                    self._generate_document(task, ai_response)
                    self._update_task_status(task, TaskStatus.COMPLETED)
                else:
                    logger.log(
                        f"Skipping interactive task '{task.name}' (interactive mode"
                        " off)",
                        "WARNING",
                    )
        elif task.is_auto == TaskExecutability.INTERACTIVE:
            if self.interactive_mode:
                analysis_result = self._analyze_task(
                    task
                )  # Analyze even interactive tasks
                ai_response = self._handle_interactive_task(
                    task, analysis_result.ambiguities
                )
                self._generate_document(task, ai_response)
                self._update_task_status(task, TaskStatus.COMPLETED)
            else:
                logger.log(
                    f"Skipping interactive task '{task.name}' (interactive mode off)",
                    "WARNING",
                )


# --- Signal Handler ---
def signal_handler(signum, frame):
    """Handles termination signals (Ctrl+C, etc.)."""
    global terminating
    logger.log("Termination signal received. Shutting down gracefully...")
    terminating = True
    STATUS_CHANGE_EVENT.set()  # Wake up any waiting threads


# --- Main UI Functions ---
def display_tasks_ui():
    """Displays tasks using Rich."""
    with tasks_db_lock:
        if not tasks_list:
            console.print("[yellow]No tasks available.[/]")
            return

        table = Table(
            "ID", "Name", "Status", "Auto", title="Current Tasks", show_lines=True
        )
        for task in tasks_list:
            auto_flag = "[AUTO]" if task.is_auto == TaskExecutability.AUTO else ""
            table.add_row(
                str(task.id), f"{task.name} {auto_flag}", task.status.value, task.is_auto.value
            )
        console.print(table)

def get_valid_input_ui(prompt_message, input_type=str, valid_options=None):
    """Gets valid input from the user using Rich prompts."""
    while True:
        try:
            if input_type == bool:
                value = Confirm.ask(prompt_message)
                return value
            else:
                value = Prompt.ask(prompt_message)
                if input_type == int:
                    value = int(value)
                elif input_type == float:
                    value = float(value)

                if valid_options and value not in valid_options:
                    console.print(
                        f"[yellow]Invalid option.  Choose from: {', '.join(map(str, valid_options))}[/]"
                    )
                    continue
                return value
        except Exception as e:
            console.print(f"[red]Invalid input: {e}[/]")

def add_task_interactive_ui(todo_watcher: TodoWatcher):
    """Adds a new task interactively."""
    console.print(Markdown("**Add New Task**"))
    name = get_valid_input_ui("Task Name: ")
    description = get_valid_input_ui("Task Description: ")
    is_auto = get_valid_input_ui("Autonomous? (yes/no): ", input_type=bool)

    # Find the next available ID
    with tasks_db_lock:
        if not tasks_list:
            new_id = 1
        else:
            new_id = max(task.id for task in tasks_list) + 1

    new_task = Task(
        id=new_id,
        name=name,
        description=description,
        status=TaskStatus.TODO,
        is_auto=TaskExecutability.AUTO if is_auto else TaskExecutability.INTERACTIVE,
    )
    with tasks_db_lock:
        tasks_list.append(new_task)
    todo_watcher.update_todo_file()  # Save to CSV
    logger.log(f"Added task: '{name}' (ID: {new_id})", "SUCCESS")

def execute_task_interactive_ui(agent: SwarmsAutoAgent):
    """Executes a task interactively."""
    with tasks_db_lock:
        todo_tasks = [task for task in tasks_list if task.status == TaskStatus.TODO]
    if not todo_tasks:
        console.print("[yellow]No tasks in 'to do' status.[/]")
        return

    console.print(Markdown("**Select a task to execute:**"))
    for i, task in enumerate(todo_tasks):
        console.print(f"  {i+1}. {task.name}")

    task_index = get_valid_input_ui(
        "Enter task number: ", input_type=int, valid_options=list(range(1, len(todo_tasks) + 1))
    )
    selected_task = todo_tasks[task_index - 1]
    agent.process_single_task(selected_task)


def delete_task_interactive_ui(todo_watcher: TodoWatcher):
    """Deletes a task interactively."""
    display_tasks_ui()  # Show current tasks
    if not tasks_list:
        return

    task_id_to_delete = get_valid_input_ui("Enter ID of task to delete: ", input_type=int)
    with tasks_db_lock:
        task_to_delete = next(
            (task for task in tasks_list if task.id == task_id_to_delete), None
        )

    if task_to_delete:
        confirm_delete = get_valid_input_ui(
            f"Delete task '{task_to_delete.name}'? (yes/no): ", input_type=bool
        )
        if confirm_delete:
            with tasks_db_lock:
                tasks_list.remove(task_to_delete)
            todo_watcher.update_todo_file()
            logger.log(f"Deleted task: '{task_to_delete.name}'", "SUCCESS")
    else:
        console.print("[yellow]Task not found.[/]")

def view_output_interactive_ui():
    """Allows viewing of generated output files."""
    output_dir = Path(OUTPUT_DIR)
    markdown_files = list(output_dir.glob("*.md"))

    if not markdown_files:
        console.print("[yellow]No output files found.[/]")
        return

    console.print(Markdown("**Select a file to view:**"))
    for i, file_path in enumerate(markdown_files):
        console.print(f"  {i+1}. {file_path.name}")

    file_index = get_valid_input_ui(
        "Enter file number: ",
        input_type=int,
        valid_options=list(range(1, len(markdown_files) + 1)),
    )
    selected_file = markdown_files[file_index - 1]

    try:
        with open(selected_file, "r", encoding="utf-8") as f:
            content = f.read()
        console.print(Markdown(f"---\n**File: {selected_file.name}**\n---\n"))
        console.print(Markdown(content))
    except Exception as e:
        logger.log(f"Error reading file {selected_file.name}: {e}", "ERROR")
        console.print(f"[red]Error reading file: {e}[/]")

def main_ui(todo_watcher: TodoWatcher, task_processor: TaskProcessor, agent: SwarmsAutoAgent):
    """Main user interface loop."""
    global AUTO_MODE

    console.print(Markdown("# Phoenix Autonomous Agent"))
    console.print("Select Mode:")
    console.print("  1. Autonomous Mode (processes 'auto' tasks automatically)")
    console.print("  2. Interactive Mode")
    mode = get_valid_input_ui("Enter 1 or 2: ", input_type=int, valid_options=[1, 2])
    AUTO_MODE = mode == 1

    if AUTO_MODE:
        console.print(
            "[green]Autonomous mode enabled. The agent will continuously process"
            " 'auto' tasks.[/]"
        )
        console.print("Press Ctrl+C to quit.")
        try:
            while not terminating:
                time.sleep(1)  # Keep main thread alive
        except KeyboardInterrupt:
            console.print("\n[yellow]KeyboardInterrupt received. Shutting down.[/]")

    else:
        console.print("[green]Interactive Mode Selected.[/]")
        while not terminating:
            display_tasks_ui()
            console.print(
                "Commands: (a)dd, (e)xecute, (d)elete, (v)iew, (r)efresh, (q)uit,"
                " (h)elp"
            )
            command = get_valid_input_ui(
                "Enter command: ", valid_options=["a", "e", "d", "v", "r", "q", "h"]
            )

            if command == "h":
                console.print(Markdown("**Commands:**"))
                console.print("  * **a**: Add new task")
                console.print("  * **e**: Execute a task")
                console.print("  * **d**: Delete a task")
                console.print("  * **v**: View output documents")
                console.print("  * **r**: Refresh tasks from CSV")
                console.print("  * **q**: Quit")
            elif command == "a":
                add_task_interactive_ui(todo_watcher)
            elif command == "e":
                execute_task_interactive_ui(agent)  # Pass the agent
            elif command == "d":
                delete_task_interactive_ui(todo_watcher)
            elif command == "v":
                view_output_interactive_ui()
            elif command == "r":
                logger.log("Manual refresh requested.")
                todo_watcher.parse_todo_file()  # Force a reload from CSV
            elif command == "q":
                break
            time.sleep(0.5)  # Short delay for responsiveness


# --- Main Function ---

def main():
    """Main entry point of the application."""
    global terminating, tasks_list
    tasks_list = []  # Initialize the global task list
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize directories and load environment variables
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        # Initialize TodoWatcher and load tasks
        todo_watcher = TodoWatcher()
        if not todo_watcher.create_template_if_not_exists():
            console.print("[red]Failed to initialize CSV file. Exiting.[/]")
            return

        todo_watcher.parse_todo_file()  # Load initial tasks

        # Initialize the SwarmsAutoAgent
        agent = SwarmsAutoAgent(interactive_mode=True) # Default to interactive

        # Start the file watcher thread
        file_watch_thread = threading.Thread(
            target=todo_watcher.watch_and_update, daemon=True
        )
        file_watch_thread.start()

        # Start the task processor thread
        task_processor = TaskProcessor(todo_watcher, agent)
        task_processor.start()

        console.print("\n[green]Welcome to Phoenix Agent![/]\n")
        main_ui(todo_watcher, task_processor, agent)  # Pass the agent

    except Exception as e:
        logger.log(f"Fatal error: {e}", "ERROR")
        console.print(f"[red]Fatal error: {e}[/]")

    finally:
        terminating = True
        if "task_processor" in locals():
            task_processor.stop()  # Stop the task processor
        logger.log("Phoenix Agent shutdown complete.")
        console.print("[green]Agent shut down.[/]")


if __name__ == "__main__":
    main()