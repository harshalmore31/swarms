import os
import time
import threading
import shutil
import csv
from datetime import datetime
from pathlib import Path
import signal
import queue

# Swarms library for agent functionality
from swarms import Agent

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- Global Settings ---
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY is not defined in environment variables.")
    exit(1)

# For testing purposes—if set to "true", simulated analysis and execution are used.
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

# --- Global Variables ---
TASKS_DB = []
tasks_db_lock = threading.RLock()
AUTO_MODE = False                   # Determines if agent runs autonomously
STATUS_CHANGE_EVENT = threading.Event()
terminating = False                 # Global flag to trigger shutdown
agent_busy = False                  # Flag used in task processing

# --- Logger (Console and File Logging) ---
class Logger:
    def __init__(self):
        self.logs_dir = Path.cwd() / 'logs'
        self.logs_dir.mkdir(exist_ok=True, parents=True)
        self.log_file = self.logs_dir / f"phoenix_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_lock = threading.Lock()

    def log(self, message, level="INFO"):
        timestamp = datetime.now().isoformat(timespec="seconds")
        entry = f"[{timestamp}] [{level}] {message}"
        with self.log_lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
            print(entry)

logger = Logger()

# --- Directory Management ---
def ensure_directories():
    base_dir = Path.cwd()
    output_dir = base_dir / 'outputs'
    todo_dir = output_dir / 'todo'
    inprogress_dir = output_dir / 'inprogress'
    done_dir = output_dir / 'done'
    for d in [output_dir, todo_dir, inprogress_dir, done_dir]:
        d.mkdir(exist_ok=True, parents=True)
    return output_dir, todo_dir, inprogress_dir, done_dir

# --- Task Database Functions ---
def generate_new_id():
    with tasks_db_lock:
        if not TASKS_DB:
            return 1
        return max(int(task['id']) for task in TASKS_DB if task.get('id', '').isdigit()) + 1

def get_tasks_by_status(status: str):
    with tasks_db_lock:
        return [task.copy() for task in TASKS_DB if task.get("status") == status]

def get_all_tasks():
    with tasks_db_lock:
        return [task.copy() for task in TASKS_DB]

def add_task(name: str, description: str, is_auto: bool = False, todo_watcher=None):
    task_id = generate_new_id()
    new_task = {
        "id": str(task_id),
        "name": name,
        "description": description,
        "status": "to do",
        "is_auto": "yes" if is_auto else "no",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    with tasks_db_lock:
        TASKS_DB.append(new_task)
    logger.log(f"ADDED task: '{name}' (ID: {task_id})")
    if todo_watcher:
        todo_watcher.update_todo_file()
    STATUS_CHANGE_EVENT.set()
    return new_task

def update_task_status(task_id: str, new_status: str, todo_watcher=None):
    updated_task = None
    with tasks_db_lock:
        for task in TASKS_DB:
            if task["id"] == task_id:
                old_status = task["status"]
                task["status"] = new_status
                task["updated_at"] = datetime.now().isoformat()
                updated_task = task.copy()
                logger.log(f"UPDATED task '{task['name']}' (ID: {task_id}) from {old_status} to {new_status}")
                break
    if updated_task:
        if todo_watcher:
            todo_watcher.update_todo_file()
        move_task_files(updated_task, old_status, new_status)
        STATUS_CHANGE_EVENT.set()
        return updated_task
    logger.log(f"Task with ID {task_id} not found", "ERROR")
    return {}

def delete_task(task_id: str, todo_watcher=None):
    deleted_task = None
    with tasks_db_lock:
        for i, task in enumerate(TASKS_DB):
            if task["id"] == task_id:
                deleted_task = TASKS_DB.pop(i)
                logger.log(f"DELETED task '{deleted_task['name']}' (ID: {task_id})")
                break
    if deleted_task:
        if todo_watcher:
            todo_watcher.update_todo_file()
        STATUS_CHANGE_EVENT.set()
        return deleted_task
    logger.log(f"Task with ID {task_id} not found for deletion", "ERROR")
    return {}

def find_task_by_id(task_id: str):
    with tasks_db_lock:
        for task in TASKS_DB:
            if task["id"] == task_id:
                return task.copy()
    return None

# --- File Management Functions ---
def move_task_files(task, old_status, new_status):
    if old_status == new_status:
        return
    output_dir, todo_dir, inprogress_dir, done_dir = ensure_directories()
    sanitized = "".join(c if c.isalnum() else "_" for c in task['name'].lower())
    source = todo_dir if old_status == "to do" else inprogress_dir if old_status == "in progress" else done_dir
    dest = todo_dir if new_status == "to do" else inprogress_dir if new_status == "in progress" else done_dir
    for file_path in source.glob(f"{sanitized}_*"):
        dest_path = dest / file_path.name
        try:
            shutil.move(str(file_path), str(dest_path))
            logger.log(f"Moved file {file_path.name} from {source.name} to {dest.name}")
        except Exception as e:
            logger.log(f"Error moving {file_path.name}: {e}", "ERROR")

def create_project_document(task_name: str, content: str, document_type: str, task_status: str):
    output_dir, todo_dir, inprogress_dir, done_dir = ensure_directories()
    target_dir = {"to do": todo_dir, "in progress": inprogress_dir, "done": done_dir}.get(task_status, output_dir)
    sanitized = "".join(c if c.isalnum() else "_" for c in task_name.lower())
    filename = f"{sanitized}_{document_type}_{int(time.time())}.md"
    file_path = target_dir / filename
    header = (
        f"# {task_name} - {document_type.capitalize()}\n"
        f"Created: {datetime.now().isoformat()}\n"
        f"Status: {task_status.upper()}\n"
        f"Type: {document_type.upper()}\n\n"
    )
    document_content = header + content
    try:
        file_path.write_text(document_content, encoding="utf-8")
        logger.log(f"Created document '{filename}' for '{task_name}' in {target_dir.name}")
        return {"success": True, "filePath": str(file_path), "fileName": filename}
    except Exception as e:
        logger.log(f"Error creating document for '{task_name}': {e}", "ERROR")
        return {"success": False, "error": str(e)}

# --- Swarms Agent Functions ---
def analyze_task(task: dict, retry_count=2):
    logger.log(f"Analyzing task '{task['name']}' (ID: {task['id']})")
    # If in test mode, simulate an analysis result.
    if TEST_MODE:
        simulated_analysis = (
            "# Analysis\n"
            "- Domain: Testing\n"
            "- Requirements: None\n"
            "- Ambiguities: None\n"
            "- Complexity: Low\n"
            "- Action Plan: Simulated steps\n"
            "- Resources: None\n"
            "- Time Estimate: 1 minute\n"
            "- Autonomous Feasibility: Yes"
        )
        logger.log(f"(TEST MODE) Simulated analysis for '{task['name']}'")
        return {
            "success": True,
            "analysis": simulated_analysis,
            "canExecuteAutonomously": True,
            "taskName": task['name'],
            "taskId": task['id']
        }
    # Otherwise, use Swarms Agent to analyze
    system_prompt = """
You are Phoenix, an advanced autonomous AI agent.
Analyze the following task and provide:
1. Domain – Which field/area?
2. Requirements – Specific needs?
3. Ambiguities – What is unclear?
4. Complexity – Low, Medium, or High?
5. Action Plan – Outline the steps.
6. Resources – What is required?
7. Time Estimate – How long for a human?
8. Autonomous Feasibility – Can it be automated?
Format your response in Markdown.
"""
    for attempt in range(retry_count + 1):
        try:
            agent = Agent(
                agent_name="Task-Analysis-Agent",
                system_prompt=system_prompt,
                model_name=DEFAULT_MODEL,
                max_loops=3
            )
            task_input = (
                f"Task Name: {task['name']}\n"
                f"Description: {task.get('description', 'No description')}\n"
                f"ID: {task['id']}\n"
                f"Status: {task['status']}"
            )
            analysis = agent.run(task_input)
            # Determine if the analysis suggests it can be executed autonomously.
            cannot_auto = any(kw in analysis.lower() for kw in ["clarification needed", "more information", "human", "cannot be automated"])
            auto_possible = (task.get("is_auto") == "yes") and (not cannot_auto)
            logger.log(f"Completed analysis for '{task['name']}'")
            return {
                "success": True,
                "analysis": analysis,
                "canExecuteAutonomously": auto_possible,
                "taskName": task['name'],
                "taskId": task['id']
            }
        except Exception as e:
            logger.log(f"Analysis error for '{task['name']}' (attempt {attempt+1}): {e}", "ERROR")
            if attempt < retry_count:
                time.sleep(5)
            else:
                return {"success": False, "analysis": str(e), "canExecuteAutonomously": False, "taskName": task['name'], "taskId": task['id']}

def determine_document_type(task, analysis):
    text = (task.get("name", "") + " " + task.get("description", "") + " " + analysis).lower()
    candidates = {
        "code": ["code", "develop", "program", "script", "implement"],
        "research": ["research", "analyze", "study", "investigate"],
        "plan": ["plan", "roadmap", "strategy"],
        "documentation": ["document", "manual", "guide"]
    }
    scores = {k: sum(word in text for word in keywords) for k, keywords in candidates.items()}
    return max(scores, key=scores.get) if sum(scores.values()) > 0 else "documentation"

def execute_task(task: dict, analysis: str, retry_count=2):
    logger.log(f"Executing task '{task['name']}' (ID: {task['id']})")
    # In test mode, simulate execution
    if TEST_MODE:
        simulated_output = (
            "# Execution Output\n"
            "This is a simulated, complete, and implementation-ready solution for the task."
        )
        doc_type = determine_document_type(task, analysis)
        document = create_project_document(task["name"], simulated_output, doc_type, task["status"])
        if document.get("success"):
            logger.log(f"(TEST MODE) Simulated execution complete for '{task['name']}'")
            return {
                "success": True,
                "documentType": doc_type,
                "documentPath": document.get("filePath"),
                "documentName": document.get("fileName")
            }
        else:
            return {"success": False, "error": document.get("error", "Document creation error")}
    # Otherwise, do real execution via Swarms Agent.
    system_prompt = f"""
You are Phoenix, an autonomous AI agent.
Task: {task['name']}
Use the following analysis details:
{analysis}
Provide a complete, structured, and implementation-ready solution in Markdown.
"""
    for attempt in range(retry_count + 1):
        try:
            agent = Agent(
                agent_name="Task-Execution-Agent",
                system_prompt=system_prompt,
                model_name=DEFAULT_MODEL,
                max_loops=3
            )
            task_input = f"Task: {task['name']}\nDescription: {task.get('description', 'No description')}\nExecute completely."
            print(f"Executing '{task['name']}'", end="")
            for _ in range(3):
                print(".", end="", flush=True)
                time.sleep(0.5)
            print()
            output_content = agent.run(task_input)
            doc_type = determine_document_type(task, analysis)
            document = create_project_document(task["name"], output_content, doc_type, task["status"])
            if document.get("success"):
                logger.log(f"Task '{task['name']}' executed successfully.")
                return {"success": True, "documentType": doc_type, "documentPath": document.get("filePath"), "documentName": document.get("fileName")}
            raise Exception(document.get("error", "Document creation error"))
        except Exception as e:
            logger.log(f"Execution error for '{task['name']}' (attempt {attempt+1}): {e}", "ERROR")
            if attempt < retry_count:
                time.sleep(5)
            else:
                return {"success": False, "error": str(e)}
    return {"success": False, "error": "Unknown error in execute_task"}

# --- TodoWatcher: Monitors and Updates the CSV File ---
class TodoWatcher:
    def __init__(self):
        self.todo_file = Path.cwd() / "todo.csv"
        self.backup_dir = Path.cwd() / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.last_modified = None
        self.lock = threading.RLock()
        self.update_interval = 3    # seconds between checks
        self.retry_count = 3
        self.retry_delay = 2

    def create_template_if_not_exists(self):
        with self.lock:
            if not self.todo_file.exists():
                self._create_template_file()
                self.last_modified = self.todo_file.stat().st_mtime
                return True
            return True

    def _create_template_file(self):
        with open(self.todo_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["id", "name", "description", "status", "is_auto"])
            now = datetime.now().isoformat()
            # Sample tasks – adjust as necessary. (Make sure "is_auto" is exactly "yes" for auto tasks.)
            writer.writerow(["1", "Research Swarms", "Research the Swarms library", "to do", "yes"])
            writer.writerow(["2", "Write Documentation", "Write documentation for Project X", "to do", "yes"])
            writer.writerow(["3", "Build Data Pipeline", "Build a Python data processing pipeline", "in progress", "no"])
            writer.writerow(["4", "Complete swarms tools", "Complete swarms tools", "in progress", "yes"])
            writer.writerow(["5", "Search for new multi-agent systems", "Search for new multi-agent systems", "in progress", "yes"])
            writer.writerow(["6", "Research new multi-agent systems", "Research new multi-agent systems", "in progress", "yes"])
        logger.log(f"Created CSV template: {self.todo_file}")

    def create_backup(self):
        if self.todo_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"todo_backup_{timestamp}.csv"
            try:
                shutil.copy2(self.todo_file, backup_file)
                logger.log(f"Backup created: {backup_file.name}")
                backups = sorted(self.backup_dir.glob("todo_backup_*.csv"))
                if len(backups) > 10:
                    for old in backups[:-10]:
                        old.unlink()
                        logger.log(f"Removed old backup: {old.name}")
            except Exception as e:
                logger.log(f"Backup error: {e}", "ERROR")

    def parse_todo_file(self):
        if not self.todo_file.exists():
            self.create_template_if_not_exists()
        for attempt in range(self.retry_count):
            try:
                tasks = []
                with open(self.todo_file, "r", encoding="utf-8") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if not all(k in row for k in ["id", "name", "description", "status", "is_auto"]):
                            logger.log(f"Skipping invalid row: {row}", "WARNING")
                            continue
                        tasks.append(row)
                with tasks_db_lock:
                    global TASKS_DB
                    TASKS_DB = tasks
                self.update_todo_file(force_write=True)
                logger.log("Parsed CSV and updated in-memory tasks.")
                return tasks
            except Exception as e:
                logger.log(f"CSV parse error (attempt {attempt+1}): {e}", "ERROR")
                time.sleep(self.retry_delay)
        logger.log("Failed to parse CSV after retries; recreating template.", "WARNING")
        self.create_template_if_not_exists()
        return []

    def check_for_changes(self):
        try:
            current_mtime = self.todo_file.stat().st_mtime
            if (self.last_modified is None) or (current_mtime > self.last_modified):
                self.last_modified = current_mtime
                return True
            return False
        except Exception as e:
            logger.log(f"Error checking CSV changes: {e}", "ERROR")
            return False

    def watch_and_update(self):
        global terminating
        while not terminating:
            try:
                if self.check_for_changes():
                    logger.log("Change detected in CSV; re-parsing tasks.")
                    self.parse_todo_file()
                    STATUS_CHANGE_EVENT.set()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.log(f"Error in CSV watcher: {e}", "ERROR")
                time.sleep(self.update_interval * 2)

    def update_todo_file(self, force_write=False):
        with self.lock:
            if not self.todo_file.exists():
                logger.log("CSV file not found for update.", "ERROR")
                return False
            for attempt in range(self.retry_count):
                try:
                    self.create_backup()
                    temp_file = self.todo_file.with_suffix('.tmp')
                    with open(temp_file, "w", newline="", encoding="utf-8") as csvfile:
                        fieldnames = ["id", "name", "description", "status", "is_auto"]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        with tasks_db_lock:
                            for task in TASKS_DB:
                                writer.writerow({f: task.get(f, "") for f in fieldnames})
                    shutil.move(str(temp_file), str(self.todo_file))
                    self.last_modified = self.todo_file.stat().st_mtime
                    logger.log("CSV updated successfully.")
                    return True
                except Exception as e:
                    logger.log(f"CSV update error (attempt {attempt+1}): {e}", "ERROR")
                    time.sleep(self.retry_delay)
            logger.log("All attempts to update CSV failed.", "ERROR")
            return False

# --- TaskProcessor: Processes Auto-Enabled Tasks ---
class TaskProcessor:
    def __init__(self, todo_watcher):
        self.todo_watcher = todo_watcher
        self.processing_queue = queue.Queue()
        self.thread = None
        self.lock = threading.RLock()
        self.active_task_id = None
        self.running = True

    def start(self):
        self.running = True
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._process_tasks, daemon=True)
            self.thread.start()
            logger.log("TaskProcessor thread started.")

    def stop(self):
        logger.log("Stopping TaskProcessor...")
        self.running = False
        STATUS_CHANGE_EVENT.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
            logger.log("TaskProcessor stopped.")

    def _check_and_queue_tasks(self):
        # Refresh TASKS_DB from CSV.
        self.todo_watcher.parse_todo_file()
        todo_tasks = get_tasks_by_status("to do")
        inprogress_tasks = get_tasks_by_status("in progress")
        # Only auto tasks (is_auto exactly "yes") are processed.
        auto_tasks = [t for t in (todo_tasks + inprogress_tasks) if t.get("is_auto") == "yes"]
        with self.lock:
            queued = set(self.processing_queue.queue)
            for task in auto_tasks:
                if task["id"] not in queued and task["id"] != self.active_task_id:
                    logger.log(f"Queueing task '{task['name']}' (ID: {task['id']})")
                    self.processing_queue.put(task["id"])

    def _handle_next_task(self):
        try:
            task_id = self.processing_queue.get(block=False)
        except queue.Empty:
            return
        with self.lock:
            self.active_task_id = task_id
        task = find_task_by_id(task_id)
        if not task:
            logger.log(f"Task ID {task_id} not found!", "ERROR")
            self.processing_queue.task_done()
            return
        logger.log(f"Processing task '{task['name']}' (ID: {task_id})")
        print(f"\nProcessing '{task['name']}'", end="")
        for _ in range(3):
            print(".", end="", flush=True)
            time.sleep(0.5)
        print()
        if task["status"] == "to do":
            self._process_todo_task(task)
        elif task["status"] == "in progress":
            self._process_inprogress_task(task)
        self.processing_queue.task_done()
        with self.lock:
            self.active_task_id = None

    def _process_todo_task(self, task):
        analysis = analyze_task(task)
        if not analysis.get("success", False):
            logger.log(f"Analysis failed for '{task['name']}'", "ERROR")
            return
        if analysis.get("canExecuteAutonomously", False):
            update_task_status(task["id"], "in progress", self.todo_watcher)
            exec_result = execute_task(task, analysis["analysis"])
            if exec_result.get("success", False):
                update_task_status(task["id"], "done", self.todo_watcher)
                logger.log(f"Task '{task['name']}' completed.", "SUCCESS")
                print(f"Task '{task['name']}' finished!")
            else:
                logger.log(f"Execution error for '{task['name']}': {exec_result.get('error')}", "ERROR")
        else:
            logger.log(f"Task '{task['name']}' requires human input; skipping autonomous execution.", "INFO")

    def _process_inprogress_task(self, task):
        logger.log(f"Resuming in-progress task '{task['name']}' (ID: {task['id']})", "INFO")
        time.sleep(2)
        update_task_status(task["id"], "done", self.todo_watcher)
        logger.log(f"In-progress task '{task['name']}' marked as done.", "SUCCESS")
        print(f"Task '{task['name']}' finished!")

    def _process_tasks(self):
        global terminating, agent_busy, AUTO_MODE
        while not terminating and self.running:
            try:
                # Wait until a status change is signaled.
                STATUS_CHANGE_EVENT.wait(timeout=10)
                STATUS_CHANGE_EVENT.clear()
                if not AUTO_MODE:
                    time.sleep(1)
                    continue
                self._check_and_queue_tasks()
                if not self.processing_queue.empty() and not agent_busy:
                    agent_busy = True
                    self._handle_next_task()
                    agent_busy = False
            except Exception as e:
                logger.log(f"Error in TaskProcessor: {e}", "ERROR")
                time.sleep(5)
                agent_busy = False

# --- Interactive UI Functions ---
def display_tasks():
    tasks = get_all_tasks()
    print("\n--- Current Tasks ---")
    if not tasks:
        print("No tasks available.")
        return
    for t in tasks:
        auto_flag = "[AUTO]" if t.get("is_auto") == "yes" else ""
        print(f"  [{t['id']}] {t['name']} - {t['status'].upper()} {auto_flag}")
    print("-" * 30)

def get_valid_input(prompt, input_type=str, valid_options=None, allow_empty=False):
    while True:
        try:
            value = input(prompt).strip()
            if not value and allow_empty:
                return None
            if input_type == int:
                value = int(value)
            elif input_type == float:
                value = float(value)
            elif input_type == bool:
                if value.lower() not in ["yes", "no"]:
                    raise ValueError("Enter 'yes' or 'no'.")
                value = (value.lower() == "yes")
            if valid_options and value not in valid_options:
                print("Invalid option. Options:", valid_options)
                continue
            return value
        except Exception as e:
            print("Invalid input:", e)

def add_task_interactive(todo_watcher):
    print("\nAdd New Task:")
    name = get_valid_input("Task Name: ")
    description = get_valid_input("Task Description: ", allow_empty=True)
    is_auto = get_valid_input("Autonomous? (yes/no): ", input_type=bool)
    add_task(name, description, is_auto, todo_watcher)

def execute_task_interactive(todo_watcher):
    todo_tasks = get_tasks_by_status("to do")
    if not todo_tasks:
        print("No tasks in 'to do' status.")
        return
    print("\nSelect a task to execute:")
    for i, t in enumerate(todo_tasks):
        print(f"  {i+1}. {t['name']}")
    sel = get_valid_input("Enter task number: ", input_type=int, valid_options=range(1, len(todo_tasks)+1))
    task = todo_tasks[sel-1]
    update_task_status(task["id"], "in progress", todo_watcher)
    analysis = analyze_task(task)
    if not analysis.get("success", False):
        logger.log(f"Analysis failed for '{task['name']}'", "ERROR")
        update_task_status(task["id"], "to do", todo_watcher)
        return
    proceed = get_valid_input("Proceed with execution? (yes/no): ", input_type=bool)
    if proceed:
        result = execute_task(task, analysis["analysis"])
        if result.get("success", False):
            update_task_status(task["id"], "done", todo_watcher)
            print(f"Task '{task['name']}' completed.")
        else:
            print(f"Execution failed for '{task['name']}'.")
    else:
        update_task_status(task["id"], "to do", todo_watcher)

def delete_task_interactive(todo_watcher):
    tasks = get_all_tasks()
    if not tasks:
        print("No tasks to delete.")
        return
    print("\nDelete a Task:")
    for i, t in enumerate(tasks):
        print(f"  {i+1}. {t['name']} ({t['status']})")
    sel = get_valid_input("Enter task number: ", input_type=int, valid_options=range(1, len(tasks)+1))
    task = tasks[sel-1]
    conf = get_valid_input(f"Delete '{task['name']}'? (yes/no): ", input_type=bool)
    if conf:
        delete_task(task["id"], todo_watcher)
        print(f"Task '{task['name']}' deleted.")

def view_output_interactive():
    output_dir, todo_dir, inprogress_dir, done_dir = ensure_directories()
    print("\nSelect Output Directory:")
    print("  1. All outputs")
    print("  2. To Do")
    print("  3. In Progress")
    print("  4. Done")
    choice = get_valid_input("Your choice: ", input_type=int, valid_options=[1, 2, 3, 4])
    selected = output_dir if choice == 1 else todo_dir if choice == 2 else inprogress_dir if choice == 3 else done_dir
    files = list(selected.glob("*.md"))
    if not files:
        print(f"No files in {selected.name}")
        return
    for i, f in enumerate(files):
        print(f"  {i+1}. {f.name}")
    sel = get_valid_input("Select file number: ", input_type=int, valid_options=range(1, len(files)+1))
    try:
        content = files[sel-1].read_text("utf-8")
        print(f"\n--- Content of {files[sel-1].name} ---")
        print(content)
        print("-" * 30)
    except Exception as e:
        logger.log(f"File read error: {e}", "ERROR")
        print(f"Error reading file: {e}")

# --- Signal Handling ---
def signal_handler(signum, frame):
    global terminating
    print("\nTermination signal received. Shutting down gracefully...")
    terminating = True

# --- Main UI ---
def main_ui(todo_watcher, task_processor):
    print("Phoenix Autonomous Agent")
    print("Select Mode:")
    print("  1. Autonomous Mode")
    print("  2. Interactive Mode")
    mode = get_valid_input("Enter 1 or 2: ", input_type=int, valid_options=[1, 2])
    global AUTO_MODE
    AUTO_MODE = (mode == 1)
    if AUTO_MODE:
        print("\nAutonomous mode enabled. The agent will continuously watch the CSV and process auto tasks.")
        print("Press Ctrl+C to quit.")
        try:
            while not terminating:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received. Preparing to shut down.")
    else:
        print("\nInteractive Mode Selected.")
        while not terminating:
            display_tasks()
            print("Commands: (a)dd, (e)xecute, (d)elete, (v)iew, (r)efresh, (q)uit, (h)elp")
            cmd = get_valid_input("Enter command: ", valid_options=["a","e","d","v","r","q","h"])
            if cmd == "h":
                print("Commands:")
                print("  a: Add new task")
                print("  e: Execute a task")
                print("  d: Delete a task")
                print("  v: View output documents")
                print("  r: Refresh tasks from CSV")
                print("  q: Quit")
            elif cmd == "a":
                add_task_interactive(todo_watcher)
            elif cmd == "e":
                execute_task_interactive(todo_watcher)
            elif cmd == "d":
                delete_task_interactive(todo_watcher)
            elif cmd == "v":
                view_output_interactive()
            elif cmd == "r":
                logger.log("Manual refresh requested.")
                todo_watcher.parse_todo_file()
            elif cmd == "q":
                break
            time.sleep(0.5)

# --- Main Function ---
def main():
    global terminating
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        ensure_directories()
        logger.log("Starting Phoenix Agent...")
        todo_watcher = TodoWatcher()
        if not todo_watcher.create_template_if_not_exists():
            print("Failed to initialize CSV file. Exiting.")
            exit(1)
        # Initial CSV load
        todo_watcher.parse_todo_file()
        # Start CSV watcher thread
        file_watch_thread = threading.Thread(target=todo_watcher.watch_and_update, daemon=True)
        file_watch_thread.start()
        # Start task processor thread
        task_processor = TaskProcessor(todo_watcher)
        task_processor.start()
        print("\nWelcome to Phoenix Agent!\n")
        main_ui(todo_watcher, task_processor)
    except Exception as e:
        logger.log(f"Fatal error: {e}", "ERROR")
        print(f"Fatal error: {e}")
    finally:
        terminating = True
        if 'task_processor' in locals():
            task_processor.stop()
        logger.log("Phoenix Agent shutdown complete.")
        print("Agent shut down.")

if __name__ == "__main__":
    main()