# --- START OF FILE eval_cr.py (MODIFIED AGAIN) ---
import os
import requests
from dotenv import load_dotenv
import json

import inspect
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.tree import Tree

# --- Load environment variables ---
load_dotenv()
API_KEY = os.getenv("SWARMS_API_KEY")
BASE_URL = "https://swarms-api-285321057562.us-east1.run.app"


# Console for rich output
console = Console()

# --- Helper Classes ---
class Category(str, Enum):
    """Categories for evaluation criteria."""
    CORRECTNESS = "Correctness"
    COMPLETENESS = "Completeness"
    RELEVANCE = "Relevance"
    NOVELTY = "Novelty"
    EFFICIENCY = "Efficiency"
    USABILITY = "Usability"
    SAFETY = "Safety"
    EXPLAINABILITY = "Explainability"
    ROBUSTNESS = "Robustness"
    CONSISTENCY = "Consistency"
    COHERENCE = "Coherence"
    CONCISENESS = "Conciseness"
    COLLABORATION = "Collaboration"
    SPECIALIZATION = "Specialization"
    INTEGRATION = "Integration"
    PARALLELISM = "Parallelism"
    OTHER = "Other"


class MeasurementMethod(str, Enum):
    """Methods for measuring criteria."""
    AUTOMATED = "Automated"
    HUMAN = "Human Evaluation"
    BENCHMARK = "Benchmark Dataset"
    REAL_WORLD = "Real-World Experiment"


@dataclass
class EvaluationResult:
    """Store the result of an evaluation."""
    score: float
    max_score: float
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def normalized_score(self) -> float:
        """Return score normalized to a 0-1 scale."""
        return self.score / self.max_score if self.max_score != 0 else 0


@dataclass
class Metric:
    """Definition of an evaluation metric."""
    name: str
    description: str
    category: Category
    measurement_method: MeasurementMethod
    evaluator_function: Callable[[Any], EvaluationResult]
    max_score: float = 5.0
    weight: float = 1.0
    

@dataclass
class CriteriaSet:
    """A collection of evaluation metrics."""
    name: str
    description: str
    metrics: List[Metric] = field(default_factory=list)
    
    def add_metric(self, metric: Metric) -> None:
        """Add a metric to the criteria set."""
        self.metrics.append(metric)
    
    def remove_metric(self, metric_name: str) -> None:
        """Remove a metric from the criteria set by name."""
        self.metrics = [m for m in self.metrics if m.name != metric_name]


class EvaluationSummary:
    """Summary of evaluation results across multiple metrics."""
    
    def __init__(self, criteria_set: CriteriaSet):
        self.criteria_set = criteria_set
        self.results: Dict[str, EvaluationResult] = {}
        self.timestamps: Dict[str, float] = {}
        self.metadata: Dict[str, Any] = {}
    
    def add_result(self, metric_name: str, result: EvaluationResult) -> None:
        """Add a result for a specific metric."""
        self.results[metric_name] = result
        self.timestamps[metric_name] = time.time()
    
    def get_result(self, metric_name: str) -> Optional[EvaluationResult]:
        """Get the result for a specific metric."""
        return self.results.get(metric_name)
    
    def get_category_scores(self) -> Dict[Category, List[Tuple[Metric, EvaluationResult]]]:
        """Group results by category."""
        category_scores = {category: [] for category in Category}
        
        for metric in self.criteria_set.metrics:
            if metric.name in self.results:
                category = metric.category
                category_scores[category].append((metric, self.results[metric.name]))
        
        return {k: v for k, v in category_scores.items() if v}  # Remove empty categories
    
    def get_overall_score(self) -> float:
        """Calculate overall weighted score."""
        if not self.results:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric in self.criteria_set.metrics:
            if metric.name in self.results:
                result = self.results[metric.name]
                weighted_sum += result.normalized_score * metric.weight
                total_weight += metric.weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class BaseReporter:
    """Base class for result reporters."""
    
    def report(self, summary: EvaluationSummary) -> None:
        """Report evaluation results."""
        raise NotImplementedError("Subclasses must implement this method")


class ConsoleReporter(BaseReporter):
    """Reporter that outputs results to the console using Rich."""
    
    def __init__(self, detailed: bool = True):
        self.detailed = detailed
        self.console = Console()
    
    def report(self, summary: EvaluationSummary) -> None:
        """Report evaluation results to the console."""
        overall_score = summary.get_overall_score()
        
        # Create main table
        table = Table(title=f"Evaluation Results: {summary.criteria_set.name}")
        table.add_column("Category", style="cyan")
        table.add_column("Metric", style="green")
        table.add_column("Score", justify="right")
        table.add_column("Weight", justify="right")
        table.add_column("Notes", style="dim")
        
        # Add rows by category
        category_scores = summary.get_category_scores()
        for category, metrics_results in category_scores.items():
            # Calculate category average for coloring
            category_avg = sum(r.normalized_score for _, r in metrics_results) / len(metrics_results)
            category_color = self._get_score_color(category_avg)
            
            for i, (metric, result) in enumerate(metrics_results):
                # Only show category name on the first row of each category
                cat_display = category.value if i == 0 else ""
                
                score_text = f"{result.score:.2f}/{result.max_score:.1f}"
                score_color = self._get_score_color(result.normalized_score)
                
                table.add_row(
                    cat_display, 
                    metric.name, 
                    f"[{score_color}]{score_text}[/{score_color}]", 
                    f"{metric.weight:.1f}",
                    result.notes or ""
                )
        
        # Overall score panel
        score_panel = Panel(
            f"[bold]{overall_score:.2%}[/bold]", 
            title="Overall Score",
            border_style=self._get_score_color(overall_score)
        )
        
        # Display results
        self.console.print("\n")
        self.console.print(score_panel)
        self.console.print(table)
        
        # Display metadata if available and detailed is True
        if self.detailed and summary.metadata:
            metadata_panel = Panel(
                Syntax(json.dumps(summary.metadata, indent=2), "json"),
                title="Metadata",
                border_style="dim"
            )
            self.console.print(metadata_panel)
    
    def _get_score_color(self, normalized_score: float) -> str:
        """Get color based on score."""
        if normalized_score >= 0.8:
            return "green"
        elif normalized_score >= 0.6:
            return "yellow"
        elif normalized_score >= 0.4:
            return "magenta"
        else:
            return "red"


class JSONReporter(BaseReporter):
    """Reporter that outputs results as JSON."""
    
    def report(self, summary: EvaluationSummary) -> str:
        """Report evaluation results as JSON string."""
        result_dict = {
            "criteria_set": summary.criteria_set.name,
            "description": summary.criteria_set.description,
            "overall_score": summary.get_overall_score(),
            "categories": {},
            "metadata": summary.metadata
        }
        
        # Group by category
        category_scores = summary.get_category_scores()
        for category, metrics_results in category_scores.items():
            result_dict["categories"][category.value] = {
                "metrics": [
                    {
                        "name": metric.name,
                        "score": result.score,
                        "max_score": result.max_score,
                        "normalized_score": result.normalized_score,
                        "weight": metric.weight,
                        "notes": result.notes,
                        "metadata": result.metadata
                    }
                    for metric, result in metrics_results
                ]
            }
        
        return json.dumps(result_dict, indent=2)


class Evaluator:
    """Main class for evaluating AI agents."""
    
    def __init__(self, criteria_set: CriteriaSet):
        self.criteria_set = criteria_set
        self.summary = EvaluationSummary(criteria_set)
        self.reporters: List[BaseReporter] = [ConsoleReporter()]
    
    def add_reporter(self, reporter: BaseReporter) -> None:
        """Add a reporter to the evaluator."""
        self.reporters.append(reporter)
    
    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set metadata for the evaluation."""
        self.summary.metadata = metadata
    
    def evaluate(self, ai_output: Any, metrics: Optional[List[str]] = None) -> EvaluationSummary:
        """
        Evaluate AI output against selected metrics.
        
        Args:
            ai_output: The output to evaluate
            metrics: List of metric names to evaluate (default: all metrics)
            
        Returns:
            EvaluationSummary: Summary of evaluation results
        """
        # Reset summary
        self.summary = EvaluationSummary(self.criteria_set)
        
        # Determine which metrics to use
        metrics_to_evaluate = self.criteria_set.metrics
        if metrics:
            metrics_to_evaluate = [m for m in self.criteria_set.metrics if m.name in metrics]
        
        # Display evaluation progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for metric in metrics_to_evaluate:
                task_id = progress.add_task(f"Evaluating {metric.name}...", total=1)
                
                try:
                    # Measure evaluation time
                    start_time = time.time()
                    result = metric.evaluator_function(ai_output)
                    elapsed = time.time() - start_time
                    
                    # Add timing information to result metadata
                    if not result.metadata:
                        result.metadata = {}
                    result.metadata["evaluation_time"] = elapsed
                    
                    self.summary.add_result(metric.name, result)
                    progress.update(task_id, completed=1)
                    
                except Exception as e:
                    console.print(f"[bold red]Error evaluating {metric.name}:[/bold red] {str(e)}")
                    progress.update(task_id, completed=1, description=f"Error evaluating {metric.name}")
        
        # Report results with all reporters
        for reporter in self.reporters:
            reporter.report(self.summary)
        
        return self.summary

# --- Helper Functions ---
def create_human_evaluation_metric(
    name: str,
    description: str,
    category: Category,
    max_score: float = 5.0,
    weight: float = 1.0,
) -> Metric:
    """Create a metric that will be evaluated by humans."""

    def human_evaluator(ai_output: Any) -> EvaluationResult:
        console.print(f"\n[bold cyan]Human Evaluation: {name}[/bold cyan]")
        console.print(f"[dim]{description}[/dim]\n")
        console.print("AI Output to evaluate:")
        console.print(Panel(str(ai_output)))

        score = float(console.input(f"Enter score (0-{max_score}): "))
        notes = console.input("Enter notes (optional): ")

        return EvaluationResult(score=score, max_score=max_score, notes=notes)

    return Metric(
        name=name,
        description=description,
        category=category,
        measurement_method=MeasurementMethod.HUMAN,
        evaluator_function=human_evaluator,
        max_score=max_score,
        weight=weight,
    )


def create_automated_metric(
    name: str,
    description: str,
    category: Category,
    evaluation_function: Callable[[Any], Tuple[float, Optional[str]]],
    max_score: float = 1.0,
    weight: float = 1.0
) -> Metric:
    """
    Create a metric that will be evaluated automatically.
    
    Args:
        evaluation_function: Function that takes AI output and returns (score, notes)
    """
    
    def automated_evaluator(ai_output: Any) -> EvaluationResult:
        score, notes = evaluation_function(ai_output)
        return EvaluationResult(score=score, max_score=max_score, notes=notes)
    
    return Metric(
        name=name,
        description=description,
        category=category,
        measurement_method=MeasurementMethod.AUTOMATED,
        evaluator_function=automated_evaluator,
        max_score=max_score,
        weight=weight
    )

# --- Swarm API Interaction Class ---
class EvaluatorSwarm:
    """Handles interactions with the Swarms API."""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}

    def run_swarm(self, swarm_config: dict) -> dict:
        """Runs a swarm with the given configuration."""
        response = requests.post(
            f"{self.base_url}/v1/swarm/completions",
            headers=self.headers,
            json=swarm_config,
        )
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()

    def health_check(self) -> dict:
        """Performs a health check on the Swarms API."""
        response = requests.get(f"{self.base_url}/health", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
# --- LLM and Swarm Evaluation Metric Creation ---
def create_llm_evaluation_metric(
    name: str,
    description: str,
    category: Category,
    llm_evaluator_prompt: str,  # Add a prompt for the LLM
    max_score: float = 5.0,
    weight: float = 1.0,
    llm_evaluator_model = "gpt-3.5-turbo"
) -> Metric:
    """
    Create a metric that will be evaluated by an LLM.

    Args:
        llm_evaluator_prompt:  System prompt for the LLM evaluator agent.  This
                               should instruct the LLM how to score the output.
        llm_evaluator_model: The OpenAI model name to use for the LLM agent
    """
    def llm_evaluator(ai_output: Any) -> EvaluationResult:
        from openai import OpenAI
        client = OpenAI()
        try:

            messages = [
                {"role": "system", "content": llm_evaluator_prompt},
                {"role": "user", "content": f"Evaluate the following AI output:\n\n{ai_output}"}
            ]

            response = client.chat.completions.create(
                model=llm_evaluator_model,
                messages=messages,
                max_tokens=100,  # Keep the evaluation concise
                temperature=0.2, # Low temperature for more deterministic scoring
                n = 1 #for distinct values
            )
            # Extract the score from the LLM's response.  We'll assume the LLM
            # outputs a single number, the score.  Robust implementations might
            # need more sophisticated parsing.
            try:
                score_str = response.choices[0].message.content.strip()
                score = float(score_str)
                notes = "" #keep notes empty to keep concise

                # if the score extracted from llm is not a number between 0 and max_score
                if not (0 <= score <= max_score):
                    raise ValueError("LLM did not output score")
                
            except (ValueError, TypeError) as e:
                # Handle cases where the LLM doesn't output a valid score.
                score = 0.0
                notes = f"Error parsing LLM response: {response.choices[0].message.content}"

            return EvaluationResult(score=score, max_score=max_score, notes=notes)

        except Exception as e:
            print(f"Error during LLM evaluation: {e}")
            return EvaluationResult(score=0.0, max_score=max_score, notes=f"LLM evaluation failed: {e}")

    return Metric(
        name=name,
        description=description,
        category=category,
        measurement_method=MeasurementMethod.AUTOMATED,
        evaluator_function=llm_evaluator,
        max_score=max_score,
        weight=weight
    )


def create_swarm_evaluation_metric(
    name: str,
    description: str,
    category: Category,
    evaluator_swarm_config: dict,
    max_score: float = 5.0,
    weight: float = 1.0,
) -> Metric:
    """
    Creates a metric that uses a Swarm (via the API) for evaluation.

    Args:
        evaluator_swarm_config:  Configuration for the evaluator swarm.
    """

    def swarm_evaluator(ai_output: Any) -> EvaluationResult:
        # Initialize the Swarm API client (only once)
        evaluator_swarm = EvaluatorSwarm(API_KEY, BASE_URL)
        
        # Construct the task for the evaluator swarm.
        evaluator_swarm_config["task"] = (
            f"Evaluate the following AI output based on {description}:\n\n{ai_output}"
            f"\n\nProvide a JSON output with a 'score' field (integer between 0 and {max_score})"
            " and a 'notes' field (string)."
        )
        evaluator_swarm_config["output_type"] = "json"

        try:
            swarm_response = evaluator_swarm.run_swarm(evaluator_swarm_config)

            # Now, try to parse the response as JSON.  Handle potential errors.
            try:
                if isinstance(swarm_response, str):
                    result_json = json.loads(swarm_response)
                elif isinstance(swarm_response, dict):
                    result_json = swarm_response
                else:
                    raise ValueError(f"Unexpected response type from swarm: {type(swarm_response)}")
                    
                score = float(result_json.get("score", 0))  # Get score, default to 0
                notes = result_json.get("notes", "")  # Get notes, default to ""

            except (json.JSONDecodeError, TypeError, ValueError) as parse_error:
                score = 0.0
                notes = f"Error parsing swarm response: {parse_error}, Response: {swarm_response}"
                console.print(f"[bold red]Error parsing swarm response:[/bold red] {parse_error}")
                console.print(f"[red]Swarm response:[/red] {swarm_response}")

            if not (0 <= score <= max_score):
                notes += f" (NOTE: Score {score} was outside valid range 0-{max_score} and has been clamped.)"
                score = max(0, min(score, max_score))  # Clamp the score

            return EvaluationResult(score=score, max_score=max_score, notes=notes)

        except requests.exceptions.RequestException as req_err:
            console.print(f"[bold red]Request error:[/bold red] {req_err}")
            return EvaluationResult(
                score=0.0,
                max_score=max_score,
                notes=f"Swarm API request failed: {req_err}",
            )
        except Exception as e:
            console.print(f"[bold red]Unexpected error during swarm evaluation:[/bold red] {e}")
            return EvaluationResult(
                score=0.0, max_score=max_score, notes=f"Swarm evaluation failed: {e}"
            )
        
    return Metric(
        name=name,
        description=description,
        category=category,
        measurement_method=MeasurementMethod.AUTOMATED,
        evaluator_function=swarm_evaluator,  # Use the swarm_evaluator
        max_score=max_score,
        weight=weight,
    )
    
# --- Example Criteria Sets ---
def get_default_criteria_set() -> CriteriaSet:
    
    # --- Evaluator Swarm Config for general evaluation ---
    general_evaluator_config = {
    "name": "General Evaluator Swarm",
    "description": "Evaluates general quality of the output.",
    "agents": [
        {
            "agent_name": "Quality Evaluator",
            "description": "Assesses overall response quality",
            "system_prompt": (
                "You are an expert evaluator, focusing on overall quality."
                " Analyze the given output and provide a JSON response containing"
                " a 'score' (0-5, 5 being best) and 'notes'."
            ),
            "model_name": "groq/mixtral-8x7b-32768",
            "role": "worker",
            "max_loops": 1,
            "max_tokens": 1024,
        },
    ],
    "max_loops": 1,
    "swarm_type": "ConcurrentWorkflow",
    "output_type": "json",  # Expect JSON output
    "return_history": False #no need to return history
}
    criteria = CriteriaSet(
        name="General AI Evaluation",
        description="General evaluation criteria for AI agents using SWARMS",
    )
    
    # Use the swarm evaluator
    criteria.add_metric(
        create_swarm_evaluation_metric(
            name="Overall Quality",
            description="the overall quality of the response",
            category=Category.CORRECTNESS,
            evaluator_swarm_config=general_evaluator_config,
            max_score=5.0,
            weight=1.0,
        )
    )
    
    # --- Add other metrics as needed, potentially with different evaluator swarm configurations ---
    return criteria



def get_swarm_criteria_set() -> CriteriaSet:
    # --- Example evaluator swarm configurations ---

    # --- Config for evaluating agent collaboration ---
    collaboration_evaluator_config = {
        "name": "Collaboration Evaluator Swarm",
        "description": "Evaluates agent collaboration in a swarm.",
        "agents": [
            {
                "agent_name": "Collaboration Analyst",
                "description": "Assesses how well agents work together.",
                "system_prompt": (
                    "You are an expert in multi-agent systems, specializing in collaboration."
                    " Analyze the provided output and history to determine how effectively"
                    " the agents collaborated. Output a JSON object containing a 'score' (0-5, 5 being best)"
                    " and 'notes' explaining your reasoning."
                ),
                "model_name": "groq/mixtral-8x7b-32768",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 1024,
            }
        ],
        "max_loops": 1,
        "swarm_type": "ConcurrentWorkflow",
        "output_type": "json",
        "return_history": True,
    }

    # Config for evaluating role specialization
    specialization_evaluator_config = {
        "name": "Specialization Evaluator Swarm",
        "description": "Evaluates role specialization in a swarm.",
        "agents": [
            {
                "agent_name": "Role Analyst",
                "description": "Assesses how well agents stick to their roles.",
                "system_prompt": (
                    "You are an expert in analyzing agent roles in multi-agent systems."
                    " Examine the output and history, and evaluate how well each agent"
                    " fulfilled its defined role.  Provide a JSON response with a 'score' (0-5,"
                    " 5 being perfect specialization) and 'notes'."
                ),
                "model_name": "groq/mixtral-8x7b-32768",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 1024,
            }
        ],
        "max_loops": 1,
        "swarm_type": "ConcurrentWorkflow",
        "output_type": "json",
        "return_history": True,
    }

    criteria = CriteriaSet(
        name="Swarm AI Evaluation",
        description="Specialized criteria for evaluating multi-agent swarm systems",
    )
    
    # --- Agent Collaboration ---
    criteria.add_metric(create_swarm_evaluation_metric(
        name="Agent Collaboration",
        description="How effectively do agents collaborate and build on each other's outputs?",
        category=Category.COLLABORATION,
        evaluator_swarm_config=collaboration_evaluator_config,
        max_score=5.0,
        weight=1.5
    ))

    # --- Role Specialization ---
    criteria.add_metric(create_swarm_evaluation_metric(
        name="Role Specialization",
        description="How well do agents fulfill their specialized roles in the swarm?",
        category=Category.SPECIALIZATION,
        evaluator_swarm_config=specialization_evaluator_config,
        max_score=5.0,
        weight=1.2
    ))

    # --- Add other swarm-specific metrics as needed, each with its own evaluator swarm ---
    
    # --- Example of an automated metric (response time) ---
    def measure_response_time(ai_output_with_metadata: Dict) -> Tuple[float, str]:
        # Assuming the AI output includes metadata with response_time_ms
        response_time = ai_output_with_metadata.get("metadata", {}).get(
            "response_time_ms", 0
        )

        # Convert to seconds and score (lower is better)
        time_sec = response_time / 1000

        # Example scoring: 0-1s: 5.0, 1-3s: 4.0, 3-5s: 3.0, 5-10s: 2.0, >10s: 1.0
        if time_sec <= 1:
            score = 5.0
        elif time_sec <= 3:
            score = 4.0
        elif time_sec <= 5:
            score = 3.0
        elif time_sec <= 10:
            score = 2.0
        else:
            score = 1.0

        return score, f"Response time: {time_sec:.2f}s"
    
    criteria.add_metric(create_automated_metric(
    name="Response Time",
    description="How quickly did the AI respond?",
    category=Category.EFFICIENCY,
    evaluation_function=measure_response_time,
    max_score=5.0,
    weight=0.5))
    
    return criteria
# --- Evaluation Functions (Examples) ---

def evaluate_agent_output(agent_output: Any, criteria_set: Optional[CriteriaSet] = None) -> EvaluationSummary:
    """
    Evaluate an agent's output using the specified criteria.
    
    Args:
        agent_output: The output to evaluate
        criteria_set: The criteria to use (default: use default criteria)
        
    Returns:
        EvaluationSummary: Summary of evaluation results
    """
    if criteria_set is None:
        criteria_set = get_default_criteria_set()
    
    evaluator = Evaluator(criteria_set)
    
    # Add JSON reporter
    evaluator.add_reporter(JSONReporter())
    
    # Run evaluation
    return evaluator.evaluate(agent_output)


def evaluate_swarm_output(swarm_output: Dict, criteria_set: Optional[CriteriaSet] = None) -> EvaluationSummary:
    """
    Evaluate a swarm's output using specialized swarm evaluation criteria.
    
    Args:
        swarm_output: The swarm output to evaluate (including history and metadata)
        criteria_set: The criteria to use (default: use swarm-specific criteria)
        
    Returns:
        EvaluationSummary: Summary of evaluation results
    """
    if criteria_set is None:
        criteria_set = get_swarm_criteria_set() #get_default_criteria_set() #SwarmEvaluationCriteria.get_criteria_set()
    
    evaluator = Evaluator(criteria_set)
    
    # Add JSON reporter
    evaluator.add_reporter(JSONReporter())
    
    # Run evaluation
    return evaluator.evaluate(swarm_output)

# --- Comparison Functions (Optional, but useful) ---
def evaluate_agent_comparison(
    agent_outputs: Dict[str, Any],
    criteria_set: Optional[CriteriaSet] = None
) -> Dict[str, EvaluationSummary]:
    """
    Compare multiple agent outputs using the same criteria.
    
    Args:
        agent_outputs: Dictionary mapping agent names to their outputs
        criteria_set: The criteria to use (default: use default criteria)
        
    Returns:
        Dict[str, EvaluationSummary]: Mapping of agent names to their evaluation summaries
    """
    if criteria_set is None:
        criteria_set = get_default_criteria_set()
    
    results = {}
    
    for agent_name, output in agent_outputs.items():
        console.print(f"\n[bold]Evaluating agent: {agent_name}[/bold]")
        
        evaluator = Evaluator(criteria_set)
        results[agent_name] = evaluator.evaluate(output)
    
    return results


def evaluate_swarm_comparison(
    swarm_outputs: Dict[str, Dict],
    criteria_set: Optional[CriteriaSet] = None
) -> Dict[str, EvaluationSummary]:
    """
    Compare multiple swarm outputs using the same criteria.
    
    Args:
        swarm_outputs: Dictionary mapping swarm names to their outputs
        criteria_set: The criteria to use (default: use swarm-specific criteria)
    # --- Comparison Functions (Optional, but useful) --- Continued
) -> Dict[str, EvaluationSummary]:
    
    Compare multiple swarm outputs using the same criteria.

    Args:
        swarm_outputs: Dictionary mapping swarm names to their outputs
        criteria_set: The criteria to use (default: use swarm-specific criteria)

    Returns:
        Dict[str, EvaluationSummary]: Mapping of swarm names to their evaluation summaries
    """
    if criteria_set is None:
        criteria_set = get_swarm_criteria_set()

    results = {}

    for swarm_name, output in swarm_outputs.items():
        console.print(f"\n[bold]Evaluating swarm: {swarm_name}[/bold]")

        evaluator = Evaluator(criteria_set)
        results[swarm_name] = evaluator.evaluate(output)

    return results


# Create a simple decorator for measuring function execution time
def measure_execution_time(func):
    """Decorator to measure execution time of a function."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time

        # If result is a dict with metadata, add execution time
        if isinstance(result, dict) and "metadata" in result:
            if result["metadata"] is None:
                result["metadata"] = {}
            result["metadata"]["execution_time"] = execution_time

        return result

    return wrapper