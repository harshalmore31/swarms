import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from swarms import Agent
from swarms.structs.devh.eval_cr import (
    Category,
    CriteriaSet,
    Metric,
    Evaluator,
    ConsoleReporter,
    JSONReporter,
    EvaluationSummary,
    create_automated_metric
)

# Load environment variables
load_dotenv()

class AgentEvaluationRunner:
    """Class to run and evaluate multiple agents on multiple tasks."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize the evaluation runner.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = {}
        self.agents = {}
        self.criteria_sets = {}
        self.tasks = {}
        
    def add_agent(self, agent_id: str, agent: Agent) -> None:
        """Add an agent to be evaluated."""
        self.agents[agent_id] = agent
        
    def add_criteria_set(self, criteria_id: str, criteria: CriteriaSet) -> None:
        """Add an evaluation criteria set."""
        self.criteria_sets[criteria_id] = criteria
        
    def add_task(self, task_id: str, prompt: str, description: str = "") -> None:
        """Add a task for evaluation."""
        self.tasks[task_id] = {
            "prompt": prompt,
            "description": description
        }
    
    def run_evaluation(self, 
                      agent_ids: Optional[List[str]] = None, 
                      task_ids: Optional[List[str]] = None,
                      criteria_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run evaluation for selected agents and tasks.
        
        Args:
            agent_ids: List of agent IDs to evaluate (None = all)
            task_ids: List of task IDs to evaluate (None = all)
            criteria_id: Criteria set ID to use (None = first available)
            
        Returns:
            Dictionary of evaluation results
        """
        # Select agents and tasks
        agents_to_run = agent_ids or list(self.agents.keys())
        tasks_to_run = task_ids or list(self.tasks.keys())
        
        # Select criteria set
        if criteria_id is None and self.criteria_sets:
            criteria_id = next(iter(self.criteria_sets.keys()))
        
        if not criteria_id or criteria_id not in self.criteria_sets:
            raise ValueError("No valid criteria set specified")
            
        criteria = self.criteria_sets[criteria_id]
        
        # Run evaluations
        results = {}
        for agent_id in agents_to_run:
            if agent_id not in self.agents:
                print(f"Warning: Agent {agent_id} not found, skipping")
                continue
                
            agent = self.agents[agent_id]
            agent_results = {}
            
            for task_id in tasks_to_run:
                if task_id not in self.tasks:
                    print(f"Warning: Task {task_id} not found, skipping")
                    continue
                    
                task = self.tasks[task_id]
                prompt = task["prompt"]
                
                print(f"\n🤖 Running agent: {agent_id} on task: {task_id}...\n")
                
                # Run the agent
                start_time = datetime.now()
                output = agent.run(prompt)
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                print(f"\n✅ Agent run complete in {execution_time:.2f}s. Evaluating output...\n")
                
                # Create evaluator with console and JSON reporters
                evaluator = Evaluator(criteria)
                evaluator.add_reporter(JSONReporter())
                
                # Run evaluation
                summary = evaluator.evaluate(output)
                
                # Store results
                agent_results[task_id] = {
                    "output": output,
                    "execution_time": execution_time,
                    "evaluation": summary,
                    "timestamp": datetime.now().isoformat()
                }
            
            results[agent_id] = agent_results
        
        # Store and return results
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "criteria_set": criteria_id,
                "num_agents": len(agents_to_run),
                "num_tasks": len(tasks_to_run)
            },
            "results": results
        }
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\n📊 Evaluation results saved to: {results_file}")
        
        return self.results
    
    def generate_report(self) -> str:
        """
        Generate a summary report of the evaluation results.
        
        Returns:
            Path to the generated report file
        """
        if not self.results:
            raise ValueError("No evaluation results to report")
        
        # Create DataFrame for analysis
        rows = []
        for agent_id, agent_results in self.results["results"].items():
            for task_id, task_result in agent_results.items():
                summary = task_result["evaluation"]
                overall_score = summary.get_overall_score()
                
                # Get category scores
                category_scores = {}
                for category, metrics_results in summary.get_category_scores().items():
                    category_avg = sum(r.normalized_score for _, r in metrics_results) / len(metrics_results)
                    category_scores[f"category_{category.value}"] = category_avg
                
                rows.append({
                    "agent_id": agent_id,
                    "task_id": task_id,
                    "overall_score": overall_score,
                    "execution_time": task_result["execution_time"],
                    **category_scores
                })
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Generate visualization
        plt.figure(figsize=(12, 8))
        
        # Overall scores by agent and task
        plt.subplot(2, 1, 1)
        pivot_df = df.pivot(index="agent_id", columns="task_id", values="overall_score")
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
        plt.title("Overall Evaluation Scores by Agent and Task")
        
        # Category scores by agent (average across tasks)
        plt.subplot(2, 1, 2)
        category_cols = [col for col in df.columns if col.startswith("category_")]
        if category_cols:
            agent_categories = df.groupby("agent_id")[category_cols].mean()
            # Rename columns to remove "category_" prefix
            agent_categories.columns = [col.replace("category_", "") for col in agent_categories.columns]
            sns.heatmap(agent_categories, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
            plt.title("Category Scores by Agent (Average Across Tasks)")
        
        plt.tight_layout()
        
        # Save visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"evaluation_report_{timestamp}.png"
        plt.savefig(report_file)
        
        # Save CSV data too
        csv_file = self.output_dir / f"evaluation_data_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\n📈 Evaluation report saved to: {report_file}")
        print(f"📋 Evaluation data saved to: {csv_file}")
        
        return str(report_file)


def main():
    """Run a batch evaluation of multiple agents on financial advice tasks."""
    print("🚀 Starting Batch Agent Evaluation")
    
    # Create evaluation runner
    runner = AgentEvaluationRunner()
    
    # Set up tasks
    runner.add_task(
        "ai_investment_table", 
        "Create a table of super high growth opportunities for AI. I have $40k to invest in ETFs, index funds, and more. Please create a table in markdown.",
        "AI investment opportunities table"
    )
    
    runner.add_task(
        "portfolio_allocation",
        "Given $40,000 to invest, provide a detailed allocation across different AI technology sectors. Include specific ETFs and index funds with allocation percentages.",
        "Portfolio allocation for AI investments"
    )
    
    # Import criteria creation function 
    from agent_eval_example import create_financial_advice_criteria
    runner.add_criteria_set("financial_advice", create_financial_advice_criteria())
    
    # Setup different agent configurations
    runner.add_agent(
        "finance_gpt4o",
        Agent(
            agent_name="Financial-GPT4o-Agent",
            agent_description="Personal finance advisor agent using GPT-4o",
            system_prompt="You are an expert financial advisor specializing in technology and AI investments. Provide clear, concise advice with specific investment recommendations.",
            max_loops=1,
            model_name="gpt-4o",
            dynamic_temperature_enabled=True,
            user_name="swarms_corp",
            return_step_meta=True,
            output_type="str",
        )
    )
    
    runner.add_agent(
        "finance_gpt35",
        Agent(
            agent_name="Financial-GPT35-Agent",
            agent_description="Personal finance advisor agent using GPT-3.5",
            system_prompt="You are an expert financial advisor specializing in technology and AI investments. Provide clear, concise advice with specific investment recommendations.",
            max_loops=1,
            model_name="gpt-3.5-turbo",
            dynamic_temperature_enabled=True,
            user_name="swarms_corp",
            return_step_meta=True,
            output_type="str",
        )
    )
    
    # Run evaluations
    results = runner.run_evaluation()
    
    # Generate report
    report_path = runner.generate_report()
    
    print(f"\n🏁 Batch evaluation completed! Report available at: {report_path}")


if __name__ == "__main__":
    main()
