import os
import requests
import json
import time
from dotenv import load_dotenv

from swarms.structs.devh.eval_cr import (
    Category,
    CriteriaSet,
    Evaluator,
    JSONReporter,
    create_automated_metric,
    create_human_evaluation_metric,
    evaluate_agent_output,
    measure_execution_time
)

# Load environment variables
load_dotenv()

API_KEY = os.getenv("SWARMS_API_KEY")
BASE_URL = "https://swarms-api-285321057562.us-east1.run.app"

headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}


def run_health_check():
    """Check if the Swarms API is operational"""
    response = requests.get(f"{BASE_URL}/health", headers=headers)
    return response.json()


@measure_execution_time
def run_single_swarm(task="What are the best etfs and index funds for ai and tech?"):
    """Run a financial analysis swarm with the given task"""
    payload = {
        "name": "Financial Analysis Swarm",
        "description": "Market analysis swarm",
        "agents": [
            {
                "agent_name": "Market Analyst",
                "description": "Analyzes market trends",
                "system_prompt": "You are a financial analyst expert.",
                "model_name": "groq/deepseek-r1-distill-qwen-32b",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
            },
            {
                "agent_name": "Economic Forecaster",
                "description": "Predicts economic trends",
                "system_prompt": "You are an expert in economic forecasting.",
                "model_name": "groq/deepseek-r1-distill-qwen-32b",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
            },
            {
                "agent_name": "Data Scientist",
                "description": "Performs data analysis",
                "system_prompt": "You are a data science expert.",
                "model_name": "groq/deepseek-r1-distill-qwen-32b",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
            },
        ],
        "max_loops": 1,
        "swarm_type": "ConcurrentWorkflow",
        "task": task,
        "output_type": "str",
        "return_history": True,
    }

    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=payload,
    )
    execution_time = time.time() - start_time

    output = response.json()
    
    # Add execution time to metadata
    if "metadata" not in output:
        output["metadata"] = {}
    output["metadata"]["execution_time"] = execution_time
    output["metadata"]["response_time_ms"] = execution_time * 1000

    return output


def get_logs():
    """Get the logs from the Swarms API"""
    response = requests.get(
        f"{BASE_URL}/v1/swarm/logs", headers=headers
    )
    output = response.json()
    return output


def get_financial_swarm_criteria() -> CriteriaSet:
    """
    Create a criteria set specifically for financial analysis swarms.
    """
    criteria = CriteriaSet(
        name="Financial Analysis Swarm Evaluation",
        description="Evaluation criteria for financial analysis swarms"
    )
    
    # Financial accuracy
    criteria.add_metric(create_human_evaluation_metric(
        name="Financial Accuracy",
        description="Are the financial facts, terms, and principles correctly applied?",
        category=Category.CORRECTNESS,
        max_score=5.0,
        weight=2.0
    ))
    
    # Investment recommendation quality
    criteria.add_metric(create_human_evaluation_metric(
        name="Investment Recommendation Quality",
        description="Are the investment recommendations appropriate, well-justified, and realistic?",
        category=Category.RELEVANCE,
        max_score=5.0,
        weight=1.8
    ))
    
    # Market analysis depth
    criteria.add_metric(create_human_evaluation_metric(
        name="Market Analysis Depth",
        description="How thorough and insightful is the market analysis?",
        category=Category.COMPLETENESS,
        max_score=5.0,
        weight=1.5
    ))
    
    # Risk assessment
    criteria.add_metric(create_human_evaluation_metric(
        name="Risk Assessment",
        description="Does the analysis properly assess and communicate investment risks?",
        category=Category.SAFETY,
        max_score=5.0,
        weight=1.7
    ))
    
    # Technical diversity
    criteria.add_metric(create_human_evaluation_metric(
        name="Technical Diversity",
        description="Does the response include a diverse set of technical options and considerations?",
        category=Category.NOVELTY,
        max_score=5.0,
        weight=1.2
    ))
    
    # Jargon clarity
    criteria.add_metric(create_human_evaluation_metric(
        name="Jargon Clarity",
        description="Are financial terms and jargon clearly explained for non-experts?",
        category=Category.EXPLAINABILITY,
        max_score=5.0,
        weight=1.0
    ))
    
    # Multi-agent collaboration
    criteria.add_metric(create_human_evaluation_metric(
        name="Agent Collaboration",
        description="How well did the different agents (Market Analyst, Economic Forecaster, Data Scientist) collaborate?",
        category=Category.CONSISTENCY,
        max_score=5.0,
        weight=1.3
    ))
    
    # Response time automated metric
    def measure_response_time(swarm_output: dict) -> tuple[float, str]:
        # Get response time from metadata
        response_time = swarm_output.get("metadata", {}).get("response_time_ms", 0)
        
        # Convert to seconds and score (lower is better)
        time_sec = response_time / 1000
        
        # Example scoring: 0-5s: 5.0, 5-10s: 4.0, 10-20s: 3.0, 20-30s: 2.0, >30s: 1.0
        if time_sec <= 5:
            score = 5.0
        elif time_sec <= 10:
            score = 4.0
        elif time_sec <= 20:
            score = 3.0
        elif time_sec <= 30:
            score = 2.0
        else:
            score = 1.0
            
        return score, f"Response time: {time_sec:.2f}s"
    
    criteria.add_metric(create_automated_metric(
        name="Response Time",
        description="How quickly did the swarm respond?",
        category=Category.EFFICIENCY,
        evaluation_function=measure_response_time,
        max_score=5.0,
        weight=0.5
    ))
    
    # Agent count metric
    def evaluate_agent_count(swarm_output: dict) -> tuple[float, str]:
        try:
            # Get agent count from the output
            agent_count = len(swarm_output.get("agents", []))
            
            # Score based on optimal range (3-5 agents is ideal for financial analysis)
            if 3 <= agent_count <= 5:
                score = 5.0
                notes = f"Optimal agent count: {agent_count}"
            elif agent_count < 3:
                score = 3.0
                notes = f"Too few agents: {agent_count} (recommended: 3-5)"
            else:  # agent_count > 5
                score = 4.0
                notes = f"Many agents: {agent_count} (may increase complexity)"
                
            return score, notes
        except:
            return 0.0, "Couldn't determine agent count"
    
    criteria.add_metric(create_automated_metric(
        name="Agent Configuration",
        description="Is the number of agents optimal for the task?",
        category=Category.EFFICIENCY,
        evaluation_function=evaluate_agent_count,
        max_score=5.0,
        weight=0.8
    ))
    
    return criteria


def analyze_swarm_performance(task="What are the best etfs and index funds for ai and tech?"):
    """
    Run a financial analysis swarm and evaluate its performance.
    
    Args:
        task: The query to send to the swarm
        
    Returns:
        dict: The evaluation results
    """
    print(f"Running financial analysis swarm with task: '{task}'")
    
    # Run the swarm
    swarm_output = run_single_swarm(task)
    
    # Get the evaluation criteria
    criteria_set = get_financial_swarm_criteria()
    
    # Create evaluator
    evaluator = Evaluator(criteria_set)
    
    # Add JSON reporter
    json_reporter = JSONReporter()
    evaluator.add_reporter(json_reporter)
    
    # Set metadata for the evaluation
    evaluator.set_metadata({
        "task": task,
        "swarm_type": "ConcurrentWorkflow",
        "agent_count": len(swarm_output.get("agents", [])),
        "execution_time": swarm_output.get("metadata", {}).get("execution_time", 0),
    })
    
    # Run evaluation
    summary = evaluator.evaluate(swarm_output)
    
    # Get the JSON output
    json_output = json_reporter.report(summary)
    
    return {
        "swarm_output": swarm_output,
        "evaluation": json.loads(json_output)
    }


if __name__ == "__main__":
    # Check API health
    health = run_health_check()
    print(f"API Health: {health}")
    
    # Run and evaluate the swarm with default task
    result = analyze_swarm_performance()
    
    # Display results
    print("\nSwarm Evaluation Results:")
    print(json.dumps(result["evaluation"], indent=2))
    
    # Save results to file
    with open("swarm_evaluation_results.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("\nFull results saved to swarm_evaluation_results.json")