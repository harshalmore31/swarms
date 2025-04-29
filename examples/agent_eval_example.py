import os
from dotenv import load_dotenv
from swarms import Agent
from swarms.structs.eval_cr import (
    Category, 
    CriteriaSet, 
    Metric, 
    Evaluator, 
    ConsoleReporter, 
    JSONReporter, 
    create_automated_metric
)
from typing import Tuple, Dict, Any, Optional, List
import re

# Load environment variables
load_dotenv()

def create_financial_advice_criteria() -> CriteriaSet:
    """Create a criteria set for evaluating financial advice agents."""
    criteria = CriteriaSet(
        name="Financial Advice Evaluation",
        description="Criteria for evaluating AI-generated financial advice"
    )
    
    # Automated metric: Check for table presence
    def check_table_presence(output: str) -> Tuple[float, str]:
        # Check for markdown table format (| --- |)
        has_table_header = bool(re.search(r'\|[\s\-]+\|', output))
        has_table_rows = len(re.findall(r'\|.*\|', output)) > 1
        
        if has_table_header and has_table_rows:
            return 5.0, "Properly formatted markdown table found"
        elif has_table_rows:
            return 3.0, "Table-like structure found but may not be properly formatted"
        else:
            return 0.0, "No table found in the output"
    
    criteria.add_metric(create_automated_metric(
        name="Table Format Check",
        description="Check if the response includes a properly formatted markdown table",
        category=Category.COMPLETENESS,
        evaluation_function=check_table_presence,
        max_score=5.0,
        weight=1.0
    ))
    
    # Automated metric: Investment diversity
    def check_investment_diversity(output: str) -> Tuple[float, str]:
        # Look for mentions of different investment types
        investment_types = ["ETF", "index fund", "stock", "bond", "mutual fund"]
        mentioned_types = []
        
        for inv_type in investment_types:
            if re.search(rf'\b{inv_type}s?\b', output, re.IGNORECASE):
                mentioned_types.append(inv_type)
        
        diversity_score = len(mentioned_types)
        notes = f"Found {diversity_score} different investment types: {', '.join(mentioned_types)}"
        
        # Score based on diversity (0-5)
        return min(diversity_score, 5.0), notes
    
    criteria.add_metric(create_automated_metric(
        name="Investment Diversity",
        description="Check for diversity of investment options presented",
        category=Category.COMPLETENESS,
        evaluation_function=check_investment_diversity,
        max_score=5.0,
        weight=1.2
    ))
    
    # Automated metric: AI focus check
    def check_ai_focus(output: str) -> Tuple[float, str]:
        # Look for mentions of AI-related investments
        ai_keywords = ["AI", "artificial intelligence", "machine learning", "ML", "deep learning",
                      "neural network", "NLP", "computer vision", "robotics"]
        
        ai_mentions = []
        for keyword in ai_keywords:
            if re.search(rf'\b{keyword}\b', output, re.IGNORECASE):
                ai_mentions.append(keyword)
        
        unique_mentions = len(set(ai_mentions))
        
        if unique_mentions >= 4:
            score = 5.0
        elif unique_mentions >= 2:
            score = 3.0
        elif unique_mentions >= 1:
            score = 1.0
        else:
            score = 0.0
            
        return score, f"Found {unique_mentions} unique AI-related terms: {', '.join(set(ai_mentions))}"
    
    criteria.add_metric(create_automated_metric(
        name="AI Focus Check",
        description="Check if the response focuses on AI-related investment opportunities",
        category=Category.RELEVANCE,
        evaluation_function=check_ai_focus,
        max_score=5.0,
        weight=1.5
    ))
    
    # Automated metric: Explanation quality
    def check_explanation_quality(output: str) -> Tuple[float, str]:
        # Look for explanatory text outside of tables
        # Remove table lines
        non_table_text = re.sub(r'\|.*\|', '', output)
        words = len(re.findall(r'\b\w+\b', non_table_text))
        
        if words >= 100:
            score = 5.0
            note = f"Comprehensive explanation with {words} words"
        elif words >= 50:
            score = 3.0
            note = f"Moderate explanation with {words} words"
        elif words >= 20:
            score = 1.0
            note = f"Minimal explanation with {words} words"
        else:
            score = 0.0
            note = f"Insufficient explanation with only {words} words"
            
        return score, note
    
    criteria.add_metric(create_automated_metric(
        name="Explanation Quality",
        description="Check the quality of explanations provided",
        category=Category.EXPLAINABILITY,
        evaluation_function=check_explanation_quality,
        max_score=5.0,
        weight=1.0
    ))
    
    # Automated metric: Budget adherence
    def check_budget_adherence(output: str) -> Tuple[float, str]:
        # Check if the response acknowledges the $40k budget
        has_budget_mention = bool(re.search(r'\$?40\s?k|\$?40,000|\$?40000', output, re.IGNORECASE))
        
        if has_budget_mention:
            return 5.0, "Budget constraint acknowledged"
        else:
            return 0.0, "No mention of the budget constraint"
    
    criteria.add_metric(create_automated_metric(
        name="Budget Adherence",
        description="Check if the agent acknowledges the budget constraint",
        category=Category.RELEVANCE,
        evaluation_function=check_budget_adherence,
        max_score=5.0,
        weight=1.0
    ))
    
    return criteria

def evaluate_agent_run(agent: Agent, prompt: str) -> Dict[str, Any]:
    """
    Run an agent with the given prompt and evaluate its output.
    
    Args:
        agent: The agent to run
        prompt: The prompt to provide to the agent
        
    Returns:
        Dict containing both the agent's output and evaluation results
    """
    print(f"\n🤖 Running agent: {agent.agent_name}...\n")
    
    # Run the agent
    output = agent.run(prompt)
    
    print("\n✅ Agent run complete. Evaluating output...\n")
    
    # Create evaluation criteria
    criteria = create_financial_advice_criteria()
    
    # Create evaluator with console and JSON reporters
    evaluator = Evaluator(criteria)
    evaluator.add_reporter(JSONReporter())
    
    # Run evaluation
    summary = evaluator.evaluate(output)
    
    return {
        "agent_name": agent.agent_name,
        "prompt": prompt,
        "output": output,
        "evaluation": summary
    }

def main():
    """Main function to demonstrate agent evaluation."""
    print("🚀 Starting Agent Evaluation Example")
    
    # Define a prompt for financial advice
    prompt = "Create a table of super high growth opportunities for AI. I have $40k to invest in ETFs, index funds, and more. Please create a table in markdown."
    
    # Initialize the agent
    agent = Agent(
        agent_name="Financial-Analysis-Agent",
        agent_description="Personal finance advisor agent",
        system_prompt="You are an expert financial advisor specializing in technology and AI investments. Provide clear, concise advice with specific investment recommendations.",
        max_loops=1,
        model_name="gpt-4o",  # Change to the model you have access to
        dynamic_temperature_enabled=True,
        user_name="swarms_corp",
        retry_attempts=3,
        context_length=8192,
        return_step_meta=True,  # Enable this to get metadata for evaluation
        output_type="str",
        auto_generate_prompt=False,
        max_tokens=4000,
        saved_state_path="agent_00.json",
        interactive=False,
        roles="director",
    )
    
    # Run the agent and evaluate its output
    result = evaluate_agent_run(agent, prompt)
    
    # Additional options: compare multiple agents
    # Compare different models
    compare_models = False
    if compare_models:
        models = ["gpt-4o", "gpt-3.5-turbo", "claude-3-opus-20240229"]
        results = {}
        
        for model in models:
            agent.model_name = model
            results[model] = evaluate_agent_run(agent, prompt)
            
        # Compare results
        print("\n📊 Model Comparison Results:")
        for model, result in results.items():
            overall_score = result["evaluation"].get_overall_score()
            print(f"- {model}: {overall_score:.2%}")

if __name__ == "__main__":
    main()
