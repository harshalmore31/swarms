import time
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import json
from pathlib import Path

from swarms.structs.devh.eval_cr import (
    Category,
    CriteriaSet,
    Evaluator,
    ConsoleReporter,
    JSONReporter,
    EvaluationSummary,
    create_automated_metric,
    create_human_evaluation_metric
)

from swarms.agents.base import BaseAgent


class AgentEvaluator:
    """
    Evaluates agent performance against defined criteria.
    Can be used to automatically evaluate agent output quality.
    """
    
    def __init__(
        self, 
        criteria_set: CriteriaSet = None,
        save_results: bool = True,
        results_dir: str = "evaluation_results"
    ):
        """
        Initialize the agent evaluator.
        
        Args:
            criteria_set: Criteria to use for evaluation (if None, default criteria will be used)
            save_results: Whether to save evaluation results to disk
            results_dir: Directory to save evaluation results
        """
        self.criteria_set = criteria_set or self._get_default_criteria()
        self.save_results = save_results
        self.results_dir = Path(results_dir)
        
        if save_results:
            self.results_dir.mkdir(exist_ok=True, parents=True)
        
        self.evaluator = Evaluator(self.criteria_set)
        self.evaluator.add_reporter(JSONReporter())
    
    def _get_default_criteria(self) -> CriteriaSet:
        """Create a default criteria set for agent evaluation."""
        criteria = CriteriaSet(
            name="Agent Response Evaluation",
            description="Default criteria for evaluating agent responses"
        )
        
        # Completeness - checks if the agent addressed the task fully
        def check_completeness(output: Any) -> Tuple[float, str]:
            # Simple heuristic: longer responses tend to be more complete
            output_str = str(output)
            word_count = len(output_str.split())
            
            if word_count > 500:
                return 5.0, f"Comprehensive response with {word_count} words"
            elif word_count > 250:
                return 4.0, f"Substantial response with {word_count} words"
            elif word_count > 100:
                return 3.0, f"Moderate response with {word_count} words"
            elif word_count > 50:
                return 2.0, f"Brief response with {word_count} words"
            else:
                return 1.0, f"Minimal response with only {word_count} words"
        
        criteria.add_metric(create_automated_metric(
            name="Response Completeness",
            description="Checks if the agent's response fully addresses the task",
            category=Category.COMPLETENESS,
            evaluation_function=check_completeness,
            max_score=5.0,
            weight=1.0
        ))
        
        # Coherence - checks if the response is structured and readable
        def check_coherence(output: Any) -> Tuple[float, str]:
            output_str = str(output)
            
            # Check for paragraph breaks
            has_paragraphs = "\n\n" in output_str
            
            # Check for structured elements
            has_list = bool(any(line.strip().startswith(("- ", "* ", "1. ")) for line in output_str.split("\n")))
            has_headers = bool(any(line.strip().startswith("#") for line in output_str.split("\n")))
            has_sections = has_headers or output_str.count("\n\n") > 2
            
            # Score based on structure
            coherence_score = 1.0
            notes = []
            
            if has_paragraphs:
                coherence_score += 1.0
                notes.append("uses paragraphs")
            
            if has_list:
                coherence_score += 1.0
                notes.append("includes lists")
                
            if has_headers:
                coherence_score += 1.0
                notes.append("includes headers")
            
            if has_sections:
                coherence_score += 1.0
                notes.append("organized into sections")
                
            return min(coherence_score, 5.0), f"Structure and readability: {', '.join(notes)}"
        
        criteria.add_metric(create_automated_metric(
            name="Response Coherence",
            description="Checks if the agent's response is well-structured and readable",
            category=Category.COHERENCE,
            evaluation_function=check_coherence,
            max_score=5.0,
            weight=1.0
        ))
        
        # Correctness - a placeholder for domain-specific correctness
        criteria.add_metric(create_human_evaluation_metric(
            name="Factual Correctness",
            description="Evaluate if the information provided is factually correct",
            category=Category.CORRECTNESS,
            max_score=5.0,
            weight=1.5
        ))
        
        return criteria
    
    def evaluate_run(
        self, 
        agent: BaseAgent, 
        prompt: str, 
        expected_output: Any = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run an agent with the given prompt and evaluate its output.
        
        Args:
            agent: The agent to run
            prompt: The prompt to provide to the agent
            expected_output: Optional expected output for automatic comparison
            metadata: Additional metadata to include with results
            
        Returns:
            Dict containing both the agent's output and evaluation results
        """
        # Run the agent
        start_time = time.time()
        output = agent.run(prompt)
        execution_time = time.time() - start_time
        
        # Create metadata
        run_metadata = {
            "timestamp": time.time(),
            "execution_time": execution_time,
            "agent_name": getattr(agent, "agent_name", str(agent.__class__.__name__)),
            "model_name": getattr(agent, "model_name", "unknown")
        }
        
        if metadata:
            run_metadata.update(metadata)
        
        if expected_output is not None:
            run_metadata["has_expected_output"] = True
            
            # Add automated comparison metric for expected output
            def compare_to_expected(actual_output: Any) -> Tuple[float, str]:
                try:
                    # For string matching, check similarity
                    if isinstance(expected_output, str) and isinstance(actual_output, str):
                        # Simple string similarity - can be improved with more sophisticated methods
                        expected_words = set(expected_output.lower().split())
                        actual_words = set(str(actual_output).lower().split())
                        
                        if len(expected_words) == 0:
                            return 0.0, "Expected output is empty"
                        
                        common_words = expected_words.intersection(actual_words)
                        similarity = len(common_words) / len(expected_words)
                        
                        # Scale to 0-5 score
                        score = similarity * 5.0
                        return score, f"Output similarity: {similarity:.2%}"
                    
                    # For dict/json matching, check key presence and value equality
                    elif isinstance(expected_output, dict) and isinstance(actual_output, dict):
                        total_keys = len(expected_output)
                        if total_keys == 0:
                            return 0.0, "Expected output is empty"
                            
                        matching_keys = 0
                        for key, value in expected_output.items():
                            if key in actual_output and actual_output[key] == value:
                                matching_keys += 1
                                
                        similarity = matching_keys / total_keys
                        score = similarity * 5.0
                        return score, f"Output match: {matching_keys}/{total_keys} keys match"
                    
                    # Default case
                    equality = expected_output == actual_output
                    return 5.0 if equality else 0.0, "Exact match" if equality else "No match"
                    
                except Exception as e:
                    return 0.0, f"Error comparing outputs: {str(e)}"
            
            # Add comparison metric temporarily for this evaluation
            comparison_metric = create_automated_metric(
                name="Expected Output Match",
                description="Compares agent output to expected output",
                category=Category.CORRECTNESS,
                evaluation_function=compare_to_expected,
                max_score=5.0,
                weight=2.0
            )
            
            self.criteria_set.add_metric(comparison_metric)
            
        # Evaluate the output
        summary = self.evaluator.evaluate(output)
        
        # Remove temporary metric if added
        if expected_output is not None:
            self.criteria_set.remove_metric("Expected Output Match")
        
        # Create result dictionary
        results = {
            "prompt": prompt,
            "output": output,
            "overall_score": summary.get_overall_score(),
            "metadata": run_metadata
        }
        
        # Save results if enabled
        if self.save_results:
            timestamp = int(time.time())
            agent_name = run_metadata["agent_name"].replace(" ", "_")
            results_file = self.results_dir / f"{agent_name}_eval_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                # Convert output to string if it's not serializable
                serializable_results = results.copy()
                if not isinstance(output, (str, int, float, bool, list, dict, type(None))):
                    serializable_results["output"] = str(output)
                
                json.dump(serializable_results, f, indent=2)
        
        return results
    
    def batch_evaluate(
        self, 
        agents: List[BaseAgent],
        prompts: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple agents on multiple prompts.
        
        Args:
            agents: List of agents to evaluate
            prompts: List of prompts to test
            metadata: Additional metadata to include with results
            
        Returns:
            Dictionary mapping agent names to their evaluation results
        """
        results = {}
        
        for agent in agents:
            agent_name = getattr(agent, "agent_name", str(agent.__class__.__name__))
            agent_results = {}
            
            for i, prompt in enumerate(prompts):
                prompt_name = f"prompt_{i+1}"
                agent_results[prompt_name] = self.evaluate_run(
                    agent=agent,
                    prompt=prompt,
                    metadata=metadata
                )
            
            results[agent_name] = agent_results
        
        return results


def evaluate_agent_performance(
    agent: BaseAgent, 
    prompts: List[str],
    criteria_set: Optional[CriteriaSet] = None,
    save_results: bool = True,
    results_dir: str = "evaluation_results",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate an agent's performance across multiple prompts.
    
    Args:
        agent: The agent to evaluate
        prompts: List of prompts to test
        criteria_set: Custom evaluation criteria (None = use default)
        save_results: Whether to save evaluation results
        results_dir: Directory to save results
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with evaluation results
    """
    # Initialize evaluator
    evaluator = AgentEvaluator(
        criteria_set=criteria_set,
        save_results=save_results,
        results_dir=results_dir
    )
    
    if verbose:
        print(f"🧪 Evaluating agent: {getattr(agent, 'agent_name', type(agent).__name__)}")
        print(f"📝 Number of test prompts: {len(prompts)}")
    
    # Run batch evaluation
    results = evaluator.batch_evaluate(
        agents=[agent],
        prompts=prompts
    )
    
    # Extract the single agent's results
    agent_name = next(iter(results.keys()))
    agent_results = results[agent_name]
    
    # Calculate aggregate metrics
    overall_scores = [result["overall_score"] for result in agent_results.values()]
    avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    execution_times = [result["metadata"]["execution_time"] for result in agent_results.values()]
    avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
    
    # Create summary
    summary = {
        "agent_name": agent_name,
        "num_prompts": len(prompts),
        "average_score": avg_score,
        "average_execution_time": avg_time,
        "score_by_prompt": {f"prompt_{i+1}": score for i, score in enumerate(overall_scores)},
        "detailed_results": agent_results
    }
    
    if verbose:
        print(f"\n✅ Evaluation complete:")
        print(f"   - Average score: {avg_score:.2%}")
        print(f"   - Average execution time: {avg_time:.2f}s")
    
    return summary


def compare_agents(
    agents: List[BaseAgent],
    prompts: List[str],
    criteria_set: Optional[CriteriaSet] = None,
    save_results: bool = True,
    results_dir: str = "evaluation_results",
    generate_report: bool = True,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple agents across the same set of prompts.
    
    Args:
        agents: List of agents to compare
        prompts: List of prompts to test with
        criteria_set: Custom evaluation criteria (None = use default)
        save_results: Whether to save evaluation results
        results_dir: Directory to save results
        generate_report: Whether to generate comparative visualization
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with comparative evaluation results
    """
    # Initialize evaluator
    evaluator = AgentEvaluator(
        criteria_set=criteria_set,
        save_results=save_results,
        results_dir=results_dir
    )
    
    if verbose:
        print(f"🔍 Comparing {len(agents)} agents across {len(prompts)} prompts")
    
    # Run batch evaluation
    results = evaluator.batch_evaluate(
        agents=agents,
        prompts=prompts
    )
    
    # Calculate summary statistics
    summary = {}
    for agent_name, agent_results in results.items():
        overall_scores = [result["overall_score"] for result in agent_results.values()]
        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
        execution_times = [result["metadata"]["execution_time"] for result in agent_results.values()]
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        summary[agent_name] = {
            "average_score": avg_score,
            "average_execution_time": avg_time,
            "scores_by_prompt": {
                f"prompt_{i+1}": result["overall_score"] 
                for i, (_, result) in enumerate(agent_results.items())
            }
        }
    
    # Generate comparison report if requested
    if generate_report:
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create directory if it doesn't exist
            report_dir = Path(results_dir) / "reports"
            report_dir.mkdir(exist_ok=True, parents=True)
            
            # Create DataFrame for visualization
            rows = []
            for agent_name, agent_summary in summary.items():
                for prompt_id, score in agent_summary["scores_by_prompt"].items():
                    rows.append({
                        "Agent": agent_name,
                        "Prompt": prompt_id,
                        "Score": score,
                        "Execution Time": results[agent_name][prompt_id]["metadata"]["execution_time"]
                    })
            
            df = pd.DataFrame(rows)
            
            # Create visualizations
            plt.figure(figsize=(12, 10))
            
            # 1. Overall score comparison
            plt.subplot(2, 1, 1)
            sns.barplot(x="Agent", y="Score", data=df, palette="viridis")
            plt.title("Average Evaluation Score by Agent")
            plt.ylim(0, 1)
            
            # 2. Score heatmap by prompt
            plt.subplot(2, 1, 2)
            pivot_df = df.pivot(index="Agent", columns="Prompt", values="Score")
            sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
            plt.title("Evaluation Scores by Agent and Prompt")
            
            plt.tight_layout()
            
            # Save visualization
            timestamp = int(time.time())
            report_file = report_dir / f"agent_comparison_{timestamp}.png"
            plt.savefig(report_file)
            
            if verbose:
                print(f"\n📊 Comparison report saved to: {report_file}")
        
        except ImportError:
            if verbose:
                print("\n⚠️ Could not generate visualization. Required packages: pandas, matplotlib, seaborn")
    
    if verbose:
        print("\n📋 Comparison Results:")
        for agent_name, agent_summary in summary.items():
            print(f"   - {agent_name}: Score={agent_summary['average_score']:.2%}, Time={agent_summary['average_execution_time']:.2f}s")
    
    return {
        "agent_summaries": summary,
        "detailed_results": results
    }


def get_domain_specific_criteria(domain: str) -> CriteriaSet:
    """
    Get domain-specific evaluation criteria.
    
    Args:
        domain: Domain name (e.g., "finance", "code", "creative", "qa")
        
    Returns:
        CriteriaSet for the specific domain
    """
    domain = domain.lower()
    
    if domain == "finance":
        return _get_finance_criteria()
    elif domain == "code" or domain == "programming":
        return _get_code_criteria()
    elif domain == "creative" or domain == "writing":
        return _get_creative_criteria()
    elif domain == "qa" or domain == "question_answering":
        return _get_qa_criteria()
    else:
        # Return default criteria if domain not recognized
        criteria = CriteriaSet(
            name=f"General {domain.capitalize()} Evaluation",
            description=f"General criteria for evaluating {domain} agents"
        )
        
        # Add basic metrics
        criteria.add_metric(create_human_evaluation_metric(
            name="Task Completion",
            description="How well did the agent complete the task?",
            category=Category.COMPLETENESS,
            max_score=5.0,
            weight=1.0
        ))
        
        criteria.add_metric(create_human_evaluation_metric(
            name="Domain Expertise",
            description=f"How much expertise in {domain} does the agent demonstrate?",
            category=Category.CORRECTNESS,
            max_score=5.0,
            weight=1.0
        ))
        
        return criteria


def _get_finance_criteria() -> CriteriaSet:
    """Get criteria for evaluating finance-related agents."""
    criteria = CriteriaSet(
        name="Financial Advice Evaluation",
        description="Criteria for evaluating financial advice agents"
    )
    
    # Add finance-specific metrics
    def check_financial_jargon(output: str) -> Tuple[float, str]:
        # List of common financial terms
        financial_terms = [
            "portfolio", "diversification", "asset allocation", "ETF", "index fund",
            "mutual fund", "stock", "bond", "equity", "debt", "return", "yield",
            "dividend", "capital gain", "volatility", "risk", "investment"
        ]
        
        # Count occurrences
        term_count = 0
        for term in financial_terms:
            if term.lower() in output.lower():
                term_count += 1
        
        # Score based on term usage
        if term_count >= 8:
            score = 5.0
        elif term_count >= 5:
            score = 4.0
        elif term_count >= 3:
            score = 3.0
        elif term_count >= 1:
            score = 2.0
        else:
            score = 1.0
        
        return score, f"Used {term_count} financial terms"
    
    criteria.add_metric(create_automated_metric(
        name="Financial Terminology",
        description="Check if appropriate financial terminology is used",
        category=Category.CORRECTNESS,
        evaluation_function=check_financial_jargon,
        max_score=5.0,
        weight=1.0
    ))
    
    criteria.add_metric(create_human_evaluation_metric(
        name="Risk Assessment",
        description="Does the advice include appropriate risk assessment?",
        category=Category.CORRECTNESS,
        max_score=5.0,
        weight=1.2
    ))
    
    criteria.add_metric(create_human_evaluation_metric(
        name="Diversification Advice",
        description="Does the advice recommend appropriate diversification?",
        category=Category.CORRECTNESS,
        max_score=5.0,
        weight=1.2
    ))
    
    return criteria


def _get_code_criteria() -> CriteriaSet:
    """Get criteria for evaluating code-related agents."""
    criteria = CriteriaSet(
        name="Code Generation Evaluation",
        description="Criteria for evaluating code generation agents"
    )
    
    # Check for code blocks
    def check_code_blocks(output: str) -> Tuple[float, str]:
        # Look for markdown code blocks and other code indicators
        code_block_pattern = r'```[a-zA-Z]*\n[\s\S]*?\n```'
        inline_code_pattern = r'`[^`]+`'
        
        code_blocks = len(re.findall(code_block_pattern, output))
        inline_code = len(re.findall(inline_code_pattern, output))
        
        if code_blocks >= 1:
            return 5.0, f"Found {code_blocks} code blocks and {inline_code} inline code segments"
        elif inline_code >= 3:
            return 3.0, f"Found {inline_code} inline code segments"
        else:
            return 1.0, "Minimal or no code found"
    
    criteria.add_metric(create_automated_metric(
        name="Code Presence",
        description="Check if the response contains code blocks",
        category=Category.COMPLETENESS,
        evaluation_function=check_code_blocks,
        max_score=5.0,
        weight=1.0
    ))
    
    # Add code-specific metrics
    criteria.add_metric(create_human_evaluation_metric(
        name="Code Correctness",
        description="Is the generated code correct and functional?",
        category=Category.CORRECTNESS,
        max_score=5.0,
        weight=1.5
    ))
    
    criteria.add_metric(create_human_evaluation_metric(
        name="Code Quality",
        description="Is the code well-structured, readable, and following best practices?",
        category=Category.COHERENCE,
        max_score=5.0,
        weight=1.2
    ))
    
    criteria.add_metric(create_human_evaluation_metric(
        name="Code Efficiency",
        description="Is the code efficient in terms of time and space complexity?",
        category=Category.EFFICIENCY,
        max_score=5.0,
        weight=1.0
    ))
    
    return criteria


def _get_creative_criteria() -> CriteriaSet:
    """Get criteria for evaluating creative writing agents."""
    criteria = CriteriaSet(
        name="Creative Writing Evaluation",
        description="Criteria for evaluating creative writing agents"
    )
    
    # Check for creative language
    def check_creative_language(output: str) -> Tuple[float, str]:
        # Simple heuristics for creative language
        sentences = output.split(".")
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Look for descriptive words (adjectives often end with these suffixes)
        descriptive_patterns = r'\w+(ful|ous|ive|cal|ent|ant|ing)\b'
        descriptive_count = len(re.findall(descriptive_patterns, output, re.IGNORECASE))
        
        # Calculate creativity score
        creativity_score = 0.0
        notes = []
        
        # Varied sentence length is good for creative writing
        if 10 <= avg_sentence_length <= 20:
            creativity_score += 2.0
            notes.append("good sentence variety")
        
        # Check for descriptive language
        if descriptive_count >= 10:
            creativity_score += 2.0
            notes.append(f"rich descriptive language ({descriptive_count} descriptive words)")
        elif descriptive_count >= 5:
            creativity_score += 1.0
            notes.append(f"some descriptive language ({descriptive_count} descriptive words)")
        
        # Check for dialogue (simple heuristic: look for quotes)
        quotes_count = output.count('"')
        if quotes_count >= 4:  # At least two pairs of quotes
            creativity_score += 1.0
            notes.append("includes dialogue")
        
        return min(creativity_score, 5.0), ", ".join(notes)
    
    criteria.add_metric(create_automated_metric(
        name="Creative Language",
        description="Check for creative and descriptive language",
        category=Category.NOVELTY,
        evaluation_function=check_creative_language,
        max_score=5.0,
        weight=1.2
    ))
    
    # Add creative writing specific metrics
    criteria.add_metric(create_human_evaluation_metric(
        name="Originality",
        description="How original and unique is the content?",
        category=Category.NOVELTY,
        max_score=5.0,
        weight=1.5
    ))
    
    criteria.add_metric(create_human_evaluation_metric(
        name="Narrative Flow",
        description="How well does the narrative flow and engage the reader?",
        category=Category.COHERENCE,
        max_score=5.0,
        weight=1.3
    ))
    
    criteria.add_metric(create_human_evaluation_metric(
        name="Character Development",
        description="How well are characters developed (if applicable)?",
        category=Category.COMPLETENESS,
        max_score=5.0,
        weight=1.0
    ))
    
    return criteria


def _get_qa_criteria() -> CriteriaSet:
    """Get criteria for evaluating question answering agents."""
    criteria = CriteriaSet(
        name="Question Answering Evaluation",
        description="Criteria for evaluating question answering agents"
    )
    
    # Check if the answer directly addresses the question
    def check_direct_answer(output: str, question: str = None) -> Tuple[float, str]:
        # This is a simplistic check - in practice, you'd want to pass in the question
        if question is None:
            return 3.0, "Cannot determine directness without original question"
        
        # Get key terms from the question (excluding stop words)
        stop_words = {"what", "where", "when", "why", "how", "is", "are", "was", "were", "do", "does", 
                     "did", "a", "an", "the", "in", "on", "at", "to", "for", "with", "by", "about"}
        
        question_terms = set(word.lower() for word in question.replace("?", "").split() if word.lower() not in stop_words)
        
        # Count how many key terms from the question appear in the first two sentences
        sentences = output.split(".")
        first_two = ".".join(sentences[:2]) if len(sentences) >= 2 else output
        
        matched_terms = sum(1 for term in question_terms if term in first_two.lower())
        match_ratio = matched_terms / len(question_terms) if question_terms else 0
        
        # Score based on how directly the answer begins
        if match_ratio >= 0.8:
            return 5.0, "Answer directly addresses the question"
        elif match_ratio >= 0.5:
            return 4.0, "Answer mostly addresses the question"
        elif match_ratio >= 0.3:
            return 3.0, "Answer somewhat addresses the question"
        elif match_ratio > 0:
            return 2.0, "Answer tangentially addresses the question"
        else:
            return 1.0, "Answer does not appear to address the question"
    
    criteria.add_metric(create_automated_metric(
        name="Direct Answer",
        description="Check if the answer directly addresses the question",
        category=Category.RELEVANCE,
        evaluation_function=check_direct_answer,
        max_score=5.0,
        weight=1.5
    ))
    
    # Add QA-specific metrics
    criteria.add_metric(create_human_evaluation_metric(
        name="Answer Accuracy",
        description="Is the provided answer factually correct?",
        category=Category.CORRECTNESS,
        max_score=5.0,
        weight=2.0
    ))
    
    criteria.add_metric(create_human_evaluation_metric(
        name="Answer Completeness",
        description="Does the answer cover all aspects of the question?",
        category=Category.COMPLETENESS,
        max_score=5.0,
        weight=1.2
    ))
    
    criteria.add_metric(create_human_evaluation_metric(
        name="Evidence/Citations",
        description="Does the answer provide evidence or citations for its claims?",
        category=Category.EXPLAINABILITY,
        max_score=5.0,
        weight=1.0
    ))
    
    return criteria


# Create decorator for evaluating agent functions
def evaluate_agent(criteria_set: Optional[CriteriaSet] = None):
    """
    Decorator to evaluate an agent's output against criteria.
    
    Args:
        criteria_set: The criteria to use for evaluation
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Run the original function
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Set up evaluator
            nonlocal criteria_set
            if criteria_set is None:
                # Try to infer domain from function name
                func_name = func.__name__.lower()
                if any(domain in func_name for domain in ["finance", "money", "invest"]):
                    criteria_set = _get_finance_criteria()
                elif any(domain in func_name for domain in ["code", "program", "develop"]):
                    criteria_set = _get_code_criteria()
                elif any domain in func_name for domain in ["creative", "write", "story"]):
                    criteria_set = _get_creative_criteria()
                elif any domain in func_name for domain in ["qa", "question"]:
                    criteria_set = _get_qa_criteria()
                else:
                    criteria_set = _get_default_criteria()
            
            # Evaluate the result
            evaluator = Evaluator(criteria_set)
            summary = evaluator.evaluate(result)
            
            # Print evaluation summary
            print(f"\nEvaluation Summary for {func.__name__}:")
            print(f"Overall Score: {summary.get_overall_score():.2f}")
            for metric in summary.metrics:
                print(f" - {metric.name}: {metric.score:.2f} ({metric.description})")
            
            return result
        
        return wrapper
    
    return decorator