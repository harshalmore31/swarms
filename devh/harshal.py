"""
EnhancedAutonomousSwarmAgent: A multi-agent system for fully autonomous task execution.

Building on the original AutonomousSwarmAgent but adding:
- Multi-model LLM integration (ChatGPT, Claude, Gemini)
- Browser automation and OCR capabilities
- Enhanced parallel execution with ReAct methodology
- Improved synthesis and minimal human intervention
"""

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Core dependencies
from swarms import Agent
from swarms.structs.conversation import Conversation

# For browser automation and OCR
import pytesseract
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Setup logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("swarm_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Defines the roles for agents within the system."""
    TASK_HANDLER = "task_handler"
    DECOMPOSER = "decomposer"
    REASONER = "reasoner"
    PARALLEL_EXECUTOR = "parallel_executor"
    SYNTHESIZER = "synthesizer"
    DATA_GATHERER = "data_gatherer"
    OCR_PROCESSOR = "ocr_processor"


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"  # ChatGPT
    ANTHROPIC = "anthropic"  # Claude
    GOOGLE = "google"  # Gemini


class ExecutionApproach(Enum):
    """Represents different execution strategies for parallel processing."""
    STANDARD_APPROACH = "standard_approach"  # Default/balanced approach
    DEEP_REASONING = "deep_reasoning"  # Emphasizes thorough reasoning
    RAPID_PROTOTYPE = "rapid_prototype"  # Emphasizes quick implementation


class EnhancedAutonomousSwarmAgent:
    """
    An enhanced autonomous multi-agent system for task execution with minimal human intervention.

    Attributes:
        default_model: Base model name for initial operations
        max_iterations: Maximum iterations for the ReAct loop
        base_path: Directory for saving agent states
        api_keys: Dictionary of API keys for different LLM providers
        browser_enabled: Whether to enable browser automation
        ocr_enabled: Whether to enable OCR processing
        human_feedback: Whether to allow minimal human feedback
    """

    def __init__(
        self,
        default_model: str = "gemini/gemini-2.0-flash",
        max_iterations: int = 5,
        base_path: Optional[str] = None,
        api_keys: Optional[Dict[str, str]] = None,
        browser_enabled: bool = True,
        ocr_enabled: bool = True,
        human_feedback: bool = True,
    ):
        """Initializes the EnhancedAutonomousSwarmAgent system."""
        self.default_model = default_model
        self.max_iterations = max_iterations
        self.base_path = Path(base_path) if base_path else Path("./autonomous_agent_states")
        self.base_path.mkdir(exist_ok=True)
        
        # Setup API keys
        self.api_keys = api_keys or {}
        self._setup_apis()
        
        # Feature flags
        self.browser_enabled = browser_enabled
        self.ocr_enabled = ocr_enabled
        self.human_feedback = human_feedback
        
        # Initialize components
        self._init_agents()
        self._init_browser() if self.browser_enabled else None
        
        # Conversation and execution tracking
        self.conversation = Conversation()
        self.current_llm_provider = LLMProvider.OPENAI  # Start with OpenAI (ChatGPT)
        self.execution_history = []
        
    def _setup_apis(self):
        """Setup API clients based on provided keys."""
        # OpenAI (ChatGPT) setup
        if "openai" in self.api_keys:
            openai.api_key = self.api_keys["openai"]
            logger.info("OpenAI API configured.")
        else:
            logger.warning("OpenAI API key not provided. Using environment variable if available.")
            
        # Anthropic (Claude) setup
        if "anthropic" in self.api_keys:
            self.anthropic_client = Anthropic(api_key=self.api_keys["anthropic"])
            logger.info("Anthropic API configured.")
        else:
            logger.warning("Anthropic API key not provided. Using environment variable if available.")
            self.anthropic_client = Anthropic()  # Will use ANTHROPIC_API_KEY env var
            
        # Google (Gemini) setup
        if "google" in self.api_keys:
            genai.configure(api_key=self.api_keys["google"])
            logger.info("Google Generative AI API configured.")
        else:
            logger.warning("Google API key not provided. Will attempt to use environment variable.")

    def _init_browser(self):
        """Initialize browser automation with Selenium."""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            self.browser = webdriver.Chrome(options=chrome_options)
            logger.info("Browser automation initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize browser: {str(e)}")
            self.browser_enabled = False

    def _init_agents(self) -> None:
        """Initializes all agents with their roles and prompts."""
        # Task Handler Agent
        self.task_handler = Agent(
            agent_name="Task-Handler",
            system_prompt=self._get_task_handler_prompt(),
            model_name=self.default_model,
            max_loops=1,
            saved_state_path=str(self.base_path / "task_handler.json"),
            verbose=True,
        )

        # Task Decomposer Agent
        self.decomposer = Agent(
            agent_name="Task-Decomposer",
            system_prompt=self._get_decomposer_prompt(),
            model_name=self.default_model,
            max_loops=1,
            saved_state_path=str(self.base_path / "decomposer.json"),
            verbose=True,
        )

        # Reasoner Agent (with multi-model support)
        self.reasoner = Agent(
            agent_name="Task-Reasoner",
            system_prompt=self._get_reasoner_prompt(),
            model_name=self.default_model,
            max_loops=self.max_iterations,
            saved_state_path=str(self.base_path / "reasoner.json"),
            verbose=True,
        )

        # Parallel Executor Agent (enhanced with three approaches)
        self.parallel_executor = Agent(
            agent_name="Parallel-Executor",
            system_prompt=self._get_parallel_executor_prompt(),
            model_name=self.default_model,
            max_loops=1,
            saved_state_path=str(self.base_path / "parallel_executor.json"),
            verbose=True,
        )

        # Synthesizer Agent
        self.synthesizer = Agent(
            agent_name="Task-Synthesizer",
            system_prompt=self._get_synthesizer_prompt(),
            model_name=self.default_model,
            max_loops=1,
            saved_state_path=str(self.base_path / "synthesizer.json"),
            verbose=True,
        )
        
        # Data Gatherer Agent (for web scraping and data collection)
        self.data_gatherer = Agent(
            agent_name="Data-Gatherer",
            system_prompt=self._get_data_gatherer_prompt(),
            model_name=self.default_model,
            max_loops=2,  # Allow for retry
            saved_state_path=str(self.base_path / "data_gatherer.json"),
            verbose=True,
        )
        
        # OCR Processor Agent
        self.ocr_processor = Agent(
            agent_name="OCR-Processor",
            system_prompt=self._get_ocr_processor_prompt(),
            model_name=self.default_model,
            max_loops=1,
            saved_state_path=str(self.base_path / "ocr_processor.json"),
            verbose=True,
        )

    def _get_task_handler_prompt(self) -> str:
        """Prompt for the Task Handler agent."""
        return """You are the Task Handler. Your role is to receive user tasks,
        validate them, and prepare them for processing.

        1. Receive the user's task and any contextual data.
        2. Validate the input (check for completeness, feasibility, etc.).
        3. Determine if the task requires web search, OCR, or other data gathering.
        4. Log the task and prepare it for the next stage (decomposition).

        Output in JSON format:
        {
            "task": "The original task description",
            "validation_status": "valid" or "invalid",
            "data_gathering_needed": true/false,
            "data_types_required": ["web_search", "ocr", "code_analysis", etc.],
            "notes": "Any relevant notes or issues"
        }
        """

    def _get_decomposer_prompt(self) -> str:
        """Prompt for the Task Decomposer agent."""
        return """You are the Task Decomposer. Your role is to break down a complex
        task into smaller, manageable sub-tasks using advanced NLP techniques.

        1. Receive a task (potentially validated by the Task Handler).
        2. Decompose the task into discrete units that can be processed independently.
        3. Prioritize the sub-tasks and determine a logical sequence.
        4. Identify dependencies between sub-tasks.
        5. Suggest the most appropriate LLM for each sub-task (gpt-4, claude, gemini).

        Output in JSON format:
        {
            "original_task": "The original task description",
            "sub_tasks": [
                {
                    "id": "subtask_1",
                    "description": "Description of the sub-task",
                    "priority": "high|medium|low",
                    "dependencies": ["subtask_id_1", "subtask_id_2"],
                    "suggested_llm": "gpt-4|claude|gemini",
                    "requires_data_gathering": true/false,
                    "data_gathering_details": {
                        "type": "web_search|ocr|code_analysis",
                        "queries": ["query1", "query2"]
                    }
                },
                ...
            ]
        }
        """

    def _get_reasoner_prompt(self) -> str:
        """Prompt for the Reasoner agent (handles ReAct and multi-model)."""
        return """You are the Task Reasoner. Your role is to apply the ReAct methodology
        to analyze, plan, and solve tasks.

        1. Generate an initial plan (high-level blueprint).
        2. Iteratively refine the plan using the ReAct methodology:
           - **Reason:** Analyze collected data, task feasibility, and identify next steps.
           - **Act:** Perform actions (e.g., suggest data gathering, call other agents).
           - **Observe:** Interpret results and update your understanding.
        3. Manage multi-model collaboration:
           - ChatGPT (gpt-4): Initial planning and blueprint generation.
           - Claude: Enhanced reasoning and refinement of the blueprint.
           - Gemini: Detailed code generation and technical implementation.
        4. Document your thought process, including alternatives considered.

        Output in JSON format at each iteration:
        {
            "iteration": 1,
            "reasoning": "Your current analysis and thought process",
            "action": {
                "type": "data_gathering|generate_code|refine_approach",
                "details": { ... }
            },
            "suggested_model": "gpt-4|claude|gemini",
            "intermediate_result": { ... },
            "next_steps": ["step1", "step2"]
        }
        """

    def _get_parallel_executor_prompt(self) -> str:
        """Prompt for the Parallel Executor agent."""
        return """You are the Parallel Executor. Your role is to manage the
        execution of multiple approaches to solve a sub-task.

        1. Receive a sub-task, its context, and reasoning from the Reasoner.
        2. Spawn three independent processes, each using a different approach:
           - **Standard Approach**: Balanced between reasoning and implementation.
           - **Deep Reasoning Approach**: Emphasizes thorough analysis and planning.
           - **Rapid Prototype Approach**: Focuses on quick implementation and iteration.
        3. Each approach should use the ReAct methodology and may use different LLMs.
        4. Collect the outputs from each approach and evaluate their effectiveness.

        Output in JSON format:
        {
            "sub_task_id": "subtask_1",
            "approaches": {
                "standard_approach": {
                    "model_used": "gpt-4|claude|gemini",
                    "reasoning": "...",
                    "implementation": "...",
                    "result": "..."
                },
                "deep_reasoning_approach": {
                    "model_used": "gpt-4|claude|gemini",
                    "reasoning": "...",
                    "implementation": "...",
                    "result": "..."
                },
                "rapid_prototype_approach": {
                    "model_used": "gpt-4|claude|gemini",
                    "reasoning": "...",
                    "implementation": "...",
                    "result": "..."
                }
            },
            "evaluation": {
                "best_approach": "standard_approach|deep_reasoning_approach|rapid_prototype_approach",
                "reasoning": "Why this approach was selected as best"
            }
        }
        """

    def _get_synthesizer_prompt(self) -> str:
        """Prompt for the Synthesizer agent."""
        return """You are the Task Synthesizer. Your role is to consolidate
        the outputs from various stages and agents into a unified solution.

        1. Receive the outputs from the parallel executors, reasoners, and other agents.
        2. Combine the results, along with logs and metadata.
        3. Create a coherent final solution that addresses the original task.
        4. Structure the final output in JSON format with clear code blocks.
        5. If human feedback is enabled, prepare options for human review.

        Output in JSON format:
        {
            "original_task": "...",
            "final_solution": {
                "summary": "A concise summary of the solution",
                "recommended_approach": "standard_approach|deep_reasoning_approach|rapid_prototype_approach",
                "code_blocks": [
                    {
                        "purpose": "...",
                        "language": "python|javascript|etc.",
                        "code": "...",
                        "explanation": "..."
                    },
                    ...
                ],
                "implementation_steps": ["step1", "step2", ...]
            },
            "alternative_solutions": [
                {
                    "approach": "...",
                    "summary": "...",
                    "code_blocks": [...]
                },
                ...
            ],
            "decision_points": [
                {
                    "question": "Decision point question",
                    "options": ["option1", "option2"],
                    "recommendation": "option1",
                    "reasoning": "Why this option is recommended"
                },
                ...
            ],
            "logs": [...],
            "metadata": {...}
        }
        """

    def _get_data_gatherer_prompt(self) -> str:
        """Prompt for the Data Gatherer agent."""
        return """You are the Data Gatherer. Your role is to collect
        relevant information from external sources.
        
        1. Receive a data gathering request with specific queries or targets.
        2. Determine the most appropriate method for gathering the data:
           - Web search: For retrieving information from websites.
           - Code repository analysis: For understanding code structure and patterns.
           - Documentation search: For finding relevant API or library information.
        3. Format the gathered data in a structured way for other agents to use.
        
        Output in JSON format:
        {
            "request": "Original data gathering request",
            "gathered_data": [
                {
                    "source": "Where this data was found",
                    "content": "The actual data content",
                    "relevance_score": 0-10,
                    "timestamp": "When this was gathered"
                },
                ...
            ],
            "screenshot_paths": ["path1.png", "path2.png"],  # If screenshots were taken
            "summary": "A concise summary of the gathered information"
        }
        """

    def _get_ocr_processor_prompt(self) -> str:
        """Prompt for the OCR Processor agent."""
        return """You are the OCR Processor. Your role is to extract text from images
        and make it usable for other agents.
        
        1. Receive image paths for processing.
        2. Extract text using OCR technology.
        3. Clean, structure, and format the extracted text.
        4. Identify code blocks, tables, and other structured content.
        
        Output in JSON format:
        {
            "image_path": "Path to the processed image",
            "extracted_text": "Full extracted text",
            "structured_content": {
                "code_blocks": [
                    {
                        "language": "python|javascript|etc.",
                        "code": "..."
                    },
                    ...
                ],
                "tables": [...],
                "lists": [...]
            },
            "confidence_score": 0-100,
            "processing_notes": "Any issues or special considerations"
        }
        """

    def _safely_parse_json(self, json_str: str) -> Dict[str, Any]:
        """Safely parses JSON strings, handling potential errors."""
        try:
            # First, try to find JSON content in the response (handling cases where there's additional text)
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}')
            
            if start_idx >= 0 and end_idx > start_idx:
                json_content = json_str[start_idx:end_idx+1]
                return json.loads(json_content)
            
            # If no JSON structure found, try parsing the whole string
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"JSONDecodeError: Invalid JSON: {json_str[:500]}...")  # Log truncated string
            
            # Attempt a basic extraction of key-value pairs
            extracted = {}
            try:
                # Look for patterns like "key": "value" or "key": value
                import re
                patterns = re.findall(r'"([^"]+)"\s*:\s*(?:"([^"]*)"|(true|false|null|\d+))', json_str)
                for match in patterns:
                    key, str_value, other_value = match
                    extracted[key] = str_value if str_value else other_value
            except Exception:
                pass
                
            return {"error": "Invalid JSON", "content": json_str, "extracted": extracted}

    async def _run_agent_async(
        self, agent: Agent, input_data: str, role: Optional[AgentRole] = None
    ) -> Dict[str, Any]:
        """Runs an agent asynchronously and handles the interaction."""
        try:
            # Update the model based on current provider if it's the reasoner
            if role == AgentRole.REASONER and agent.agent_name == "Task-Reasoner":
                agent.model_name = self._get_model_for_provider(self.current_llm_provider)
                logger.info(f"Using {agent.model_name} for reasoning step")
            
            # Execute the agent
            agent_response = agent.run(input_data)
            
            # Log the interaction
            self.conversation.add(role=agent.agent_name, content=agent_response)
            
            # Parse and return the response
            parsed_response = self._safely_parse_json(agent_response)
            
            # Check if we need to switch LLM providers based on the reasoner's suggestion
            if role == AgentRole.REASONER and "suggested_model" in parsed_response:
                suggested_model = parsed_response["suggested_model"]
                if "gpt" in suggested_model.lower():
                    self.current_llm_provider = LLMProvider.OPENAI
                elif "claude" in suggested_model.lower():
                    self.current_llm_provider = LLMProvider.ANTHROPIC
                elif "gemini" in suggested_model.lower():
                    self.current_llm_provider = LLMProvider.GOOGLE
            
            return parsed_response
        except Exception as e:
            logger.exception(f"Error running agent {agent.agent_name}: {e}")
            return {
                "error": f"Agent {agent.agent_name} failed: {str(e)}",
                "agent_name": agent.agent_name,
                "role": role.value if role else None,
            }

    def _get_model_for_provider(self, provider: LLMProvider) -> str:
        """Get the appropriate model string for the given provider."""
        if provider == LLMProvider.OPENAI:
            return "gpt-4"
        elif provider == LLMProvider.ANTHROPIC:
            return "claude-3-opus-20240229"  # Use the latest Claude model
        elif provider == LLMProvider.GOOGLE:
            return "gemini-1.5-pro"  # Use Gemini Pro
        else:
            return self.default_model

    async def _collect_data(self, queries: List[str], data_type: str) -> Dict[str, Any]:
        """Collects data using browser automation and/or OCR."""
        if not self.browser_enabled and data_type in ['web_search', 'screenshot']:
            logger.warning("Browser automation is disabled but web search was requested")
            return {"error": "Browser automation is disabled", "data": []}
        
        results = []
        screenshot_paths = []
        
        try:
            if data_type == "web_search":
                for query in queries:
                    # Perform web search using Google
                    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
                    self.browser.get(search_url)
                    time.sleep(2)  # Allow page to load
                    
                    # Extract search results
                    search_results = self.browser.find_elements(By.CSS_SELECTOR, "div.g")
                    for idx, result in enumerate(search_results[:3]):  # Get top 3 results
                        try:
                            title = result.find_element(By.CSS_SELECTOR, "h3").text
                            link = result.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                            snippet = result.find_element(By.CSS_SELECTOR, "div.VwiC3b").text
                            
                            results.append({
                                "source": link,
                                "title": title,
                                "content": snippet,
                                "relevance_score": 10 - idx,  # Simple relevance scoring
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                            # Take a screenshot of the result
                            screenshot_path = f"./screenshots/search_result_{len(screenshot_paths)}.png"
                            os.makedirs("./screenshots", exist_ok=True)
                            result.screenshot(screenshot_path)
                            screenshot_paths.append(screenshot_path)
                        except Exception as e:
                            logger.error(f"Error extracting search result: {str(e)}")
            
            elif data_type == "screenshot":
                for query in queries:
                    # Navigate to the URL
                    self.browser.get(query)  # In this case, query is a URL
                    time.sleep(3)  # Allow page to load
                    
                    # Take full page screenshot
                    screenshot_path = f"./screenshots/page_{len(screenshot_paths)}.png"
                    os.makedirs("./screenshots", exist_ok=True)
                    self.browser.save_screenshot(screenshot_path)
                    screenshot_paths.append(screenshot_path)
                    
                    results.append({
                        "source": query,
                        "content": f"Screenshot taken of {query}",
                        "relevance_score": 8,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # If OCR is enabled, process the screenshot
                    if self.ocr_enabled:
                        ocr_result = await self._process_ocr(screenshot_path)
                        if "error" not in ocr_result:
                            results.append({
                                "source": f"OCR from {query}",
                                "content": ocr_result["extracted_text"][:500] + "...",  # Truncate long text
                                "relevance_score": 9,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            })
        
        except Exception as e:
            logger.exception(f"Error in data collection: {str(e)}")
            return {
                "error": f"Data collection failed: {str(e)}",
                "data": results,
                "screenshot_paths": screenshot_paths
            }
        
        return {
            "gathered_data": results,
            "screenshot_paths": screenshot_paths,
            "summary": f"Collected {len(results)} data points and {len(screenshot_paths)} screenshots"
        }

    async def _process_ocr(self, image_path: str) -> Dict[str, Any]:
        """Process an image with OCR to extract text."""
        if not self.ocr_enabled:
            logger.warning("OCR processing is disabled but was requested")
            return {"error": "OCR processing is disabled"}
        
        try:
            # Load the image
            image = Image.open(image_path)
            
            # Extract text using Tesseract OCR
            extracted_text = pytesseract.image_to_string(image)
            
            # Basic detection of code blocks (simple heuristic)
            code_blocks = []
            current_block = []
            in_code_block = False
            
            for line in extracted_text.split('\n'):
                # Very simple heuristic: lines with special characters likely code
                code_indicators = ['{', '}', '()', '[];', '+=', '==', 'def ', 'function', 'class', 'import ', 'from ']
                line_is_code = any(indicator in line for indicator in code_indicators)
                
                if line_is_code and not in_code_block:
                    in_code_block = True
                    current_block = [line]
                elif in_code_block and (line_is_code or line.strip().startswith('    ')):
                    current_block.append(line)
                elif in_code_block:
                    in_code_block = False
                    if current_block:
                        language = self._detect_language('\n'.join(current_block))
                        code_blocks.append({
                            "language": language,
                            "code": '\n'.join(current_block)
                        })
                    current_block = []
            
            # Add the last block if there is one
            if in_code_block and current_block:
                language = self._detect_language('\n'.join(current_block))
                code_blocks.append({
                    "language": language,
                    "code": '\n'.join(current_block)
                })
            
            # Use OCR processor agent for more sophisticated processing
            ocr_input = json.dumps({
                "image_path": image_path,
                "raw_text": extracted_text,
                "preliminary_code_blocks": code_blocks
            })
            
            ocr_agent_response = await self._run_agent_async(
                self.ocr_processor, ocr_input, AgentRole.OCR_PROCESSOR
            )
            
            return ocr_agent_response
            
        except Exception as e:
            logger.exception(f"Error in OCR processing: {str(e)}")
            return {
                "error": f"OCR processing failed: {str(e)}",
                "image_path": image_path
            }

    def _detect_language(self, code_snippet: str) -> str:
        """Detect the programming language of a code snippet."""
        # Very simple heuristic-based detection
        if 'def ' in code_snippet and ':' in code_snippet:
            return "python"
        elif '{' in code_snippet and '}' in code_snippet and ';' in code_snippet:
            if 'function' in code_snippet or 'var' in code_snippet or 'const' in code_snippet:
                return "javascript"
            elif 'public' in code_snippet or 'class' in code_snippet:
                return "java"
            else:
                return "c-like"
        elif '<' in code_snippet and '>' in code_snippet and '</' in code_snippet:
            return "html"
        else:
            return "unknown"

    async def _execute_approaches(
        self, 
        sub_task: Dict[str, Any], 
        reasoning_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute three parallel approaches for a sub-task."""
        approaches = {}
        
        # Create executor input for each approach
        for approach in [
            ExecutionApproach.STANDARD_APPROACH,
            ExecutionApproach.DEEP_REASONING,
            ExecutionApproach.RAPID_PROTOTYPE
        ]:
            # Adjust LLM provider based on approach
            if approach == ExecutionApproach.DEEP_REASONING:
                self.current_llm_provider = LLMProvider.ANTHROPIC  # Use Claude for deep reasoning
            elif approach == ExecutionApproach.RAPID_PROTOTYPE:
                self.current_llm_provider = LLMProvider.GOOGLE  # Use Gemini for rapid prototyping
            else:
                self.current_llm_provider = LLMProvider.OPENAI  # Use ChatGPT as default
                
            # Create customized input for the approach
            executor_input = json.dumps({
                "sub_task": sub_task,
                "reasoning_context": reasoning_context,
                "approach": approach.value,
                "model": self._get_model_for_provider(self.current_llm_provider)
            })
                
            # Execute the approach
            logger.info(f"Executing {approach.value} for sub_task {sub_task.get('id', 'unknown')}")
            approach_result = await self._run_agent_async(
                self.parallel_executor,
                executor_input,
                AgentRole.PARALLEL_EXECUTOR
            )
                
            approaches[approach.value] = approach_result
        
        # Evaluate which approach performed best
        evaluation = {
            "best_approach": ExecutionApproach.STANDARD_APPROACH.value,  # Default
            "reasoning": "Standard approach selected as default."
        }
        
        # Simple evaluation logic (can be enhanced)
        approach_scores = {}
        for approach_name, approach_result in approaches.items():
            # Skip approaches with errors
            if "error" in approach_result:
                approach_scores[approach_name] = 0
                continue
                
            # Basic scoring (can be expanded with more sophisticated metrics)
            score = 0
            if "implementation" in approach_result and approach_result["implementation"]:
                score += 3  # Having an implementation is good
            if "reasoning" in approach_result and len(approach_result.get("reasoning", "")) > 100:
                score += 2  # Detailed reasoning is valued
            if "result" in approach_result and approach_result["result"]:
                score += 5  # Having a concrete result is best
                
            approach_scores[approach_name] = score
            
        # Find the best scoring approach
        if approach_scores:
            best_approach = max(approach_scores.items(), key=lambda x: x[1])[0]
            evaluation = {
                "best_approach": best_approach,
                "reasoning": f"Selected based on higher implementation quality and results (score: {approach_scores[best_approach]})."
            }
            
        return {
            "sub_task_id": sub_task.get("id", "unknown"),
            "approaches": approaches,
            "evaluation": evaluation
        }

    async def _process_subtask(self, sub_task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sub-task through the ReAct loop and parallel execution."""
        logger.info(f"Processing sub-task: {sub_task.get('id', 'unknown')}: {sub_task.get('description', '')}")
        
        # Check if data gathering is required
        if sub_task.get("requires_data_gathering", False):
            data_gathering_details = sub_task.get("data_gathering_details", {})
            data_type = data_gathering_details.get("type", "web_search")
            queries = data_gathering_details.get("queries", [])
            
            if queries:
                logger.info(f"Gathering data for sub-task {sub_task.get('id', 'unknown')}")
                gathered_data = await self._collect_data(queries, data_type)
                
                # Update the sub-task with the gathered data
                sub_task["gathered_data"] = gathered_data
        
        # Prepare input for the reasoner
        reasoner_input = json.dumps({
            "sub_task": sub_task,
            "iteration": 1,
            "previous_reasoning": None,
            "gathered_data": sub_task.get("gathered_data", {})
        })
        
        # Run the ReAct loop through the reasoner
        reasoning_context = await self._run_agent_async(
            self.reasoner, reasoner_input, AgentRole.REASONER
        )
        
        # Execute multiple approaches for the sub-task
        execution_results = await self._execute_approaches(sub_task, reasoning_context)
        
        # Combine reasoning and execution results
        result = {
            "sub_task": sub_task,
            "reasoning_context": reasoning_context,
            "execution_results": execution_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Record in execution history
        self.execution_history.append(result)
        
        return result

    async def _process_subtasks_in_order(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process sub-tasks in order, respecting dependencies."""
        results = []
        completed_subtasks = set()
        pending_subtasks = {subtask["id"]: subtask for subtask in subtasks}
        
        while pending_subtasks:
            # Find tasks that can be executed (dependencies satisfied)
            executable_tasks = []
            for task_id, task in list(pending_subtasks.items()):
                dependencies = set(task.get("dependencies", []))
                if dependencies.issubset(completed_subtasks):
                    executable_tasks.append(task)
                    del pending_subtasks[task_id]
            
            # If no tasks can be executed but there are pending tasks, there's a dependency cycle
            if not executable_tasks and pending_subtasks:
                logger.warning("Dependency cycle detected. Breaking cycle arbitrarily.")
                task_id, task = next(iter(pending_subtasks.items()))
                executable_tasks.append(task)
                del pending_subtasks[task_id]
            
            # Execute tasks in parallel that can be executed simultaneously
            execution_tasks = [self._process_subtask(task) for task in executable_tasks]
            task_results = await asyncio.gather(*execution_tasks)
            
            # Update completed tasks and collect results
            for task, result in zip(executable_tasks, task_results):
                completed_subtasks.add(task["id"])
                results.append(result)
                logger.info(f"Completed sub-task: {task['id']}")
                
        return results

    async def execute_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a task autonomously using the multi-agent system.
        
        Args:
            task_description: Description of the task to be performed
            context: Additional context or parameters for the task
            
        Returns:
            A dictionary containing the final solution and execution details
        """
        logger.info(f"Executing task: {task_description}")
        
        # Prepare task input with context
        task_input = {
            "task": task_description,
            "context": context or {}
        }
        
        # 1. Task Handling: Validate and prepare the task
        task_handler_input = json.dumps(task_input)
        task_handler_result = await self._run_agent_async(
            self.task_handler, task_handler_input, AgentRole.TASK_HANDLER
        )
        
        if task_handler_result.get("validation_status") == "invalid":
            logger.warning(f"Task validation failed: {task_handler_result.get('notes', '')}")
            return {
                "status": "failed",
                "reason": f"Task validation failed: {task_handler_result.get('notes', '')}",
                "task": task_description
            }
        
        # 2. Task Decomposition: Break down the task into sub-tasks
        decomposer_input = json.dumps(task_handler_result)
        decomposition_result = await self._run_agent_async(
            self.decomposer, decomposer_input, AgentRole.DECOMPOSER
        )
        
        subtasks = decomposition_result.get("sub_tasks", [])
        if not subtasks:
            logger.warning("Task decomposition failed: No sub-tasks generated")
            return {
                "status": "failed", 
                "reason": "Task decomposition failed: No sub-tasks generated",
                "task": task_description
            }
        
        # 3. Process all sub-tasks in dependency order
        subtask_results = await self._process_subtasks_in_order(subtasks)
        
        # 4. Synthesize results into a final solution
        synthesizer_input = json.dumps({
            "original_task": task_description,
            "task_handler_result": task_handler_result,
            "decomposition_result": decomposition_result,
            "subtask_results": subtask_results,
            "execution_history": self.execution_history,
            "human_feedback_enabled": self.human_feedback
        })
        
        synthesis_result = await self._run_agent_async(
            self.synthesizer, synthesizer_input, AgentRole.SYNTHESIZER
        )
        
        # 5. Handle human feedback if enabled and requested
        if self.human_feedback and "decision_points" in synthesis_result:
            synthesis_result = await self._collect_human_feedback(synthesis_result)
        
        # Prepare final result
        final_result = {
            "status": "success",
            "task": task_description,
            "solution": synthesis_result.get("final_solution", {}),
            "alternatives": synthesis_result.get("alternative_solutions", []),
            "execution_details": {
                "subtasks": len(subtasks),
                "execution_time": time.time(),  # Will be updated at the end
                "model_usage": {
                    "openai": 0,
                    "anthropic": 0,
                    "google": 0
                }
            }
        }
        
        # Update execution time
        final_result["execution_details"]["execution_time"] = time.time() - final_result["execution_details"]["execution_time"]
        
        logger.info(f"Task completed: {task_description}")
        return final_result

    async def _collect_human_feedback(self, synthesis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Collect human feedback on decision points if enabled."""
        decision_points = synthesis_result.get("decision_points", [])
        human_responses = {}
        
        if not decision_points:
            return synthesis_result
        
        logger.info(f"Requesting human feedback on {len(decision_points)} decision points")
        
        for i, decision_point in enumerate(decision_points):
            question = decision_point.get("question", f"Decision {i+1}")
            options = decision_point.get("options", [])
            recommendation = decision_point.get("recommendation")
            
            if not options:
                continue
                
            # Display the question and options
            print(f"\n--- Decision Required: {question} ---")
            for j, option in enumerate(options):
                if option == recommendation:
                    print(f"{j+1}. {option} (Recommended)")
                else:
                    print(f"{j+1}. {option}")
            
            # Get human input with a timeout
            print("\nEnter your choice (number), or press Enter to accept recommendation: ")
            human_choice = input()
            
            # Process the choice
            try:
                if human_choice.strip():
                    choice_idx = int(human_choice) - 1
                    if 0 <= choice_idx < len(options):
                        human_responses[question] = options[choice_idx]
                    else:
                        human_responses[question] = recommendation
                else:
                    human_responses[question] = recommendation
            except ValueError:
                human_responses[question] = recommendation
                
        # Update synthesis result with human responses
        if human_responses:
            synthesis_result["human_feedback"] = human_responses
            
        return synthesis_result

    def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a task through the EnhancedAutonomousSwarmAgent (synchronous wrapper).
        
        Args:
            task_description: Description of the task to be performed
            context: Additional context or parameters for the task
            
        Returns:
            A dictionary containing the final solution and execution details
        """
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.execute_task(task_description, context))
        return result

    def cleanup(self):
        """Clean up resources used by the agent."""
        if hasattr(self, 'browser') and self.browser_enabled:
            try:
                self.browser.quit()
                logger.info("Browser session closed.")
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")