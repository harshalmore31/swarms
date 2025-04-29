import asyncio
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import aiohttp
from dotenv import load_dotenv
from rich.console import Console

from swarms import Agent
from swarms.agents.reasoning_duo import ReasoningDuo
from swarms.structs.conversation import Conversation
from swarms.utils.any_to_str import any_to_str
from swarms.utils.formatter import formatter
# from swarms.utils.str_to_dict import str_to_dict

from litellm.utils import convert_to_dict as str_to_dict
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)

console = Console()
load_dotenv()

# Number of worker threads for concurrent operations
MAX_WORKERS = (
    os.cpu_count() * 2
)  # Optimal number of workers based on CPU cores

###############################################################################
# 1. System Prompts for Each Scientist Agent
###############################################################################


def format_exa_results(json_data: Dict[str, Any]) -> str:
    """Formats Exa.ai search results into structured text"""
    if "error" in json_data:
        return f"### Error\n{json_data['error']}\n"

    # Pre-allocate formatted_text list with initial capacity
    formatted_text = []

    # Extract search metadata
    search_params = json_data.get("effectiveFilters", {})
    query = search_params.get("query", "General web search")
    formatted_text.append(
        f"### Exa Search Results for: '{query}'\n\n---\n"
    )

    # Process results
    results = json_data.get("results", [])

    if not results:
        formatted_text.append("No results found.\n")
        return "".join(formatted_text)

    def process_result(
        result: Dict[str, Any], index: int
    ) -> List[str]:
        """Process a single result in a thread-safe manner"""
        title = result.get("title", "No title")
        url = result.get("url", result.get("id", "No URL"))
        published_date = result.get("publishedDate", "")

        # Handle highlights efficiently
        highlights = result.get("highlights", [])
        highlight_text = (
            "\n".join(
                (
                    h.get("text", str(h))
                    if isinstance(h, dict)
                    else str(h)
                )
                for h in highlights[:3]
            )
            if highlights
            else "No summary available"
        )

        return [
            f"{index}. **{title}**\n",
            f"   - URL: {url}\n",
            f"   - Published: {published_date.split('T')[0] if published_date else 'Date unknown'}\n",
            f"   - Key Points:\n      {highlight_text}\n\n",
        ]

    # Process results concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_result = {
            executor.submit(process_result, result, i + 1): i
            for i, result in enumerate(results)
        }

        # Collect results in order
        processed_results = [None] * len(results)
        for future in as_completed(future_to_result):
            idx = future_to_result[future]
            try:
                processed_results[idx] = future.result()
            except Exception as e:
                console.print(
                    f"[bold red]Error processing result {idx + 1}: {str(e)}[/bold red]"
                )
                processed_results[idx] = [
                    f"Error processing result {idx + 1}: {str(e)}\n"
                ]

    # Extend formatted text with processed results in correct order
    for result_text in processed_results:
        formatted_text.extend(result_text)

    return "".join(formatted_text)


async def _async_exa_search(
    query: str, **kwargs: Any
) -> Dict[str, Any]:
    """Asynchronous helper function for Exa.ai API requests"""
    api_url = "https://api.exa.ai/search"
    headers = {
        "x-api-key": os.getenv("EXA_API_KEY"),
        "Content-Type": "application/json",
    }

    payload = {
        "query": query,
        "useAutoprompt": True,
        "numResults": kwargs.get("num_results", 10),
        "contents": {
            "text": True,
            "highlights": {"numSentences": 2},
        },
        **kwargs,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    return {
                        "error": f"HTTP {response.status}: {await response.text()}"
                    }
                return await response.json()
    except Exception as e:
        return {"error": str(e)}


def exa_search(query: str, **kwargs: Any) -> str:
    """Performs web search using Exa.ai API with concurrent processing"""
    try:
        # Run async search in the event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response_json = loop.run_until_complete(
                _async_exa_search(query, **kwargs)
            )
        finally:
            loop.close()

        # Format results concurrently
        formatted_text = format_exa_results(response_json)

        return formatted_text

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        console.print(f"[bold red]{error_msg}[/bold red]")
        return error_msg


# Define the research tools schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_topic",
            "description": "Conduct an in-depth search on a specified topic or subtopic, generating a comprehensive array of highly detailed search queries tailored to the input parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "depth": {
                        "type": "integer",
                        "description": "Indicates the level of thoroughness for the search. Values range from 1 to 3, where 1 represents a superficial search and 3 signifies an exploration of the topic.",
                    },
                    "detailed_queries": {
                        "type": "array",
                        "description": "An array of highly specific search queries that are generated based on the input query and the specified depth. Each query should be designed to elicit detailed and relevant information from various sources.",
                        "items": {
                            "type": "string",
                            "description": "Each item in this array should represent a unique search query that targets a specific aspect of the main topic, ensuring a comprehensive exploration of the subject matter.",
                        },
                    },
                },
                "required": ["depth", "detailed_queries"],
            },
        },
    },
]

RESEARCH_AGENT_PROMPT = """
You are an advanced research agent specialized in conducting deep, comprehensive research across multiple domains.
Your task is to:

1. Break down complex topics into searchable subtopics
2. Generate diverse search queries to explore each subtopic thoroughly
3. Identify connections and patterns across different areas of research
4. Synthesize findings into coherent insights
5. Identify gaps in current knowledge and suggest areas for further investigation

For each research task:
- Consider multiple perspectives and approaches
- Look for both supporting and contradicting evidence
- Evaluate the credibility and relevance of sources
- Track emerging trends and recent developments
- Consider cross-disciplinary implications

Output Format:
- Provide structured research plans
- Include specific search queries for each subtopic
- Prioritize queries based on relevance and potential impact
- Suggest follow-up areas for deeper investigation
"""

SUMMARIZATION_AGENT_PROMPT = """
You are an expert information synthesis and summarization agent designed for producing clear, accurate, and insightful summaries of complex information. Your core capabilities include:

INFORMATION PROCESSING:
- Identify and extract key concepts, themes, and insights from any given content
- Recognize patterns, relationships, and hierarchies within information
- Filter out noise while preserving crucial context and nuance
- Handle multiple sources and perspectives simultaneously

SUMMARIZATION APPROACH:
1. Multi-level Structure
   - Provide an executive summary (1-2 sentences)
   - Follow with key findings (3-5 bullet points)
   - Include detailed insights with supporting evidence
   - End with implications or next steps when relevant

2. Quality Standards
   - Maintain factual accuracy and precision
   - Preserve important technical details and terminology
   - Avoid oversimplification of complex concepts
   - Include quantitative data when available
   - Cite or reference specific sources when summarizing claims

3. Clarity & Accessibility
   - Use clear, concise language
   - Define technical terms when necessary
   - Structure information logically
   - Use formatting to enhance readability
   - Maintain appropriate level of technical depth for the audience

4. Synthesis & Analysis
   - Identify conflicting information or viewpoints
   - Highlight consensus across sources
   - Note gaps or limitations in the information
   - Draw connections between related concepts
   - Provide context for better understanding

OUTPUT REQUIREMENTS:
- Begin with a clear statement of the topic or question being addressed
- Use consistent formatting and structure
- Clearly separate different levels of detail
- Include confidence levels for conclusions when appropriate
- Note any areas requiring additional research or clarification

Remember: Your goal is to make complex information accessible while maintaining accuracy and depth. Prioritize clarity without sacrificing important nuance or detail."""


# Initialize the research agent
research_agent = Agent(
    agent_name="Deep-Research-Agent",
    agent_description="Specialized agent for conducting comprehensive research across multiple domains",
    system_prompt=RESEARCH_AGENT_PROMPT,
    max_loops=1,  # Allow multiple iterations for thorough research
    tools_list_dictionary=tools,
)


reasoning_duo = ReasoningDuo(
    system_prompt=SUMMARIZATION_AGENT_PROMPT, output_type="string"
)


class DeepResearchSwarm:
    def __init__(
        self,
        research_agent: Agent = research_agent,
        max_loops: int = 1,
        nice_print: bool = True,
        output_type: str = "json",
    ):
        self.research_agent = research_agent
        self.max_loops = max_loops
        self.nice_print = nice_print
        self.output_type = output_type

        self.reliability_check()

        self.conversation = Conversation()

    def reliability_check(self):
        """
        Check the reliability of the query
        """
        if self.max_loops < 1:
            raise ValueError("max_loops must be greater than 0")

        formatter.print_panel(
            "DeepResearchSwarm is booting up...", "blue"
        )
        formatter.print_panel("Reliability check passed", "green")

    def get_queries(self, query: str) -> List[str]:
        """
        Generate a list of detailed search queries based on the input query.

        Args:
            query (str): The main research query to explore

        Returns:
            List[str]: A list of detailed search queries
        """

        self.conversation.add(role="User", content=query)

        # Get the agent's response
        agent_output = self.research_agent.run(query)

        self.conversation.add(
            role=self.research_agent.agent_name, content=agent_output
        )

        # Convert the string output to dictionary
        output_dict = str_to_dict(agent_output)

        # Print the conversation history
        if self.nice_print:
            to_do_list = any_to_str(output_dict)
            formatter.print_panel(to_do_list, "blue")

        # Extract the detailed queries from the output
        if (
            isinstance(output_dict, dict)
            and "detailed_queries" in output_dict
        ):
            return output_dict["detailed_queries"]

        return []

    def step(self, query: str):
        """
        Execute a single research step.

        Args:
            query (str): The research query to process
        """
        queries = self.get_queries(query)

        for query in queries:
            results = exa_search(query)
            self.conversation.add(
                role="User",
                content=f"Search results for {query}: \n {results}",
            )

            reasoning_output = reasoning_duo.run(results)
            self.conversation.add(
                role=reasoning_duo.agent_name,
                content=reasoning_output,
            )

        return history_output_formatter(
            self.conversation, type="json"
        )


# Example usage
if __name__ == "__main__":
    swarm = DeepResearchSwarm()
    print(
        swarm.step("What is the active tarrif situation with mexico?")
    )
