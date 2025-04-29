"""
Example usage of WebAgent for performing a web search.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from swarms.structs.devh.wbagent import WebAgent

console = Console()

async def main():
    # Initialize WebAgent
    agent = WebAgent(
        max_iterations=5,
        headless=False,  # Set to True for headless operation
        base_path="./web_agent_states"
    )

    # Example instruction
    instruction = """
    1. Navigate to Google
    2. Search for 'Playwright Python tutorial'
    3. Click the first search result
    4. Wait for the page to load
    """

    try:
        console.print("[bold green]Starting web automation...[/bold green]")
        result = await agent.run(instruction)

        if result["status"] == "completed":
            console.print("[bold green]Web automation completed successfully![/bold green]")
            console.print("Results:", result["results"])
        else:
            console.print("[bold red]Web automation failed![/bold red]")
            console.print(f"Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        console.print(f"[bold red]Error during execution: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 7):
        console.print("[bold red]Error: Python 3.7 or higher is required[/bold red]")
        sys.exit(1)

    # Ensure OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error: OPENAI_API_KEY environment variable not set[/bold red]")
        sys.exit(1)

    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Automation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Fatal error: {str(e)}[/bold red]")
        sys.exit(1)