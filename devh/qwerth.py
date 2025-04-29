from typing import Dict
from dotenv import load_dotenv
import json
from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
# from swarms.utils.str_to_dict import str_to_dict
def str_to_dict(s: str, retries: int = 3) -> Dict:
    """
    Converts a JSON string to dictionary.

    Args:
        s (str): The JSON string to be converted.
        retries (int): The number of times to retry parsing the string in case of a JSONDecodeError. Default is 3.

    Returns:
        Dict: The parsed dictionary from the JSON string.

    Raises:
        json.JSONDecodeError: If the string cannot be parsed into a dictionary after the specified number of retries.
    """
    for attempt in range(retries):
        try:
            # Run json.loads directly since it's fast enough
            return json.loads(s)
        except json.JSONDecodeError as e:
            if attempt < retries - 1:
                continue  # Retry on failure
            else:
                raise e  # Raise the error if all retries fail

load_dotenv()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Retrieve the current stock price and related information for a specified company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol of the company, e.g. AAPL for Apple Inc.",
                    },
                    "include_history": {
                        "type": "boolean",
                        "description": "Indicates whether to include historical price data along with the current price.",
                    },
                    "time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Optional parameter to specify the time for which the stock data is requested, in ISO 8601 format.",
                    },
                },
                "required": [
                    "ticker",
                    "include_history",
                    "time",
                ],
            },
        },
    }
]


agent = Agent(
    agent_name="Financial-Analysis-Agent",
    model_name="gpt-4o",
    agent_description="Personal finance advisor agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    tools_list_dictionary=tools,
)

out = agent.run(
    "What is the current stock price for Apple Inc. (AAPL)? Include historical price data.",
)

print(out)

# print(type(out))

# print(str_to_dict(out))

# print(type(str_to_dict(out)))