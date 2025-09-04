from swarms import Agent

# Initialize the agent
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    model_name="claude-sonnet-4-20250514",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    streaming_on=False,
    print_on=False,
)

out = agent.run(
    task="What are the top five best energy stocks across nuclear, solar, gas, and other energy sources?",
)

print(out)
