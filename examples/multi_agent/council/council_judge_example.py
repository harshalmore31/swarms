from swarms import Agent, CouncilAsAJudge

# Create a base agent
base_agent = Agent(
    agent_name="Financial-Analysis-Agent",
    system_prompt="You are a financial expert helping users analyze invoices and provide tax-saving strategies.",
    model_name="gpt-4o",
    max_loops=1,
)

# Run the base agent
user_query = "How can I save my tax, as per my invoice? analyze the figures in the image."
model_output = base_agent.run(
    user_query,
)

# Create and run the council
panel = CouncilAsAJudge(
    model_name="gpt-4o"
)
results = panel.run(f"QUERY: {user_query}\n\nRESPONSE: {model_output}")
print(results)
