from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent  # deprecated warning is expected

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

llm = ChatOpenAI(model="gpt-4o")

# IMPORTANT: pass the model via `model=llm` and don't use `state_modifier` here
agent = create_react_agent(
    model=llm,
    tools=[get_weather],
)

result = agent.invoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]})
print(result["messages"][-1].content)
