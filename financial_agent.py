from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

web_search_agent = Agent(
    name="Web search agent",
    role="Search the web",
    model=Groq(id='llama-3.3-70b-versatile'),
    tools=[DuckDuckGo()],
    instructions=["Always include resources"],
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance agent",
    model=Groq(id='llama-3.3-70b-versatile'),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        company_news=True,
    )],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("Summarize analuyts recommendations and share latest news for AAPL", stream=True)