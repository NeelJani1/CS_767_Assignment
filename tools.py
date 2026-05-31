from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun , DuckDuckGoSearchResults , Tool
from langchain_community.utilities import WikipediaAPIWrapper
from datetime import datetime


search = DuckDuckGoSearchRun()
search_tool = Tool(
    name = "Search_tool",
    func = search.run,
    description="search web for upto date information",
)