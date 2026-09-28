"""tools.py.

External tool definitions and execution handlers for the CS 767 Intelligent
Planning & Research Agent.

Tools included:
    - Search_tool: Live internet information retrieval via DuckDuckGo.
    - Wikipedia: Encyclopedic search and summary lookup via Wikipedia API.
    - Save_to_txt_file: Timestamped file logging for conversation memory and itineraries.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import wikipedia
from langchain_community.tools import DuckDuckGoSearchRun, Tool, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# 1. Set standard User-Agent for Wikipedia API compliance
WIKIPEDIA_USER_AGENT = os.getenv(
    "WIKIPEDIA_USER_AGENT",
    "CS767-PlanningAgent/1.0 (njan320@aucklanduni.ac.nz)",
)
try:
    wikipedia.set_user_agent(WIKIPEDIA_USER_AGENT)
except Exception as e:
    # Non-fatal warning if user-agent cannot be set
    print(f"[Warning] Failed to set Wikipedia user agent: {e}")


def save_to_txt_file(
    data: str,
    filename: str | None = None,
) -> str:
    """Save conversation history or itinerary data to a text file with a timestamp.

    Args:
        data: The formatted string content to append to the log.
        filename: Target file path. Defaults to CONVERSATION_LOG_FILE env var or
          'conversation_history.txt'.

    Returns:
        Confirmation message detailing status, file location, and timestamp.
    """
    target_path = filename or os.getenv("CONVERSATION_LOG_FILE", "conversation_history.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"---Conversation History---\nTime: {timestamp}:\n{data.strip()}\n{'-' * 50}\n"

    try:
        # Ensure parent directories exist if a custom nested path is passed
        parent_dir = os.path.dirname(target_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(target_path, "a", encoding="utf-8") as file:
            file.write(formatted_text)

        return f"Conversation history saved to {target_path} at {timestamp}."
    except OSError as err:
        return f"Error saving history to {target_path}: {err}"


def safe_search(query: str) -> str:
    """Run a DuckDuckGo search query with error handling."""
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        if not result or not result.strip():
            return "No relevant web search results found for this query."
        return result
    except Exception as err:
        return f"Web search encountered an error: {err}"


def safe_wiki(query: str) -> str:
    """Query Wikipedia with custom wrapper limits and error handling."""
    try:
        api_wrapper = WikipediaAPIWrapper(
            top_k_results=3,
            doc_content_chars_limit=300,
        )
        wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
        result = wiki.run(query)
        if not result or not result.strip():
            return "No Wikipedia articles matched your search query."
        return result
    except Exception as err:
        return f"Wikipedia search encountered an error: {err}"


# 2. Tool Wrappers for LangChain Agent Integration
save_tool = Tool(
    name="Save_to_txt_file",
    func=save_to_txt_file,
    description=(
        "Saves the conversation history or research itinerary to a text file with a timestamp. "
        "Input should be a string containing the text to save."
    ),
)

search_tool = Tool(
    name="Search_tool",
    func=safe_search,
    description="Search the web for up-to-date information, news, travel options, and real-time facts.",
)

wiki_tool = Tool(
    name="Wikipedia",
    func=safe_wiki,
    description="Search Wikipedia for encyclopedic context, background history, definitions, and landmarks.",
)


def get_default_tools() -> list[Any]:
    """Return the list of default active tools used by the planning agent."""
    return [search_tool, wiki_tool]
