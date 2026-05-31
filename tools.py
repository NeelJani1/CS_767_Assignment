import wikipedia
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, Tool
from langchain_community.utilities import WikipediaAPIWrapper
from datetime import datetime

# 1. Set the User-Agent for the underlying Wikipedia package
wikipedia.set_user_agent("SearchBot/1.0 (njan320@aucklanduni.ac.nz)")

def save_to_txt_file(data: str, filename: str = "conversation_history.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"---Conversation History---\nTime: {timestamp}:\n{data}\n{'-'*50}\n"

    with open(filename, "a", encoding="utf-8") as file:
        file.write(formatted_text)

    return f"Conversation history saved to {filename} at {timestamp}."

save_tool = Tool(
    name="Save_to_txt_file",
    func=save_to_txt_file,
    description="Saves the conversation history to a text file with a timestamp. Input should be a string.",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="Search_tool",
    func=search.run,
    description="Search web for up-to-date information.",
)

# 2. Use LangChain's built-in Wikipedia wrappers
api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_limit=300)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)