from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from tools import search_tool, wiki_tool , save_tool ,save_to_txt_file
from langchain_classic.memory import ConversationBufferMemory

import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

load_dotenv()

class ResponseSchema(BaseModel):
    topic: str
    summary: str
    source: list[str]
    tools_used: list[str]
#llm = ChatOpenAI(model="gpt-4o")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.7)

parser = PydanticOutputParser(pydantic_object=ResponseSchema)



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """You are a helpful research assistant. 
        Use your tools to find accurate information based on the user's query.
        
        IMPORTANT RULES:
        1. You must ALWAYS provide your final answer as a valid JSON object.
        2. Do not include any conversational text, greetings, or markdown outside of the JSON block.
        3. Your final output must strictly follow these schema instructions:
        
        {format_instructions}
        """),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm, 
    prompt=prompt, 
    tools=tools
    )
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

print("==================================================")
print("🤖 Planning & Research Agent Initialized.")
print("Type 'exit' or 'quit' to end the conversation.")
print("==================================================\n")

while True:
    query = input("What can I help you with?\nUser: ")
    
    if query.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break

    try:
        raw_response = agent_executor.invoke({"query": query})
        
        try:
            output_raw = raw_response.get("output")
            
            # Stitch list chunks together if Gemini returns a list
            if isinstance(output_raw, list):
                extracted_text = ""
                for chunk in output_raw:
                    if isinstance(chunk, str):
                        extracted_text += chunk
                    elif isinstance(chunk, dict) and "text" in chunk:
                        extracted_text += chunk["text"]
                output_string = extracted_text
            else:
                output_string = str(output_raw)

            # Parse the text into JSON
            structured_response = parser.parse(output_string)
            
            print("\n--- PARSED RESPONSE ---")
            print("Topic:", structured_response.topic)
            print("Summary:", structured_response.summary)
            print("Source:", structured_response.source)
            print("Tools Used:", structured_response.tools_used)
            
            # Manually save the response using your existing function
            log_data = (
                f"Query: {query}\n"
                f"Topic: {structured_response.topic}\n"
                f"Summary: {structured_response.summary}\n"
                f"Sources: {', '.join(structured_response.source)}\n"
            )
            save_result = save_to_txt_file(log_data)
            print(f"\n[System] {save_result}\n")
            
        except Exception as e:
            print("\nError parsing response into Pydantic schema:", e)
            print("Output string attempted to parse was:\n", output_string)

    except Exception as e:
        print(f"\nAgent execution failed. Error: {e}")