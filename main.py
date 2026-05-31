from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from tools import search_tool

load_dotenv()

class ResponseSchema(BaseModel):
    topic: str
    summary: str
    source: list[str]
    tools_used: list[str]

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash",temperature=0.7)

parser = PydanticOutputParser(pydantic_object=ResponseSchema)



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """
        You are a helpful assistant that provides concise summaries of topics.
        And No Extra text other than the summary.\n{format_instructions}
        """)
        ,
        ("placeholder", "{chat_history}"),
        ("human", "Provide a summary of the topic: {topic}"),
        ("placeholder", "{agent_scratchpad}")
    ]
    ).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool]
agent = create_tool_calling_agent(
    llm=llm, 
    prompt=prompt, 
    tools=tools
    )

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("what can I help you with? ")
raw_response = agent_executor.invoke({"query": query })

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("Error parsing response:", e , "Raw response was:", raw_response)

print(structured_response.topic)
print(structured_response.summary)
print(structured_response.source)
print(structured_response.tools_used)