import base64
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from tools import search_tool, wiki_tool , save_tool ,save_to_txt_file
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import MessagesPlaceholder

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
#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite",temperature=0.0)

#llm = ChatGroq(model="llama3-70b-8192", temperature=0.7)


parser = PydanticOutputParser(pydantic_object=ResponseSchema)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """You are an intelligent task planning and research assistant.
        Use your tools to find accurate information based on the user's query.
        You have vision capabilities and can analyze images provided by the user.
        
        IMPORTANT RULES:
        1. You must ALWAYS provide your final answer as a valid JSON object.
        2. Do not include any conversational text, greetings, or markdown outside of the JSON block.
        3. Your final output must strictly follow these schema instructions:
        
        {format_instructions}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="image_input"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
).partial(format_instructions=parser.get_format_instructions())


memory = ConversationBufferMemory(memory_key="chat_history", input_key="input", return_messages=True)


tools = [search_tool, wiki_tool]

agent = create_tool_calling_agent(
    llm=llm, 
    prompt=prompt, 
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

def encode_image(image_path):
    """Converts an image to a base64 string so the LLM can read it."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
print("==================================================")
print("🤖 Planning & Research Agent Initialized.")
print("Type 'exit' or 'quit' to end the conversation.")
print("==================================================\n")

while True:
    print("-" * 40)
    text_query = input("What can I help you with?\nUser: ")
    
    if text_query.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break

    image_path = input("Attach an image path (or press Enter to skip): ").strip().strip('\"\'')
    
    # 1. Prepare the Image properly
    if image_path and os.path.exists(image_path):
        print(f"[System] Processing image: {image_path}...")
        base64_image = encode_image(image_path)
        
        # We put the image in a REAL message object so LangChain doesn't break it
        image_msg = [HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ])]
    elif image_path:
        print(f"[Warning] Could not find image at {image_path}. Continuing with text only.")
        image_msg = []
    else:
        image_msg = []

    try:
        # 2. Give the AI a default text prompt if you just press Enter
        safe_text_query = text_query if text_query.strip() != "" else "Please analyze the attached image."

        # 3. Send BOTH the text and the image to the agent!
        raw_response = agent_executor.invoke({
            "input": safe_text_query,
            "image_input": image_msg  # <-- Sending the image through the secret door!
        })
        
        try:
            output_raw = raw_response.get("output")
            
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

            structured_response = parser.parse(output_string)
            
            print("\n--- PARSED RESPONSE ---")
            print("Topic:", structured_response.topic)
            print("Summary:", structured_response.summary)
            print("Source:", structured_response.source)
            print("Tools Used:", structured_response.tools_used)
            
            log_data = (
                f"Query: {text_query} (Image Attached: {bool(image_path)})\n"
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