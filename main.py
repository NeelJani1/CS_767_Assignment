"""main.py.

Intelligent Multimodal Planning & Research Agent
Course: CS 767 Intelligent Software Agents
Author: Neel Jani

An autonomous agent utilizing the ReAct (Reasoning and Acting) framework,
LangChain, Google Gemini, and multi-tool execution with strict Pydantic JSON safety.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
import sys
import warnings
from typing import Any

from dotenv import load_dotenv
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

# Graceful import compatibility for standard LangChain and classic wrappers
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain.memory import ConversationBufferMemory
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
    from langchain_classic.memory import ConversationBufferMemory

from tools import get_default_tools, save_to_txt_file

# Suppress benign library deprecation notices for clean terminal UX
warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Load environment variables from .env file
load_dotenv()


# =====================================================================
# 1. Output Schema & Safety Verification
# =====================================================================
class ResponseSchema(BaseModel):
    """Strict JSON response format enforced via PydanticOutputParser."""

    topic: str = Field(
        ...,
        description="The core topic or entity identified in the user query/image.",
    )
    summary: str = Field(
        ...,
        description="A detailed, factual itinerary, research summary, or response.",
    )
    source: list[str] = Field(
        default_factory=list,
        description="List of information sources, websites, or articles referenced.",
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="Names of tools invoked by the agent (e.g. Search_tool, Wikipedia).",
    )


def clean_json_output(raw_output: Any) -> str:
    """Sanitize agent output strings and extract pure JSON content.

    Handles chunked lists, Markdown code block delimiters (```json ... ```),
    and leading/trailing whitespace.
    """
    if isinstance(raw_output, list):
        chunks = []
        for chunk in raw_output:
            if isinstance(chunk, str):
                chunks.append(chunk)
            elif isinstance(chunk, dict) and "text" in chunk:
                chunks.append(str(chunk["text"]))
        text = "".join(chunks)
    else:
        text = str(raw_output).strip()

    # Extract JSON between markdown fences ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Extract outermost JSON object if present
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx : end_idx + 1].strip()

    return text.strip()


# =====================================================================
# 2. Image Processing & Multimodal Helpers
# =====================================================================
def encode_image(image_path: str) -> tuple[str, str]:
    """Read a local image and return base64 encoded data with its MIME type.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        Tuple of (base64_string, mime_type).
    """
    resolved_path = os.path.abspath(os.path.expanduser(image_path.strip().strip("'\"")))
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Image not found at path: {resolved_path}")

    mime_type, _ = mimetypes.guess_type(resolved_path)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"

    with open(resolved_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")

    return encoded, mime_type


# =====================================================================
# 3. Agent & LLM Initialization
# =====================================================================
def get_llm(temperature: float = 0.0) -> Any:
    """Initialize the configured LLM provider (default: Google Gemini)."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI

            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            return ChatOpenAI(model=model, temperature=temperature)
        except ImportError as err:
            raise ImportError(
                "langchain-openai is required for OpenAI provider. Run: pip install langchain-openai"
            ) from err

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq

            model = os.getenv("GROQ_MODEL", "llama3-70b-8192")
            return ChatGroq(model=model, temperature=temperature)
        except ImportError as err:
            raise ImportError(
                "langchain-groq is required for Groq provider. Run: pip install langchain-groq"
            ) from err

    # Default provider: Google Gemini
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Check for Gemini API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.startswith("your_"):
        print(
            "\n[Notice] GOOGLE_API_KEY is not set or using placeholder value.\n"
            "         Please configure your key in .env (see .env.example) or environment.\n"
            "         Get a free key at: https://aistudio.google.com/\n"
        )

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
    )


def create_agent(
    tools: list[Any] | None = None,
    verbose: bool | None = None,
) -> tuple[AgentExecutor, PydanticOutputParser]:
    """Assemble the ReAct prompt, tools, memory, and AgentExecutor."""
    active_tools = tools if tools is not None else get_default_tools()
    is_verbose = (
        verbose if verbose is not None else (os.getenv("AGENT_VERBOSE", "true").lower() == "true")
    )

    llm = get_llm(temperature=float(os.getenv("AGENT_TEMPERATURE", "0.0")))
    parser = PydanticOutputParser(pydantic_object=ResponseSchema)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an intelligent task planning and research assistant.
Use your tools to find accurate information based on the user's query.
You have vision capabilities and can analyze images provided by the user.

IMPORTANT RULES:
1. You must ALWAYS provide your final answer as a valid JSON object.
2. Do not include any conversational text, greetings, or markdown outside of the JSON block.
3. Your final output must strictly follow these schema instructions:

{format_instructions}
""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="image_input"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="input",
        return_messages=True,
    )

    agent = create_tool_calling_agent(
        llm=llm,
        prompt=prompt,
        tools=active_tools,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=active_tools,
        verbose=is_verbose,
        memory=memory,
        handle_parsing_errors=True,
    )

    return executor, parser


# =====================================================================
# 4. Interactive CLI Loop
# =====================================================================
def run_cli() -> None:
    """Run interactive terminal session for the planning and research agent."""
    print("=" * 60)
    print("🤖 Intelligent Multimodal Planning & Research Agent (CS 767)")
    print("   Powered by Google Gemini & LangChain ReAct Framework")
    print("   Commands: Type 'exit' or 'quit' to end session.")
    print("=" * 60 + "\n")

    try:
        agent_executor, parser = create_agent()
    except Exception as err:
        print(f"[Fatal] Failed to initialize agent: {err}")
        sys.exit(1)

    while True:
        try:
            print("-" * 50)
            text_query = input("What can I help you with?\nUser: ").strip()

            if text_query.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye! Safe travels.")
                break

            if not text_query:
                print("[Info] Please enter a valid question or goal.")
                continue

            raw_image_path = input("Attach an image path (or press Enter to skip): ").strip()

            # Handle image input
            image_msg: list[HumanMessage] = []
            image_attached = False

            if raw_image_path:
                try:
                    b64_img, mime_type = encode_image(raw_image_path)
                    print(f"[System] Image successfully attached ({mime_type}).")
                    image_msg = [
                        HumanMessage(
                            content=[
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime_type};base64,{b64_img}"},
                                }
                            ]
                        )
                    ]
                    image_attached = True
                except Exception as img_err:
                    print(
                        f"[Warning] Could not process image: {img_err}. Continuing with text only."
                    )

            # Invoke Agent Executor
            safe_text_query = text_query if text_query else "Please analyze the attached image."
            raw_response = agent_executor.invoke(
                {
                    "input": safe_text_query,
                    "image_input": image_msg,
                }
            )

            # Extract and parse response safely
            output_raw = raw_response.get("output", "")
            sanitized_json_str = clean_json_output(output_raw)

            try:
                structured_response = parser.parse(sanitized_json_str)

                print("\n" + "=" * 25 + " PARSED RESPONSE " + "=" * 25)
                print(f"📌 Topic     : {structured_response.topic}")
                print(f"📝 Summary   : {structured_response.summary}")
                print(f"🔗 Sources   : {', '.join(structured_response.source) or 'None'}")
                print(f"🛠️  Tools Used: {', '.join(structured_response.tools_used) or 'None'}")
                print("=" * 67)

                # Persist to conversation log
                log_data = (
                    f"Query: {text_query} (Image Attached: {image_attached})\n"
                    f"Topic: {structured_response.topic}\n"
                    f"Summary: {structured_response.summary}\n"
                    f"Sources: {', '.join(structured_response.source)}\n"
                    f"Tools Used: {', '.join(structured_response.tools_used)}\n"
                )
                save_msg = save_to_txt_file(log_data)
                print(f"\n[System] {save_msg}\n")

            except Exception as parse_err:
                print(f"\n[Warning] Schema validation notice: {parse_err}")
                print("Raw Agent Output:")
                print(sanitized_json_str)

        except KeyboardInterrupt:
            print("\n\nSession interrupted by user. Goodbye!")
            break
        except Exception as loop_err:
            print(f"\n[Error] Agent execution failed: {loop_err}\n")


if __name__ == "__main__":
    run_cli()
