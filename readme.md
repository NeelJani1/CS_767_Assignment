# Intelligent Multimodal Planning & Research Agent 🤖

**Course:** CS 767 Intelligent Software Agents  
**Goal:** Design and implement a software agent prototype that can perceive input, make decisions, and take actions toward a goal.

## 🎥 Demo Video 
[**Click here to watch the 2-min Demo Video**](https://drive.google.com/file/d/1HFRvBOcnLBepkx14RdPtprNDDann1R-7/view?usp=drive_link)

## 🧠 System Overview
This project is a **Goal-Oriented Planning & Research Agent** built using the ReAct (Reasoning and Acting) framework. It acts as an autonomous assistant capable of taking high-level user goals (e.g., "Plan a 7-day trip around this landmark"), researching up-to-date information, and formatting the output into a strictly safe JSON schema.

### Core Capabilities (Assignment Requirements Met):
* **Perceive (Multimodal Input):** The agent accepts both conversational text and local image files (via Base64 encoding), allowing it to "see" and analyze user-provided images.
* **Decide (LLM & Tools):** Powered by Google's Gemini Flash models, the agent autonomously decides when its internal knowledge is insufficient and routes queries to external tools.
* **Act (Tool Execution):** The agent utilizes `DuckDuckGoSearchRun` for live web data and `Wikipedia` for encyclopedic context. Finally, it uses local file system operations to automatically save generated itineraries.
* **Memory:** Implements LangChain's `ConversationBufferMemory`, allowing the agent to maintain context over multiple turns (e.g., modifying a 2-day trip to a 7-day trip without restating the destination).
* **Safety Mechanism:** Uses `PydanticOutputParser` to enforce strict JSON output validation. This prevents prompt-injection and hallucinated formatting, ensuring the output is always machine-readable and predictable.

---

## ⚙️ Reproduction Instructions

### 1. Prerequisites
* Python 3.10+
* A valid Google Gemini API Key. Get one for free at [Google AI Studio](https://aistudio.google.com/).

### 2. Environment Setup
Clone this repository and navigate to the project folder:
```bash
git clone https://github.com/NeelJani1/CS_767_Assignment.git
cd CS_767_Assignment

Create a .env file in the root directory and add your Google API key:

GOOGLE_API_KEY="your_api_key_here"

3. Install Dependencies

Install the required LangChain and tool libraries:

pip install -r requirements.txt

4. Running the Agent

Run the main script from your terminal:

python main.py

  - Text queries: Type your goal/question and press Enter.
  - Image queries: When prompted, provide the absolute local path to an image
    (e.g., /home/neel/CS_767_Assignment/images.jpg) or drag-and-drop the image
    into the terminal.

🔄 Design Evolution & Commit Checkpoints

As required by the assignment, here is the evolution of the system design and
the challenges overcome during development:

  - Commit 1: Initial ReAct setup with Gemini and basic tools
      - Design: Connected Gemini API and set up the LangChain AgentExecutor with
        DuckDuckGo and Wikipedia tools.
      - Improvement: Realized that standard LLM text generation was too
        unpredictable for automated file saving.
  - Commit 2: Implemented Pydantic Output Parser as a Safety Mechanism
      - Design: Added a strict Pydantic ResponseSchema (Topic, Summary, Sources,
        Tools). Forced temperature=0.0 for deterministic outputs.
      - Improvement: Encountered a bug where Gemini returned chunked list
        strings; added custom Python stitching logic to sanitize the output
        before parsing.
  - Commit 3: Integrated ConversationBufferMemory for contextual awareness
      - Design: Transitioned from a single-run script to a while True loop to
        allow the agent to refine plans based on iterative human feedback.
  - Commit 4: Added Multimodal Vision capabilities via Base64 encoding
      - Design: Built a helper function to encode local images to Base64 to
        allow the agent to perceive visual environments.
      - Improvement: Discovered a major limitation where LangChain's Memory
        buffer crashed when fed image dictionaries.
  - Commit 5: Fixed LangChain Memory/Vision routing bug (Final Polish)
      - Design: Restructured the Prompt Template. Routed text inputs strictly to
        the Memory buffer (input), while creating a "secret backdoor"
        placeholder (image_input) to safely pass visual data directly to the
        Gemini model. Fixed terminal quote-escaping for drag-and-drop file
        paths.

Developed for CS 767 by Neel Jani

