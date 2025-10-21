import asyncio
import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import google_search            # <- import the built-in tool
from google.genai import types

# Load environment variables
load_dotenv()

# Define sub-agents with tools
agent1 = LlmAgent(
    name="ResearchAgent1",
    model="gemini-2.0-flash",           # your model
    description="Agent researching topic A",
    instruction="You are a research assistant for topic A. Provide a concise summary. Use web search if needed.",
    output_key="result_A",
    tools=[google_search]                # <— assign the tool here
)

agent2 = LlmAgent(
    name="ResearchAgent2",
    model="gemini-2.0-flash",
    description="Agent researching topic B",
    instruction="You are a research assistant for topic B. Provide a concise summary. Use web search if needed.",
    output_key="result_B",
    tools=[google_search]                # <— also assign the tool
)

# Create the ParallelAgent that runs both sub-agents concurrently
parallel_agent = ParallelAgent(
    name="ParallelResearchAgent",
    sub_agents=[agent1, agent2]
)

# Summary agent to aggregate results
summary_agent = LlmAgent(
    name="SummaryAgent",
    model="gemini-2.0-flash",
    description="Summarizes the combined results",
    instruction=(
        "You are given results from topic A (in {result_A}) "
        "and topic B (in {result_B}). Produce a combined summary."
    ),
    output_key="combined_summary"
    # tools omitted here since summarization may not need search
)

# Full workflow: first the parallel fan-out, then summarise
workflow_agent = SequentialAgent(
    name="ResearchWorkflow",
    sub_agents=[parallel_agent, summary_agent]
)

async def main():
    APP_NAME = "my_app"
    USER_ID  = "user_123"
    SESSION_ID = "sess_001"

    # Initialize memory state
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    # Create the agent runner
    runner = Runner(
        agent=workflow_agent,
        app_name=APP_NAME,
        session_service=session_service
    )

    # Create user content requesting research on both topics
    content = types.Content(
        role="user",
        parts=[types.Part(text="Please research topics Deep Agents and Shallow Agents and provide a combined summary.")]
    )

    final_response = None
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text
            print(final_response)
    
    #print("Final output:", final_response)

if __name__ == "__main__":
    asyncio.run(main())
