from typing import Literal

from langchain_core.messages import AIMessage

from llm import llm
from state import AssistantState


def _extract_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "".join(parts)

    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return str(content["content"])

    return str(content)


def get_conversation(state: AssistantState):
    conversation = ""

    for message in state["messages"]:
        conversation += f"{message.type}: {message.content}\n"

    return conversation


def supervisor_agent(state: AssistantState):

    conversation = get_conversation(state)

    prompt = f"""
You are the supervisor of a multi-agent AI assistant.

Study the conversation and determine which specialist should handle
the user's latest request.

Available specialists:

explain
- Explanations
- Teaching
- Educational questions
- Technical concepts

plan
- Plans
- Schedules
- Steps
- Strategies
- Workflows

write
- Emails
- Messages
- Posts
- Letters
- Written content

general
- General questions
- Casual questions
- Anything that does not fit the other categories

Conversation:

{conversation}

Latest user request:

{state["user_input"]}

Return only one word:

explain
plan
write
general
"""

    response = llm.invoke(prompt)

    route = _extract_text(response.content).strip().lower()

    return {
        "route": route
    }


def explain_agent(state: AssistantState):

    conversation = get_conversation(state)

    prompt = f"""
You are the teaching specialist in a multi-agent AI assistant.

Use the previous conversation when useful.

Conversation:

{conversation}

Latest user request:

{state["user_input"]}

Explain the topic clearly and simply.

Use examples when useful.
"""

    response = llm.invoke(prompt)

    return {
        "specialist_response": response.content
    }


def planner_agent(state: AssistantState):

    conversation = get_conversation(state)

    prompt = f"""
You are the planning specialist.

Use the conversation history when necessary.

Conversation:

{conversation}

Latest user request:

{state["user_input"]}

Create a practical and clear plan for the user.
"""

    response = llm.invoke(prompt)

    return {
        "specialist_response": response.content
    }


def writer_agent(state: AssistantState):

    conversation = get_conversation(state)

    prompt = f"""
You are the writing specialist.

Use the previous conversation for context when needed.

Conversation:

{conversation}

Latest user request:

{state["user_input"]}

Create the written content requested by the user.
"""

    response = llm.invoke(prompt)

    return {
        "specialist_response": response.content
    }


def general_agent(state: AssistantState):

    conversation = get_conversation(state)

    prompt = f"""
You are a helpful general AI assistant.

Use the conversation history when relevant.

Conversation:

{conversation}

Latest request:

{state["user_input"]}

Answer clearly.
"""

    response = llm.invoke(prompt)

    return {
        "specialist_response": response.content
    }


def review_agent(state: AssistantState):

    prompt = f"""
You are the final reviewer of a multi-agent AI assistant.

The user asked:

{state["user_input"]}

The specialist produced:

{state["specialist_response"]}

Improve the answer.

Check that it:

- answers the user's request
- is clear
- avoids unnecessary repetition
- corrects obvious mistakes
- keeps important information

Return only the improved final answer.
"""

    response = llm.invoke(prompt)

    final_answer = response.content

    return {
        "final_response": final_answer,
        "messages": [
            AIMessage(content=final_answer)
        ]
    }


def route_request(
    state: AssistantState
) -> Literal["explain", "plan", "write", "general"]:

    route = state["route"]

    if route == "explain":
        return "explain"

    if route == "plan":
        return "plan"

    if route == "write":
        return "write"

    return "general"