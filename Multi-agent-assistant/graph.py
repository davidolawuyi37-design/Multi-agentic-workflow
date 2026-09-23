from langgraph.graph import StateGraph, START, END
from state import AssistantState
from agents import(
    supervisor_agent,
    explain_agent,
    planner_agent,
    writer_agent,
    general_agent,
    review_agent,
    route_request
)

builder = StateGraph(AssistantState)

builder.add_node("supervisor", supervisor_agent)
builder.add_node("explain", explain_agent)
builder.add_node("plan", planner_agent)
builder.add_node("write", writer_agent)
builder.add_node("general", general_agent)
builder.add_node("review", review_agent)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_request,
    {
        "explain": "explain",
        "plan": "plan",
        "write": "write",
        "general": "general"
    }
)

builder.add_edge("explain", "review")
builder.add_edge("plan", "review")
builder.add_edge("write", "review")
builder.add_edge("general", "review")

builder.add_edge("review", END)

assistant_graph = builder.compile()

