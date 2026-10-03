from graph import AssistantState, assistant_graph
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig

print("=" * 50)
print("MULTI-AGENT AI ASSISTANT")
print("=" * 50)

print("Type 'exit' to stop.\n")

config: RunnableConfig = {
    "configurable": {
        "thread_id": "david-conversation"
    }
}


while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Assistant: Goodbye!")
        break


    result = assistant_graph.invoke(

        {
            "messages": [
                HumanMessage(content=user_input)
            ],

            "user_input": user_input,

            "route": "",

            "specialist_response": "",

            "final_response": ""
        },

        config=config
    )


    print()
    print("Agent selected:", result["route"])
    print()

    print("Assistant:")
    print(result["final_response"])

    print()