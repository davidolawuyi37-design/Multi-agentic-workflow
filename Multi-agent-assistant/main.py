from graph import assistant_graph

print("=" * 50)
print("MULTI-AGENT AI ASSISTANT")
print("=" * 50)

print("Type 'exit' to stop.\n")


while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Assistant: Goodbye!")
        break

    result = assistant_graph.invoke({
        "user_input": user_input,
        "route": "",
        "specialist_response": "",
        "final_response": ""
    })

    print()
    print("Agent selected:", result["route"])
    print()
    print("Assistant:")
    print(result["final_response"])
    print()