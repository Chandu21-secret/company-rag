from src.chat import ask_question


print("================================")
print("COMPANY RAG CHATBOT")
print("================================")
print("Type 'exit' to quit.")


# ==================================================
# CHAT MEMORY
# ==================================================

chat_history = []


# ==================================================
# CHAT LOOP
# ==================================================

while True:

    question = input("\nYou: ").strip()


    # Exit
    if question.lower() in [
        "exit",
        "quit"
    ]:

        print("Goodbye!")

        break


    # Empty input
    if not question:

        continue


    try:

        # ------------------------------------------
        # Ask question with history
        # ------------------------------------------

        answer = ask_question(
            question,
            chat_history
        )


        # ------------------------------------------
        # Print answer
        # ------------------------------------------

        print(
            "\nAI:",
            answer
        )


        # ------------------------------------------
        # Save conversation
        # ------------------------------------------

        chat_history.append({

            "question": question,

            "answer": answer
        })


        # ------------------------------------------
        # Keep last 10 turns only
        # ------------------------------------------

        if len(chat_history) > 10:

            chat_history = chat_history[-10:]


    except Exception as e:

        print(
            "\nError:",
            e
        )