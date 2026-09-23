from flask import Flask, request, jsonify
from src.chat import ask_question
import os

app = Flask(__name__)

# ==================================================
# CHAT MEMORY
# ==================================================

chat_history = []


# ==================================================
# HOME
# ==================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "Company RAG Chatbot API is running"
    })


# ==================================================
# CHAT API
# ==================================================

@app.route("/chat", methods=["POST"])
def chat():

    global chat_history

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Request body is required"
            }), 400

        question = data.get("question", "").strip()

        if not question:
            return jsonify({
                "error": "Question is required"
            }), 400

        # ------------------------------------------
        # Ask question
        # ------------------------------------------

        answer = ask_question(
            question,
            chat_history
        )

        # ------------------------------------------
        # Save conversation
        # ------------------------------------------

        chat_history.append({
            "question": question,
            "answer": answer
        })

        # ------------------------------------------
        # Keep last 10 conversations
        # ------------------------------------------

        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        return jsonify({
            "question": question,
            "answer": answer
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )