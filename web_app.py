from flask import Flask, render_template, request, jsonify

from src.chat import ask_question


# ==================================================
# FLASK APP
# ==================================================

app = Flask(__name__)


# ==================================================
# HOME
# ==================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==================================================
# ASK
# ==================================================

@app.route(
    "/ask",
    methods=["POST"]
)
def ask():

    try:

        # ------------------------------------------
        # Read JSON
        # ------------------------------------------

        data = request.get_json(
            silent=True
        )


        if not isinstance(data, dict):

            return jsonify({
                "answer": "Invalid request."
            }), 400


        # ------------------------------------------
        # Question
        # ------------------------------------------

        question = str(

            data.get(
                "question",
                ""
            )

        ).strip()


        # ------------------------------------------
        # Chat history
        # ------------------------------------------

        history = data.get(
            "history",
            []
        )


        if not isinstance(
            history,
            list
        ):

            history = []


        # ------------------------------------------
        # Empty question
        # ------------------------------------------

        if not question:

            return jsonify({
                "answer": "Please enter a question."
            }), 400


        # ------------------------------------------
        # Keep only recent history
        #
        # This prevents unnecessarily large
        # requests and improves response speed.
        # ------------------------------------------

        history = history[-5:]


        # ------------------------------------------
        # Ask AI
        # ------------------------------------------

        print(
            f"QUESTION: {question}",
            flush=True
        )


        answer = ask_question(

            question,

            history
        )


        # ------------------------------------------
        # Empty response protection
        # ------------------------------------------

        if answer is None:

            answer = (
                "I could not generate an answer. "
                "Please try again."
            )


        answer = str(
            answer
        ).strip()


        if not answer:

            answer = (
                "I could not generate an answer. "
                "Please try again."
            )


        print(
            f"ANSWER: {answer}",
            flush=True
        )


        return jsonify({

            "answer": answer

        }), 200


    # ==================================================
    # ERROR HANDLING
    # ==================================================

    except Exception as e:

        import traceback


        print(
            "",
            flush=True
        )

        print(
            "================================",
            flush=True
        )

        print(
            "WEB APP ERROR",
            flush=True
        )

        print(
            "================================",
            flush=True
        )

        print(
            "ERROR:",
            repr(e),
            flush=True
        )


        traceback.print_exc()


        print(
            "================================",
            flush=True
        )


        return jsonify({

            "answer": (
                "Something went wrong. "
                "Please try again."
            )

        }), 500


# ==================================================
# HEALTH CHECK
# ==================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status": "ok"

    }), 200


# ==================================================
# RUN LOCALLY
# ==================================================

if __name__ == "__main__":

    print()
    print(
        "================================"
    )
    print(
        "COMPANY AI ASSISTANT"
    )
    print(
        "================================"
    )
    print(
        "Server:"
    )
    print(
        "http://127.0.0.1:5000"
    )
    print(
        "================================"
    )
    print()


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False,

        threaded=True
    )