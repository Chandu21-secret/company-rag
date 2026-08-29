from flask import Flask, render_template, request, jsonify

from src.chat import ask_question


app = Flask(__name__)


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/ask",
    methods=["POST"]
)
def ask():

    data = request.get_json()

    question = (
        data.get(
            "question",
            ""
        )
        .strip()
    )

    history = data.get(
        "history",
        []
    )


    if not question:

        return jsonify({
            "answer": "Please enter a question."
        })


    try:

        answer = ask_question(
            question,
            history
        )


        return jsonify({
            "answer": answer
        })


    except Exception as e:

        return jsonify({
            "answer": f"Error: {e}"
        }), 500


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
