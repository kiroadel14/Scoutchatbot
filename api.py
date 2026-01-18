from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot_logic import answer_question

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "")

    answer = answer_question(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
