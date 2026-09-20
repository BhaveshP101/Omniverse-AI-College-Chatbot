from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

#  Import your existing chatbot logic
import chatbot  

app = Flask(__name__)
CORS(app)

# ---------------- HOME ROUTE ----------------
@app.route("/")
def home():
    return send_file("index.html")


# ---------------- CHAT API ----------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_msg = data.get("message", "")

        if not user_msg.strip():
            return jsonify({"response": "Please type something."})
        
        # Call your chatbot directly
        response = chatbot.chatbot_reply(user_msg)

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"response": "Error processing request."})


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(debug=True)