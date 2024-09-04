from flask import Flask, request, jsonify
from modules.model import Model


app = Flask(__name__)


@app.route("/api/loans/approve", methods=["POST"])
def approve():
    loan_data = request.get_json()
    return jsonify(loan_data)


if __name__ == "__main__":
    model = Model()
    model.train()
    # app.run(debug=True)
