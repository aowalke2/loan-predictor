from flask import Flask, request, jsonify
from model import read_data, process_data

app = Flask(__name__)


@app.route("/api/loans/approve", methods=["POST"])
def approve():
    loan_data = request.get_json()
    return jsonify(loan_data)


if __name__ == "__main__":
    data = read_data()
    data = process_data(data)
    print(data.head())
    # app.run(debug=True)
