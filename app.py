from flask import Flask, request
from modules.loan import LoanHandler


app = Flask(__name__)


@app.route("/api/loans/approve", methods=["POST"])
def approve():
    loan_data = request.get_json()
    handler = LoanHandler()
    result = handler.handle_request(loan_data)
    return result


if __name__ == "__main__":
    # Un-comment to train
    # model = Model()
    # model.train()

    # Comment out to train
    app.run(port=8000, debug=True)
