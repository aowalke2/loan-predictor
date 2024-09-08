from flask import Flask, request
from modules.loan import LoanHandler


app = Flask(__name__)


@app.route("/api/loans/approve", methods=["POST"])
def approve():
    loan_data = request.get_json()
    handler = LoanHandler()
    result = handler.handle_request(loan_data)
    return "will pay loan" if result == 1 else "will default"


if __name__ == "__main__":
    # model = Model()
    # model.train()

    app.run(port=8000, debug=True)
