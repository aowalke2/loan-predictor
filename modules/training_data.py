import pandas as pd
import numpy as np


class TrainingData:
    def __init__(self):
        self.processed = False
        self.data = None

    def process(self):
        data = pd.read_csv("lending_club_loan_two.csv")
        data.loc[
            (data.home_ownership == "ANY") | (data.home_ownership == "NONE"),
            "home_ownership",
        ] = "OTHER"

        data["pub_rec"] = data.pub_rec.apply(self.__pub_rec)
        data["mort_acc"] = data.mort_acc.apply(self.__mort_acc)
        data["pub_rec_bankruptcies"] = data.pub_rec_bankruptcies.apply(
            self.__pub_rec_bankruptcies
        )
        data["loan_status"] = data.loan_status.map({"Fully Paid": 1, "Charged Off": 0})

        data.drop("emp_title", axis=1, inplace=True)
        data.drop("emp_length", axis=1, inplace=True)
        data.drop("title", axis=1, inplace=True)

        total_acc_avg = data.groupby(by="total_acc").mean(numeric_only=True).mort_acc
        data["mort_acc"] = data.apply(
            lambda x: self.__fill_mort_acc(
                total_acc_avg, x["total_acc"], x["mort_acc"]
            ),
            axis=1,
        )
        data.dropna(inplace=True)

        term_values = {" 36 months": 36, " 60 months": 60}
        data["term"] = data.term.map(term_values)

        data.drop("grade", axis=1, inplace=True)
        dummies = [
            "sub_grade",
            "verification_status",
            "purpose",
            "initial_list_status",
            "application_type",
            "home_ownership",
        ]
        data = pd.get_dummies(data, columns=dummies, drop_first=True)

        # data["zip_code"] = data.address.apply(lambda x: x[-5:])
        # data = pd.get_dummies(data, columns=["zip_code"], drop_first=True)
        data.drop("address", axis=1, inplace=True)

        data.drop("issue_d", axis=1, inplace=True)

        data["earliest_cr_line"] = (
            data["earliest_cr_line"].astype("datetime64[ns]").dt.year
        )

        self.data = data
        self.processed = True

    def write_processed(self):
        if not self.processed:
            return
        self.data.to_csv("processed.csv")

    def __pub_rec(self, number):
        if number == 0.0:
            return 0
        else:
            return 1

    def __mort_acc(self, number):
        if number == 0.0:
            return 0
        elif number >= 1.0:
            return 1
        else:
            return number

    def __pub_rec_bankruptcies(self, number):
        if number == 0.0:
            return 0
        elif number >= 1.0:
            return 1
        else:
            return number

    def __fill_mort_acc(self, total_acc_avg, total_acc, mort_acc):
        if np.isnan(mort_acc):
            return total_acc_avg[total_acc].round()
        else:
            return mort_acc
