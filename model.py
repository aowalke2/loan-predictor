import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC


def pub_rec(number):
    if number == 0.0:
        return 0
    else:
        return 1


def mort_acc(number):
    if number == 0.0:
        return 0
    elif number >= 1.0:
        return 1
    else:
        return number


def pub_rec_bankruptcies(number):
    if number == 0.0:
        return 0
    elif number >= 1.0:
        return 1
    else:
        return number


def fill_mort_acc(total_acc_avg, total_acc, mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc].round()
    else:
        return mort_acc


def read_data():
    data = pd.read_csv("lending_club_loan_two.csv")
    return data


def write_processed_data(data):
    data.to_csv("processed.csv")


def process_data(data):
    data["pub_rec"] = data.pub_rec.apply(pub_rec)
    data["mort_acc"] = data.mort_acc.apply(mort_acc)
    data["pub_rec_bankruptcies"] = data.pub_rec_bankruptcies.apply(pub_rec_bankruptcies)
    data["loan_status"] = data.loan_status.map({"Fully Paid": 1, "Charged Off": 0})

    data.drop("emp_title", axis=1, inplace=True)
    data.drop("emp_length", axis=1, inplace=True)
    data.drop("title", axis=1, inplace=True)

    total_acc_avg = data.groupby(by="total_acc").mean(numeric_only=True).mort_acc
    data["mort_acc"] = data.apply(
        lambda x: fill_mort_acc(total_acc_avg, x["total_acc"], x["mort_acc"]), axis=1
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

    data["zip_code"] = data.address.apply(lambda x: x[-5:])
    data = pd.get_dummies(data, columns=["zip_code"], drop_first=True)
    data.drop("address", axis=1, inplace=True)

    data.drop("issue_d", axis=1, inplace=True)

    data["earliest_cr_line"] = data["earliest_cr_line"].astype("datetime64[ns]").dt.year
    return data


def train(data):
    train, test = train_test_split(data, test_size=0.33, random_state=77)

    train = train[train["annual_inc"] <= 250000]
    train = train[train["dti"] <= 50]
    train = train[train["open_acc"] <= 40]
    train = train[train["total_acc"] <= 80]
    train = train[train["revol_util"] <= 120]
    train = train[train["revol_bal"] <= 250000]
