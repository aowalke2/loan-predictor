import pandas as pd
import numpy as np
import pickle
from modules.data import TrainingData

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from sklearn.ensemble import RandomForestClassifier


class Model:
    def __init__(self):
        self.scaler_file_name = "pickles/min_max_scaler.pickle"
        self.model_file_name = "pickles/random_forest.pickle"
        self.scaler = None
        self.model = None

    def train(self):
        print("Processing Data\n")
        lending_club = TrainingData()
        lending_club.process()
        data = lending_club.data

        print("Training Model\n")
        train, test = train_test_split(data, test_size=0.33, random_state=77)

        train = train[train["annual_inc"] <= 250000]
        train = train[train["dti"] <= 50]
        train = train[train["open_acc"] <= 40]
        train = train[train["total_acc"] <= 80]
        train = train[train["revol_util"] <= 120]
        train = train[train["revol_bal"] <= 250000]

        X_train, y_train = train.drop("loan_status", axis=1), train.loan_status
        X_test, y_test = test.drop("loan_status", axis=1), test.loan_status

        scaler = MinMaxScaler()
        pickle.dump(scaler, open(self.scaler_file_name, "wb"))
        self.scaler = scaler

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = np.array(X_train).astype(np.float32)
        X_test = np.array(X_test).astype(np.float32)
        y_train = np.array(y_train).astype(np.float32)
        y_test = np.array(y_test).astype(np.float32)

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        pickle.dump(model, open(self.model_file_name, "wb"))
        self.model = model

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # self.__print_score(y_train, y_train_pred, train=True)
        self.__print_score(y_test, y_test_pred, train=False)

    def predict(self, x):
        if not self.scaler:
            self.scaler = pickle.load(open(self.scaler_file_name, "rb"))
        x = self.scaler.transform(x)
        x = np.array(x).astype(np.float32)

        if not self.model:
            self.model = pickle.load(open(self.model_file_name, "rb"))
        return self.model.predict(x)

    def __print_score(self, true, pred, train=True):
        if train:
            clf_report = pd.DataFrame(
                classification_report(true, pred, output_dict=True)
            )
            print("Train Result:\n================================================")
            print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
            print("_______________________________________________")
            print(f"CLASSIFICATION REPORT:\n{clf_report}")
            print("_______________________________________________")
            print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")

        elif train == False:
            clf_report = pd.DataFrame(
                classification_report(true, pred, output_dict=True)
            )
            print("Test Result:\n================================================")
            print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
            print("_______________________________________________")
            print(f"CLASSIFICATION REPORT:\n{clf_report}")
            print("_______________________________________________")
            print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
