from enum import Enum
from modules.model import Model
from collections import OrderedDict
import pandas as pd


class SubGrade(Enum):
    A2 = 0
    A3 = 1
    A4 = 2
    A5 = 3
    B1 = 4
    B2 = 5
    B3 = 6
    B4 = 7
    B5 = 8
    C1 = 9
    C2 = 10
    C3 = 11
    C4 = 12
    C5 = 13
    D1 = 14
    D2 = 15
    D3 = 16
    D4 = 17
    D5 = 18
    E1 = 19
    E2 = 20
    E3 = 21
    E4 = 22
    E5 = 23
    F1 = 24
    F2 = 25
    F3 = 26
    F4 = 27
    F5 = 28
    G1 = 29
    G2 = 30
    G3 = 31
    G4 = 32
    G5 = 33


class VerificationStatus(Enum):
    SourceVerified = 0
    Verified = 1


class Purpose(Enum):
    credit_card = 0
    debt_consolidation = 1
    educational = 2
    home_improvement = 3
    house = 4
    major_purchase = 5
    medical = 6
    moving = 7
    other = 8
    renewable_energy = 9
    small_business = 10
    vacation = 11
    wedding = 12


class ApplicationType(Enum):
    INDIVIDUAL = 0
    JOINT = 1


class HomeOwnership(Enum):
    OTHER = 0
    OWN = 1
    RENT = 2


class LoanHandler:
    def handle_request(self, loan_data):
        data_map = self.__build_request_data_map(loan_data)
        application = self.__build_request_dataframe(data_map)

        model = Model()
        return model.predict(application)

    def __build_request_data_map(self, loan_data):
        data = self.__get_data_map()
        for key in loan_data.keys():
            if key == "sub_grade":
                data_key = "sub_grade_" + SubGrade(loan_data[key]).name
                data[data_key] = 1
            elif key == "verification_status":
                if VerificationStatus(loan_data[key]) == 0:
                    data_key = "verification_status_Source Verified"
                else:
                    data_key = "verification_status_Verified"
                data[data_key] = 1
            elif key == "purpose":
                data_key = "purpose_" + Purpose(loan_data[key]).name
                data[data_key] = 1
            elif key == "application_type":
                data_key = "application_type_" + ApplicationType(loan_data[key]).name
                data[data_key] = 1
            elif key == "home_ownership":
                data_key = "home_ownership_" + HomeOwnership(loan_data[key]).name
                data[data_key] = 1
            else:
                data[key] = loan_data[key]
        return data

    def __build_request_dataframe(self, data_map):
        application = pd.DataFrame([data_map])

        sub_grade_columns = [col for col in application.columns if "sub_grade" in col]
        application[sub_grade_columns] = application[sub_grade_columns].astype(bool)

        verification_status_columns = [
            col for col in application.columns if "verification_status" in col
        ]
        application[verification_status_columns] = application[
            verification_status_columns
        ].astype(bool)

        purpose_columns = [col for col in application.columns if "purpose" in col]
        application[purpose_columns] = application[purpose_columns].astype(bool)

        application_type_columns = [
            col for col in application.columns if "application_type" in col
        ]
        application[application_type_columns] = application[
            application_type_columns
        ].astype(bool)

        home_ownership_columns = [
            col for col in application.columns if "home_ownership" in col
        ]
        application[home_ownership_columns] = application[
            home_ownership_columns
        ].astype(bool)
        return application

    def __get_data_map(self):
        keys = [
            "loan_amnt",
            "term",
            "int_rate",
            "installment",
            "annual_inc",
            "dti",
            "earliest_cr_line",
            "open_acc",
            "pub_rec",
            "revol_bal",
            "revol_util",
            "total_acc",
            "mort_acc",
            "pub_rec_bankruptcies",
            "sub_grade_A2",
            "sub_grade_A3",
            "sub_grade_A4",
            "sub_grade_A5",
            "sub_grade_B1",
            "sub_grade_B2",
            "sub_grade_B3",
            "sub_grade_B4",
            "sub_grade_B5",
            "sub_grade_C1",
            "sub_grade_C2",
            "sub_grade_C3",
            "sub_grade_C4",
            "sub_grade_C5",
            "sub_grade_D1",
            "sub_grade_D2",
            "sub_grade_D3",
            "sub_grade_D4",
            "sub_grade_D5",
            "sub_grade_E1",
            "sub_grade_E2",
            "sub_grade_E3",
            "sub_grade_E4",
            "sub_grade_E5",
            "sub_grade_F1",
            "sub_grade_F2",
            "sub_grade_F3",
            "sub_grade_F4",
            "sub_grade_F5",
            "sub_grade_G1",
            "sub_grade_G2",
            "sub_grade_G3",
            "sub_grade_G4",
            "sub_grade_G5",
            "verification_status_Source Verified",
            "verification_status_Verified",
            "purpose_credit_card",
            "purpose_debt_consolidation",
            "purpose_educational",
            "purpose_home_improvement",
            "purpose_house",
            "purpose_major_purchase",
            "purpose_medical",
            "purpose_moving",
            "purpose_other",
            "purpose_renewable_energy",
            "purpose_small_business",
            "purpose_vacation",
            "purpose_wedding",
            "initial_list_status_w",
            "application_type_INDIVIDUAL",
            "application_type_JOINT",
            "home_ownership_OTHER",
            "home_ownership_OWN",
            "home_ownership_RENT",
        ]
        data = OrderedDict()
        for key in keys:
            data[key] = 0
        return data
