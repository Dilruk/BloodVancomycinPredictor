# TODO: New Dataset 2022
import copy
import numpy as np
import argparse
import pandas as pd
from sklearn import preprocessing

def parse_args():
    parser = argparse.ArgumentParser(description="Run Neural FM.")
    parser.add_argument('--path', nargs='?', default='data_2',
                        help='Input data path.')
    parser.add_argument('--loss_type', nargs='?', default='square_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    return parser.parse_args()


class LoadData(object):
    # Three files are needed in the path
    def __init__(self, path):
        self.path = path + "/"
        self.datafile = self.path + "all_data_04feb.csv"

        self.bwd_label_scalar = 0.01
        # prev_dosage

        df_total = pd.read_csv(self.datafile)

        # TODO: Normalize the data
        df_normalized, user_features, fwd_features, fwd_label, bwd_features, bwd_label = self.normalize_data(df_total)

        # TODO: Divide the train, test, validation after normalizing
        df_train, df_valid, df_test = self.get_train_valid_test_data(df_normalized, 0.7, 0.2)

        # TODO: Get forward and backward features, labels, etc. for train, validation and test sets
        self.train_fwd_user_features, self.train_df_fwd_features, self.train_fwd_labels, self.train_fwd_subject_ids, self.train_fwd_hadm_ids = self.get_fwd_data(df_train.copy(), user_features, fwd_features, fwd_label)
        self.train_bwd_user_features, self.train_df_bwd_features, self.train_bwd_labels, self.train_bwd_subject_ids, self.train_bwd_hadm_ids = self.get_bwd_data(df_train.copy(), user_features, bwd_features, bwd_label)

        self.valid_fwd_user_features, self.valid_df_fwd_features, self.valid_fwd_labels, self.valid_fwd_subject_ids, self.valid_fwd_hadm_ids = self.get_fwd_data(df_valid.copy(), user_features, fwd_features, fwd_label)
        self.valid_bwd_user_features, self.valid_df_bwd_features, self.valid_bwd_labels, self.valid_bwd_subject_ids, self.valid_bwd_hadm_ids = self.get_bwd_data(df_valid.copy(), user_features, bwd_features, bwd_label)

        self.test_fwd_user_features, self.test_df_fwd_features, self.test_fwd_labels, self.test_fwd_subject_ids, self.test_fwd_hadm_ids = self.get_fwd_data(df_test.copy(), user_features, fwd_features, fwd_label)
        self.test_bwd_user_features, self.test_df_bwd_features, self.test_bwd_labels, self.test_bwd_subject_ids, self.test_bwd_hadm_ids = self.get_bwd_data(df_test.copy(), user_features, bwd_features, bwd_label)

        self.Train_data, self.Validation_data, self.Test_data = self.finalize_data()
        self.no_userstate_features = len(self.Train_data['FWD_User_X'][0])
        self.no_forward_features = len(self.Validation_data['FWD_X'][0])
        self.no_backward_features = len(self.Test_data['BWD_X'][0])
        print('Loaded Data')

    def get_fwd_data(self, df, user_features, fwd_features, fwd_label):

        # Create Forward prediction instances
        df_fwd = copy.deepcopy(df)
        # remove instance where the next is a D type event, or if this is the lsat step (so should be a V at the next step).
        df_fwd = df_fwd[df_fwd['next_event_type'] == 'V']
        # remove instances where a dosage is not influencing the V. There are some cases where no D was recorded before a V. So should remove them
        df_fwd = df_fwd[(df_fwd['prev_dose_amount'] + df_fwd['second_last_dose_amount']) > 0]

        df_fwd = copy.deepcopy(df_fwd.reset_index(drop=True))

        return df_fwd[user_features], df_fwd[fwd_features], df_fwd[fwd_label], df_fwd.subject_id, df_fwd.hadm_id

    def get_bwd_data(self, df, user_features, bwd_features, bwd_label):

        # Create Forward prediction instances
        df_bwd = copy.deepcopy(df)
        # remove instance where the next is a D type event, or if this is the lsat step (so should be a V at the next step).
        df_bwd = df_bwd[df_bwd['event_type'] == 'V']
        # remove instances where a dosage is not influencing the V. There are some cases where no D was recorded before a V. So should remove them
        df_bwd = df_bwd[df_bwd['prev_dose_amount'] > 0]

        df_bwd = copy.deepcopy(df_bwd.reset_index(drop=True))

        return df_bwd[user_features], df_bwd[bwd_features], df_bwd[bwd_label], df_bwd.subject_id, df_bwd.hadm_id

    def get_train_valid_test_data(self, df_total, train_prob=0.7, valid_prob=0.1):
        hadm_all = list(df_total['hadm_id'].unique())
        len_hadm_all = len(hadm_all)
        train_hadm_ids = np.random.choice(hadm_all, int(train_prob * len_hadm_all), replace=False)
        hadm_list_temp = [i for i in hadm_all if i not in train_hadm_ids]
        valid_hadm_ids = np.random.choice(hadm_list_temp, int(valid_prob * len_hadm_all), replace=False)
        test_hadm_ids = [i for i in hadm_list_temp if i not in valid_hadm_ids]

        df_train = df_total[df_total['hadm_id'].isin(train_hadm_ids)]
        df_valid = df_total[df_total['hadm_id'].isin(valid_hadm_ids)]
        df_test = df_total[df_total['hadm_id'].isin(test_hadm_ids)]

        return df_train, df_valid, df_test

    def finalize_data(self):
        Train_data, Validation_data, Test_data = {}, {}, {}

        Train_data['FWD_User_X'] = self.train_fwd_user_features.values.tolist()        # data to determine user state at t
        Train_data['FWD_X'] = self.train_df_fwd_features.values.tolist()                       # other non-patient specific data needed for the forward prediction. E.g. dosage, time gap from t to t+1
        Train_data['FWD_Y'] = self.train_fwd_labels.values.tolist()                       # forward prediction label - vancomycin level at t+1
        Train_data['FWD_subject_ids'] = self.train_fwd_subject_ids.values.tolist()
        Train_data['FWD_hadm_ids'] = self.train_fwd_hadm_ids.values.tolist()
        Train_data['BWD_User_X'] = self.train_bwd_user_features.values.tolist()       # data to determine user state at t+1
        Train_data['BWD_X'] = self.train_df_bwd_features.values.tolist()                      # other non-patient specific data needed for the backward prediction. E.g. time gap
        Train_data['BWD_Y'] = self.train_bwd_labels.values.tolist()
        Train_data['BWD_subject_ids'] = self.train_bwd_subject_ids.values.tolist()
        Train_data['BWD_hadm_ids'] = self.train_bwd_hadm_ids.values.tolist()

        Validation_data['FWD_User_X'] = self.valid_fwd_user_features.values.tolist()
        Validation_data['FWD_X'] = self.valid_df_fwd_features.values.tolist()
        Validation_data['FWD_Y'] = self.valid_fwd_labels.values.tolist()
        Validation_data['FWD_subject_ids'] = self.valid_fwd_subject_ids.values.tolist()
        Validation_data['FWD_hadm_ids'] = self.valid_fwd_hadm_ids.values.tolist()
        Validation_data['BWD_User_X'] = self.valid_bwd_user_features.values.tolist()
        Validation_data['BWD_X'] = self.valid_df_bwd_features.values.tolist()
        Validation_data['BWD_Y'] = self.valid_bwd_labels.values.tolist()
        Validation_data['BWD_subject_ids'] = self.valid_bwd_subject_ids.values.tolist()
        Validation_data['BWD_hadm_ids'] = self.valid_bwd_hadm_ids.values.tolist()

        Test_data['FWD_User_X'] = self.test_fwd_user_features.values.tolist()
        Test_data['FWD_X'] = self.test_df_fwd_features.values.tolist()
        Test_data['FWD_Y'] = self.test_fwd_labels.values.tolist()
        Test_data['FWD_subject_ids'] = self.test_fwd_subject_ids.values.tolist()
        Test_data['FWD_hadm_ids'] = self.test_fwd_hadm_ids.values.tolist()
        Test_data['BWD_User_X'] = self.test_bwd_user_features.values.tolist()
        Test_data['BWD_X'] = self.test_df_bwd_features.values.tolist()
        Test_data['BWD_Y'] = self.test_bwd_labels.values.tolist()
        Test_data['BWD_subject_ids'] = self.test_bwd_subject_ids.values.tolist()
        Test_data['BWD_hadm_ids'] = self.test_bwd_hadm_ids.values.tolist()

        return Train_data, Validation_data, Test_data

    def normalize_data(self, df_features):
        # Remove events with 0 corresponding values to avoid possible errornous data capturing. E.g., D type with 0 dosage

        df_features = df_features[~((df_features['event_type'] == 'D') & (df_features['prev_dose_amount'] == 0) & (df_features['time_since_last_dosage'] == 0))]
        df_features = df_features[~((df_features['event_type'] == 'V') & (df_features['current_vanco_level'] == 0))]

        # replace the 0.0 values in time since last dosage from a small value. Because they are not missing values, but dosage given at those times.
        df_features['time_since_last_dosage'] = df_features['time_since_last_dosage'].replace(0.0, 1.0)
        df_features['time_since_last_vanco'] = df_features['time_since_last_vanco'].replace(0.0, 1.0)
        df_features['time_since_second_last_vanco'] = df_features['time_since_second_last_vanco'].replace(0.0, 1.0)
        df_features['time_since_third_last_vanco'] = df_features['time_since_third_last_vanco'].replace(0.0, 1.0)
        df_features['time_since_second_last_dose'] = df_features['time_since_second_last_dose'].replace(0.0, 1.0)
        df_features['time_since_prev_bun_creatinine'] = df_features['time_since_prev_bun_creatinine'].replace(0.0, 1.0)

        # Delete rows with nan values
        df = df_features.fillna(0)

        # STEP 1: Converting all categorical variables to one hot encoding
        categorical_columns = ['gender', 'insurance', 'marital_status', 'ethnicity']

        gender_onehot_c = pd.get_dummies(df['gender'], prefix='gender')
        insurance_onehot_c = pd.get_dummies(df['insurance'], prefix='insurance')
        marital_status_onehot_c = pd.get_dummies(df['marital_status'], prefix='marital_status')
        ethnicity_onehot_c = pd.get_dummies(df['ethnicity'], prefix='ethnicity')
        categorical_features = list(gender_onehot_c.columns) + (list(insurance_onehot_c.columns)) + (list(marital_status_onehot_c.columns)) + (list(ethnicity_onehot_c.columns))

        df = pd.concat([df.reset_index(drop=True), gender_onehot_c.reset_index(drop=True), insurance_onehot_c.reset_index(drop=True), marital_status_onehot_c.reset_index(drop=True), ethnicity_onehot_c.reset_index(drop=True)], axis=1, sort=False)

        # STEP 2: Normalizing all numerical variables
        numerical_columns = ['time_since_last_dosage', 'prev_dose_amount', 'next_vanco_level', 'time_to_next_vanco', 'time_since_last_vanco', 'prev_vanco_level', 'second_last_vanco_amount', 'time_since_second_last_vanco', 'third_last_vanco_amount', 'time_since_third_last_vanco', 'second_last_dose_amount', 'time_since_second_last_dose', 'current_vanco_level', 'prev_bun_amount', 'prev_creatinine_amount', 'time_since_prev_bun_creatinine', 'age', 'average_weight', 'height', 'age_score', 'charlson_comorbidity_index']
        df[numerical_columns] = self.normalize_df(df[numerical_columns])

        # STEP 3: Convert boolean columns
        boolean_columns = ['dosage_btw_current_and_last_vanco_level', 'dosage_btw_current_and_second_last_vanco_level', 'dosage_btw_current_and_third_last_vanco_level']
        df[boolean_columns] = df[boolean_columns].replace(True, 1.0)
        df[boolean_columns] = df[boolean_columns].replace(False, 0.0)

        # STEP 4: binary columns (1/0)
        binary_columns = ['myocardial_infarct', 'congestive_heart_failure', 'peripheral_vascular_disease', 'cerebrovascular_disease', 'dementia', 'chronic_pulmonary_disease', 'rheumatic_disease', 'peptic_ulcer_disease', 'mild_liver_disease', 'diabetes_without_cc', 'diabetes_with_cc', 'paraplegia', 'renal_disease', 'malignant_cancer', 'severe_liver_disease', 'metastatic_solid_tumor', 'aids']
        df[binary_columns] = 1.0 * df[binary_columns]

        all_features = categorical_features + numerical_columns + boolean_columns + binary_columns

        fwd_label = ['next_vanco_level']
        user_features = ['prev_bun_amount', 'prev_creatinine_amount', 'time_since_prev_bun_creatinine', 'age', 'average_weight', 'height', 'age_score', 'charlson_comorbidity_index'] + binary_columns
        temp = copy.deepcopy(fwd_label) + (copy.deepcopy(user_features))
        fwd_features = list(copy.deepcopy(set(all_features).difference(set(temp))))

        bwd_features = ['time_since_last_dosage', 'next_vanco_level', 'time_to_next_vanco', 'current_vanco_level']
        bwd_label = ['prev_dose_amount']

        return df, user_features, fwd_features, fwd_label, bwd_features, bwd_label

    def normalize_df(self, df):
        x = df.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)
        return df


if __name__ == '__main__':

    args = parse_args()
    data = LoadData(args.path)
