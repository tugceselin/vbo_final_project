import streamlit as st
import pandas as pd
import numpy as np
import category_encoders as ce
from streamlit_option_menu import option_menu
from PIL import Image
import streamlit as st
import pickle
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
# !pip install lightgbm
from lightgbm import LGBMClassifier
import itertools
from matplotlib import rc, rcParams
from matplotlib import cm
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

image = Image.open('/Users/tugceselin/PycharmProjects/dsmlbc_9/vbo_project_disease_prediction_ml/images/company_logo.jpeg')
st.image(image, width=330)


df_lung = pd.read_csv("/Users/tugceselin/PycharmProjects/dsmlbc_9/vbo_project_disease_prediction_ml/survey lung cancer.csv")
df_heart = pd.read_csv("/Users/tugceselin/PycharmProjects/dsmlbc_9/vbo_project_disease_prediction_ml/heart.csv")


# LUNG CANCER
##############
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
	quartile1 = dataframe[col_name].quantile(q1)
	quartile3 = dataframe[col_name].quantile(q3)
	interquantile_range = quartile3 - quartile1
	up_limit = quartile3 + 1.5 * interquantile_range
	low_limit = quartile1 - 1.5 * interquantile_range
	return low_limit, up_limit

outlier_thresholds(df_lung, "AGE")

def replace_with_thresholds(dataframe, variable):
	low_limit, up_limit = outlier_thresholds(dataframe, variable)
	dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
	dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df_lung, "AGE")

def age_prep(dataframe):
	dataframe.loc[(dataframe['AGE'] < 30), "RISK_AGE_DIM"] = 'LOW RISK'
	dataframe.loc[(dataframe['AGE'] >= 30) & (dataframe['AGE'] <= 45), "RISK_AGE_DIM"] = 'MID RISK'
	dataframe.loc[(dataframe['AGE'] > 45), "RISK_AGE_DIM"] = 'HIGH RISK'

age_prep(df_lung)

def change_types(df):
	df.loc[(df['LUNG_CANCER'] == "YES"), 'LUNG_CANCER'] = 1
	df.loc[(df['LUNG_CANCER'] == "NO"), 'LUNG_CANCER'] = 0
	df["LUNG_CANCER"] = df["LUNG_CANCER"].astype("int64")

change_types(df_lung)

def grab_col_names(dataframe, cat_th=10, car_th=20):
	"""

	Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
	Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

	Parameters
	------
		dataframe: dataframe
				Değişken isimleri alınmak istenilen dataframe
		cat_th: int, optional
				numerik fakat kategorik olan değişkenler için sınıf eşik değeri
		car_th: int, optinal
				kategorik fakat kardinal değişkenler için sınıf eşik değeri

	Returns
	------
		cat_cols: list
				Kategorik değişken listesi
		num_cols: list
				Numerik değişken listesi
		cat_but_car: list
				Kategorik görünümlü kardinal değişken listesi

	Examples
	------
		import seaborn as sns
		df = sns.load_dataset("iris")
		print(grab_col_names(df))


	Notes
	------
		cat_cols + num_cols + cat_but_car = toplam değişken sayısı
		num_but_cat cat_cols'un içerisinde.
		Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

	"""
	
	# cat_cols, cat_but_car
	cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
	num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
	               dataframe[col].dtypes != "O"]
	cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
	               dataframe[col].dtypes == "O"]
	cat_cols = cat_cols + num_but_cat
	cat_cols = [col for col in cat_cols if col not in cat_but_car]
	
	# num_cols
	num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
	num_cols = [col for col in num_cols if col not in num_but_cat]
	
	# print(f"Observations: {dataframe.shape[0]}")
	# print(f"Variables: {dataframe.shape[1]}")
	# print(f'cat_cols: {len(cat_cols)}')
	# print(f'num_cols: {len(num_cols)}')
	# print(f'cat_but_car: {len(cat_but_car)}')
	# print(f'num_but_cat: {len(num_but_cat)}')
	return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df_lung)

def create_new_features(df):
	df.loc[(df["ALCOHOL CONSUMING"] + df["FATIGUE "] == 4), "STRESSED_PERSON"] = "YES"
	df.loc[(df["ALCOHOL CONSUMING"] + df["FATIGUE "] < 4), "STRESSED_PERSON"] = "NO"
	df.loc[(df["ALCOHOL CONSUMING"] + df["SMOKING"] == 4), "IS_HEALHTY"] = "NO"
	df.loc[(df["ALCOHOL CONSUMING"] + df["SMOKING"] < 4), "IS_HEALHTY"] = "YES"
	df.loc[(df["SMOKING"] + df["WHEEZING"] == 4), "BREATHING_PROBLEM"] = "YES"
	df.loc[(df["SMOKING"] + df["WHEEZING"] < 4), "BREATHING_PROBLEM"] = "NO"
	df.loc[(df["SMOKING"] == 1) & (df["ALCOHOL CONSUMING"] == 2), "SECONDHAND_SMOKER"] = "YES"
	df.loc[(df["SMOKING"] == 2) & (df["ALCOHOL CONSUMING"] == 1), "SECONDHAND_SMOKER"] = "NO"
	df.loc[(df["SMOKING"] == 1) & (df["ALCOHOL CONSUMING"] == 1), "SECONDHAND_SMOKER"] = "NO"
	df.loc[(df["SMOKING"] == 2) & (df["ALCOHOL CONSUMING"] == 2), "SECONDHAND_SMOKER"] = "NO"

create_new_features(df_lung)

def label_encoder_prep(dataframe, binary_col):
	labelencoder = LabelEncoder()
	dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
	return dataframe

binary_cols = [col for col in df_lung.columns if df_lung[col].nunique() == 2]
len(binary_cols)

for col in binary_cols:
	label_encoder_prep(df_lung, col)
rs = RobustScaler()
df_lung["AGE"] = rs.fit_transform(df_lung[["AGE"]])

# HEART ATTACK
##############

def thall():
	for i in range(len(df_heart["thall"])):
		if df_heart["thall"][i] == 0:
			df_heart["thall"][i] = 2
	for i in range(len(df_heart["thall"])):
		if df_heart["thall"][i] == 1:
			df_heart["thall"][i] = 0
		elif df_heart["thall"][i] == 2:
			df_heart["thall"][i] = 1
		elif df_heart["thall"][i] == 3:
			df_heart["thall"][i] = 2

thall()

def print_info(dataframe):
	info = pd.DataFrame(columns=["Variables", "n_Valid", "n_Missing", "Type", "Unique_observations"])
	info["Variables"] = pd.DataFrame(dataframe.columns)
	
	for i in range(len(info["n_Valid"])):
		info["n_Valid"][i] = dataframe[dataframe.columns[i]].notnull().sum()
	
	for i in range(len(info["n_Missing"])):
		info["n_Missing"][i] = dataframe[dataframe.columns[i]].isnull().sum()
	
	for i in range(len(info["Type"])):
		info["Type"][i] = dataframe[dataframe.columns[i]].dtype
	
	for i in range(len(info["Unique_observations"])):
		info["Unique_observations"][i] = dataframe[dataframe.columns[i]].nunique()
	
	return info

def data_description(dataframe):
	for i in range(len(dataframe)):
		if dataframe["cp"][i] == 0:
			dataframe["cp"][i] = "0:typical angina"
		elif dataframe["cp"][i] == 1:
			dataframe["cp"][i] = "1:atypical angina"
		elif dataframe["cp"][i] == 2:
			dataframe["cp"][i] = "2:non-anginal pain"
		elif dataframe["cp"][i] == 3:
			dataframe["cp"][i] = "3:asymptomatic"
	dataframe["sex"] = ["1:m" if dataframe["sex"][i] == 1 else "0:f" for i in range(len(dataframe))]
	dataframe["fbs"] = ["1:true" if dataframe["fbs"][i] == 1 else "0:false" for i in range(len(dataframe))]
	dataframe["exng"] = ["1:yes" if dataframe["exng"][i] == 1 else "0:no" for i in range(len(dataframe))]
	for i in range(len(dataframe)):
		if dataframe["slp"][i] == 0:
			dataframe["slp"][i] = "0:downsloping"
		elif dataframe["slp"][i] == 1:
			dataframe["slp"][i] = "1:flat"
		elif dataframe["slp"][i] == 2:
			dataframe["slp"][i] = "2:upsloping"
	for i in range(len(dataframe)):
		if dataframe["thall"][i] == 0:
			dataframe["thall"][i] = "0:fixed defect"
		elif dataframe["thall"][i] == 1:
			dataframe["thall"][i] = "1:normal"
		elif dataframe["thall"][i] == 2:
			dataframe["thall"][i] = "2:reversable defect"
	for i in range(len(dataframe)):
		if dataframe["restecg"][i] == 0:
			dataframe["restecg"][i] = "0:left ventricular hypertrophy"
		elif dataframe["restecg"][i] == 1:
			dataframe["restecg"][i] = "1:normal"
		elif dataframe["restecg"][i] == 2:
			dataframe["restecg"][i] = "2:ST-T wave abnormality"
	for i in range(len(dataframe)):
		if dataframe["cp"][i] == 0:
			dataframe["cp"][i] = "0: asymptomatic"
		elif dataframe["cp"][i] == 1:
			dataframe["cp"][i] = " 1: atypical angina"
		elif dataframe["cp"][i] == 2:
			dataframe["cp"][i] = "2: non-anginal pain"
		elif dataframe["cp"][i] == 3:
			dataframe["cp"][i] = "3: typical angina"
	dataframe["output"] = [
		"0:More chance of heart disease" if dataframe["output"][i] == 0 else "1:Less chance of heart disease" for i in
		range(len(dataframe))]
	
	heart_info = {"age": ["Age of the patient", "-"],
	              "sex": ["Sex of the patient", "0:female, 1:male"],
	              "cp": ["Chest pain type", "0:asymptomatic, 1:atypical angina, 2:non-anginal pain, 3:typical angina"],
	              "trtbps": ["Resting blood pressure (in mm Hg on admission to the hospital)", "-"],
	              "chol": ["Cholesterol measurement in mg/dl", "-"],
	              "fbs": ["(fasting blood sugar > 120 mg/dl)", "1:true, 0:false"],
	              "restecg": ["Resting electrocardiographic results",
	                          "0:showing probable or definite left ventricular hypertrophy by Estes' criteria, 1:normal, 2:having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)"],
	              "thalachh": ["Maximum heart rate achieved", "-"],
	              "exng": ["Exercise induced angina", "1:yes, 0:no"],
	              "oldpeak": ["ST depression induced by exercise relative to rest", "-"],
	              "slp": ["The slope of the peak exercise ST segment", "0:downsloping, 1:flat, 2:upsloping"],
	              "caa": ["Number of major vessels (0-3) colored by flourosopy", "-"],
	              "thall": ["Thalassemia", "0:fixed defect, 1:normal, 2:reversable defect"],
	              "output": "Diagnosis of heart disease (angiographic disease status)"}
	dataf = print_info(dataframe)
	datad = pd.DataFrame(heart_info).T
	datad.reset_index(inplace=True)
	datad.columns = ["Variables", "Description", "Values"]
	data = pd.merge(dataf, datad, on="Variables")
	
	return data

data = data_description(df_heart)

# CATBOOST

##############################################################################################
# CatBoost
###############################################################################################
dff = pd.read_csv("/Users/tugceselin/PycharmProjects/dsmlbc_9/vbo_project_disease_prediction_ml/heart.csv")

import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve, \
    roc_curve
import category_encoders as ce
from scipy.stats import shapiro

import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def thall():
    for i in range(len(dff["thall"])):
        if dff["thall"][i] == 0:
            dff["thall"][i] = 2
    for i in range(len(dff["thall"])):
        if dff["thall"][i] == 1:
            dff["thall"][i] = 0
        elif dff["thall"][i] == 2:
            dff["thall"][i] = 1
        elif dff["thall"][i] == 3:
            dff["thall"][i] = 2
thall()

df_2 = dff.copy()


def print_info(dataframe):
    info = pd.DataFrame(columns=["Variables", "n_Valid", "n_Missing", "Type", "Unique_observations"])
    info["Variables"] = pd.DataFrame(dataframe.columns)

    for i in range(len(info["n_Valid"])):
        info["n_Valid"][i] = dataframe[dataframe.columns[i]].notnull().sum()

    for i in range(len(info["n_Missing"])):
        info["n_Missing"][i] = dataframe[dataframe.columns[i]].isnull().sum()

    for i in range(len(info["Type"])):
        info["Type"][i] = dataframe[dataframe.columns[i]].dtype

    for i in range(len(info["Unique_observations"])):
        info["Unique_observations"][i] = dataframe[dataframe.columns[i]].nunique()

    return info

def data_description(dataframe):
    for i in range(len(df_2)):
        if dataframe["cp"][i] == 0:
            dataframe["cp"][i] = "0:typical angina"
        elif dataframe["cp"][i] == 1:
            dataframe["cp"][i] = "1:atypical angina"
        elif dataframe["cp"][i] == 2:
            dataframe["cp"][i] = "2:non-anginal pain"
        elif dataframe["cp"][i] == 3:
            dataframe["cp"][i] = "3:asymptomatic"
    dataframe["sex"] = ["1:m" if dataframe["sex"][i] == 1 else "0:f" for i in range(len(df_2))]
    dataframe["fbs"] = ["1:true" if dataframe["fbs"][i] == 1 else "0:false" for i in range(len(df_2))]
    dataframe["exng"] = ["1:yes" if dataframe["exng"][i] == 1 else "0:no" for i in range(len(df_2))]
    for i in range(len(df_2)):
        if dataframe["slp"][i] == 0:
            dataframe["slp"][i] = "0:downsloping"
        elif dataframe["slp"][i] == 1:
            dataframe["slp"][i] = "1:flat"
        elif dataframe["slp"][i] == 2:
            dataframe["slp"][i] = "2:upsloping"
    for i in range(len(df_2)):
        if dataframe["thall"][i] == 0:
            dataframe["thall"][i] = "0:fixed defect"
        elif dataframe["thall"][i] == 1:
            dataframe["thall"][i] = "1:normal"
        elif dataframe["thall"][i] == 2:
            dataframe["thall"][i] = "2:reversable defect"
    for i in range(len(df_2)):
        if dataframe["restecg"][i] == 0:
            dataframe["restecg"][i] = "0:left ventricular hypertrophy"
        elif dataframe["restecg"][i] == 1:
            dataframe["restecg"][i] = "1:normal"
        elif dataframe["restecg"][i] == 2:
            dataframe["restecg"][i] = "2:ST-T wave abnormality"
    for i in range(len(df_2)):
        if dataframe["cp"][i] == 0:
            dataframe["cp"][i] = "0: asymptomatic"
        elif dataframe["cp"][i] == 1:
            dataframe["cp"][i] = " 1: atypical angina"
        elif dataframe["cp"][i] == 2:
            dataframe["cp"][i] = "2: non-anginal pain"
        elif dataframe["cp"][i] == 3:
            dataframe["cp"][i] = "3: typical angina"
    dataframe["output"] = ["0:More chance of heart disease" if dataframe["output"][i] == 0 else "1:Less chance of heart disease" for i in range(len(df_2))]

    heart_info = {"age": ["Age of the patient", "-"],
                  "sex": ["Sex of the patient", "0:female, 1:male"],
                  "cp": ["Chest pain type", "0:asymptomatic, 1:atypical angina, 2:non-anginal pain, 3:typical angina"],
                  "trtbps": ["Resting blood pressure (in mm Hg on admission to the hospital)", "-"],
                  "chol": ["Cholesterol measurement in mg/dl", "-"],
                  "fbs": ["(fasting blood sugar > 120 mg/dl)", "1:true, 0:false"],
                  "restecg": ["Resting electrocardiographic results",
                              "0:showing probable or definite left ventricular hypertrophy by Estes' criteria, 1:normal, 2:having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)"],
                  "thalachh": ["Maximum heart rate achieved", "-"],
                  "exng": ["Exercise induced angina", "1:yes, 0:no"],
                  "oldpeak": ["ST depression induced by exercise relative to rest", "-"],
                  "slp": ["The slope of the peak exercise ST segment", "0:downsloping, 1:flat, 2:upsloping"],
                  "caa": ["Number of major vessels (0-3) colored by flourosopy", "-"],
                  "thall": ["Thalassemia", "0:fixed defect, 1:normal, 2:reversable defect"],
                  "output": "Diagnosis of heart disease (angiographic disease status)"}
    dataf = print_info(dataframe)
    datad = pd.DataFrame(heart_info).T
    datad.reset_index(inplace=True)
    datad.columns = ["Variables", "Description", "Values"]
    data = pd.merge(dataf, datad, on="Variables")

    return data
data = data_description(df_2)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri
                (sınıf sayısı çok olduğundan ölçüm değeri taşımayan değişkneler)

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df_2)
cat_cols.remove("output")
del cat_but_car

# HEART ATTACK MODEL-CATBOOST
#############################
X_first = dff.drop("output", axis=1)
X_heart = X_first.copy()

# for col in cat_cols:
#     X_heart[col] = X_heart[col].astype("object")

y_heart = dff["output"]


# catboost encoder
cbe_encoder = ce.cat_boost.CatBoostEncoder()
X_heart = cbe_encoder.fit_transform(X_heart, y_heart)

catboost_model_full = CatBoostClassifier(bootstrap_type='Bayesian', depth=3, iterations=300, learning_rate=0.05,
                                         verbose=False).fit(X_heart, y_heart)


########################################################################################################
########################################################################################################
########################################################################################################


# MODEL
y = df_lung["LUNG_CANCER"]
X = df_lung.drop(["LUNG_CANCER"], axis=1)

#! pip install imblearn
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=17)

# Parametre gridi :
##################

final_model_lightgbm = LGBMClassifier(colsample_bytree=0.7,
                                      learning_rate=0.1,
                                      max_depth=7,
                                      min_data_in_leaf=30,
                                      n_estimators=500,
                                      num_leaves=10,
                                      random_state=17).fit(X_train, y_train)


# def plot_importance(model, feature, num=len(X_heart), save=False):
#     feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": feature.columns})
#     plt.figure(figsize=(10,10))
#     sns.set(font_scale=1)
#     sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
#                                                                      ascending=False)[0:num]) # bu num erideki değişken kadar görselleştirme yapar ancak biz buradan kaç tane gözlem istediğimizi ayarlayabiliriz
#
#     plt.title("Features")
#     plt.tight_layout()
#     plt.show()
#     if save:
#         plt.saveig("importances.png") #True dersek ilgili görseli kaydeder
#
# plot_importance(catboost_model_full, X_heart)

############################ Download Dosyasi ############################

with st.sidebar:
	selected = option_menu('Disease Prediction System With Artificial Intelligence',
							['Heart Attack Prediction',
	                        'Lung Cancer Prediction'],
	                       icons=['heart', 'disease'],
	                       default_index=0)
	
if selected == 'Lung Cancer Prediction':
	st.title("Download File for Lung Cancer Prediction")
	dataframe = pd.DataFrame(columns=df_lung.columns)
	dataframe = dataframe.drop("LUNG_CANCER", axis=1).to_csv()
	st.download_button(
			label="Download data as CSV for uploading",
			data=dataframe,
			file_name='lung_cancer_features.csv',
			mime='text/csv',)
	
else:
	st.title("Download File for Heart Attack Prediction")
	dataframe = pd.DataFrame(columns=df_heart.columns)
	dataframe = dataframe.drop("output",axis=1).to_csv()
	st.download_button(
		label="Download data as CSV for uploading",
		data=dataframe,
		file_name='heart_attack_features.csv',
		mime='text/csv',)


################################################################################################################

with st.sidebar:
	option  = option_menu("Choose your Disease",
	                       ["Lung Cancer","Heart Attack"])
	
if option == "Lung Cancer":
	st.header("Prediction System of Lung Cancer")
	uploaded_file = st.file_uploader("Upload your input CSV file for Lung Cancer", type=["csv"])
	
	if uploaded_file == None:
		st.info('Awaiting for CSV file to be uploaded.')
	else:
		dataframe = pd.read_csv(uploaded_file)
		predicted_uploaded_file = final_model_lightgbm.predict(dataframe)
		result_df = pd.DataFrame(columns=["Result"], data=predicted_uploaded_file, index=dataframe.index)
		final_dataframe = pd.concat([dataframe, result_df], axis=1)
		st.dataframe(final_dataframe)
		image = Image.open(
			'/Users/tugceselin/PycharmProjects/dsmlbc_9/vbo_project_disease_prediction_ml/images/feature_importance_lung.png')
		st.image(image, caption='Feature Importance of Lung Cancer')
		st.write("Costs On Average")
		st.write("Operation: 35.000$")
		st.write("Chemotherapy: 7.000$")
		st.write("Radiation Therapy: 5.000$")
		st.write("Medicare: 3.000$")
		st.write("Other Routine Expenses: 2.000$")
		st.info("Total cost per patient: 52.000$")
		st.info("Number of patient according to dataset: 8")
		st.info("Expected Cash inflow: 416.000$")
		final_dataframe = final_dataframe.to_csv()
		st.write("""You may download your results from here.""")
		st.download_button(
			label="Download your result as CSV",
			data=final_dataframe,
			file_name='your_results_for_lung_cancer.csv',
			mime='text/csv', )


else:
	st.header("Prediction System of Heart Attack")
	uploaded_file = st.file_uploader("Upload your input CSV file for Heart Attack", type=["csv"])
	
	if uploaded_file == None:
		st.info('Awaiting for CSV file to be uploaded.')
	else:
		dataframe = pd.read_csv(uploaded_file)
		predicted_uploaded_file = catboost_model_full.predict(dataframe)
		result_df = pd.DataFrame(columns=["Result"], data=predicted_uploaded_file, index=dataframe.index)
		final_dataframe = pd.concat([dataframe, result_df], axis=1)
		st.dataframe(final_dataframe)
		image = Image.open('/Users/tugceselin/PycharmProjects/dsmlbc_9/vbo_project_disease_prediction_ml/images/heart_attack_features_imp.png')
		st.image(image, caption='Feature Importance of Lung Cancer')
		final_dataframe = final_dataframe.to_csv()
		st.write("""You may download your results from here.""")
		st.download_button(
			label="Download your result as CSV",
			data=final_dataframe,
			file_name='your_results_for_heart_attack.csv',
			mime='text/csv', )


