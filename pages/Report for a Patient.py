# !pip install lightgbm
import pandas as pd
import streamlit as st
from lightgbm import LGBMClassifier
from sklearn.preprocessing import RobustScaler
from streamlit_option_menu import option_menu
from sklearn import preprocessing as prep

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# DATA IMPORT
df = pd.read_csv("/Users/tugceselin/PycharmProjects/dsmlbc_9/vbo_project_disease_prediction_ml/survey lung cancer.csv")

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
	quartile1 = dataframe[col_name].quantile(q1)
	quartile3 = dataframe[col_name].quantile(q3)
	interquantile_range = quartile3 - quartile1
	up_limit = quartile3 + 1.5 * interquantile_range
	low_limit = quartile1 - 1.5 * interquantile_range
	return low_limit, up_limit

outlier_thresholds(df, "AGE")

def replace_with_thresholds(dataframe, variable):
	low_limit, up_limit = outlier_thresholds(dataframe, variable)
	dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
	dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "AGE")

def age_prep(dataframe):
	dataframe.loc[(dataframe['AGE'] < 30), "RISK_AGE_DIM"] = 'LOW RISK'
	dataframe.loc[(dataframe['AGE'] >= 30) & (dataframe['AGE'] <= 45), "RISK_AGE_DIM"] = 'MID RISK'
	dataframe.loc[(dataframe['AGE'] > 45), "RISK_AGE_DIM"] = 'HIGH RISK'

age_prep(df)

def change_types(df):
	df.loc[(df['LUNG_CANCER'] == "YES"), 'LUNG_CANCER'] = 1
	df.loc[(df['LUNG_CANCER'] == "NO"), 'LUNG_CANCER'] = 0
	df["LUNG_CANCER"] = df["LUNG_CANCER"].astype("int64")
	
change_types(df)

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

cat_cols, num_cols, cat_but_car = grab_col_names(df)

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
	
create_new_features(df)

def label_encoder_prep(dataframe, binary_col):
	labelencoder = prep.LabelEncoder()
	dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
	return dataframe

binary_cols = [col for col in df.columns if df[col].nunique() == 2]
len(binary_cols)

for col in binary_cols:
	label_encoder_prep(df, col)
rs = RobustScaler()
df["AGE"] = rs.fit_transform(df[["AGE"]])

############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

###############################################################################################
# DATA - HEART ATTACK
###############################################################################################
dff = pd.read_csv("/Users/tugceselin/PycharmProjects/dsmlbc_9/vbo_project_disease_prediction_ml/heart.csv")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import category_encoders as ce

import matplotlib
#matplotlib.use('Qt5Agg')


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

##############################################################################################
# CatBoost
###############################################################################################
X_first = dff.drop("output", axis=1)
X_heart = X_first.copy()

for col in cat_cols:
    X_heart[col] = X_heart[col].astype("object")

y_heart = dff["output"]


# catboost encoder
cbe_encoder = ce.cat_boost.CatBoostEncoder()
X_heart = cbe_encoder.fit_transform(X_heart, y_heart)

catboost_model_full = CatBoostClassifier(bootstrap_type='Bayesian', depth=3, iterations=300, learning_rate=0.05,
                                         verbose=False).fit(X_heart, y_heart)

# new_patient = np.array([63, 1, 2, 172, 252, 0, 1, 150, 0, 0.05, 0, 1, 1])
#
# def heart_scaler(X_first, Y):
#     X_new = X_first.append(pd.DataFrame([new_patient], columns=X_first.columns), ignore_index=True)
#     for col in cat_cols:
#         X_new[col] = X_new[col].astype("object")
#     Y[303] = 0
#     cbe_encoder = ce.cat_boost.CatBoostEncoder()
#     X_scaled = cbe_encoder.fit_transform(X_new, Y)
#     new_patient_scaled = X_scaled.iloc[303].to_numpy()
#     return new_patient_scaled

# new_patient_scaled = heart_scaler(X_first, y_heart)
#
# catboost_model_full.predict(new_patient_scaled)

# heart_prediction_origin = np([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])






########################################################################################################################################################################################################################
########################################################################################################################################################################################################################
# MODEL

y = df["LUNG_CANCER"]
X = df.drop(["LUNG_CANCER"], axis=1)

#! pip install imblearn
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=17)

# Parametre gridi :
##################

# lgbm_params = {
# 	"learning_rate": [0.01, 0.1],
# 	"n_estimators": [500, 550, 600],
# 	"colsample_bytree": [0.7, 0.8, 0.1],
# 	"num_leaves": [5, 7, 10],
# 	"min_data_in_leaf": [30, 40, 50],
# 	"max_depth": [7, 8, 9]}

final_model_lightgbm = LGBMClassifier(colsample_bytree=0.7,
                                      learning_rate=0.1,
                                      max_depth=7,
                                      min_data_in_leaf=30,
                                      n_estimators=500,
                                      num_leaves=10,
                                      random_state=17).fit(X_train,y_train)
############################################################################################################################################################################################################
############################################################################################################################################################################################################

# OPTION MENU
with st.sidebar:
	selected = option_menu('Disease Prediction System With Artificial Intelligence',
	
	                       ['Heart Attack Prediction',
	                        'Lung Cancer Prediction'],
	                       icons=['activity', 'lungs-fill'],
	                       default_index=0)

# Heart Disease Prediction Page
if (selected == 'Heart Attack Prediction'):
	
	# page title
	st.title('Heart Attack Prediction with Artificial Intelligence')
	
	col1, col2, col3 = st.columns(3)
	with col1:
		age = st.number_input('Age',min_value=0,max_value=71)
	with col2:
		sex = st.number_input('Sex',min_value=0,max_value=1)
	with col3:
		cp = st.number_input('Chest Pain types',min_value=0.0,max_value=1.0)
	with col1:
		trestbps = st.number_input('Resting Blood Pressure',min_value=60.0,max_value=180.0)
	with col2:
		chol = st.number_input('Serum Cholestoral in mg/dl',min_value=130.0,max_value=360.0)
	with col3:
		fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl',min_value=0.0,max_value=1.0)
	with col1:
		restecg = st.number_input('Resting Electrocardiographic results',min_value=0.0,max_value=1.0)
	with col2:
		thalach = st.number_input('Maximum Heart Rate achieved',min_value=60.0,max_value=190.0)
	with col3:
		exang = st.number_input('Exercise Induced Angina',min_value=0.0,max_value=1.0)
	with col1:
		oldpeak = st.number_input('ST depression induced by exercise',min_value=0.0,max_value=1.0)
	with col2:
		slope = st.number_input('Slope of the peak exercise ST segment',min_value=0.0,max_value=1.0)
	with col3:
		ca = st.number_input('Major vessels colored by flourosopy',min_value=0.0,max_value=1.0)
	with col2:
		thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect',min_value=0.0,max_value=1.0)
	
	# code for Prediction
	heart_diagnosis = ''
	
	# creating a button for Prediction
	
	if st.button('Heart Disease Test Result'):

		heart_prediction = catboost_model_full.predict(
			[[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
		
		if (heart_prediction[0] == 1):
			heart_diagnosis = 'The person is having heart disease'
		else:
			heart_diagnosis = 'The person does not have any heart disease'
	
	st.success(heart_diagnosis)
	

# Heart Disease Prediction Page
if (selected == 'Lung Cancer Prediction'):
	
	# page title
	st.title('Lung Cancer Prediction with Artificial Intelligence')
	
	col1, col2, col3 = st.columns(3)
	
	with col1:
		GENDER = st.number_input("GENDER",min_value=0,max_value=1)
		
	with col2:
		AGE = st.number_input("AGE",min_value=1.0,max_value=70.0)
		
	with col3:
		SMOKING = st.number_input('SMOKING',min_value=0,max_value=1)
	
	with col1:
		YELLOW_FINGERS = st.number_input('YELLOW_FINGERS',min_value=0,max_value=1)
	
	with col2:
		ANXIETY = st.number_input('ANXIETY',min_value=0,max_value=1)
	
	with col3:
		PEER_PRESSURE = st.number_input('PEER_PRESSURE',min_value=0,max_value=1)
	
	with col1:
		CHRONIC_DISEASE = st.number_input('CHRONIC DISEASE',min_value=0,max_value=1)
	
	with col2:
		FATIGUE = st.number_input('FATIGUE',min_value=0,max_value=1)
	
	with col3:
		ALLERGY = st.number_input('ALLERGY',min_value=0,max_value=1)
	
	with col1:
		WHEEZING = st.number_input('WHEEZING',min_value=0,max_value=1)
	
	with col2:
		ALCOHOL_CONSUMING = st.number_input('ALCOHOL CONSUMING',min_value=0,max_value=1)
	
	with col3:
		COUGHING = st.number_input('COUGHING',min_value=0,max_value=1)
	
	with col1:
		SHORTNESS_OF_BREATH = st.number_input('SHORTNESS OF BREATH',min_value=0,max_value=1)
	
	with col2:
		SWALLOWING_DIFFICULTY = st.number_input("SWALLOWING DIFFICULTY",min_value=0,max_value=1)
	
	with col3:
		CHEST_PAIN = st.number_input("CHEST_PAIN",min_value=0,max_value=1)
	
	with col1:
		RISK_AGE_DIM = st.number_input('RISK_AGE_DIM',min_value=0,max_value=1)
	
	with col2:
		STRESSED_PERSON = st.number_input('STRESSED_PERSON',min_value=0,max_value=1)
	
	with col3:
		IS_HEALHTY = st.number_input('IS_HEALHTY',min_value=0,max_value=1)
		
	with col1:

		BREATHING_PROBLEM = st.number_input('BREATHING_PROBLEM',min_value=0,max_value=1)
		
	with col2:
		SECONDHAND_SMOKER = st.number_input('SECONDHAND_SMOKER',min_value=0,max_value=1)

	# code for Prediction
	lung_cancer = ''
	
	# creating a button for Prediction
	if st.button('Lung Cancer Test Results'):
		lung_cancer_pred = final_model_lightgbm.predict([[GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE,
														  FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH,
														  SWALLOWING_DIFFICULTY, CHEST_PAIN, RISK_AGE_DIM,STRESSED_PERSON,IS_HEALHTY,BREATHING_PROBLEM,SECONDHAND_SMOKER]])
		if (lung_cancer_pred[0] == 1):
			lung_cancer = 'The person is having lung cancer disease'
		else:
			lung_cancer = 'The person does not have any lung cancer disease'
	
	st.success(lung_cancer)




