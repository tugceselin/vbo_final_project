from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
#!pip install streamlit_option_menu
from streamlit_option_menu import option_menu
#! pip install streamlit_pandas_profiling
#! pip install pandas_profiling
import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

image = Image.open('/Users/tugceselin/PycharmProjects/dsmlbc_9/vbo_project_disease_prediction_ml/images/company_logo.jpeg')
st.image(image,width=330)

with st.sidebar.header('1. Upload your CSV data'):
	uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
	st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

if uploaded_file is not None:
	@st.cache
	def load_csv():
		csv = pd.read_csv(uploaded_file)
		return csv
	
	
	df = load_csv()
	pr = ProfileReport(df, explorative=True)
	st.header('**Exploratory Data Analysis**')
	st.write(df)
	st.write('---')
	st.header('**Pandas Profiling Report**')
	st_profile_report(pr)
else:
	st.info('Awaiting for CSV file to be uploaded.')
	if st.button('Press to use Example Dataset'):
		# Example data
		@st.cache
		def load_data():
			a = pd.DataFrame(
				np.random.rand(100, 5),
				columns=['a', 'b', 'c', 'd', 'e']
			)
			return a
		
		
		df = load_data()
		pr = ProfileReport(df, explorative=True)
		st.header('**Input DataFrame**')
		st.write(df)
		st.write('---')
		st.header('**Pandas Profiling Report**')
		st_profile_report(pr)
		


	
	
	




        
        








