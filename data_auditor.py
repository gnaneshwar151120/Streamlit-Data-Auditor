import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image 
from uuid import uuid4
from streamlit.proto.Selectbox_pb2 import Selectbox
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder


# st.title('Data Auditor')
html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px">
		<h1 style="color:{};text-align:center;">Data Auditor </h1>
		</div>
		<br>
		"""
st.markdown(html_temp.format('#000000','white'),unsafe_allow_html=True)
def prompt_file_upload():
    file_handler = st.file_uploader('Upload the CSV', type=['csv'])

    if file_handler is not None:
       
        csv_filename = f'{uuid4().hex}.csv'
       
        with open(csv_filename, 'wb') as fh:
            fh.write(file_handler.read())
           
        return csv_filename
    else:
        return None
   
   
@st.cache
def get_dataframe(csv_filename):
    return pd.read_csv(csv_filename)
           
           
csv_filename = prompt_file_upload()
def get_singleton(data: list):
    if len(data) == 1:
        return data[0]
    
    return data



def selectbox_with_default(text, values, default="DEFAULT", sidebar=False):
    func = st.sidebar.selectbox #if sidebar else st.selectbox
    return func(text, [default, *values])

if csv_filename is not None:

    options = ['1. EDA','2. PyPlot', '3. Build','4. About']
    DEFAULT = 'SELECT AN OPTION BELOW'
    selected_option = selectbox_with_default('Select an Option', options, default=DEFAULT)
   
    if selected_option == DEFAULT:
        st.warning("Please fill all the fields !")
        raise st.StopException

    if selected_option == options[0]:
        st.info('Exploratory Data Analysis')
        st.subheader('THE WHOLE DATA')
        df = get_dataframe(csv_filename)
        st.dataframe(df)
        st.subheader("Select the type of EDA")
        EDA=st.radio("options",('Show Shape','Head','Tail','Summary','valuecount','Show Selected Columns','Correlation Plot(Matplotlib)','Correlation Plot(Seaborn)','Pie Plot'))
        if EDA== 'Show Shape':
            st.dataframe(df.shape)
        if EDA=='Head':
            st.dataframe(df.head())
        if EDA=='Tail':
            st.dataframe(df.tail())
        if EDA=='Summary':
            st.dataframe(df.describe())
        if EDA=='valuecount':
            st.dataframe(df.iloc[:,-1].value_counts())
        if EDA== 'Show Selected Columns':
            selected_columns = st.multiselect("Select Columns", df.columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)
        if EDA== 'Correlation Plot(Matplotlib)':
            plt.matshow(df.corr())
            st.pyplot()
        if EDA== 'Correlation Plot(Seaborn)':
            st.write(sns.heatmap(df.corr(),annot=True))
            st.pyplot()
        if EDA== 'Pie Plot':
            all_columns = df.columns.to_list()
            column_to_plot = st.selectbox("Select 1 Column",all_columns)
            fig, ax = plt.subplots()
            pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
            st.write(pie_plot)
            st.pyplot(fig)


    if selected_option == options[1]:
        st.info("Data Visualization")
        df = get_dataframe(csv_filename)
        
        if st.checkbox("Show Value Counts"):
            fig, ax = plt.subplots()
            st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
            st.pyplot(fig)
        all_columns_names = df.columns.tolist()
        type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
        selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)
        if st.button("Generate Plot"):
            st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))
            if type_of_plot == 'area':
                
                cust_data = df[selected_columns_names]
                st.area_chart(cust_data)
            if type_of_plot == 'bar':
                
                cust_data = df[selected_columns_names]
                st.bar_chart(cust_data)
            if type_of_plot == 'line':
                
                cust_data = df[selected_columns_names]
                st.line_chart(cust_data)
            if type_of_plot == 'hist' or 'box' or 'kde':
                if len(selected_columns_names)>1:
                    st.warning('select one coloumn')
                else:
                    fig, ax = plt.subplots()
                    cust_plot= df[get_singleton(selected_columns_names)].plot(kind=type_of_plot)
                    st.write(cust_plot)
                    st.pyplot(fig)
            
        
    if selected_option == options[2]:
            st.info("Building ML Models")
            df = get_dataframe(csv_filename)
            df_cat= df.select_dtypes(include = 'object').astype(str).copy()
            onehot=OneHotEncoder(handle_unknown ='ignore')
            df_hot=onehot.fit_transform(df_cat)
            for i in df_cat:
                del df[i]
            Y=df.iloc[:,-1].copy()
            df.drop(df.columns[[-1]],axis=1,inplace=True)
            import scipy.sparse as sp
            A=sp.csr_matrix(df.values)
            df=sp.csr_matrix.transpose(A)
            df=df.T
            df=sp.hstack((df,df_hot))
            X=df.copy()
            from scipy.sparse import csr_matrix
            X=csr_matrix(X)
            seed = 7        
                
			
            models = []
            models.append(('LR', LogisticRegression()))
            models.append(('KNN', KNeighborsClassifier()))
            models.append(('CART', DecisionTreeClassifier()))
            models.append(('SVM', SVC()))
            model_names = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = 'accuracy'
            for name, model in models:
                kfold = model_selection.KFold(n_splits=10, random_state=seed ,shuffle=True)
                cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
                model_names.append(name)
                model_mean.append(cv_results.mean())
                model_std.append(cv_results.std())
                
                accuracy_results = {"model name":name,"model_accuracy":cv_results.mean(),"standard deviation":cv_results.std()}
                all_models.append(accuracy_results)
             
                            
            
            if st.checkbox("Metrics As Table"):
                st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Algo","Mean of Accuracy","Std"]))
                
            
            if st.checkbox("Metrics As JSON"):
                st.json(all_models)
                
    if selected_option == options[3]:
        st.success("About")
        st.subheader("Creators")
        col1, col2, col3, col4 = st.beta_columns(4)
        image=('gnani.jpeg')
        Gnan = Image.open(image)
        col1.header("Gnaneshwar")
        col1.image(Gnan, use_column_width=True)
        image1=('sam.jpeg')
        sam = Image.open(image1)
        col2.header("Sam ")
        col2.image(sam, use_column_width=True)
        image2=('janhvi.jpeg')
        Jahnvi = Image.open(image2)
        col3.header("Janhvi")
        col3.image(Jahnvi, use_column_width=True)
        image3=('binayak.jpeg')
        Binayak = Image.open(image3)
        col4.header("Binayak")
        col4.image(Binayak, use_column_width=True)
        with st.beta_expander("What is Data Auditor?"):
            st.write('''DATA AUDITOR is a web application, Which process: \n
            1.EDA(Exploratory data analysis) \n
            2.PyPlot(Data vizualization) \n
            3.Model Bulding(ML models using 5 algorithms) \n ''')
        with st.beta_expander("What is EDA(Exploratory data analysis)?"):
            st.write('''Exploratory data analysis (EDA) is used by data scientists to analyze and investigate data sets and summarize their
            main characteristics, often employing data visualization methods. It helps determine how best to manipulate data sources to get the 
            answers you need, making it easier for data scientists to discover patterns, 
            spot anomalies, test a hypothesis, or check assumptions.EDA is primarily used to see what data can reveal beyond the 
            formal modeling or hypothesis testing task and provides a provides a better understanding of data set variables and the 
            relationships between them. It can also help determine if the statistical techniques you are considering for data analysis 
            are appropriate. Originally developed by American mathematician John Tukey in the 1970s, EDA techniques continue to be a widely
            used method in the data discovery process today.''')
        with st.beta_expander("Options in EDA"):
            st.write('''
                     1.Show Shape - Displays number of rows and coloumns \n
                    2.Head - Display top 5 rows \n
                    3.Tail - Display last 5 rows \n
                    4.Summary - Display cout,Mean,Standard deviation,Min and Max values \n
                    5.valuecount - Display count of unique values \n
                    6.Show Selected Columns - Display the coloums you select \n
                    7.Correlation Plot(Matplotlib) - Display dependence of multiple variables\n
                    8.Correlation Plot(Seaborn) \n
                    9.Pie Plot - Display the pie chart of selected coloumn data''')
        with st.beta_expander("What are Pyplots?"):
            st.write('''Plots contain the s a collection of functions that make matplotlib work like MATLAB. 
            Each pyplot function makes some change to a figure: e.g., creates a figure, creates a plotting area 
            in a figure, plots some lines in a plotting area, decorates the plot with labels, etc.
            In matplotlib.pyplot various states are preserved across function calls, so that it keeps track of
            things like the current figure and plotting area, and the plotting functions are directed to the 
            current axes (please note that "axes" here and in most places in the documentation refers to the 
            axes part of a figure and not the strict mathematical term for more than one axis).''')
        with st.beta_expander("Options in pyplot"):
            st.write('''
                    1.area \n 
                    2.bar \n
                    3.line, \n
                    4.hist\n 
                    5.box \n
                    6.kde''') 
        with st.beta_expander("What are Builds?"):
            st.write('''Machine learning consists of algorithms that can automate analytical model building. Using algorithms that 
            iteratively learn from data, machine learning models facilitate computers to find hidden insights from Big Data without 
            being explicitly programmed where to look''')
        #with st.beta_container("How Data auditor Works"):
