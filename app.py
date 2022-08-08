#Importing Libraries:
import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
#import time

#This part of code creates a sidebar and even gives it a heading 
st.sidebar.header('User Input Parameters')

Page=st.sidebar.selectbox('Select Page:',['Home Page','About the Data','Flow-Chart','Visualization','Predictor'])

if(Page=='Home Page'):
    #The below code is used to write stuff on Web App
    st.write("""
    # Predictiction of Insurance Costs for Individuals

    This app predicts the Insurance Costs of individuals staying in **_United States of America_**

    **_Project Aim_**
    
    The main aim of the project is to predict the Insurance Cost of the people in the United
    States of America within different regions with the help of specific attributes. Since Insurance
    Cost varies from one individual to another and it depends on different features, we will try to
    find correlation between the features and insurance cost charges.
    """)
    #The below code will show an image and below it there will be caption
    image=Image.open('capture.JPG')
    st.image(image)

if(Page=='Flow-Chart'):
    #The below code is used to write stuff on Web App
    st.write("""
    # Predictiction of Insurance Costs for Individuals

    This app predicts the Insurance Costs of individuals staying in **_United States of America_**

    **_Project Flow Chart_**
    
    The project has been done in different phases as shown below.
    
    """)
    #The below code will show an image and below it there will be caption
    image=Image.open('FlowChart.JPG')
    st.image(image)

if(Page=='About the Data'):
    st.write("""
    # Cost of Insurance Predictor

    This web app predicts the costs of Insurance based on the features given below for indivduals residing in **_United States of America_**.
    """)
    st.write("""
    # About The Data
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    
    st.write("""

    **Context**

    This dataset is created for prediction of the **_Insurance Costs_**.

    **Content**

    The dataset contains several parameters which are considered important during the prediction of **_Insurance Costs_**.
    The parameters included are :

    1) age: age of primary beneficiary

    2) sex: insurance contractor gender, female, male

    3) bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
    objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9

    4) children: Number of children covered by health insurance / Number of dependents

    5) smoker: Smoking

    6) region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.

    7) charges: Individual medical costs billed by health insurance

    **Acknowledgements**

    This dataset is available on Github here : https://github.com/stedy/Machine-Learning-with-R-datasets
    
    The kaggle dataset is available here : https://www.kaggle.com/mirichoi0218/insurance

    **Inspiration**

    Can you accurately predict insurance costs?

    **Tags**

    education, health, finance, insurance, healthcare
 
    """)

if(Page=='Predictor'):
    st.write("""
    # Cost of Insurance Predictor

    This web app predicts the costs of Insurance based on the features given below for indivduals residing in **_United States of America_**.
    """)
    st.write("""
    # Predictor
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")
    Predictor_Page=st.sidebar.selectbox('Select Model:',['All Variables-Random Forest Regression','Backward Elimination-Linear Regression','Forward Selection-Random Forest Regression'])

    if(Predictor_Page=='All Variables-Random Forest Regression'):
        #Creting sliders for every feature to be used
        def user_input_features():
            age=st.sidebar.slider('Age ',18,100)
            sex=st.sidebar.slider('Sex: 0=Female, 1=Male ',0,1)
            bmi=st.sidebar.number_input('BMI ',10.00,100.00)
            children=st.sidebar.slider('No. of Children/Dependents ',0,5)
            smoker=st.sidebar.slider('Smoker: 0=N0, 1=Yes ',0,1)
            region=st.sidebar.slider('Regions: 0-NE, 1=NW, 2=SE, 3=SW ',0,3)
            #University_Rating=st.sidebar.slider('University Rating',1.0,5.0)
            #LOR=st.sidebar.slider('LOR',1.0,5.0)
            #SOP=st.sidebar.slider('SOP',1.0,5.0)
            #Research=st.sidebar.slider('Research Done',0,1)
            data={'age':age,
              'sex':sex,
              'bmi':bmi,
              'children':children,
              'smoker':smoker,
              'region':region
              #'University Rating':University_Rating,
              #'LOR':LOR,
              #'SOP':SOP,
              #'Research':Research
              }
            features = pd.DataFrame(data, index=[0])
            return features

        df = user_input_features()
    
        #Loading the data and cleaning it
        data_path=r'C:\Users\Rutu Desai\AppData\Local\Programs\Python\Python37\Project\Insurance Cost\insurance.csv'
        dataset=pd.read_csv(data_path)
        from sklearn.preprocessing import LabelEncoder
        le=LabelEncoder()
        le.fit(dataset.sex)
        dataset.sex=le.transform(dataset.sex)
        le.fit(dataset.smoker)
        dataset.smoker=le.transform(dataset.smoker)
        le.fit(dataset.region)
        dataset.region=le.transform(dataset.region)
        #data.drop(['Serial No.'],axis=1,inplace=True)
        #data.rename(columns={'LOR ':'LOR','Chance of Admit ':'Chance of Admit'},inplace=True)


        #Preparing the data for Model
        X=dataset.iloc[:,[0,1,2,3,4,5]]
        y=dataset.iloc[:,-1]


        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

        #from sklearn.linear_model import LinearRegression
        #from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import accuracy_score,mean_squared_error
        #regressors=[['Linear Regression :',LinearRegression()],
        #['Decision Tree Regression :',DecisionTreeRegressor()],
        #['Random Forest Regression :',RandomForestRegressor()],
        #]
        regressors=[['Random Forest Regression',RandomForestRegressor()]]
        reg_pred=[]
        accuracies=[]
        print('Results...\n')
        for name,model in regressors:
            model=model
            model.fit(X_train,y_train)
            predictions = model.predict(X_test)
            rms=np.sqrt(mean_squared_error(y_test, predictions))
            reg_pred.append(rms)
            accuracy= model.score(X_test,y_test)
            accuracies.append(accuracy)
        st.write('Model Name : ',name)
        st.write('RMS Score : ',rms)
        st.write('Accuracy : ',accuracy)
        
        st.subheader('User Input parameters')
        st.write(df)
        y_val=model.predict(df)
        st.subheader('Predicted Insurance Cost :')
        st.write(y_val)
    
    if(Predictor_Page=='Backward Elimination-Linear Regression'):
        def user_input_features():
            age=st.sidebar.slider('Age ',18,100)
            #sex=st.sidebar.slider('Sex: 0=Female, 1=Male ',0,1)
            #bmi=st.sidebar.slider('BMI ',10,100)
            #children=st.sidebar.slider('No. of Children ',0,5)
            smoker=st.sidebar.slider('Smoker: 0=N0, 1=Yes ',0,1)
            region=st.sidebar.slider('Regions: 0-NE, 1=NW, 2=SE, 3=SW ',0,3)
            #University_Rating=st.sidebar.slider('University Rating',1.0,5.0)
            #LOR=st.sidebar.slider('LOR',1.0,5.0)
            #SOP=st.sidebar.slider('SOP',1.0,5.0)
            #Research=st.sidebar.slider('Research Done',0,1)
            data={'age':age,
              'smoker':smoker,
              'region':region
              #'University Rating':University_Rating,
              #'LOR':LOR,
              #'SOP':SOP,
              #'Research':Research
              }
            features = pd.DataFrame(data, index=[0])
            return features

        df = user_input_features()
    
        #Loading the data and cleaning it
        data_path=r'C:\Users\Rutu Desai\AppData\Local\Programs\Python\Python37\Project\Insurance Cost\insurance.csv'
        dataset=pd.read_csv(data_path)
        from sklearn.preprocessing import LabelEncoder
        le=LabelEncoder()
        le.fit(dataset.sex)
        dataset.sex=le.transform(dataset.sex)
        le.fit(dataset.smoker)
        dataset.smoker=le.transform(dataset.smoker)
        le.fit(dataset.region)
        dataset.region=le.transform(dataset.region)
        #data.drop(['Serial No.'],axis=1,inplace=True)
        #data.rename(columns={'LOR ':'LOR','Chance of Admit ':'Chance of Admit'},inplace=True)

          #Preparing the data for Model
        X=dataset.iloc[:,[0,4,5]]
        y=dataset.iloc[:,-1]


        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

        from sklearn.linear_model import LinearRegression
        #from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import accuracy_score,mean_squared_error
        #regressors=[['Linear Regression :',LinearRegression()],
        #['Decision Tree Regression :',DecisionTreeRegressor()],
        #['Random Forest Regression :',RandomForestRegressor()],
        #]
        regressors=[['Linear Regression',LinearRegression()]]
        reg_pred=[]
        accuracies=[]
        print('Results...\n')
        for name,model in regressors:
            model=model
            model.fit(X_train,y_train)
            predictions = model.predict(X_test)
            rms=np.sqrt(mean_squared_error(y_test, predictions))
            reg_pred.append(rms)
            accuracy= model.score(X_test,y_test)
            accuracies.append(accuracy)
        st.write('Model Name : ',name)
        st.write('RMS Score : ',rms)
        st.write('Accuracy : ',accuracy)
        
        st.subheader('User Input parameters')
        st.write(df)
        y_val=model.predict(df)
        st.subheader('Predicted Insurance Cost :')
        st.write(y_val)

    if(Predictor_Page=='Forward Selection-Random Forest Regression'):
        def user_input_features():
            age=st.sidebar.slider('Age ',18,100)
            #sex=st.sidebar.slider('Sex: 0=Female, 1=Male ',0,1)
            bmi=st.sidebar.number_input('BMI ',10.00,100.00)
            #children=st.sidebar.slider('No. of Children ',0,5)
            smoker=st.sidebar.slider('Smoker: 0=N0, 1=Yes ',0,1)
            region=st.sidebar.slider('Regions: 0-NE, 1=NW, 2=SE, 3=SW ',0,3)
            #University_Rating=st.sidebar.slider('University Rating',1.0,5.0)
            #LOR=st.sidebar.slider('LOR',1.0,5.0)
            #SOP=st.sidebar.slider('SOP',1.0,5.0)
            #Research=st.sidebar.slider('Research Done',0,1)
            data={'age':age,
              'bmi':bmi,
              'smoker':smoker,
              'region':region
              #'University Rating':University_Rating,
              #'LOR':LOR,
              #'SOP':SOP,
              #'Research':Research
              }
            features = pd.DataFrame(data, index=[0])
            return features

        df = user_input_features()
    
        #Loading the data and cleaning it
        data_path=r'C:\Users\Rutu Desai\AppData\Local\Programs\Python\Python37\Project\Insurance Cost\insurance.csv'
        dataset=pd.read_csv(data_path)
        from sklearn.preprocessing import LabelEncoder
        le=LabelEncoder()
        le.fit(dataset.sex)
        dataset.sex=le.transform(dataset.sex)
        le.fit(dataset.smoker)
        dataset.smoker=le.transform(dataset.smoker)
        le.fit(dataset.region)
        dataset.region=le.transform(dataset.region)
        #data.drop(['Serial No.'],axis=1,inplace=True)
        #data.rename(columns={'LOR ':'LOR','Chance of Admit ':'Chance of Admit'},inplace=True)

        #Preparing the data for Model
        X=dataset.iloc[:,[0,2,4,5]]
        y=dataset.iloc[:,-1]


        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

        #from sklearn.linear_model import LinearRegression
        #from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import accuracy_score,mean_squared_error
        #regressors=[['Linear Regression :',LinearRegression()],
        #['Decision Tree Regression :',DecisionTreeRegressor()],
        #['Random Forest Regression :',RandomForestRegressor()],
        #]
        regressors=[['Random Forest Regression',RandomForestRegressor()]]
        reg_pred=[]
        accuracies=[]
        print('Results...\n')
        for name,model in regressors:
            model=model
            model.fit(X_train,y_train)
            predictions = model.predict(X_test)
            rms=np.sqrt(mean_squared_error(y_test, predictions))
            reg_pred.append(rms)
            accuracy= model.score(X_test,y_test)
            accuracies.append(accuracy)
        st.write('Model Name : ',name)
        st.write('RMS Score : ',rms)
        st.write('Accuracy : ',accuracy)
        
        st.subheader('User Input parameters')
        st.write(df)
        y_val=model.predict(df)
        st.subheader('Predicted Insurance Cost :')
        st.write(y_val)
    #model2=RandomForestRegressor(n_estimators=100,random_state=0)
    #model2.fit(X,y)
    #st.subheader('User Input parameters')
    #st.write(df)
    #y_val=model2.predict(df)
    #st.write('The Chance of Admission is:',y_val)


    

if(Page=="Visualization"):
    st.write("""
    # Cost of Insurance Predictor

    This web app predicts the costs of Insurance based on the features given below for indivduals residing in **_United States of America_**.
    """)
    st.write("""
    # Visualizations
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")
    #The below code will show an image and below it there will be caption
    
#st.write("""------------------------------------------------------------------------------------------------------------------------------------""")
    image=Image.open('1.JPG')
    st.image(image,caption='Correlation Map of numerical/quantitative variables with Charge')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('2.JPG')
    st.image(image,caption='Correlation Map of all(including qualitative) variables with Charge')
    st.write("""
    According to the Heatmap ,it is clear that Smoker is the most important features related to Insurance Cost
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('3.JPG')
    st.image(image,caption='How are differnt Features distributed')
    st.write("""
    Distplot showing Distribution of the Charges.
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")


    image=Image.open('4.JPG')
    st.image(image,caption='Percentage of People who are Smoker or Non-Smoker')
    st.write("""
    We can see that majority of the people are Non-Smokers!
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    
    image=Image.open('5.JPG')
    st.image(image,caption='Scatterplot showing how Smoking increase the Charges')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('6.JPG')
    st.image(image,caption='Violin Plot showin Sex wise Smokers and Non-Smokers v/s Charges')

    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")
    
    image=Image.open('7.JPG')
    st.image(image,caption='Distribution of Charges for Smokers and Non-Smokers respectively')
    st.write("""
    According to the plots, it is clear smoking leads to higher Insurance Charges!
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")
    image=Image.open('8.JPG')
    st.image(image,caption='Scatterplots showing how Age wise Charges distribute for Smokers and Non-Smokers')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('9.JPG')
    st.image(image,caption='Scatterplots showing how Age wise Charges distribute for Smokers and Non-Smokers respectively')
    st.write("""
    For Smokers, charges are almost in the double range compared to Non-Smokers!
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('10.JPG')
    st.image(image,caption='Boxplots to show how charges are distributed for Female and Male respectively')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('11.JPG')
    st.image(image,caption='Distribution of BMI variable')
    st.write("""
    We can see that distribution of BMI variable is almost Normal!
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    
    image=Image.open('12.JPG')
    st.image(image,caption='Region wise countplots from dataset')
    st.write("""
    Majority of people in dataset are from SouthEast, dataset looks almost balanced.""")
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('13.JPG')
    st.image(image,caption='Scatterplot to show how BMI and Charges relate on the basis of Region')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")
    
    image=Image.open('14.JPG')
    st.image(image,caption='Scatterplot to show how Age and Charges relate on the basis of Region')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('15.JPG')
    st.image(image,caption='Pairplot to show how each quantitative variable correlate with another variable')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('16.JPG')
    st.image(image,caption='Bar charts to show the distribution of smokers and non smokers region wise')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('17.JPG')
    st.image(image,caption='Sexwise comparison of Smokers')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")
    

    image=Image.open('18.JPG')
    st.image(image,caption='Bar Chart to show how many Smokers and Non-Smokers are there while considering Sex')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    
    image=Image.open('19.JPG')
    st.image(image,caption='Jointplot to show how Age is related to Insurance Charges')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('20.JPG')
    st.image(image,caption='Jointplot to show how Region is related to Insurance Charges')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")
    
    image=Image.open('21.JPG')
    st.image(image,caption='Violinplot to show how Sex wise Charges are distributed')
    st.write("""
    It is clear that it won't much depend on the Sex of an Individual
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")
    image=Image.open('22.JPG')
    st.image(image,caption='Boxplot showing distribution of BMI for female and male respectively')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('23.JPG')
    st.image(image,caption='Scatterplot showing relationship between Age, BMI and Smoker or not')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('24.JPG')
    st.image(image,caption='How is the distribution of Age amongst Male and Female using Boxplot')
    st.write("""
    It shows that age is balanced for both male as well as female
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")


    image=Image.open('25.JPG')
    st.image(image,caption='Region wise charges for Smokers and Non-Smokers')
    st.write("""
    Clearly Smokers in each region have really high Insurance Charges
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")
    
    
    image=Image.open('26.JPG')
    st.image(image,caption='Age wise distribution of Smokers and their Insurance Charges')
    st.write("""
    It is sad to see that people with age <25 have so much Insurance Charge
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('27.JPG')
    st.image(image,caption='Regionwise distribution of Charges on the basis of No. of Childrens/Dependents')
    st.write("""
    Generally, more dependents should have more Insurance Charge, but this is not the case over here.
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")
    
    image=Image.open('28.JPG')
    st.image(image,caption='Violinplot to show how No. of Childrens/Dependents vary with Charges for Smokers and Non-Smokers')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('29.JPG')
    st.image(image,caption='lmplot to show relationship between age and charges')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('30.JPG')
    st.image(image,caption='lmplot to show relationship between bmi and charges')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('31.JPG')
    st.image(image,caption='lmplot to show relationship between children/dependent and charges')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")


    image=Image.open('32.JPG')
    st.image(image,caption='RMS Score and Accuracies for different Regression Models for All Variables')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    
    image=Image.open('33.JPG')
    st.image(image,caption='RMS Score and Accuracies for different Regression Models for variables selected using Backward Elimination')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('34.JPG')
    st.image(image,caption='RMS Score and Accuracies for different Regression Models for variables selected using Forward Selection')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")
    


#    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    

st.write('**Made with :heart: using Streamlit** - By : Anchala Krishnan, Atharva Adbe, Rutu Desai, Shubham Kokane')

