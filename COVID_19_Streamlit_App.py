#########################################################################
########### Section 0: Import Packages and Define Functions #############
#########################################################################

# Import Necessary Packages
import streamlit as st 
import streamlit.components.v1 as components
import pandas as pd 
import numpy as np
import shap
import pickle 
from PIL import Image
import matplotlib.pyplot as plt


# Set Display Options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(suppress=True)

# Define Variables for use later on in App
#io_path = r"C:\Users\Eric\Desktop\Eric All Files\Python Projects\COVID-19 Projects"

# Load the XGBoost Classifier
xgb_clf = pickle.load(open("xgb_clf.pkl", "rb"))

#xgb_clf = pickle.load(open(io_path+r"\xgb_clf.pkl", "rb"))

# Defining a Function to show shaply plots in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Define A Function to return predicted outcome string
def human_outcome(probability):

    if probability < 0.35:
        return "Survives"
    elif probability > 0.65:
        return "Does Not Survive"
    else:
        return "Uncertain"

# Define A Function to Create a Testing Observation for the Model to Predict
def prepare_prediction_data(symptom_time=4, lab_case=1, med_cond=0, race='White', gender='Male', age=30, hospital=0, icu=0):
    
    """
    Function Name: prepare_prediction_data
    
    Arguments:
    
    - symptom_time: int. An integer that represents the number of days that it took for COVID-19 Symptoms to develop.
    
    - lab_case: int. A 1 or 0 flag to indicate if the COVID case was detected via lab results, or just a probable case without
                     the person actually getting a test.
                     
    - med_cond: int. A 1 or 0 flag to indicate if the patient had an underlying health condition.
    
    - race: str. A string that represents the race of the patient. The acceptable inputs to this argument are one of
                 ['White','Black','Hispanic','Asian','Multiple','Native Hawaiian/Pacific Islander','Indigenous']. Unknown
                 race is the reference category.
                 
    - gender: str. A string representing the gender of the patient. Should be one of ['Male','Female']. Unknown/Other gender
                   is the reference category
                   
    - age: int. An integer that represents the age of the patient in years.
    
    - hospital: int. A 1 or 0 flag indicating if the patient was hospitalized for symptoms related to COVID-19
    
    - icu: int. A 1 or 0 flag indicating if the patitent was placed in an ICU for symptoms related to COVID-19
    
    Description:
    
    This function takes in arguments about an individual patient's COVID-19 Symptoms, demographic information, and hospitalization
    status, and returns a Pandas DataFrame with one row that represents the input vector that will be passed into the trained
    XGBClassifier Model.
    """
    
    
    pred_frame = pd.DataFrame(data={'Symptom_Development_Time':np.nan}, index=[0])
    
    pred_frame['Symptom_Development_Time'] = int(symptom_time)
    pred_frame['Lab_Confirmed_Case'] = int(lab_case)
    pred_frame['MedCond_yes'] = int(med_cond)
    pred_frame['Sex_M'] = np.where(gender == 'Male',   [1],[0])
    pred_frame['Sex_F'] = np.where(gender == 'Female', [1],[0])
    pred_frame['Age_0 - 9 Years']   = np.where(age < 10, [1],[0])
    pred_frame['Age_10 - 19 Years'] = np.where(age >= 10 and age < 20, [1],[0])
    pred_frame['Age_20 - 29 Years'] = np.where(age >= 20 and age < 30, [1],[0])
    pred_frame['Age_30 - 39 Years'] = np.where(age >= 30 and age < 40, [1],[0])
    pred_frame['Age_40 - 49 Years'] = np.where(age >= 40 and age < 50, [1],[0])
    pred_frame['Age_50 - 59 Years'] = np.where(age >= 50 and age < 60, [1],[0])
    pred_frame['Age_60 - 69 Years'] = np.where(age >= 60 and age < 70, [1],[0])
    pred_frame['Age_70 - 79 Years'] = np.where(age >= 70 and age <80, [1],[0])
    pred_frame['Age_80+ Years']     = np.where(age >= 80, [1],[0])
    pred_frame['Race_Hispanic/Latino'] = np.where(race == 'Hispanic', [1],[0])
    pred_frame['Race_Black']           = np.where(race == 'Black', [1],[0])
    pred_frame['Race_White']           = np.where(race == 'White', [1],[0])
    pred_frame['Race_Multiple/Other']  = np.where(race == 'Multiple', [1],[0])
    pred_frame['Race_Native Hawaiian/Other Pacific Islander'] = np.where(race == 'Native Hawaiian/Pacific Islander', [1],[0])
    pred_frame['Race_Asian']           = np.where(race == 'Asian', [1],[0])
    pred_frame['Race_American Indian/Alaska Native']          = np.where(race == 'Indigneous', [1],[0])
    pred_frame['Hosp_yn'] = int(hospital)
    pred_frame['ICU_yn'] = int(icu)
    
    return pred_frame

#########################################################################
########################## Section 1: Main App ##########################
#########################################################################

# Start of the Main App. Write the Header and give some background info
st.markdown("# COVID-19 Survival Prediction App",unsafe_allow_html=True)
st.image("https://southkingstownri.com/ImageRepository/Document?documentID=3809", use_column_width=True)
st.markdown("""This web application uses machine learning to predict if a person diagnosed with COVID-19 will survive having the disease. It uses the COVID-19 Case Surveillance Public Use Data which is available from the CDC. That data is updated monthly, and it contains anonymized patient data from people who have contracted COVID-19. The dataset includes information about the patient, including demographic information like age, race and gender, as well as information about if the patient has an underlying health condition, if they were admitted to the hospital, and if they ultimately passed away from COVID-19. The full data set can be found __[here](https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data/vbim-akqf)__.

This web app uses the data that the user provides as inputs to an XGBoost Classifier, which then returns a probability of death, and a predicted outcome status, namely Survives, Unsure, and Does Not Survive. These three outcome statuses are based on the probability of death, and are just a short hand interpretation of the probability that the model returns.

**It should be noted that the results of this model should in no way, shape, or from be construed as medical advice. Any predictions made by this model are purely theoretical, and should not influence any decision regarding your physical health. If you have been in contact with someone who has COVID-19, or have contracted COVID-19 yourself, contact your local health provider for treatment. Please wear a mask, wash your hands, social distance, avoid large gatherings, and get vaccinated when it is your turn.**
 """, unsafe_allow_html=True)

# Provide the User Instructions about how to use the app
st.markdown("""## Instructions:

Use the sidebar by clicking on the arrow in the top left-hand corner of the screen to adjust the variables for a patient to predict their probability of survival if they contract COVID-19.

The variables that you can adjust are listed below:

1. Hospitalization Indicator
2. Intensive Care Indicator
3. Lab Confirmed Case Indicator
4. Underlying Medical Condition Indicator
5. Gender
6. Race
7. Age
8. Symptom Development Time in Days
""")

# Create a Sidebar where users can input data
st.sidebar.markdown("""### Adjust the Patient Conditions""")

hosp_yn = st.sidebar.checkbox(label='Hospital Y/N')

icu_yn = st.sidebar.checkbox(label='Intensive Care Y/N')

lab_case = st.sidebar.checkbox(label='Lab Confirmed COVID Case Y/N')

med_cond = st.sidebar.checkbox(label='Underlying Medical Condition Y/N')

gender = st.sidebar.selectbox(label='Patient Gender', 
                             options=['Male','Female','Other'], 
                             index=0)

race = st.sidebar.selectbox(label='Patient Race',
                            options=['White','Black','Hispanic','Asian',
                                     'Multiple','Native Hawaiian/Pacific Islander',
                                     'Indigneous'], index=0)

age = st.sidebar.slider(label='Age', min_value=0, max_value=120, value=30,
step=1)

symptom_time = st.sidebar.slider(label='Symptom Development Time', 
                                 min_value=0, 
                                 max_value=30, 
                                 value=5,
                                 step=1)

# Create the Vector of Test Data
test_val = prepare_prediction_data(symptom_time=symptom_time, lab_case=lab_case, 
                                   med_cond=med_cond, race=race, gender=gender,
                                    age=age, hospital=hosp_yn, icu=icu_yn)

# Show the user what variables they are using to predict survival
st.markdown("### This is the set of variables that the model will use to predict survival:")
st.dataframe(test_val,height=200)

st.markdown("### This is what the Model thinks will happen:")

outcome_df = pd.DataFrame(data={'Predicted Outcome':'Survives','Probability of Death':1}, index=[0])
outcome_df['Probability of Death'] = xgb_clf.predict_proba(test_val)[0][1]
outcome_df['Predicted Outcome'] = outcome_df['Probability of Death'].apply(human_outcome)

st.dataframe(outcome_df)

st.markdown("""## Understanding the Results:

### Impact of the Selected Varaibles on the Predicted Patient Outcome

This plot shows the contributions of each individual variable in the model in terms of how much it increased or decreased the probability of death of this specific individual with the symptom characteristics that you selected. Variables in blue mean that that variable decreased the probability of death, and variables in red mean that that variable increased the probability of death.
 """, unsafe_allow_html=True)

explainer = shap.TreeExplainer(xgb_clf)
shap_values = explainer.shap_values(test_val)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
st_shap(shap.force_plot(explainer.expected_value, shap_values, test_val,link='logit'),150)

st.markdown("""### Shapley Values Summary Plot 

Below is an Image of the Shapley Values over all variables when this model was fit on the training data. This plot shows what the model thinks the most important variables are when making predictions. The color bar on the right hand side of the chart corresponds to the value of the individual variable on the y axis. Red means high, and blue means low.

For example, we can see that for high values of Hosp_yn, meaning someone was hospitalized, the model thinks that is very important in determining if someone will live or die, and being admitted to the hospital increases the probability of death. Conversely, the model thinks if someone is between 0 and 9 years old, that they are much less likey to die from COVID-19.

<b><center>Shapley Value Summary Plot</center></b>
""", unsafe_allow_html=True)

st.image('https://raw.githubusercontent.com/ericvan1325/Streamlit-COVID-19-App/main/Shapley_Values_From_Model_JPEG.jpg')
