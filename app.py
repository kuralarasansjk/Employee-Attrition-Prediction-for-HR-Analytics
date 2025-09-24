import streamlit as st
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# --- Page Title and Introduction ---
st.title("Employee Attrition Prediction for HR Analytics")
st.markdown("This application predicts the likelihood of an employee leaving the company.")
st.markdown("Please provide the employee's information in the sidebar to get a prediction.")

# --- Load Pre-trained Model and Features ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('attrition_model.pkl')
        feature_names = joblib.load('model_features.pkl')
        return model, feature_names
    except FileNotFoundError:
        st.error("Error: Model files not found.")
        st.warning("Please run `python train_model.py` first to create the model files.")
        st.stop()

model, feature_names = load_artifacts()

# --- Sidebar for User Input ---
st.sidebar.header("Employee Information")

# Helper function to create dropdowns
def create_dropdown(label, options):
    return st.sidebar.selectbox(label, options)

# Get unique values for dropdowns (based on your original dataset)
# These lists are hardcoded since the app doesn't access the raw CSV file
business_travel_options = ['Non-Travel', 'Travel_Frequently', 'Travel_Rarely']
department_options = ['Human Resources', 'Research & Development', 'Sales']
education_field_options = ['Life Sciences', 'Medical', 'Marketing', 'Other', 'Technical Degree', 'Human Resources']
gender_options = ['Female', 'Male']
jobrole_options = ['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative']
maritalstatus_options = ['Divorced', 'Married', 'Single']
overtime_options = ['No', 'Yes']

# Mapping for user's job level names to numeric values
job_level_map = {
    'Trainee': 1, 'Junior': 1, 'Associate': 2, 'Senior': 3, 'Manager': 4, 'Director': 5, 'VP': 5, 'Other': 1
}
job_level_options = list(job_level_map.keys())

# Create input widgets as per the user's request
user_input = {}
user_input['Age'] = st.sidebar.slider("Age", 18, 65, 30)
user_input['BusinessTravel'] = create_dropdown("Business Travel", business_travel_options)
user_input['Department'] = create_dropdown("Department", department_options)
user_input['EducationField'] = create_dropdown("Education Field", education_field_options)
user_input['JobSatisfaction'] = st.sidebar.select_slider("Job Satisfaction (Level 1-4)", options=[1, 2, 3, 4], value=3)
user_input['Gender'] = create_dropdown("Gender", gender_options)
user_input['JobRole'] = create_dropdown("Job Role", jobrole_options)
user_input['WorkLifeBalance'] = st.sidebar.select_slider("Work Life Balance (Level 1-4)", options=[1, 2, 3, 4], value=3)
user_input['MaritalStatus'] = create_dropdown("Marital Status", maritalstatus_options)
user_input_job_level = create_dropdown("Job Level", job_level_options)
user_input['JobLevel'] = job_level_map[user_input_job_level]
user_input['YearsAtCompany'] = st.sidebar.slider("Years at Company", 0, 40, 5)
user_input['DistanceFromHome'] = st.sidebar.slider("Distance from Home (miles)", 1, 29, 10)
user_input['OverTime'] = create_dropdown("OverTime", overtime_options)
user_input['YearsInCurrentRole'] = st.sidebar.slider("Years in Current Role", 0, 20, 2)
user_input['NumCompaniesWorked'] = st.sidebar.slider("Number of Companies Worked", 0, 9, 1)
user_input['MonthlyIncomeLPA'] = st.sidebar.slider("Monthly Income (LPA)", 2.0, 24.0, 5.0, step=1.0)
user_input['YearsSinceLastPromotion'] = st.sidebar.slider("Years Since Last Promotion", 0, 15, 1)

# --- Prediction Logic ---
if st.sidebar.button("Predict Attrition"):
    # Create a DataFrame from the user's input
    input_df = pd.DataFrame([user_input])
    
    # One-hot encode the categorical features from user input
    categorical_features_input = [
        'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 
        'MaritalStatus', 'OverTime'
    ]
    input_encoded = pd.get_dummies(input_df, columns=categorical_features_input, drop_first=True)
    
    # Align the columns of the input with the trained model's features
    input_aligned = input_encoded.reindex(columns=feature_names, fill_value=0)
    
    # Make a prediction
    prediction = model.predict(input_aligned)
    prediction_proba = model.predict_proba(input_aligned)
    
    st.subheader("Prediction Result")
    
    # Display the result
    if prediction[0] == 1:
        st.error(f"**This employee is likely to leave the company.**")
        st.markdown(f"**Likelihood: {prediction_proba[0][1] * 100:.2f}%**")
    else:
        st.success(f"**This employee is likely to stay.**")
        st.markdown(f"**Likelihood: {prediction_proba[0][0] * 100:.2f}%**")
    
    st.info("💡 **Common Sense Check:** The model prediction takes into account various factors. For instance, low monthly income, low job satisfaction, and a lack of recent promotion generally increase the likelihood of attrition.")