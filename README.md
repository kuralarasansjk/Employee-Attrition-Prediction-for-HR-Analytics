Project Title
Employee Attrition Prediction for HR Analytics 📊

Project Description
This is a machine learning project that uses an employee dataset to predict whether an employee is likely to leave a company (attrition). The goal is to provide HR departments with a predictive tool to proactively identify employees at risk of attrition based on key factors like job satisfaction, salary, and work-life balance.

The project is built with Python and uses the following libraries:

scikit-learn: For building and training the machine learning model.

Pandas: For data handling and preprocessing.

Streamlit: For creating an interactive web application that allows users to input data and get real-time predictions.

How It Works
The project operates in two main stages:

Model Training: The train_model.py script trains a Random Forest Classifier on the provided HR-Employee-Attrition.csv dataset. After training, the model and its feature columns are saved as .pkl files (attrition_model.pkl and model_features.pkl). This separates the training process from the application itself, making the web app load and run much faster.

Interactive Application: The app.py script loads the pre-trained model and provides a user-friendly interface. Users can input employee details through a sidebar, and the app uses the loaded model to predict the likelihood of attrition, providing a clear result and confidence score.

How to Use
Clone the repository: git clone [repository URL]

Install the necessary libraries: pip install -r requirements.txt (or pip install streamlit pandas scikit-learn joblib)

First, train the model by running the training script: python train_model.py

Then, launch the Streamlit app: streamlit run app.py
