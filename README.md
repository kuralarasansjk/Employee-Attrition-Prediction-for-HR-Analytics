# Employee Attrition Prediction for HR Analytics 📊

This project uses a machine learning model to predict employee attrition, helping HR departments proactively identify and address factors that may lead to employees leaving the company.

## Project Description

This is an end-to-end HR analytics project built with Python, Pandas, scikit-learn, and Streamlit. The core of the project is a Random Forest Classifier model trained on the provided employee attrition dataset.

The project workflow is as follows:
1.  **Model Training (`train_model.py`)**: A script that loads the dataset, preprocesses it, and trains a RandomForest model. The trained model and a list of its features are saved as `.pkl` files for efficient use.
2.  **Interactive App (`app.py`)**: A Streamlit web application that loads the pre-trained model and provides a user-friendly interface. Users can input various employee details, and the app will provide a real-time prediction of whether the employee is likely to stay or leave.

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Install dependencies:**
    ```bash
    pip install streamlit pandas scikit-learn joblib
    ```
3.  **Run the training script to create the model files:**
    ```bash
    python train_model.py
    ```
4.  **Launch the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
The application will open in your web browser.
