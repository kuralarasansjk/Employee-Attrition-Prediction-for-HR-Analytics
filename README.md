# Employee Attrition Prediction for HR Analytics

## Overview
An interactive web application built with Streamlit that helps HR professionals predict employee attrition using machine learning.

## Features
- 🔮 Real-time attrition predictions
- 📊 Interactive input controls
- 💡 Common sense validation of predictions
- 📈 Probability scores for prediction confidence

## Technical Stack
- Python 3.7+
- Streamlit
- Scikit-learn
- Pandas
- Joblib

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd Kuralarasan_Employee_Attrition_Prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. First, train the model:
```bash
python train_model.py
```

2. Launch the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and navigate to `http://localhost:8501`

## Input Parameters

### Demographics
- **Age**: 18-65 years
- **Gender**: Female/Male
- **Marital Status**: Single/Married/Divorced

### Job Related
- **Department**: HR/R&D/Sales
- **Job Role**: Multiple positions available
- **Job Level**: Trainee to VP
- **Years at Company**: 0-40 years
- **Distance from Home**: 1-29 miles

### Work Conditions
- **Business Travel**: Non-Travel/Frequent/Rare
- **Work Life Balance**: Scale 1-4
- **Overtime**: Yes/No
- **Job Satisfaction**: Scale 1-4

### Career Metrics
- **Monthly Income**: 2-24 LPA
- **Years Since Last Promotion**: 0-15 years
- **Number of Companies Worked**: 0-9

## Project Structure
```
├── app.py                  # Main Streamlit application
├── train_model.py         # Model training script
├── attrition_model.pkl    # Trained model
├── model_features.pkl     # Model features
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)