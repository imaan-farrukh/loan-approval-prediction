# 🏦 LendWise — Loan Approval Prediction

An end-to-end Machine Learning project that predicts loan approval outcomes using applicant information, financial details, and credit history. The application is built using Python, Scikit-learn, and Streamlit.

## 1. Project Overview

LendWise is a Machine Learning application designed to predict whether a loan application is likely to be approved or rejected. It combines a trained Logistic Regression model with a user-friendly web interface.

Users can enter applicant details, view the model's prediction and estimated approval and rejection probabilities, and download a prediction report.

## 2. Problem Statement

Loan approval decisions depend on several factors, including applicant income, co-applicant income, loan amount, credit history, education, and property area.

The objective of this project is to use historical loan application data to build a classification model that predicts loan approval outcomes based on applicant characteristics.

## 3. Dataset Description

The project uses a loan prediction dataset containing applicant information and historical loan decisions.

- **Number of records:** 614
- **Input features:** 15 after preprocessing
- **Target variable:** `Loan_Status`
- **Problem type:** Binary classification

The target variable was encoded as:

- Approved (`Y`) = 1
- Rejected (`N`) = 0

The dataset contains demographic, employment, financial, credit history, and property-area information.

## 4. Data Preprocessing

The following preprocessing steps were performed to prepare the dataset for Machine Learning:

1. Handling missing values.
2. Converting categorical variables into numerical representations.
3. Encoding the target variable into binary values.
4. Applying one-hot encoding to property area and dependent categories.
5. Separating independent variables (`X`) from the target variable (`y`).
6. Preparing the final set of 15 input features for model training.

These steps transform the original dataset into a numerical format suitable for classification.

## 5. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was used to understand the dataset's structure and examine the available applicant attributes.

The dataset includes:

- Applicant demographics
- Education and employment status
- Applicant and co-applicant income
- Loan amount and repayment term
- Credit history
- Property area
- Historical loan approval status

These variables provide the information used by the model to learn patterns associated with historical loan decisions.

## 6. Model Selection

**Algorithm: Logistic Regression**

Logistic Regression was selected for this binary classification task. It estimates the probability of an application belonging to one of two classes.

The model produces one of the following predictions:

- `1` — Loan approved
- `0` — Loan rejected

The model also provides estimated class probabilities that the application displays as approval and rejection percentages.

## 7. Model Training

The model was trained using the preprocessed dataset in Google Colab.

The workflow involved:

1. Separating the input features and target variable.
2. Training a Logistic Regression classifier.
3. Evaluating the model's predictive performance.
4. Saving the trained model and feature names using Joblib.
5. Integrating the saved model into a Streamlit application.

The trained model is saved as `loan_model.pkl`. The application loads this file and arranges the input values according to the model's expected feature order before generating predictions.

## 8. Model Evaluation

The model achieved approximately **86.2% accuracy** in the previously recorded evaluation.

Accuracy measures the proportion of predictions that correctly match the actual labels. However, accuracy alone does not fully describe classification performance, particularly when the model may perform differently on approved and rejected applications.

The model's results should therefore be interpreted in the context of the dataset and evaluation procedure used during training.

## 9. Application Features

The Streamlit application provides the following features:

- User-friendly applicant information form
- Personal and financial input fields
- Credit history and property-area selection
- Loan approval or rejection prediction
- Estimated approval and rejection probabilities
- Applicant information summary
- Downloadable prediction report in CSV format

## 10. Technologies Used

- **Python** — Programming language
- **Pandas** — Data manipulation and processing
- **Scikit-learn** — Machine Learning model
- **Joblib** — Saving and loading the trained model
- **Streamlit** — Interactive web application
- **Google Colab** — Model development and training
- **Visual Studio Code** — Local development
- **GitHub** — Version control and project hosting

## 11. Project Structure

```text
LoanPredictionApp/
├── app.py
├── loan_model.pkl
├── requirements.txt
└── README.md
```

- `app.py` — Streamlit application and prediction logic
- `loan_model.pkl` — Saved Machine Learning model and feature names
- `requirements.txt` — Python dependencies required to run the application
- `README.md` — Project documentation

## 12. Installation and Usage

### Prerequisites

Python and pip must be installed on your computer.

### Step 1: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 2: Run the application

```bash
python -m streamlit run app.py
```

### Step 3: Use the application

Open the local URL displayed in the terminal. Enter the applicant's details and click **Analyze Loan Application** to view the predicted class and estimated probabilities.

Ensure that `loan_model.pkl` is present in the project directory.

## 13. Prediction Results

After the applicant's information is submitted, the application displays:

- Predicted loan approval or rejection class
- Estimated approval probability
- Estimated rejection probability
- Applicant information summary
- Option to download the prediction report

The output demonstrates how a trained Machine Learning model can be integrated into an interactive application for making predictions from user-provided data.

## 14. Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- Historical patterns may not generalize to every applicant or lending environment.
- Accuracy alone does not capture all types of classification errors.
- Input encoding and feature order must remain consistent with the model's training process.
- Predicted probabilities are estimates and are not guarantees of actual loan approval.
- The application is an educational project and should not be used as the sole basis for real financial decisions.

## 15. Future Improvements

Potential improvements include:

- Comparing Logistic Regression with other classification algorithms.
- Evaluating precision, recall, F1-score, and confusion matrix results.
- Improving model validation and probability calibration.
- Strengthening consistency between training preprocessing and application inputs.
- Deploying the application online for wider accessibility.

## 16. Conclusion

LendWise demonstrates an end-to-end Machine Learning workflow, from data preprocessing and model training to model evaluation and application development.

By integrating Logistic Regression with Streamlit, the project provides an interactive way to explore loan approval predictions. It also demonstrates practical skills in Python programming, data preparation, Machine Learning, and application development.

## Disclaimer

LendWise is an educational Machine Learning project. Its predictions are estimates based on historical data and should not be treated as actual lending decisions or financial advice.
