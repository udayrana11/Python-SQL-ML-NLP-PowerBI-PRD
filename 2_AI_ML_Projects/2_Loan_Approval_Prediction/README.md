# 🏦 Loan Approval Prediction – Data Preprocessing & Modeling

This project aims to prepare and analyze a loan approval dataset to build predictive models that classify whether a loan will be approved.

---

## 📌 Objective

To perform end-to-end data preprocessing, handle imbalances, and train multiple classification models (Logistic Regression, XGBoost, SVM, etc.) for loan approval prediction.


---

## 🔍 Steps Performed

### 1. Exploratory Data Analysis (EDA)
- Plotted distributions (Histograms, KDE, Boxplots) for numeric columns.
- Used count plots for categorical features (Gender, Education, etc.).
- Generated a correlation heatmap.

### 2. Data Cleaning
- Dropped rows with critical missing values (`Credit_History`, `Gender`, `Married`).
- Filled remaining missing values with mode (e.g., `Self_Employed`, `Loan_Amount_Term`).
- Mapped `3+` dependents to `4` and handled datatype consistency.
- Removed duplicates and corrected data types.

### 3. Outlier Removal & Transformation
- Used IQR method to remove outliers from:
  - `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`
- Applied log transformation (log(x + 1)) to reduce skewness and normalize:
  - `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`

### 4. Feature Engineering
- Created new features:
  - `Log_ApplicantIncome`, `Log_CoapplicantIncome`, `Log_LoanAmount`
- Applied label encoding to:
  - `Gender`, `Married`, `Education`, `Self_Employed`, `Property_Area`, `Loan_Status`

### 5. Handling Imbalanced Data
- Used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance classes in `Loan_Status`.

### 6. Train-Test Split
- Performed 80/20 split using `train_test_split`.

---

## 🔧 Libraries Used

- `Pandas`, `NumPy`
- `Matplotlib`, `Seaborn`
- `scikit-learn`
- `xgboost`
- `imbalanced-learn`

---

## 🚀 Model Evaluation

- Train and compare the following models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Support Vector Machine (SVM)
- Evaluate model performance using:
  - Accuracy
  - ROC-AUC
  - Confusion Matrix
  - F1-score

---

## 📂 Dataset

> **Note**: Due to size limitations, the dataset is not included in this repository. You can download it here:

[📥 Click here to download the dataset (Google Drive)](https://drive.google.com/file/d/1VRffer1vnCA7vB5xUaz7cMsu1ymC5H3k/view?usp=drive_link)

---

## 📸 Visuals

 `/images` folder: https://drive.google.com/drive/folders/14a2V9qMFj_FPDsC_Eqxpfrtz1zBDu6oO?usp=drive_link



---

## 📝 Author

**Udaybhan Singh Rana**  
🔗 [LinkedIn](https://www.linkedin.com/in/udaybhan-rana/)

---