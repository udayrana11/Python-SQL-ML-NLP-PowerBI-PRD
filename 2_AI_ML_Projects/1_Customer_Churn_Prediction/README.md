# 📉 Telco Customer Churn Prediction

This project focuses on building a machine learning model to predict customer churn for a telecommunications company. The goal is to predict whether a customer will churn (leave the service) based on various features like service type, contract length, and monthly charges.

---

## 📂 Dataset

- **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Rows:** 7,043  
- **Target:** `Churn` (Yes/No)

---

## 🧰 Libraries Used

- `pandas`, `numpy`: Data manipulation and analysis
- `matplotlib`, `seaborn`: Data visualization
- `scikit-learn`: Machine learning models, preprocessing, evaluation, and metrics
  - Models: `LogisticRegression`, `RandomForestClassifier`, `DecisionTreeClassifier`, `KNeighborsClassifier`, `SVC`, `GaussianNB`
  - Preprocessing: `LabelEncoder`, `StandardScaler`
  - Evaluation: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `classification_report`
- `joblib`: Model serialization for saving and loading the trained models

---

## ⚙️ Workflow

### 1. Data Preprocessing
- **Data cleaning**: Removed `customerID` column, converted `TotalCharges` to numeric (replacing blanks with the mode of `MonthlyCharges`).
- **Missing values**: Handled by replacing blank `TotalCharges` with the mode of `MonthlyCharges`.
- **Categorical Encoding**: 
  - Binary features were encoded using `LabelEncoder`.
  - Multiclass features were encoded using `pd.get_dummies`.

### 2. Exploratory Data Analysis (EDA)
- **Important features**: `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`
- **Visualizations**: Histograms, boxplots, and heatmaps to explore distributions and relationships.

### 3. Feature Scaling & Splitting
- **Train-test split**: Split the data into training and test sets (80/20) using stratified sampling to maintain the distribution of churned vs. non-churned customers.
- **Scaling**: Used `StandardScaler` to normalize features for better model performance.

---

## 🤖 Models Trained

### Models Used:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **K-Nearest Neighbors**
- **Support Vector Machine**
- **Naive Bayes**

### Model Performance:
| Model                  | CV Accuracy | Test Accuracy | Precision | Recall | F1 Score | ROC AUC |
|------------------------|-------------|---------------|-----------|--------|----------|---------|
| ✅ Logistic Regression | 0.7985      | 0.8162        | 0.6792    | 0.5791 | 0.6252   | ✅ 0.8616 |
| Support Vector Machine | 0.7930      | 0.8070        | ✅ 0.6850 | 0.5013 | 0.5789   | 0.8132   |
| Random Forest          | 0.7843      | 0.7977        | 0.6642    | 0.4772 | 0.5554   | 0.8349   |
| K-Nearest Neighbors    | 0.7543      | 0.7779        | 0.6000    | 0.4826 | 0.5349   | 0.7767   |
| Naive Bayes            | 0.7488      | 0.7566        | 0.5275    | ✅ 0.7721 | 0.6268   | 0.8430   |
| Decision Tree          | 0.7244      | 0.7253        | 0.4821    | 0.5067 | 0.4941   | 0.6571   |

## ✅ Actionable Insights

### 🔹 Start with Logistic Regression
- Offers the **best overall balance** of performance across:
  - **Accuracy**
  - **F1 Score**
  - **ROC AUC**
- Ideal as a **baseline model**.

### 🔹 Explore SVM Further
- Delivers the **highest precision**, which is useful when:
- Reducing **false positives** is a priority (e.g., churn prediction where wrongly flagging non-churners can cost trust).

### 🔹 For Random Forest & Decision Tree
- **Action**: Normalize or log-transform the `TotalCharges` variable.
  - These models are sensitive to **skewed distributions**.
  - Reevaluate their performance after this transformation.

### 🔹 Use Naive Bayes Only if Recall is Critical
- Suitable for scenarios where **minimizing false negatives** matters more than precision.
- Example: **Healthcare alerts**, **fraud detection**, or **early risk identification**.


---

## 🔧 Hyperparameter Tuning

## 🎯 Post-Tuning Evaluation & Insights

After hyperparameter tuning (GridSearchCV on 5 folds), here's the performance summary:

| Model                  | Best Hyperparameters                                                  | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|------------------------|-----------------------------------------------------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression    | `{'C': 100, 'penalty': 'l2', 'solver': 'liblinear'}`                 | 0.8062   | ✅ 0.6897  | 0.5222 | ✅ 0.5944 | ✅ 0.8593 |
| Support Vector Machine | `{'svc__C': 10, 'svc__gamma': 0.001, 'svc__kernel': 'rbf'}`          | 0.8020   | 0.6831    | 0.5065 | 0.5817   | 0.8533  |
| Random Forest          | `{'max_depth': 10, 'max_features': 'log2', 'min_samples_split': 10}`| 0.7984   | 0.6911    | 0.4674 | 0.5576   | 0.8552  |

### 📊 Confusion Matrices:
- **Logistic Regression**: `[[936, 90], [183, 200]]`
- **SVM**: `[[936, 90], [189, 194]]`
- **Random Forest**: `[[946, 80], [204, 179]]`

---

## 🧠 Observations

1. **Model performance remains consistent**, with slight gains in precision post-tuning.
2. **Logistic Regression** remains the **most balanced** across all key metrics:
   - Accuracy: 80.62%
   - Precision: **↑ 68.97%** (from 66.77%)
   - ROC AUC: 0.8593
3. **SVM** shows improved precision but still suffers from lower recall.
4. **Random Forest** has high precision but reduced recall, making it less balanced overall.

---

## ✅ Recommendation

> Use **Tuned Logistic Regression** as your **primary model** – it provides the **best trade-off** between precision, recall, and overall AUC.

---

## 🔜 Next Steps

- For **even higher precision**, consider SVM or Random Forest depending on your use-case tolerance for lower recall.
- **Normalize or log-transform `TotalCharges`** if using tree-based models to reduce skewness and improve generalization.
- Re-run tuning with normalized data for deeper model optimization.


---

## 📝 Author

**Udaybhan Singh Rana**  
🔗 [LinkedIn](https://www.linkedin.com/in/udaybhan-rana/)

---