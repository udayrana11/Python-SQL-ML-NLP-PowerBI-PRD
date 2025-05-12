# Retail Sales Prediction - Regression Models

A regression-based machine learning project to predict outlet sales using the BigMart dataset. This repository explores multiple linear and non-linear regression algorithms with detailed evaluation, visualization, and model tuning.

---

## Project Summary

We aim to predict **Item_Outlet_Sales** based on a variety of item and outlet features using several regression models. Our goal is to build a robust, interpretable, and accurate model that generalizes well to unseen data.

---

## 🔧 Models Used & Performance Summary

| Model                         | CV R²   | Test R² | Test RMSE | Test MAE |
|------------------------------|---------|---------|-----------|----------|
| Polynomial Regression (d=2)  | **0.5900** | **0.5778** | 1141.65    | **815.78** |
| Random Forest                | 0.5581  | 0.5522  | 1175.76   | 824.17   |
| XGBoost                      | 0.5149  | 0.5017  | 1240.25   | 866.26   |
| Linear, Ridge, Lasso, Bayes  | ~0.504  | ~0.489  | 1255.82   | 945.00+  |
| SVR                          | 0.0329  | 0.0343  | 1647.43   | Very High|

✅ **Polynomial Regression** achieved the best generalization performance.

---

## Key Insights

- **Polynomial Regression (degree 2)** outperformed all other models by effectively capturing non-linear interactions.
- **Tree-based models (Random Forest & XGBoost)** overfit the training data. Regularization needed.
- **SVR severely underfit**, failing to capture signal in data.
- **Linear models (Ridge, Lasso, Bayesian Ridge)** serve as stable baselines.

---

## Recommendations & Next Steps

### ✅ Immediate Fixes
- **Transform Skewed Features** (`Item_Visibility`, `Item_MRP`) using `log` or `Box-Cox`.
- **Handle Outliers** in `Item_Visibility` and `Item_Outlet_Sales` using capping or trimming.
- **Regularize Trees**: Tune `max_depth`, `min_samples_leaf`, and XGBoost’s `eta`, `subsample`.

### 🔄 Iterative Improvements
| Step | Goal | Metric |
|------|------|--------|
| Re-train with transforms | Reduce skew | Δ CV R² ≥ +0.02 |
| Outlier cleaning | Lower error variance | MAE ↓ 5–10% |
| Hyperparameter tuning | Prevent overfitting | Train R² – CV R² gap ≤ 0.10 |
| Robust CV | Ensure stability | CV R² std dev < ±0.02 |
| Try new models | Benchmarking | RMSE/R² improvements |

---

## Tech Stack

- Python (Pandas, NumPy, Scikit-Learn, XGBoost, Seaborn, Matplotlib)
- Jupyter Notebooks
- Regression models: Linear, Ridge, Lasso, Bayesian Ridge, Polynomial, SVR, Random Forest, XGBoost

---

## Future Scope

- Implement SHAP for model explainability
- Feature engineering (interactions, encodings)
- Ensemble stacking
- Incorporate external features (e.g., region-based sales trends)

---

## 📝 Author

**Udaybhan Singh Rana**  
🔗 [LinkedIn](https://www.linkedin.com/in/udaybhan-rana/)

---