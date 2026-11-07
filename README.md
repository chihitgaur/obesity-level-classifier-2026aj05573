# Obesity Level Classification - ML Assignment 2 (M.Tech AIML, BITS WILP)

## a. Problem statement
Predict a person's obesity level (7 categories) from eating habits, physical condition and
basic body measurements, using five classification algorithms, and compare them with
Accuracy, AUC, Precision, Recall, F1 and MCC. The models are served through a Streamlit app.

## b. Dataset description
- Source: UCI Machine Learning Repository - *Estimation of obesity levels based on eating habits and physical condition* (id 544)
- Instances: 2,111 | Features: 16 | Target: `NObeyesdad` (7 classes)
- Numeric features (8): Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE
- Categorical features (8): Gender, family_history_with_overweight, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS
- Classes: Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III
- Missing values: none. Classes are roughly balanced (272-351 rows each).
- About 77% of rows are synthetic (generated with SMOTE), the rest come from a real survey.
- Preprocessing: StandardScaler on numeric columns, One-Hot encoding on categorical columns (inside a sklearn Pipeline, saved with each model).
- Split: 80/20 stratified (random_state=2026) -> 1,688 train / 423 test rows. The test rows are in `test_data.csv`.

## c. GitHub repository
https://github.com/chihitgaur/obesity-level-classifier-2026aj05573

Live Streamlit app: https://obesity-level-classifier-2026aj05573-gfcnekc2w5gll2yrkr63wn.streamlit.app/

## d. Models used

### Comparison table (on the 423-row test split)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8794 | 0.9882 | 0.8785 | 0.8794 | 0.8782 | 0.8594 |
| Decision Tree | 0.9267 | 0.9573 | 0.9263 | 0.9267 | 0.9264 | 0.9144 |
| KNN | 0.8227 | 0.9652 | 0.8290 | 0.8227 | 0.8092 | 0.7965 |
| Naive Bayes | 0.5626 | 0.8898 | 0.5887 | 0.5626 | 0.5122 | 0.5098 |
| Support Vector Machine | 0.9574 | 0.9974 | 0.9582 | 0.9574 | 0.9576 | 0.9504 |

Precision, Recall and F1 are weighted averages; AUC is one-vs-rest, weighted.

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy of 0.879 with the second highest AUC (0.988), so it ranks the classes very well even though its hard predictions are not perfect. A linear boundary gets most rows right, but it mixes up neighbouring classes (for example Overweight Level I vs II) because the obesity categories follow BMI, which is not linear in Height and Weight. |
| Decision Tree | Second best model in accuracy (0.927) and MCC (0.914). It can split directly on Weight and Height, which fits how the labels were created. Its AUC (0.957) is lower than Logistic Regression because leaf nodes give coarse probabilities, and a depth-limited tree can still overfit the synthetic rows. |
| KNN | Accuracy of 0.823 with k=7. It depends on distance, so it needs scaling, and the many one-hot columns add extra distance that is not related to the obesity level. Its F1 (0.809) is lower than its accuracy, which means some classes are predicted worse than others. |
| Naive Bayes | Weakest model by a big margin (accuracy 0.563, MCC 0.510). It assumes the features are independent and Gaussian, but Height and Weight are strongly linked and the one-hot columns are not Gaussian, so the assumption does not hold on this dataset. |
| Support Vector Machine | Best on all six metrics (accuracy 0.957, AUC 0.997, MCC 0.950). The RBF kernel can draw the curved boundaries between the BMI-like classes, and scaling of the inputs helps it. |
| **Overall winner** | **Support Vector Machine.** It has the highest score on every metric, and its near-perfect AUC and MCC show it is good across all seven classes and not just the common ones. Decision Tree is a close second and is easier to interpret. |

## How to run locally
```
pip install -r requirements.txt
python train_models.py      # retrains and saves models into model/
streamlit run app.py
```

## Repository structure
```
app.py              Streamlit app
train_models.py     
requirements.txt
test_data.csv      
model/              
results/metrics.csv 
data_raw/           
```
