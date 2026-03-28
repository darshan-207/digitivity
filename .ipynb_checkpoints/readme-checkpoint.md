# Mental Health in Tech — Predictive Analysis
> *Can we predict whether a tech employee will seek mental health treatment — before they need to?*

**Author:** Darshan S | M.Sc. Data Science | Coimbatore Institute of Technology  
**Submitted for:** Digitivity Solutions — Data Science Internship Assessment | March 2026

---

## Why This Dataset?

Digitivity Solutions builds healthcare applications that bridge the gap between patients and care providers. Rather than picking a generic dataset, this project was designed to align directly with Digitivity's product vision.

Mental health conditions cost the global economy an estimated **$1 trillion per year** in lost productivity. In the tech industry, stigma and lack of awareness prevent employees from seeking help early. A predictive tool that identifies at-risk employees from workplace survey signals is exactly the kind of digital health product Digitivity builds.

This is not just an academic exercise — it is a **proof-of-concept for a real product.**

---

## Project Highlights

| What | Detail |
|---|---|
| Dataset | OSMI Mental Health in Tech Survey (Kaggle) |
| Records | 1,251 cleaned responses |
| Features | 25 original + 3 engineered |
| Models trained | 4 (Logistic Regression, Random Forest, XGBoost, MLP Neural Network) |
| Best model | Random Forest (Tuned) — 78.1% accuracy, 0.850 AUC-ROC |
| Bonus | NLP sentiment analysis on free-text employee comments |
| Visualizations | 8 saved figures |

---

## Results

| Model | CV Accuracy | Test Accuracy | F1 Score | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | 0.699 ± 0.018 | 0.721 | 0.718 | 0.786 |
| XGBoost | 0.692 ± 0.029 | 0.725 | 0.725 | 0.799 |
| MLP Neural Network | 0.700 ± 0.042 | 0.749 | 0.755 | 0.785 |
| **Random Forest (Tuned)** | **0.726 ± 0.023** | **0.781** | — | **0.850** |

All models evaluated with **5-fold stratified cross-validation** for robust, unbiased estimates — including the neural network.

---

## Key Findings

**1. Work interference is the most actionable signal**  
Employees whose mental health frequently interferes with work seek treatment at **85%** vs only **14.2%** for those unaffected — a 71-point gap. This is a real-time, monitorable workplace signal.

**2. Family history nearly doubles treatment likelihood**  
74% of employees with a family history of mental illness seek treatment vs 35.4% without. The strongest static predictor in the dataset.

**3. Employer culture has measurable impact**  
A composite `employer_support_score` (engineered feature combining benefits + wellness + encouragement) ranked **6th in feature importance** — validating that workplace culture collectively drives help-seeking behaviour beyond any single policy.

**4. Deep learning is not always the answer**  
The MLP's cross-validation variance (±0.042) was nearly double Random Forest's (±0.023), and its training curve showed classic overfitting on 1,251 samples. Random Forest was the correct model for this problem — and the experiment proves it empirically.

**5. Treatment-seekers write more positively**  
NLP sentiment analysis on 308 free-text comments revealed treatment-seeking employees write with higher polarity (0.117 vs 0.055) — awareness and self-resolution, not just distress, drives help-seeking.

---

## Feature Engineering

Three new features were engineered beyond the raw dataset:

| Feature | Description | Importance Rank |
|---|---|---|
| `employer_support_score` | Composite of benefits + wellness_program + seek_help | 6th |
| `high_risk` | Binary flag: family history AND frequent work interference | 8th |
| `age_group` | Career-stage buckets (18–25, 26–35, 36–45, 46+) | 15th |

---

## Project Structure

```
digitivity/
├── notebooks/
│   ├── mental_health_analysis.ipynb   ← main submission notebook
│   ├── eda_exploration.ipynb          ← EDA scratch work
│   └── nlp_sentiment.ipynb            ← NLP bonus analysis
├── data/
│   ├── survey.csv                     ← raw Kaggle dataset (unmodified)
│   └── survey_cleaned.csv             ← cleaned output
├── outputs/
│   ├── eda_demographics.png           ← Figure 1
│   ├── correlation_heatmap.png        ← Figure 2
│   ├── workplace_factors.png          ← Figure 3
│   ├── model_comparison.png           ← Figure 4
│   ├── confusion_matrices.png         ← Figure 5
│   ├── shap_summary.png               ← Figure 6 (feature importance)
│   ├── nlp_sentiment.png              ← Figure 7
│   └── dl_training_curve.png          ← Figure 8
└── README.md
```

---

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/darshan-207/digitivity.git
cd digitivity
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow textblob shap
```

**3. Download the dataset**  
Download `survey.csv` from Kaggle:  
https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey  
Place it in the `data/` folder.

**4. Run the notebook**
```bash
jupyter notebook notebooks/mental_health_analysis.ipynb
```

Run `Kernel → Restart & Run All` to execute all cells cleanly from top to bottom.

---

## Notebook Structure

The main notebook `mental_health_analysis.ipynb` is structured as a narrative — each section builds on the previous with markdown explanations written for a non-technical reader.

| Section | Description |
|---|---|
| 1. Problem framing | Business context, why this matters for Digitivity |
| 2. Data loading & audit | Shape, dtypes, null analysis, class balance |
| 3. Data cleaning | Age filtering, gender standardization, null imputation |
| 4. EDA | 3 figures — demographics, correlation heatmap, workplace factors |
| 5. Feature engineering | 3 new features with business rationale |
| 6. Model training | 4 models with 5-fold CV — LR, RF, XGBoost, MLP |
| 7. Hyperparameter tuning | RandomizedSearchCV on Random Forest |
| 8. Model comparison | ROC curves, confusion matrices, full comparison table |
| 9. Feature importance | Random Forest Gini importance — all 25 features ranked |
| 10. NLP bonus | TextBlob sentiment on employee free-text comments |
| 11. Conclusion | Findings, product recommendations, limitations |

---

## Product Recommendations for Digitivity

| Recommendation | Implementation |
|---|---|
| Deploy a screening API | Wrap tuned RF as a FastAPI endpoint for HR platforms |
| Monitor 3 key signals | Work interference + family history + employer support score |
| Add sentiment layer | VADER/TextBlob on pulse survey free-text — weekly trending |
| Target 25–35 age group | Highest density of at-risk employees in tech |
| Frame positively | App messaging should normalise help-seeking, not crisis-flag |

---

## Limitations

- Dataset is from 2014 — mental health attitudes in tech have shifted significantly post-COVID
- Self-reported survey data introduces response bias
- 75% of free-text comments were missing — NLP findings are directional, not conclusive
- A production system would require HIPAA / data privacy compliance

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.0-lightblue)
![Scikit-learn](https://img.shields.io/badge/ScikitLearn-1.3-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![TextBlob](https://img.shields.io/badge/TextBlob-NLP-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)

---

*"The goal is not to predict mental illness — it is to predict the conditions under which people feel safe enough to seek help."*