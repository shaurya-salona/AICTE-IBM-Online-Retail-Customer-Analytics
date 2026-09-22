# Online Retail Sales & Customer Analytics with AI

### AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026
**Organized by:** BharatCares & Edunet Foundation in collaboration with IBM SkillsBuild  
**Author:** Shaurya Salona  
**Domain:** E-Commerce / Online Retail Analytics & Predictive Machine Learning  
**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit  

---

## 1. Project Title
**Online Retail Sales & Customer Analytics with AI**  
*Exploratory Data Analysis, KPI Synthesis, RFM Customer Segmentation, and Leakage-Free Predictive Churn Modeling with Logistic Regression.*

---

## 2. Project Description
This repository contains the complete, production-grade internship capstone project analyzing a real-world enterprise transactional dataset from an online retail giftware merchant. The system processes over 540,000 transaction records to synthesize commercial KPIs, analyze seasonal purchasing trajectories, segment customers using an empirical RFM (Recency, Frequency, Monetary) matrix, and deploy an early-warning Machine Learning classifier for customer churn prevention.

---

## 3. Problem Statement
Online e-commerce and wholesale retailers operate in a non-contractual environment where customers do not formally close their accounts; they simply become inactive. Without structured transactional intelligence:
1. **Silent Customer Attrition:** Management cannot distinguish between routine purchasing intervals and high-risk account churn until valuable customer lifetime value is permanently lost.
2. **Homogeneous Customer Treatment:** Marketing expenditures are squandered when low-spend one-time buyers receive identical campaigns to high-volume commercial wholesale accounts.
3. **Inventory Volatility:** Seasonal spikes (particularly Q4 pre-holiday surges) cause acute stockouts on core evergreen SKUs, driving buyers to competitor suppliers.

---

## 4. Project Objectives
* **Data Auditing & Cleaning:** Clean 541,909 raw records by removing duplicates, handling cancellations, pruning administrative accounting adjustments, and isolating verified customer accounts.
* **Store-Level KPI Synthesis:** Calculate Total Gross Revenue, Total Units Sold, Order Volume, Average Order Value (AOV), and Repeat Customer Rate.
* **Exploratory & Diagnostic Analytics:** Identify seasonality trends, intra-day hourly trading curves, product Pareto distributions, and domestic vs. export market dynamics.
* **Empirical RFM Customer Segmentation:** Segment 4,338 verified customers into actionable tiers (*Champions*, *Loyal Customers*, *Potential Loyalists*, *At Risk*, *Lost*, *Promising*).
* **Leakage-Safe Machine Learning:** Formulate a behavioral churn proxy across a two-period time split (9-month observation vs. 3.5-month evaluation window) and train a standardized **Logistic Regression** classifier with zero data leakage.
* **Prescriptive Action Plan:** Prescribe four evidence-based retention and inventory strategies (Finding $\rightarrow$ Implication $\rightarrow$ Action).

---

## 5. Dataset Description
* **Dataset Source:** [Kaggle E-Commerce Data (Direct Download)](https://www.kaggle.com/datasets/carrie1/ecommerce-data?resource=download)
* **Local Project File:** `data/data.csv` (the exact dataset downloaded from Kaggle above)
* **Volume:** 541,909 rows and 8 attributes.
* **Period:** December 1, 2010 to December 9, 2011 (1 year, 1 week, 2 days).
* **Context:** UK-based registered online retailer supplying unique all-occasion giftware to wholesale and retail consumers across 38 countries.

### Data Schema
| Column | Type | Description |
| :--- | :--- | :--- |
| `InvoiceNo` | Object / String | 6-digit invoice code; cancellations prefixed with `'C'` |
| `StockCode` | Object / String | 5-digit unique product identifier |
| `Description` | Object / String | Product name |
| `Quantity` | Integer (int64) | Units per line item (negative values denote cancellations/returns) |
| `InvoiceDate` | Datetime / String | Transaction timestamp (`MM/DD/YYYY HH:MM`) |
| `UnitPrice` | Float (float64) | Price per unit in GBP (£); non-positives reflect adjustments |
| `CustomerID` | Float / Integer | 5-digit customer identifier; 135,080 records have missing customer IDs in the raw dataset. |
| `Country` | Object / String | Country of client residence / business registration |

> **Note on Verified Customer Count (4,372 Raw vs. 4,338 Clean):**  
> While the raw dataset logs **4,372** distinct non-null `CustomerID` values, exactly **34** of those accounts contain zero completed commercial purchases (88 cancellation rows and 1 zero-price adjustment). Excluding accounts that never completed a legitimate purchase leaves exactly **4,338** verified purchasing clients for RFM and machine learning modeling, preventing mathematical corruption of recency and frequency scores.


---

## 6. Technologies Used
* **Programming Language:** Python 3.9+
* **Data Processing:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (`LogisticRegression`, `StandardScaler`, `train_test_split`, `metrics`)
* **Serialization:** `joblib`
* **Interactive Dashboard:** `streamlit`
* **Report Generation:** `python-docx`
* **Interactive Notebook:** `jupyter`, `nbformat`, `nbconvert`

---

## 7. Project Structure
```text
AICTE-IBM-Online-Retail-Customer-Analytics/
│
├── data/
│   └── data.csv                                              # Original raw dataset (unmodified)
│
├── notebooks/
│   └── ShauryaSalona_Online_Retail_Customer_Analytics.ipynb   # Fully executed 19-section notebook
│
├── models/
│   ├── churn_model.pkl                                       # Serialized Logistic Regression model
│   └── scaler.pkl                                            # Serialized StandardScaler
│
├── report/
│   ├── ShauryaSalona_ProjectReport.docx                      # Formatted academic report (DOCX)
│   └── figures/                                              # High-resolution generated charts
│       ├── fig1_monthly_revenue.png
│       ├── fig2_hourly_distribution.png
│       ├── fig3_top_products.png
│       ├── fig4_geographic_distribution.png
│       ├── fig5_rfm_segments.png
│       ├── fig6_ml_evaluation.png
│       └── fig7_coefficients.png
│
├── app.py                                                    # Interactive Streamlit Web Dashboard
├── requirements.txt                                          # Project dependencies
└── README.md                                                 # Project documentation
```

---

## 8. Installation & Setup

### Prerequisites
* Python 3.9 or higher installed on your system.
* Virtual environment recommended.

### Step-by-Step Setup
```bash
# 1. Clone or navigate to the project directory
cd "Online Retail Sales & Customer Analytics with AI"

# 2. (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate       # On macOS/Linux
# .\venv\Scripts\activate      # On Windows

# 3. Install required dependencies
pip install -r requirements.txt
```

---

## 9. How to Run the Notebook
The notebook has been fully executed and contains all visual outputs, tables, and print statements. To re-run or inspect the notebook interactively:

```bash
# Launch Jupyter Notebook
jupyter notebook notebooks/ShauryaSalona_Online_Retail_Customer_Analytics.ipynb
```
Or execute headlessly from the command line:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/ShauryaSalona_Online_Retail_Customer_Analytics.ipynb
```

---

## 10. Machine Learning Methodology (Leakage-Free Churn)

### Leakage-Safe Time-Based Churn Formulation
In non-contractual retail, churn is modeled as a **behavioral churn proxy** rather than permanent account closure:
1. **Observation Period (Feature Window):** `2010-12-01` to `2011-08-31` (9 months, 3,317 active customers).
   * All behavioral features (`Recency`, `Frequency`, `Monetary`, `Tenure`, `AvgOrderValue`, `TotalUnits`) are computed **strictly prior to September 1, 2011**.
2. **Evaluation Period (Churn Target Window):** `2011-09-01` to `2011-12-09` (~3.5 months).
   * $\text{Churn} = 1$: Customer placed **0 orders** in the evaluation period ($n = 1,365$, $41.15\%$).
   * $\text{Churn} = 0$: Customer placed $\ge 1$ order in the evaluation period ($n = 1,952$, $58.85\%$).
3. **Zero Data Leakage:** No transactions from September 1, 2011 onward enter the feature calculation matrix.
4. **Model:** Standardized `LogisticRegression` with stratified 80/20 train/test split.

---

## 11. Key Verified Results

### Executive Macro KPIs
* **Total Gross Revenue:** £10,642,110.80
* **Total Units Sold:** 5,572,420 units
* **Total Commercial Orders:** 19,960 invoices
* **Average Order Value (AOV):** £533.17 (Median: £240.25)
* **Average Basket Size:** 279.2 units per order
* **Catalog Products (SKUs):** 3,922 active lines
* **Total Verified Customers:** 4,338 clients
* **Repeat Customer Rate:** **65.57%** (2,845 repeat buyers)

### RFM Segmentation Summary
| Customer Segment | Customer Count | Percentage (%) | Mean Recency | Mean Frequency | Mean Monetary (£) | Revenue Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Champions / High-Value** | 876 | 20.19% | 13.4 days | 12.1 orders | £6,723.83 | **66.2%** |
| **Loyal Customers** | 647 | 14.92% | 23.7 days | 3.7 orders | £1,072.73 | 7.8% |
| **Potential Loyalists** | 665 | 15.33% | 25.3 days | 1.4 orders | £751.28 | 5.6% |
| **At Risk** | 759 | 17.50% | 84.0 days | 3.3 orders | £1,255.39 | 10.7% |
| **Lost / Hibernating** | 1,084 | 24.99% | 247.0 days | 1.6 orders | £650.63 | 7.9% |
| **Promising / Needs Attention**| 307 | 7.08% | 85.2 days | 1.0 orders | £473.44 | 1.6% |

### Machine Learning Churn Model Performance
Evaluated on held-out test set ($n = 664$ customers):

| Metric | Score | Operational Significance |
| :--- | :---: | :--- |
| **Accuracy** | **66.57%** | Correct overall classification rate across retained and churned cohorts |
| **Precision** | **58.25%** | When flagged as churn, 58.3% actually cease purchasing (low false alarm cost) |
| **Recall (Sensitivity)** | **65.93%** | Model captures 65.9% (approx. 2 out of 3) of all churning accounts |
| **F1-Score** | **61.86%** | Balanced harmonic mean between precision and coverage |
| **ROC-AUC Score** | **0.7340** | Strong discriminative ability separating churning from active accounts |

#### Confusion Matrix Breakdown
* **True Negatives (TN):** 262 (Retained clients correctly predicted)
* **False Positives (FP):** 129 (Retained clients flagged as churn)
* **False Negatives (FN):** 93 (Churning clients missed)
* **True Positives (TP):** 180 (Churning clients correctly captured)

#### Standardized Logistic Regression Coefficients
* `AvgOrderValue`: $+1.5312$ (Sporadic large single orders without frequency elevate churn risk)
* `Recency`: $+0.4992$ (Every 30-day increase in inactivity elevates churn odds by ~65%)
* `Monetary`: $+0.0617$
* `Tenure`: $-0.2866$ (Longer account relationship decreases churn risk)
* `Frequency`: $-0.9710$ (Higher order frequency strongly protects against churn)
* `TotalUnits`: $-1.0022$ (Higher volume ordered strongly protects against churn)

---

## 12. Prescriptive Business Recommendations
1. **Automated 60-Day Re-engagement Triggers:** Target the 759 *At-Risk* accounts (average spend £1,255) at day 60 with automated replenishment alerts and a 5% repeat wholesale incentive.
2. **Dedicated VIP Account Management:** Dedicate personal account managers and guaranteed stock reservations to the 876 *Champions* who generate two-thirds of store cash flow (£5.89M).
3. **European Wholesale Freight Subsidies:** Leverage the high £1,200+ AOV of European export buyers (Netherlands, EIRE, Germany, France) by offering tiered pallet shipping discounts on orders over £1,000.
4. **Q4 Evergreen Inventory Buffer:** Mandate a 45-day safety stock buffer by July 31st for top revenue gift lines (`REGENCY CAKESTAND 3 TIER`, `WHITE HANGING HEART T-LIGHT HOLDER`) to capture the £1.5M November holiday surge.

---

## 13. Project Limitations
* **Behavioral Proxy:** In non-contractual commerce, churn is defined via an evaluation window proxy rather than contractual termination.
* **Missing Customer Identifiers:** 24.9% of transactions lacked an identifiable `CustomerID` in the raw data, limiting customer-level RFM tracking to verified client accounts.
* **December Truncation:** Transaction records end on December 9, 2011, providing partial-month data for December.

---

## 14. Future Scope
* Benchmark ensemble algorithms (Random Forest, XGBoost, LightGBM) with hyperparameter tuning.
* Implement Market Basket Association Rule Mining (Apriori algorithm) to discover product cross-sell affinities.
* Deploy continuous retraining pipelines (MLOps) with automated monthly batch score recalculation.

---

## 15. Interactive Dashboard (Streamlit App)
An optional interactive dashboard is available in `app.py`:
```bash
streamlit run app.py
```
Features:
* Executive KPI Overview
* Interactive Sales Trends & Country Revenue Filters
* Customer RFM Segment Explorer
* Live Real-Time Customer Churn Risk Predictor using the saved model artifacts.

---

## 16. Author & Academic Attribution
* **Candidate:** Shaurya Salona
* **Internship:** AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026
* **Organization:** BharatCares & Edunet Foundation in collaboration with IBM SkillsBuild
* **Submission Date:** September 2026
