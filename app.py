"""
Streamlit Web Dashboard: Online Retail Sales & Customer Analytics with AI
Author: Shaurya Salona
Internship: AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 (BharatCares)
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Online Retail Sales & Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished interface
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F62FE;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #525252;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F4F7FB;
        border-left: 5px solid #0F62FE;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .kpi-title {
        font-size: 0.9rem;
        color: #6E6E6E;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #161616;
    }
</style>
""", unsafe_allow_html=True)

# Title Header
st.markdown('<div class="main-title">Online Retail Sales & Customer Analytics with AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AICTE | IBM SkillsBuild Internship 2026 Capstone Project • Author: <b>Shaurya Salona</b></div>', unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    paths = ["data/data.csv", "../data/data.csv", "data.csv"]
    p = None
    for cand in paths:
        if os.path.exists(cand):
            p = cand
            break
    if p is None:
        return None
    
    df = pd.read_csv(p, encoding='latin1')
    df_clean = df.drop_duplicates().copy()
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    df_sales = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)].copy()
    df_sales['TotalAmount'] = df_sales['Quantity'] * df_sales['UnitPrice']
    df_sales['YearMonth'] = df_sales['InvoiceDate'].dt.to_period('M').astype(str)
    df_sales['Hour'] = df_sales['InvoiceDate'].dt.hour
    return df_sales

@st.cache_resource
def load_model_artifacts():
    model_paths = ["models/churn_model.pkl", "../models/churn_model.pkl"]
    scaler_paths = ["models/scaler.pkl", "../models/scaler.pkl"]
    m_path = next((p for p in model_paths if os.path.exists(p)), None)
    s_path = next((p for p in scaler_paths if os.path.exists(p)), None)
    if m_path and s_path:
        model = joblib.load(m_path)
        scaler = joblib.load(s_path)
        return model, scaler
    return None, None

df_sales = load_data()
model, scaler = load_model_artifacts()

if df_sales is None:
    st.error("Dataset 'data.csv' not found. Please ensure data is in the 'data/' folder.")
    st.stop()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/combo-chart--v1.png", width=64)
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select View:",
    ["Executive KPI Dashboard", "Sales Trends & Country Diagnostics", "RFM Customer Segmentation", "Real-Time Churn Risk Predictor", "Project Documentation"]
)

# ----------------------------------------------------
# 1. Executive KPI Dashboard
# ----------------------------------------------------
if menu == "Executive KPI Dashboard":
    st.subheader("Executive Macro KPIs (Cleaned Commercial Operations)")
    
    col1, col2, col3, col4 = st.columns(4)
    total_rev = df_sales['TotalAmount'].sum()
    total_orders = df_sales['InvoiceNo'].nunique()
    total_units = df_sales['Quantity'].sum()
    aov = total_rev / total_orders
    
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Gross Sales Revenue</div><div class="kpi-value">£{total_rev*1e-6:.2f}M</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Commercial Orders</div><div class="kpi-value">{total_orders:,}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Units Dispatched</div><div class="kpi-value">{total_units*1e-6:.2f}M</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Average Order Value</div><div class="kpi-value">£{aov:.2f}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    row2_col1, row2_col2 = st.columns([2, 1])
    with row2_col1:
        st.markdown("### Monthly Sales Trajectory")
        monthly = df_sales.groupby('YearMonth')['TotalAmount'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.lineplot(data=monthly, x='YearMonth', y='TotalAmount', marker='o', color='#0F62FE', linewidth=2.5, ax=ax)
        ax.set_title("Gross Revenue by Month (Dec 2010 - Dec 2011)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Revenue (£ Millions)")
        ax.set_xlabel("Year-Month")
        plt.xticks(rotation=45)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"£{x*1e-6:.2f}M"))
        st.pyplot(fig)
        plt.close()

    with row2_col2:
        st.markdown("### Top 5 Global Markets")
        country_rev = df_sales.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False).head(5).reset_index()
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=country_rev, y='Country', x='TotalAmount', palette='Blues_r', ax=ax2)
        ax2.set_title("Revenue by Top Countries (£)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Total Revenue (£)")
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"£{x*1e-3:.0f}k"))
        st.pyplot(fig2)
        plt.close()

# ----------------------------------------------------
# 2. Sales Trends & Country Diagnostics
# ----------------------------------------------------
elif menu == "Sales Trends & Country Diagnostics":
    st.subheader("Geographical & Product Diagnostic Analysis")
    
    countries = ["All"] + sorted(df_sales['Country'].unique().tolist())
    selected_country = st.selectbox("Filter by Country:", countries)
    
    filtered_df = df_sales if selected_country == "All" else df_sales[df_sales['Country'] == selected_country]
    
    st.write(f"Showing **{len(filtered_df):,}** transactions for **{selected_country}** (Revenue: **£{filtered_df['TotalAmount'].sum():,.2f}**)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Hourly Ordering Velocity")
        hourly = filtered_df.groupby('Hour')['InvoiceNo'].nunique().reset_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=hourly, x='Hour', y='InvoiceNo', color='#0F62FE', ax=ax)
        ax.set_title(f"Orders by Hour of Day ({selected_country})", fontsize=11, fontweight='bold')
        ax.set_ylabel("Order Count")
        st.pyplot(fig)
        plt.close()
        
    with col2:
        st.markdown("### Top 10 Products by Revenue")
        top_p = filtered_df.groupby('Description')['TotalAmount'].sum().reset_index().sort_values('TotalAmount', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=top_p, y=top_p['Description'].str[:25], x='TotalAmount', palette='Greens_r', ax=ax)
        ax.set_title(f"Top 10 SKUs by Revenue ({selected_country})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Revenue (£)")
        st.pyplot(fig)
        plt.close()

# ----------------------------------------------------
# 3. RFM Customer Segmentation
# ----------------------------------------------------
elif menu == "RFM Customer Segmentation":
    st.subheader("RFM Customer Segmentation Analysis (4,338 Verified Clients)")
    
    df_cust = df_sales.dropna(subset=['CustomerID']).copy()
    df_cust['CustomerID'] = df_cust['CustomerID'].astype(int)
    snap = df_cust['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df_cust.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snap - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalAmount': 'Monetary'})

    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
    rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].astype(int).sum(axis=1)

    def assign_seg(row):
        r = int(row['R_Score'])
        f = int(row['F_Score'])
        score = int(row['RFM_Score'])
        if score >= 11: return 'Champions / High-Value'
        elif r >= 3 and f >= 3: return 'Loyal Customers'
        elif r >= 3 and f <= 2: return 'Potential Loyalists'
        elif r == 2 and f >= 2: return 'At Risk'
        elif r == 1: return 'Lost / Hibernating'
        else: return 'Promising / Needs Attention'

    rfm['Customer_Segment'] = rfm.apply(assign_seg, axis=1)
    
    seg_summary = rfm.groupby('Customer_Segment').agg(
        Count=('Recency', 'count'),
        AvgRecency=('Recency', 'mean'),
        AvgFrequency=('Frequency', 'mean'),
        AvgMonetary=('Monetary', 'mean'),
        TotalRevenue=('Monetary', 'sum')
    ).reset_index().sort_values('TotalRevenue', ascending=False)
    
    seg_summary['RevenueShare (%)'] = (seg_summary['TotalRevenue'] / rfm['Monetary'].sum()) * 100
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Customer Count per Segment")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=seg_summary, y='Customer_Segment', x='Count', palette='viridis', ax=ax)
        ax.set_title("Customer Counts by RFM Segment", fontsize=11, fontweight='bold')
        st.pyplot(fig)
        plt.close()
        
    with col2:
        st.markdown("### Revenue Share per Segment")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.pie(seg_summary['RevenueShare (%)'], labels=seg_summary['Customer_Segment'],
               autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2', len(seg_summary)),
               wedgeprops=dict(width=0.45, edgecolor='white'))
        ax.set_title("Revenue Contribution (%)", fontsize=11, fontweight='bold')
        st.pyplot(fig)
        plt.close()
        
    st.markdown("### Segment Profile Table")
    st.dataframe(seg_summary.style.format({
        'AvgRecency': '{:.1f} days',
        'AvgFrequency': '{:.1f} orders',
        'AvgMonetary': '£{:,.2f}',
        'TotalRevenue': '£{:,.2f}',
        'RevenueShare (%)': '{:.2f}%'
    }), use_container_width=True)

# ----------------------------------------------------
# 4. Real-Time Churn Risk Predictor
# ----------------------------------------------------
elif menu == "Real-Time Churn Risk Predictor":
    st.subheader("Customer Churn Risk Predictor (Logistic Regression)")
    st.markdown("""
    This machine learning inference module uses the serialized **Logistic Regression** model trained on the leakage-free 9-month observation window.
    Enter customer behavioral attributes to evaluate churn risk probability:
    """)
    
    if model is None or scaler is None:
        st.error("Trained model or scaler not found in 'models/' directory. Please ensure 'models/churn_model.pkl' and 'models/scaler.pkl' are present.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            recency = st.slider("Recency (Days since last purchase):", min_value=1, max_value=300, value=65)
            frequency = st.slider("Frequency (Number of past orders):", min_value=1, max_value=50, value=2)
        with col2:
            monetary = st.number_input("Monetary (Total spend in £):", min_value=10.0, max_value=50000.0, value=850.0, step=50.0)
            tenure = st.slider("Tenure (Days since first order):", min_value=1, max_value=300, value=180)
        with col3:
            total_units = st.number_input("Total Units Purchased:", min_value=1, max_value=20000, value=450, step=10)
            avg_order_value = monetary / frequency
            st.metric("Computed Avg Order Value (AOV)", f"£{avg_order_value:.2f}")

        if st.button("Predict Churn Risk", type="primary"):
            features = np.array([[recency, frequency, monetary, tenure, avg_order_value, total_units]])
            features_scaled = scaler.transform(features)
            churn_prob = model.predict_proba(features_scaled)[0, 1]
            churn_pred = model.predict(features_scaled)[0]
            
            st.markdown("---")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Predicted Churn Probability", f"{churn_prob*100:.1f}%")
                if churn_pred == 1:
                    st.error("⚠️ HIGH CHURN RISK — Account predicted to become inactive.")
                else:
                    st.success("✅ LOW CHURN RISK — Customer predicted to remain active.")
                    
            with res_col2:
                st.markdown("#### Recommended Operational Action:")
                if churn_prob > 0.60:
                    st.warning("Immediate CRM Intervention: Customer inactivity exceeds safety threshold. Send personalized replenishment email with 5% volume incentive.")
                elif churn_prob > 0.40:
                    st.info("Watchlist: Schedule routine account follow-up and highlight newly arrived seasonal catalog lines.")
                else:
                    st.success("Healthy Account: Routine service. Candidate for VIP wholesale tier or credit term extension.")

# ----------------------------------------------------
# 5. Project Documentation
# ----------------------------------------------------
elif menu == "Project Documentation":
    st.subheader("Academic Capstone Documentation")
    st.markdown("""
    ### Project Overview
    * **Project Name:** Online Retail Sales & Customer Analytics with AI
    * **Candidate:** Shaurya Salona
    * **Internship:** AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026
    * **Partner Organization:** BharatCares & Edunet Foundation
    
    ### Key Verified Highlights
    * **Total Analyzed Records:** 541,909 raw records (524,878 clean commercial sales).
    * **Total Gross Revenue:** £10.64 Million across 19,960 invoices and 38 countries.
    * **Verified Customer Base:** 4,338 unique clients with 65.57% repeat order rate.
    * **RFM Champions:** 876 accounts (20.2%) generate 66.2% of total store revenue (£5.89M).
    * **ML Model:** Leakage-safe Logistic Regression classifier achieving **66.57% Accuracy, 65.93% Recall, and 0.7340 ROC-AUC**.
    
    ### Repository Structure
    * `notebooks/ShauryaSalona_Online_Retail_Customer_Analytics.ipynb` (Executed Notebook)
    * `report/ShauryaSalona_Online_Retail_Customer_Analytics_ProjectReport.docx` (Formal Academic Report)
    * `models/` (Saved `churn_model.pkl` and `scaler.pkl`)
    * `requirements.txt` & `README.md`
    """)

st.markdown("---")
st.caption("AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 • Shaurya Salona")
