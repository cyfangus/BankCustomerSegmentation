# Bank Customer Segmentation

**Portfolio Project | Machine Learning | Data Science for Banking**

## Overview

This project applies clustering and supervised ML techniques to real-world retail banking data, with the aim of segmenting customers for actionable business insights. By integrating traditional RFM (Recency, Frequency, Monetary) features and advanced demographic analysis, the project illustrates how banks can identify, understand, and prioritize their most valuable customer segments to drive revenue and retention.

## Dataset

- Source: [Kaggle Open Dataset](https://www.kaggle.com/datasets)
- 1M+ transactions by 800K+ customers at an Indian bank
- Features: Transaction & customer details, including gender, location, balance, and age

## Key Steps

1. **Data Preparation & Cleaning**
   - Drop missing data for robust modeling
   - Feature transformation (log-scale balance and amounts)
   - Demographic binning and grouping for high-cardinality variables

2. **Customer Segmentation**
   - Aggregate behavioral metrics per customer (transaction sums, recency, frequency)
   - K-Means clustering (n=4) based on standardized features
   - Profiles: Premier Clients, Mass Market Customers, Dormant/At-Risk, Transactional/Emerging

3. **Segment Profiling & Demographic Analysis**
   - Revenue contribution and population share per segment
   - Statistical testing (Chi-square, Kruskal–Wallis) for gender, location, age differences
   - Geographically targeted business recommendations

4. **Supervised Model for Segment Prediction**
   - LightGBM classifier with engineered features and class weighting
   - Performance evaluation (precision, recall, F1 before/after class weights)
   - Insights on model effectiveness for operational deployment

## Results

- **Premier Clients** (20% of customers) generate >80% of revenue
- Segment assignment strongly linked to gender, age, and geography
- Class weighting in supervised models boosts recall for high-value client targeting

## Usage

- Run the notebook (`BankCustomerSegmentation.ipynb`) in Jupyter or Google Colab
- Follow cell-by-cell workflow: Load data, aggregate features, cluster, profile, and evaluate models
- Use results to inform bank marketing, retention, and product strategies

## Dependencies

- Python 3.9+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, scipy, lightgbm, shap

## Author

- **cyfangus**
- Portfolio: [cyfangus.github.io](https://cyfangus.github.io)
- LinkedIn: [profile](https://linkedin.com/in/cyfangus)

## License

See LICENSE for details.

