# Customer Personality Segmentation System

## Overview

An AI-powered customer segmentation system built using Machine Learning and Streamlit.

This project analyzes customer behavior, segments customers into different groups, and provides:

- Real-time customer prediction
- Product recommendations
- Marketing strategy suggestions
- Interactive business dashboard

---

## Features

- Customer Segmentation using K-Means Clustering
- Real-Time Segment Prediction
- AI Product Recommendation Engine
- Marketing Strategy Generator
- Interactive Streamlit Dashboard
- PCA Visualization
- Business Insights & Analytics

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn

---

## Machine Learning Techniques

- K-Means Clustering
- PCA (Principal Component Analysis)
- Feature Scaling
- Outlier Detection
- Feature Engineering

---

## Project Structure

customer-personality-segmentation-system/
│
├── app/
│ └── streamlit_app.py
│
├── data/
│ ├── marketing_campaign.csv
│ └── customer_segments.csv
│
├── models/
│ ├── kmeans_model.pkl
│ └── scaler.pkl
│
├── notebooks/
│ └── 01_EDA.ipynb
│
├── requirements.txt
└── README.md

---

## Run Project Locally

```bash
pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py