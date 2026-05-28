import joblib
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#custom Design

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #312e81
    );
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111827,
        #1e3a8a
    );
    border-right: 2px solid #3b82f6;
}

/* Titles */
h1, h2, h3 {
    color: #ffffff;
    font-weight: 700;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 25px;
    font-size: 16px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(
        90deg,
        #7c3aed,
        #2563eb
    );
}

/* Input Boxes */
.stNumberInput input,
.stTextInput input,
.stSelectbox div {
    background-color: rgba(255,255,255,0.08) !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
}

/* Success Box */
.stSuccess {
    background-color: rgba(34,197,94,0.2);
    border: 1px solid #22c55e;
    border-radius: 12px;
    color: white;
}

/* Charts */
canvas {
    border-radius: 15px;
}

/* Download Button */
.stDownloadButton > button {
    background: linear-gradient(
        90deg,
        #f59e0b,
        #ef4444
    );
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

/* Section Containers */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Smooth Animations */
* {
    transition: all 0.3s ease;
}

</style>
""", unsafe_allow_html=True)

# Page Config
st.set_page_config(
    page_title="Customer Segmentation System",
    layout="wide"
)

# Title
st.title("✨ Customer Personality Segmentation System")

# Load Dataset
df = pd.read_csv("data/customer_segments.csv")

model = joblib.load("models/kmeans_model.pkl")

scaler = joblib.load("models/scaler.pkl")

# Recommendation 

def get_recommendations(segment):

    recommendations = {

        'Premium Customers': [
            'Luxury Skincare Kit',
            'Premium Fashion Collection',
            'High-End Electronics'
        ],

        'Budget Customers': [
            'Discount Coupons',
            'Affordable Combo Packs',
            'Budget Grocery Kits'
        ],

        'Family Customers': [
            'Family Meal Packs',
            'Kids Products',
            'Household Essentials'
        ],

        'Loyal Customers': [
            'VIP Membership',
            'Exclusive Rewards',
            'Early Access Products'
        ]
    }

    return recommendations.get(segment, [])

# Sidebar
st.sidebar.header("Filter Customers")

selected_segment = st.sidebar.selectbox(
    "Select Customer Segment",
    options=df['Customer_Segment'].unique()
)

# Filter Data
filtered_df = df[
    df['Customer_Segment'] == selected_segment
]

# KPI Metrics
col1, col2, col3 = st.columns(3)

col1.metric(
    "👥 Customers",
    len(filtered_df)
)

col2.metric(
    "💰 Average Income",
    round(filtered_df['Income'].mean(), 2)
)

col3.metric(
    "🛒 Average Spending",
    round(filtered_df['Total_Spending'].mean(), 2)
)

# Dataset Preview
st.subheader("📁 Filtered Customer Dataset")

st.dataframe(filtered_df.head())

# Charts
col4, col5 = st.columns(2)

with col4:

    fig, ax = plt.subplots(figsize=(6,4))

    sns.histplot(
        filtered_df['Income'],
        bins=20,
        ax=ax
    )

    plt.title("Income Distribution")

    st.pyplot(fig)

with col5:

    fig2, ax2 = plt.subplots(figsize=(6,4))

    sns.scatterplot(
        x=filtered_df['Income'],
        y=filtered_df['Total_Spending'],
        ax=ax2
    )

    plt.title("Income vs Spending")

    st.pyplot(fig2)

# Cluster Summary
st.subheader("📊 Segment Summary")

summary = filtered_df[
    [
        'Income',
        'Age',
        'Total_Spending',
        'Total_Purchases'
    ]
].describe()

st.dataframe(summary)

# Download Button
csv = filtered_df.to_csv(index=False)

st.download_button(
    label="Download Filtered Data",
    data=csv,
    file_name='filtered_customers.csv',
    mime='text/csv'
)

st.header("🤖 Predict Customer Segment")

income = st.number_input(
    "Enter Income",
    min_value=0
)

age = st.number_input(
    "Enter Age",
    min_value=18
)

total_purchases = st.number_input(
    "Enter Total Purchases",
    min_value=0
)

family_size = st.number_input(
    "Enter Family Size",
    min_value=1
)

recency = st.number_input(
    "Enter Recency",
    min_value=0
)

if st.button("Predict Segment"):

    input_data = np.array([
        [
            income,
            age,
            0,
            total_purchases,
            family_size,
            recency
        ]
    ])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    cluster_names = {
        0: 'Budget Customers',
        1: 'Premium Customers',
        2: 'Family Customers',
        3: 'Loyal Customers'
    }

    predicted_segment = cluster_names[
        prediction[0]
    ]

    st.success(
        f"Predicted Segment: {predicted_segment}"
    )

    st.subheader("Recommended Products")

    products = get_recommendations(
        predicted_segment
    )

    for product in products:

        st.write("✅", product)