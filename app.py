import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
@st.cache_data
def load_data():
    users_df = pd.read_csv("users.csv")
    interactions_df = pd.read_csv("interactions.csv")
    return users_df, interactions_df

users_df, interactions_df = load_data()

# Load trained XGBoost Model
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("recommendation_model.json")
    return model

model = load_model()

# Function to generate recommendations
def generate_recommendations(member_id, top_n=10):
    """Returns top N recommended profiles for a given member ID"""
    
    # Get user details
    user_row = users_df[users_df["Member_ID"] == member_id]
    if user_row.empty:
        return None, "Member ID not found."
    
    user_data = user_row.iloc[0]
    
    # Filter opposite gender profiles
    opposite_gender = 1 if user_data["Gender"] == 0 else 0
    candidate_profiles = users_df[users_df["Gender"] == opposite_gender].copy()

    # Ensure we exclude past interactions
    interacted_profiles = interactions_df[interactions_df["Member_ID"] == member_id]["Target_ID"].tolist()
    candidate_profiles = candidate_profiles[~candidate_profiles["Member_ID"].isin(interacted_profiles)]
    
    if candidate_profiles.empty:
        return None, "No profiles available for recommendation."

    # Feature Engineering for Recommendations
    candidate_profiles["Age_Diff"] = abs(candidate_profiles["Age"] - user_data["Age"])
    candidate_profiles["Same_Caste"] = (candidate_profiles["Caste"] == user_data["Caste"]).astype(int)
    candidate_profiles["Same_Sect"] = (candidate_profiles["Sect"] == user_data["Sect"]).astype(int)
    candidate_profiles["Same_State"] = (candidate_profiles["State"] == user_data["State"]).astype(int)
    
    # Popularity score
    interaction_counts = interactions_df["Target_ID"].value_counts()
    candidate_profiles["Target_Popularity"] = candidate_profiles["Member_ID"].map(interaction_counts).fillna(0)

    # Define feature set
    feature_columns = ["Age_Diff", "Same_Caste", "Same_Sect", "Same_State", "Target_Popularity"]
    X_test = candidate_profiles[feature_columns]

    # Get predictions
    candidate_profiles["Score"] = model.predict_proba(X_test)[:, 1]

    # Rank recommendations
    recommendations = candidate_profiles.sort_values(by="Score", ascending=False).head(top_n)

    return recommendations, None

# Streamlit UI
st.title("💍 AI Matchmaking System")

# Member ID Input
member_id = st.number_input("Enter Member ID:", min_value=int(users_df["Member_ID"].min()), 
                            max_value=int(users_df["Member_ID"].max()), step=1)

# Button to fetch recommendations
if st.button("Get Recommendations"):
    recommendations, error = generate_recommendations(member_id, top_n=10)
    
    if error:
        st.warning(error)
    else:
        st.subheader("🔍 Top Matches:")
        st.dataframe(recommendations[["Member_ID", "Age", "Caste", "Sect", "State", "Score"]])

        # 📊 Visualization
        st.subheader("📊 Recommendation Insights")

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Age Distribution
        sns.histplot(recommendations["Age"], bins=10, kde=True, ax=ax[0])
        ax[0].set_title("Age Distribution of Recommended Profiles")

        # Caste Distribution
        sns.countplot(y=recommendations["Caste"], order=recommendations["Caste"].value_counts().index, ax=ax[1])
        ax[1].set_title("Caste Distribution")

        st.pyplot(fig)
