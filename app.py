import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.express as px
from collections import Counter
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="MatchVerse - AI Matchmaking", layout="wide")
st.title("🔍 MatchVerse - AI Matchmaking Recommendations")

# File paths
USERS_CSV_PATH = "users.csv"
INTERACTIONS_CSV_PATH = "interactions.csv"
MODEL_JSON_PATH = "recommendation_model.json"

@st.cache_data
def load_users():
    return pd.read_csv(USERS_CSV_PATH)

@st.cache_data
def load_interactions():
    return pd.read_csv(INTERACTIONS_CSV_PATH)

users_df = load_users()
interactions_df = load_interactions()

# Label Encoding for categorical features
label_encoders = {}
for col in ["Gender", "Marital_Status", "Sect", "Caste", "State"]:
    le = LabelEncoder()
    users_df[col] = le.fit_transform(users_df[col])
    label_encoders[col] = le

# Load trained XGBoost model
bst = xgb.Booster()
bst.load_model(MODEL_JSON_PATH)

# Features used in the model
MODEL_FEATURES = ["Age_Diff", "Same_Caste", "Same_Sect", "Same_State", "Target_Popularity"]

# Precompute Target Popularity
interaction_counts = interactions_df["Target_ID"].value_counts()

def get_recommendations(member_id):
    """Fetch recommendations based on the trained model."""
    user_row = users_df[users_df["Member_ID"] == member_id]
    if user_row.empty:
        return {"error": "User not found"}

    user_meta = user_row.iloc[0]
    user_details = {
        "Member_ID": int(user_meta["Member_ID"]),
        "Gender": label_encoders["Gender"].inverse_transform([int(user_meta["Gender"])])[0],
        "Age": int(user_meta["Age"]),
        "Marital_Status": label_encoders["Marital_Status"].inverse_transform([int(user_meta["Marital_Status"])])[0],
        "Sect": label_encoders["Sect"].inverse_transform([int(user_meta["Sect"])])[0],
        "Caste": label_encoders["Caste"].inverse_transform([int(user_meta["Caste"])])[0],
        "State": label_encoders["State"].inverse_transform([int(user_meta["State"])])[0],
    }

    # Get opposite gender
    opposite_gender_encoded = 1 - user_meta["Gender"]
    eligible_profiles = users_df[users_df["Gender"] == opposite_gender_encoded].copy()

    # Exclude past interactions
    interacted_users = set(interactions_df[interactions_df["Member_ID"] == member_id]["Target_ID"])
    fresh_profiles = eligible_profiles[~eligible_profiles["Member_ID"].isin(interacted_users)].copy()

    if fresh_profiles.empty:
        return {"user_details": user_details, "recommended_profiles": [], "statistics": {}}

    # Feature Engineering
    fresh_profiles["Age_Diff"] = abs(fresh_profiles["Age"] - user_meta["Age"])
    fresh_profiles["Same_Caste"] = (fresh_profiles["Caste"] == user_meta["Caste"]).astype(int)
    fresh_profiles["Same_Sect"] = (fresh_profiles["Sect"] == user_meta["Sect"]).astype(int)
    fresh_profiles["Same_State"] = (fresh_profiles["State"] == user_meta["State"]).astype(int)
    fresh_profiles["Target_Popularity"] = fresh_profiles["Member_ID"].map(interaction_counts).fillna(0)

    # Prepare data for model
    X_test = fresh_profiles[MODEL_FEATURES]
    dtest = xgb.DMatrix(X_test)

    # Predict scores
    fresh_profiles["Score"] = bst.predict(dtest)

    # Get Top 100 recommendations
    recommended_profiles_df = fresh_profiles.sort_values(by="Score", ascending=False).head(100)

    # Decode categorical features
    for col in ["Gender", "Marital_Status", "Sect", "Caste", "State"]:
        recommended_profiles_df[col] = label_encoders[col].inverse_transform(recommended_profiles_df[col].astype(int))

    # Statistics
    statistics = {
        "age_distribution": dict(Counter(recommended_profiles_df["Age"])),
        "state_distribution": dict(Counter(recommended_profiles_df["State"])),
        "caste_distribution": dict(Counter(recommended_profiles_df["Caste"])),
    }

    return {
        "user_details": user_details,
        "recommended_profiles": recommended_profiles_df.to_dict(orient="records"),
        "statistics": statistics,
    }

# Streamlit UI
member_id = st.text_input("Enter Member ID:", "")

if st.button("Get Recommendations"):
    if member_id.isdigit():
        member_id = int(member_id)
        result = get_recommendations(member_id)
        
        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("📌 User Details")
            st.json(result["user_details"])

            # Show recommended profiles
            st.subheader("🎯 Recommended Profiles")
            recommendations_df = pd.DataFrame(result["recommended_profiles"])
            if not recommendations_df.empty:
                st.dataframe(recommendations_df)

                # 📊 Charts
                st.subheader("📊 Statistics")

                # Age Distribution
                age_df = pd.DataFrame(result["statistics"]["age_distribution"].items(), columns=["Age", "Count"])
                if not age_df.empty:
                    fig_age = px.bar(age_df, x="Age", y="Count", title="Age Distribution", color="Count", color_continuous_scale="Blues")
                    st.plotly_chart(fig_age, use_container_width=True)

                # Caste Distribution
                caste_df = pd.DataFrame(result["statistics"]["caste_distribution"].items(), columns=["Caste", "Count"])
                if not caste_df.empty:
                    fig_caste = px.bar(caste_df, x="Caste", y="Count", title="Caste Distribution", color="Count", color_continuous_scale="Reds")
                    st.plotly_chart(fig_caste, use_container_width=True)

                # State Distribution
                state_df = pd.DataFrame(result["statistics"]["state_distribution"].items(), columns=["State", "Count"])
                if not state_df.empty:
                    fig_state = px.bar(state_df, x="State", y="Count", title="State Distribution", color="Count", color_continuous_scale="Greens")
                    st.plotly_chart(fig_state, use_container_width=True)
            else:
                st.warning("No recommendations found for this user.")
    else:
        st.error("Please enter a valid numeric Member ID.")
