import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.express as px
from collections import Counter
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="MatchVerse - AI Matchmaking", layout="wide")
st.title("🔍 MatchVerse - AI Matchmaking Recommendations")

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

label_encoders = {}
for col in ["Gender", "Marital_Status", "Sect", "Caste", "State"]:
    le = LabelEncoder()
    users_df[col] = le.fit_transform(users_df[col])
    label_encoders[col] = le

bst = xgb.Booster()
bst.load_model(MODEL_JSON_PATH)

MODEL_FEATURES = ["Age_Diff", "Same_Caste", "Same_Sect", "Same_State", "Target_Popularity"]

interaction_counts = interactions_df["Target_ID"].value_counts()

def get_recommendations(member_id):
    user_row = users_df[users_df["Member_ID"] == member_id]
    if user_row.empty:
        return {"error": "User not found"}

    user_details = user_row.iloc[0][["Member_ID", "Gender", "Age", "Marital_Status", "Sect", "Caste", "State"]].to_dict()

    for col in ["Gender", "Marital_Status", "Sect", "Caste", "State"]:
        user_details[col] = label_encoders[col].inverse_transform([user_details[col]])[0]

    user_gender = user_row.iloc[0]["Gender"]
    user_age = user_row.iloc[0]["Age"]
    user_caste = user_row.iloc[0]["Caste"]
    user_sect = user_row.iloc[0]["Sect"]
    user_state = user_row.iloc[0]["State"]

    opposite_gender_encoded = 1 - user_gender
    eligible_profiles = users_df[users_df["Gender"] == opposite_gender_encoded].copy()

    interacted_users = set(interactions_df[interactions_df["Member_ID"] == member_id]["Target_ID"])
    fresh_profiles = eligible_profiles[~eligible_profiles["Member_ID"].isin(interacted_users)]

    if fresh_profiles.empty:
        return {"user_details": user_details, "recommended_profiles": [], "statistics": {}}

    fresh_profiles["Age_Diff"] = abs(fresh_profiles["Age"] - user_age)
    fresh_profiles["Same_Caste"] = (fresh_profiles["Caste"] == user_caste).astype(int)
    fresh_profiles["Same_Sect"] = (fresh_profiles["Sect"] == user_sect).astype(int)
    fresh_profiles["Same_State"] = (fresh_profiles["State"] == user_state).astype(int)
    fresh_profiles["Target_Popularity"] = fresh_profiles["Member_ID"].map(interaction_counts).fillna(0)

    X_test = fresh_profiles[MODEL_FEATURES]
    dtest = xgb.DMatrix(X_test)
    preds = bst.predict(dtest)

    fresh_profiles["Score"] = preds
    recommended_profiles_df = fresh_profiles.sort_values(by="Score", ascending=False).head(100)

    for col in ["Gender", "Marital_Status", "Sect", "Caste", "State"]:
        recommended_profiles_df[col] = label_encoders[col].inverse_transform(recommended_profiles_df[col])

    age_distribution = dict(Counter(recommended_profiles_df["Age"]))
    sect_distribution = dict(Counter(recommended_profiles_df["Sect"]))
    state_distribution = dict(Counter(recommended_profiles_df["State"]))
    caste_distribution = dict(Counter(recommended_profiles_df["Caste"]))

    return {
        "user_details": user_details,
        "recommended_profiles": recommended_profiles_df.to_dict(orient="records"),
        "statistics": {
            "age_distribution": age_distribution,
            "sect_distribution": sect_distribution,
            "state_distribution": state_distribution,
            "caste_distribution": caste_distribution,
        },
    }

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

            st.subheader("🎯 Recommended Profiles")
            st.dataframe(pd.DataFrame(result["recommended_profiles"]))

            st.subheader("📊 Statistics")

            age_df = pd.DataFrame(result["statistics"]["age_distribution"].items(), columns=["Age", "Count"])
            fig_age = px.bar(age_df, x="Age", y="Count", title="Age Distribution", color="Count", color_continuous_scale="Blues")
            st.plotly_chart(fig_age, use_container_width=True)

            caste_df = pd.DataFrame(result["statistics"]["caste_distribution"].items(), columns=["Caste", "Count"])
            fig_caste = px.bar(caste_df, x="Caste", y="Count", title="Caste Distribution", color="Count", color_continuous_scale="Reds")
            st.plotly_chart(fig_caste, use_container_width=True)

            state_df = pd.DataFrame(result["statistics"]["state_distribution"].items(), columns=["State", "Count"])
            fig_state = px.bar(state_df, x="State", y="Count", title="State Distribution", color="Count", color_continuous_scale="Greens")
            st.plotly_chart(fig_state, use_container_width=True)

            sect_df = pd.DataFrame(result["statistics"]["sect_distribution"].items(), columns=["Sect", "Count"])
            fig_sect = px.bar(sect_df, x="Sect", y="Count", title="Sect Distribution", color="Count", color_continuous_scale="Purples")
            st.plotly_chart(fig_sect, use_container_width=True)
    else:
        st.error("Please enter a valid numeric Member ID.")
