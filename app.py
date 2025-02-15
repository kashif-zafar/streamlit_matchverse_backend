import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.express as px
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# Streamlit App Title
st.set_page_config(page_title="MatchVerse - AI Matchmaking Recommendations", layout="wide")
st.title("🔍MatchVerse - AI Matchmaking Recommendations")

# **Set local file paths**
users_csv_path = "users.csv"
model_json_path = "recommendation_model.json"



# **Load user metadata**
users_df = pd.read_csv("users.csv")

# **Label Encoding for categorical variables**
label_encoders = {}
for col in ["Gender", "Marital_Status", "Sect", "Caste", "State"]:
    le = LabelEncoder()
    users_df[col] = le.fit_transform(users_df[col])
    label_encoders[col] = le

# **Load trained XGBoost model**
bst = xgb.Booster()
bst.load_model("recommendation_model.json")

def get_recommendations(member_id):
    """Fetch recommendations for a given user."""
    user_row = users_df[users_df["Member_ID"] == member_id]
    if user_row.empty:
        return {"error": "User not found"}

    user_details = user_row.iloc[0][["Member_ID", "Gender", "Age", "Marital_Status", "Sect", "Caste", "State"]].to_dict()

    # **Decode categorical values for human readability**
    for col in ["Gender", "Marital_Status", "Sect", "Caste", "State"]:
        user_details[col] = label_encoders[col].inverse_transform([user_details[col]])[0]

    user_gender = user_row.iloc[0]["Gender"]
    user_age = user_row.iloc[0]["Age"]

    # **Get opposite gender**
    opposite_gender_encoded = 1 - user_gender

    # **Filter opposite-gender users**
    eligible_profiles = users_df[users_df["Gender"] == opposite_gender_encoded].copy()
    if eligible_profiles.empty:
        return {"user_details": user_details, "recommended_profiles": [], "statistics": {}}

    # **Compute required features**
    eligible_profiles["Age_Diff"] = abs(eligible_profiles["Age"] - user_age)
    eligible_profiles["Same_Caste"] = (eligible_profiles["Caste"] == user_row.iloc[0]["Caste"]).astype(int)
    eligible_profiles["Same_Sect"] = (eligible_profiles["Sect"] == user_row.iloc[0]["Sect"]).astype(int)
    eligible_profiles["Same_State"] = (eligible_profiles["State"] == user_row.iloc[0]["State"]).astype(int)

    # **Ensure missing features are added with zeros**
    model_features = bst.feature_names
    for feature in model_features:
        if feature not in eligible_profiles.columns:
            eligible_profiles[feature] = 0

    # **Convert to DMatrix for XGBoost**
    X_test = eligible_profiles[model_features]
    dtest = xgb.DMatrix(X_test)

    # **Get predictions**
    preds = bst.predict(dtest)

    # **Rank profiles by prediction score**
    ranked_profiles = sorted(
        zip(eligible_profiles["Member_ID"], preds), key=lambda x: x[1], reverse=True
    )[:100]

    recommended_ids = [profile_id for profile_id, _ in ranked_profiles]

    # **Get recommended profiles**
    recommended_profiles_df = users_df[users_df["Member_ID"].isin(recommended_ids)].copy()

    # **Decode categorical values for recommended profiles**
    for col in ["Gender", "Marital_Status", "Sect", "Caste", "State"]:
        recommended_profiles_df[col] = label_encoders[col].inverse_transform(recommended_profiles_df[col])

    # **Compute statistics**
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

# **Streamlit UI**
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

            # **Plot graphs**
            st.subheader("📊 Statistics")

            # **Age Distribution**
            age_df = pd.DataFrame(result["statistics"]["age_distribution"].items(), columns=["Age", "Count"])
            fig_age = px.bar(age_df, x="Age", y="Count", title="Age Distribution", color="Count", color_continuous_scale="Blues")
            st.plotly_chart(fig_age, use_container_width=True)

            # **Caste Distribution**
            caste_df = pd.DataFrame(result["statistics"]["caste_distribution"].items(), columns=["Caste", "Count"])
            fig_caste = px.bar(caste_df, x="Caste", y="Count", title="Caste Distribution", color="Count", color_continuous_scale="Reds")
            st.plotly_chart(fig_caste, use_container_width=True)

            # **State Distribution**
            state_df = pd.DataFrame(result["statistics"]["state_distribution"].items(), columns=["State", "Count"])
            fig_state = px.bar(state_df, x="State", y="Count", title="State Distribution", color="Count", color_continuous_scale="Greens")
            st.plotly_chart(fig_state, use_container_width=True)

            # **Sect Distribution**
            sect_df = pd.DataFrame(result["statistics"]["sect_distribution"].items(), columns=["Sect", "Count"])
            fig_sect = px.bar(sect_df, x="Sect", y="Count", title="Sect Distribution", color="Count", color_continuous_scale="Purples")
            st.plotly_chart(fig_sect, use_container_width=True)

    else:
        st.error("Please enter a valid numeric Member ID.")
