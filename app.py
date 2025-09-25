import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================
# Streamlit App Title
# ==========================
st.set_page_config(page_title="NeuroMirror Personality Explorer", layout="wide")
st.title("🪞 NeuroMirror: Personality Exploration")

# ==========================
# Upload Dataset
# ==========================
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # ==========================
    # Pick 3 Random People
    # ==========================
    sample_df = df.sample(n=3, random_state=42).copy()

    # Convert required columns to numeric (safely)
    cols = [
        "Time_spent_Alone",
        "Stage_fear",
        "Social_event_attendance",
        "Going_outside",
        "Drained_after_socializing",
        "Friends_circle_size",
        "Post_frequency"
    ]
    for c in cols:
        sample_df[c] = pd.to_numeric(sample_df[c], errors="coerce")

    # ==========================
    # Personality Classification
    # ==========================
    def classify_person(row):
        score = 0
        try:
            if pd.notna(row['Time_spent_Alone']) and row['Time_spent_Alone'] > 5:
                score -= 1
            if pd.notna(row['Stage_fear']) and row['Stage_fear'] > 5:
                score -= 1
            if pd.notna(row['Social_event_attendance']) and row['Social_event_attendance'] > 5:
                score += 1
            if pd.notna(row['Going_outside']) and row['Going_outside'] > 5:
                score += 1
            if pd.notna(row['Drained_after_socializing']) and row['Drained_after_socializing'] > 3:
                score -= 1
            if pd.notna(row['Friends_circle_size']) and row['Friends_circle_size'] > 5:
                score += 1
            if pd.notna(row['Post_frequency']) and row['Post_frequency'] > 3:
                score += 1
        except Exception as e:
            return f"Error: {e}"
        return "Extrovert" if score > 0 else "Introvert"

    sample_df["Personality"] = sample_df.apply(classify_person, axis=1)

    st.subheader("🧠 Personality Predictions (3 Random People)")
    st.dataframe(sample_df[cols + ["Personality"]])

    # ==========================
    # Unique & Interactive Visualizations
    # ==========================

    st.subheader("📊 Unique Visualizations")

    # 1. Radar Chart (Personality Profile)
    st.markdown("### 🕸️ Personality Radar (click to explore)")
    radar = px.line_polar(
        sample_df.melt(id_vars="Personality", value_vars=cols),
        r="value", theta="variable", color="Personality", line_close=True,
        hover_name="variable"
    )
    st.plotly_chart(radar, use_container_width=True)

    # 2. Treemap (Personality Factors)
    st.markdown("### 🌳 Treemap of Factors Contributing to Personality")
    melted = sample_df.melt(id_vars="Personality", value_vars=cols)
    treemap = px.treemap(
        melted,
        path=["Personality", "variable"],
        values="value",
        color="value",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(treemap, use_container_width=True)

    # 3. Interactive Scatter Matrix
    st.markdown("### 🔎 Explore Relationships Between Features")
    scatter = px.scatter_matrix(
        sample_df, dimensions=cols, color="Personality",
        title="Scatter Matrix - Personality Traits"
    )
    st.plotly_chart(scatter, use_container_width=True)

    # ==========================
    # Deep Dive (on click selection)
    # ==========================
    st.subheader("🔍 Deep Dive into a Single Person")

    person_choice = st.selectbox("Select a Person", sample_df.index)
    person_data = sample_df.loc[person_choice]

    st.write(f"### Personality: **{person_data['Personality']}**")
    st.json(person_data.to_dict())

else:
    st.info("👆 Upload a CSV file to begin analysis.")

