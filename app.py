import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff

# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(page_title="Neuro Mirror", layout="wide")

st.title("🧠 Neuro Mirror - Personality Data Explorer")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV only)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Add Serial Number
    df.insert(0, "Serial_No", range(1, len(df) + 1))

    # ------------------------------
    # Rule-based Personality Classification
    # ------------------------------
    def classify_person(row):
        score = 0
        if row["Time_spent_Alone"] > 5: score -= 1
        if row["Stage_fear"] > 5: score -= 1
        if row["Social_event_attendance"] > 5: score += 1
        if row["Going_outside"] > 5: score += 1
        if row["Drained_after_socializing"] > 5: score -= 1
        if row["Friends_circle_size"] > 5: score += 1
        if row["Post_frequency"] > 5: score += 1
        return "Extrovert" if score > 0 else "Introvert"

    df["Personality"] = df.apply(classify_person, axis=1)

    st.success("✅ Data uploaded and processed successfully!")

    # ------------------------------
    # Sidebar Navigation
    # ------------------------------
    st.sidebar.title("📊 Choose a Graph")
    option = st.sidebar.radio(
        "Select one", 
        (
            "Dataset Preview",
            "Scatter Plot",
            "Box Plot",
            "Heatmap",
            "Treemap",
            "Radar Chart",
            "Sunburst Chart"
        )
    )

    # ------------------------------
    # Dataset Preview
    # ------------------------------
    if option == "Dataset Preview":
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10))

    # ------------------------------
    # Scatter Plot (Basic)
    # ------------------------------
    elif option == "Scatter Plot":
        st.subheader("Scatter Plot - Friends Circle vs Social Attendance")
        fig = px.scatter(
            df, 
            x="Friends_circle_size", 
            y="Social_event_attendance", 
            color="Personality",
            size="Post_frequency",
            hover_data=["Serial_No"]
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Box Plot (Basic)
    # ------------------------------
    elif option == "Box Plot":
        st.subheader("Box Plot - Time spent Alone by Personality")
        fig = px.box(df, x="Personality", y="Time_spent_Alone", color="Personality")
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Heatmap (Basic)
    # ------------------------------
    elif option == "Heatmap":
        st.subheader("Heatmap - Correlation Matrix")
        corr = df.drop(columns=["Serial_No"]).corr()
        fig = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="Viridis",
            showscale=True
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Treemap (Advanced)
    # ------------------------------
    elif option == "Treemap":
        st.subheader("Treemap - Personality Breakdown by Social Attendance")
        fig = px.treemap(
            df,
            path=["Personality", "Social_event_attendance"],
            values="Friends_circle_size",
            color="Post_frequency",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Radar Chart (Advanced)
    # ------------------------------
    elif option == "Radar Chart":
        st.subheader("Radar Chart - Average Traits by Personality")
        features = [
            "Time_spent_Alone", "Stage_fear", "Social_event_attendance",
            "Going_outside", "Drained_after_socializing",
            "Friends_circle_size", "Post_frequency"
        ]
        avg_df = df.groupby("Personality")[features].mean().reset_index()

        fig = px.line_polar(
            avg_df.melt(id_vars="Personality"), 
            r="value", 
            theta="variable", 
            color="Personality", 
            line_close=True
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Sunburst Chart (Advanced)
    # ------------------------------
    elif option == "Sunburst Chart":
        st.subheader("Sunburst - Personality → Friends Circle → Posting Frequency")
        fig = px.sunburst(
            df,
            path=["Personality", "Friends_circle_size", "Post_frequency"],
            values="Social_event_attendance",
            color="Personality",
            color_discrete_map={"Introvert": "blue", "Extrovert": "orange"}
        )
        st.plotly_chart(fig, use_container_width=True)

