import streamlit as st
import pandas as pd
import plotly.express as px

# ------------------------------
# Load Dataset
# ------------------------------
st.title("NeuroMirror Data Exploration")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Try converting numeric columns safely
    numeric_cols = [
        "Time_spent_Alone",
        "Stage_fear",
        "Social_event_attendance",
        "Going_outside",
        "Drained_after_socializing",
        "Friends_circle_size",
        "Post_frequency"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # ------------------------------
    # Classification Function
    # ------------------------------
    def classify_person(row):
        score = 0
        if pd.notnull(row["Time_spent_Alone"]) and row["Time_spent_Alone"] > 5: score -= 1
        if pd.notnull(row["Stage_fear"]) and row["Stage_fear"] > 5: score -= 1
        if pd.notnull(row["Social_event_attendance"]) and row["Social_event_attendance"] > 5: score += 1
        if pd.notnull(row["Going_outside"]) and row["Going_outside"] > 5: score += 1
        if pd.notnull(row["Drained_after_socializing"]) and row["Drained_after_socializing"] > 5: score -= 1
        if pd.notnull(row["Friends_circle_size"]) and row["Friends_circle_size"] > 5: score += 1
        if pd.notnull(row["Post_frequency"]) and row["Post_frequency"] > 5: score += 1
        return "Extrovert" if score > 0 else "Introvert"

    df["Personality"] = df.apply(classify_person, axis=1)

    # ------------------------------
    # Sidebar Navigation
    # ------------------------------
    st.sidebar.title("Choose a Visualization")
    options = [
        "Scatter Plot",
        "Box Plot",
        "Treemap",
        "Radar Chart",
        "Sunburst Chart",
        "Heatmap"
    ]
    choice = st.sidebar.radio("Go to", options)

    # ------------------------------
    # Visualizations
    # ------------------------------
    if choice == "Scatter Plot":
        st.subheader("Scatter Plot: Alone Time vs Friends Circle")
        fig = px.scatter(
            df,
            x="Time_spent_Alone",
            y="Friends_circle_size",
            color="Personality",
            size="Post_frequency",
            hover_data=["Stage_fear", "Social_event_attendance"]
        )
        st.plotly_chart(fig, use_container_width=True)

    elif choice == "Box Plot":
        st.subheader("Box Plot: Stage Fear by Personality")
        fig = px.box(df, x="Personality", y="Stage_fear", points="all", color="Personality")
        st.plotly_chart(fig, use_container_width=True)

    elif choice == "Treemap":
        st.subheader("Treemap: Social Events and Personality")
        fig = px.treemap(df, path=["Personality"], values="Social_event_attendance", color="Personality")
        st.plotly_chart(fig, use_container_width=True)

    elif choice == "Radar Chart":
        st.subheader("Radar Chart: Mean Scores by Personality")
        radar_data = df.groupby("Personality")[numeric_cols].mean().reset_index()
        radar_melted = radar_data.melt(id_vars="Personality", var_name="Attribute", value_name="Value")
        fig = px.line_polar(
            radar_melted,
            r="Value",
            theta="Attribute",
            color="Personality",
            line_close=True
        )
        st.plotly_chart(fig, use_container_width=True)

    elif choice == "Sunburst Chart":
        st.subheader("Sunburst: Personality Breakdown by Going Outside")
        fig = px.sunburst(df, path=["Personality", "Going_outside"], values="Friends_circle_size", color="Personality")
        st.plotly_chart(fig, use_container_width=True)

    elif choice == "Heatmap":
        st.subheader("Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Please upload a CSV file to begin.")
