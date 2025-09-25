import streamlit as st
import pandas as pd
import plotly.express as px

# ------------------------------
# Title
# ------------------------------
st.title("NeuroMirror Data Exploration")

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ------------------------------
    # Convert Numeric Columns
    # ------------------------------
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
    # Rule-Based Classification
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
    # Random 3 Persons
    # ------------------------------
    st.subheader("3 Randomly Selected Persons")
    if len(df) >= 3:
        sample_df = df.sample(3)
    else:
        sample_df = df
    st.dataframe(sample_df)

    # ------------------------------
    # Sidebar Navigation
    # ------------------------------
    st.sidebar.title("Choose a Visualization")
    options = [
        "Scatter Plot",
        "Animated Scatter",
        "Box Plot",
        "Treemap",
        "Radar Chart",
        "Sunburst Chart",
        "Heatmap"
    ]
    choice = st.sidebar.radio("Go to", options)

    # ------------------------------
    # Session state for filtered data
    # ------------------------------
    if "filtered_df" not in st.session_state:
        st.session_state.filtered_df = df.copy()

    filtered_df = st.session_state.filtered_df

    # ------------------------------
    # Function to update filtered dataframe
    # ------------------------------
    def update_filtered(selected_points, x_col, y_col):
        if selected_points is not None and len(selected_points["points"]) > 0:
            indices = [p["pointIndex"] for p in selected_points["points"]]
            st.session_state.filtered_df = filtered_df.iloc[indices]
        else:
            st.session_state.filtered_df = df.copy()

    # ------------------------------
    # Scatter Plot
    # ------------------------------
    if choice == "Scatter Plot":
        st.subheader("Scatter Plot: Alone Time vs Friends Circle")
        fig = px.scatter(
            filtered_df,
            x="Time_spent_Alone",
            y="Friends_circle_size",
            color="Personality",
            size="Post_frequency",
            hover_data=["Stage_fear", "Social_event_attendance"]
        )
        selected_points = st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Animated Scatter
    # ------------------------------
    elif choice == "Animated Scatter":
        st.subheader("Animated Scatter: Alone Time vs Friends Circle over Stage Fear")
        df_anim = filtered_df.dropna(subset=["Stage_fear", "Time_spent_Alone", "Friends_circle_size"])
        fig = px.scatter(
            df_anim,
            x="Time_spent_Alone",
            y="Friends_circle_size",
            color="Personality",
            size="Post_frequency",
            animation_frame="Stage_fear",
            hover_data=["Social_event_attendance", "Going_outside"]
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Box Plot
    # ------------------------------
    elif choice == "Box Plot":
        st.subheader("Box Plot: Stage Fear by Personality")
        df_box = filtered_df.dropna(subset=["Stage_fear", "Personality"])
        fig = px.box(
            df_box,
            x="Personality",
            y="Stage_fear",
            points="all",
            color="Personality"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Treemap
    # ------------------------------
    elif choice == "Treemap":
        st.subheader("Treemap: Social Events and Personality")
        fig = px.treemap(
            filtered_df,
            path=["Personality"],
            values="Social_event_attendance",
            color="Personality"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Radar Chart
    # ------------------------------
    elif choice == "Radar Chart":
        st.subheader("Radar Chart: Mean Scores by Personality")
        radar_data = filtered_df.groupby("Personality")[numeric_cols].mean().reset_index()
        radar_melted = radar_data.melt(id_vars="Personality", var_name="Attribute", value_name="Value")
        fig = px.line_polar(
            radar_melted,
            r="Value",
            theta="Attribute",
            color="Personality",
            line_close=True
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Sunburst Chart
    # ------------------------------
    elif choice == "Sunburst Chart":
        st.subheader("Sunburst: Personality Breakdown by Going Outside")
        fig = px.sunburst(
            filtered_df,
            path=["Personality", "Going_outside"],
            values="Friends_circle_size",
            color="Personality"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Heatmap
    # ------------------------------
    elif choice == "Heatmap":
        st.subheader("Correlation Heatmap")
        corr = filtered_df[numeric_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Show filtered data for clicks
    # ------------------------------
    st.markdown("---")
    st.subheader("Filtered Data Based on Interactions")
    st.dataframe(filtered_df)

else:
    st.info("👆 Please upload a CSV file to begin.")
