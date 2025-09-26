import streamlit as st
import pandas as pd
import plotly.express as px

# ------------------------------
# Title
# ------------------------------
st.title("NeuroMirror Data Exploration (Interactive Version)")

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
    # Personality Filter
    # ------------------------------
    personality_choice = st.sidebar.selectbox(
        "Filter by Personality",
        ["All", "Introvert", "Extrovert"]
    )

    if personality_choice != "All":
        filtered_df = df[df["Personality"] == personality_choice]
    else:
        filtered_df = df.copy()

    # ------------------------------
    # Random 3 Persons
    # ------------------------------
    st.subheader("3 Randomly Selected Persons")
    if len(filtered_df) >= 3:
        sample_df = filtered_df.sample(3)
    else:
        sample_df = filtered_df
    st.dataframe(sample_df)

    # ------------------------------
    # Sidebar: Choose Visualization
    # ------------------------------
    st.sidebar.title("Choose an Exploratory Visualization")
    options = [
        "Radar Chart",
        "Violin Plots",
        "Multi-level Sunburst",
        "Correlation Heatmap"
    ]
    choice = st.sidebar.radio("Select Visualization", options)

    # ------------------------------
    # Radar Chart
    # ------------------------------
    if choice == "Radar Chart":
        st.subheader("Radar Chart: Mean Traits by Personality")
        if filtered_df.empty:
            st.warning("No data to display.")
        else:
            radar_data = filtered_df.groupby("Personality")[numeric_cols].mean().reset_index()
            radar_melted = radar_data.melt(id_vars="Personality", var_name="Trait", value_name="Value")
            fig = px.line_polar(
                radar_melted,
                r="Value",
                theta="Trait",
                color="Personality",
                line_close=True,
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Violin Plots
    # ------------------------------
    elif choice == "Violin Plots":
        st.subheader("Violin Plots: Trait Distributions by Personality")
        if filtered_df.empty:
            st.warning("No data to display.")
        else:
            df_violin = filtered_df.melt(id_vars="Personality", value_vars=numeric_cols,
                                         var_name="Trait", value_name="Value")
            fig = px.violin(
                df_violin,
                x="Trait",
                y="Value",
                color="Personality",
                box=True,
                points="all"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Multi-level Sunburst
    # ------------------------------
    elif choice == "Multi-level Sunburst":
        st.subheader("Multi-level Sunburst: Personality → Going_outside → Stage_fear")
        df_sun = filtered_df.copy()
        if "Stage_fear" in df_sun.columns and df_sun["Stage_fear"].dropna().nunique() >= 3:
            df_sun["Stage_fear_bin"] = pd.cut(df_sun["Stage_fear"], bins=3, labels=["Low", "Medium", "High"])
        else:
            df_sun["Stage_fear_bin"] = df_sun["Stage_fear"].fillna("Unknown")
        fig = px.sunburst(
            df_sun,
            path=["Personality", "Going_outside", "Stage_fear_bin"],
            values="Friends_circle_size",
            color="Personality"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Correlation Heatmap
    # ------------------------------
    elif choice == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        if filtered_df.empty:
            st.warning("No data to display.")
        else:
            corr = filtered_df[numeric_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info(" Please upload a CSV file to begin.")
