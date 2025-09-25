import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Title
# ------------------------------
st.title("NeuroMirror Advanced Data Exploration")

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
    st.sidebar.title("Choose an Exploratory Visualization")
    options = [
        "Animated Radar Chart",
        "Cluster Map (PCA)",
        "Animated Violin Plots",
        "Multi-level Sunburst",
        "Interactive Correlation Heatmap"
    ]
    choice = st.sidebar.radio("Select Visualization", options)

    # ------------------------------
    # Filtered data for interactivity
    # ------------------------------
    if "filtered_df" not in st.session_state:
        st.session_state.filtered_df = df.copy()
    filtered_df = st.session_state.filtered_df

    # ------------------------------
    # Animated Radar Chart
    # ------------------------------
    if choice == "Animated Radar Chart":
        st.subheader("Animated Radar Chart: Mean Traits per Personality")
        radar_data = filtered_df.groupby("Personality")[numeric_cols].mean().reset_index()
        radar_melted = radar_data.melt(id_vars="Personality", var_name="Trait", value_name="Value")
        fig = px.line_polar(
            radar_melted,
            r="Value",
            theta="Trait",
            color="Personality",
            line_close=True,
            animation_frame="Personality",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Cluster Map (PCA)
    # ------------------------------
    elif choice == "Cluster Map (PCA)":
        st.subheader("Cluster Map of Persons (PCA Projection)")
        pca_data = filtered_df.dropna(subset=numeric_cols)
        if len(pca_data) >= 2:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(pca_data[numeric_cols])
            pca = PCA(n_components=2)
            components = pca.fit_transform(scaled)
            pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
            pca_df["Personality"] = pca_data["Personality"].values
            fig = px.scatter(
                pca_df,
                x="PC1",
                y="PC2",
                color="Personality",
                hover_data=pca_data[numeric_cols]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data for PCA clustering.")

    # ------------------------------
    # Animated Violin Plots
    # ------------------------------
    elif choice == "Animated Violin Plots":
        st.subheader("Animated Violin Plots: Trait Distributions by Personality")
        df_violin = filtered_df.melt(id_vars="Personality", value_vars=numeric_cols,
                                     var_name="Trait", value_name="Value")
        fig = px.violin(
            df_violin,
            x="Trait",
            y="Value",
            color="Personality",
            box=True,
            points="all",
            animation_frame="Personality"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Multi-level Sunburst
    # ------------------------------
    elif choice == "Multi-level Sunburst":
        st.subheader("Multi-level Sunburst: Personality → Going_outside → Stage_fear")
        df_sun = filtered_df.copy()
        df_sun["Stage_fear_bin"] = pd.cut(df_sun["Stage_fear"], bins=3, labels=["Low", "Medium", "High"])
        fig = px.sunburst(
            df_sun,
            path=["Personality", "Going_outside", "Stage_fear_bin"],
            values="Friends_circle_size",
            color="Personality"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Interactive Correlation Heatmap
    # ------------------------------
    elif choice == "Interactive Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        corr = filtered_df[numeric_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Please upload a CSV file to begin.")
