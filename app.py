import streamlit as st
import pandas as pd
import random
import plotly.express as px

st.set_page_config(page_title="Neuro Mirror", layout="wide")
st.title("Neuro Mirror - Personality Analysis")

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    # Replace with your dataset file name
    df = pd.read_csv("personality_behavior_data.csv") 
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

df = load_data()
st.subheader("Full Dataset Preview")
st.dataframe(df)

# -------------------------------
# Step 2: Randomly select 3 persons
# -------------------------------
st.subheader("Randomly Selected 3 Persons")
sample_df = df.sample(n=3, random_state=random.randint(0,1000))
st.dataframe(sample_df)

# -------------------------------
# Step 3: Personality Classifier
# -------------------------------
def classify_person(row):
    score = 0
    # Example rule-based scoring
    if row['Time_spent_Alone'] > 5:
        score -= 1
    if row['Going_outside'] > 3:
        score += 1
    if row['Drained_after_socializing'] > 3:
        score -= 1
    if row['Friends_circle_size'] > 5:
        score += 1
    if row['Social_event_attendance'] > 3:
        score += 1
    if row['Stage_fear'] > 3:
        score -= 1
    if row['Post_frequency'] > 5:
        score += 1
    return "Extrovert" if score > 0 else "Introvert"

sample_df['Personality'] = sample_df.apply(classify_person, axis=1)
st.subheader("Personality Predictions")
st.dataframe(sample_df[['Time_spent_Alone','Stage_fear','Social_event_attendance',
                        'Going_outside','Drained_after_socializing','Friends_circle_size',
                        'Post_frequency','Personality']])

# -------------------------------
# Step 4: Unique Visualizations
# -------------------------------
st.subheader("Unique Visualizations")

# 1. Radar Chart for each selected person
for i, row in sample_df.iterrows():
    categories = ['Time_spent_Alone','Stage_fear','Social_event_attendance',
                  'Going_outside','Drained_after_socializing','Friends_circle_size',
                  'Post_frequency']
    values = [row[cat] for cat in categories]
    values += values[:1]  # Close the radar loop

    fig = px.line_polar(r=values, theta=categories + [categories[0]],
                        line_close=True, title=f"Radar Personality Chart - Person {i+1}")
    fig.update_traces(fill='toself')
    st.plotly_chart(fig, use_container_width=True)

# 2. Heatmap of selected persons
st.subheader("Heatmap of Selected Persons")
heatmap_df = sample_df[categories]
fig2 = px.imshow(heatmap_df, text_auto=True, aspect="auto",
                 labels=dict(x="Traits", y="Person Index", color="Score"))
st.plotly_chart(fig2, use_container_width=True)
