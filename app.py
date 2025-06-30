import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("Adaptive AI Talent Mapping Tool")

# File uploader
uploaded_file = st.file_uploader("Upload Employee Data (CSV)", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Employee Data Preview")
    st.dataframe(data.head())

    # Trait weight sliders (summing to 100%)
    st.write("### Adjust Trait Weights (Total must be 100%)")
    con_weight = st.slider("Conscientiousness Weight", 0, 100, 33)
    adapt_weight = st.slider("Adaptability Weight", 0, 100 - con_weight, (100 - con_weight - 33) // 2)
    collab_weight = 100 - con_weight - adapt_weight

    # Display weights
    st.write(f"Conscientiousness: {con_weight}%, Adaptability: {adapt_weight}%, Collaboration: {collab_weight}%")

    # Normalize trait scores and calculate Weighted_Potential
    data['Weighted_Potential'] = (
        (con_weight * data['Conscientiousness'] / 100) +
        (adapt_weight * data['Adaptability'] / 100) +
        (collab_weight * data['Collaboration'] / 100)
    )

    # Talent map
    st.write("### Talent Map: Performance vs. Potential")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Sales_Revenue', y='Weighted_Potential', hue='Department', size='Project_Success_Rate', data=data, ax=ax)
    plt.xlabel('Sales Revenue ($)')
    plt.ylabel('Weighted Potential Score')
    st.pyplot(fig)

    # Role recommendations (recalculate based on weighted traits)
    weighted_traits = data[['Conscientiousness', 'Adaptability', 'Collaboration']].multiply([con_weight/100, adapt_weight/100, collab_weight/100], axis=1)
    similarity = cosine_similarity(weighted_traits, role_traits)
    data['Recommended_Role'] = [role_profiles['Role'][np.argmax(row)] for row in similarity]

    st.write("### Role Recommendations")
    st.dataframe(data[['Employee_ID', 'Role', 'Recommended_Role', 'Aspiration']])

    # Succession planning
    st.write("### Succession Planning: Top Leadership Candidates")
    top_candidates = data[data['Recommended_Role'].isin(['Manager', 'Director'])][['Employee_ID', 'Weighted_Potential', 'Department']].sort_values(by='Weighted_Potential', ascending=False).head(5)
    st.dataframe(top_candidates)
else:
    st.write("Please upload a CSV file to begin.")

# Display static talent map (from Colab)
st.write("### Sample Talent Map (Generated from AI Model)")
st.image('talent_map.png', caption="Sample 9-box grid showing performance vs. potential.")

# Define role profiles (same as in Colab for consistency)
role_profiles = pd.DataFrame({
    'Role': ['Manager', 'Tech Lead', 'Senior Analyst', 'Director'],
    'Conscientiousness': [90, 80, 85, 95],
    'Adaptability': [85, 90, 80, 90],
    'Collaboration': [90, 85, 80, 95]
})
role_traits = role_profiles[['Conscientiousness', 'Adaptability', 'Collaboration']].values