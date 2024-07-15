import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.set_page_config(
    page_title="Medical Recommendation ",
    page_icon="⛑️",  # You can use any emoji as the icon
    layout="centered"  # You can choose between "centered" and "wide"
)
# Load CSV data for symptoms, descriptions, precautions, medications, workouts, and diets
symptoms_dict = pd.read_csv("DataSets/symptoms_dict.csv")  # Assuming this contains your symptoms dictionary
diseases_list = pd.read_csv("DataSets/diseases_list.csv")  # Assuming this contains your diseases dictionary
precautions_df = pd.read_csv("DataSets/precautions_df.csv")
workout_df = pd.read_csv("DataSets/workout_df.csv")
description_df = pd.read_csv("DataSets/description.csv")
medications_df = pd.read_csv('DataSets/medications.csv')
diets_df = pd.read_csv("DataSets/diets.csv")

# Load the trained SVM model
svc_model = pickle.load(open('svc.pkl', 'rb'))

# Function to predict disease based on symptoms
def get_predicted_disease(symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms:
        if symptom in symptoms_dict['Symptom'].values:
            input_vector[symptoms_dict[symptoms_dict['Symptom'] == symptom].index[0]] = 1

    predicted_disease = diseases_list[diseases_list['Disease_Code'] == svc_model.predict([input_vector])[0]]['Disease_Name'].values[0]
    return predicted_disease

# Helper function to fetch recommendations for a disease
def get_recommendations_for_disease(predicted_disease):
    desc = description_df[description_df['Disease'] == predicted_disease]['Description'].values[0]

    pre = precautions_df[precautions_df['Disease'] == predicted_disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.tolist()[0]

    med = medications_df[medications_df['Disease'] == predicted_disease]['Medication'].values.tolist()

    die = diets_df[diets_df['Disease'] == predicted_disease]['Diet'].values.tolist()

    wrkout = workout_df[workout_df['disease'] == predicted_disease]['workout'].values.tolist()

    return desc, pre, med, die, wrkout

# Function to get recommendations based on selected symptoms
def get_recommendations(selected_symptoms):
    predicted_disease = get_predicted_disease(selected_symptoms)
    desc, pre, med, die, wrkout = get_recommendations_for_disease(predicted_disease)

    recommended_info = {
        'Disease': predicted_disease,
        'Description': desc,
        'Precautions': pre,
        'Medications': med,
        'Workout': wrkout,
        'Diets': die
    }
    return recommended_info

# Streamlit App
st.title("Medical Recommendation🩺")

# Load the CSV file containing symptoms
num_symptoms = pd.read_csv('DataSets/Symptom-severity.csv')
symptoms_list = num_symptoms['Symptom'].tolist()

# Display multiselect dropdown with search functionality
selected_symptoms = st.multiselect("Select 1 or up to 5 symptoms:", symptoms_list, max_selections=5)

# Button to trigger recommendation
if st.button("Recommend me"):
    if selected_symptoms:
        st.write("<h5 style='color: lightgreen;'>Your selected Symptoms</h5>", unsafe_allow_html=True)

        for symptom in selected_symptoms:
            st.write(f"😷 {symptom}")
        
        # Call function to get recommendations
        recommended_info = get_recommendations(selected_symptoms)

        # Display recommended information in separate tables
        # st.subheader("Recommended Disease")
        st.markdown("<h3 style='color: skyblue;'>Predicted disease</h3>", unsafe_allow_html=True)
        st.table(pd.DataFrame({'Disease': [recommended_info['Disease']]}))

        st.markdown("<h3 style='color: skyblue;'>Description</h3>", unsafe_allow_html=True)
        styled_description_df = pd.DataFrame({'Description': [recommended_info['Description']]}).style.set_properties(**{'color': 'yellow'}, subset=['Description'])
        st.table(styled_description_df)

        st.markdown("<h3 style='color: skyblue;'>Avoid Things 🚫</h3>", unsafe_allow_html=True)
        st.table(pd.DataFrame({'Avoid doing': recommended_info['Precautions']}))

        st.markdown("<h3 style='color: skyblue;'>Medications 💊</h3>", unsafe_allow_html=True)
        st.table(pd.DataFrame({'Medications': recommended_info['Medications']}))

        st.markdown("<h3 style='color: skyblue;'>Precautions</h3>", unsafe_allow_html=True)
        st.table(pd.DataFrame({'Precautions': recommended_info['Workout']}))

        st.markdown("<h3 style='color: skyblue;'>Diets</h3>", unsafe_allow_html=True)
        st.table(pd.DataFrame({'Diets': recommended_info['Diets']}))

# To run the Streamlit app, use the command: `streamlit run your_script_name.py`
