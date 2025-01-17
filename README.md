# Medical-Recommendation

## Table of Contents

- [Project Overview](#project-overview)
- [Models and Tools Used](#models-and-tools-used)
- [Datasets Used](#datasets-used)
- [Installation and Usage](#installation-and-usage)
- [Contributing](#contributing)

## Project Overview

The Medicine Recommendation System is a comprehensive application designed to predict diseases based on user-input symptoms and recommend appropriate medicines, precautions, workouts, and diets. The system leverages machine learning models to provide accurate predictions and recommendations, ensuring users receive helpful guidance for managing their health.

## Models and Tools Used

- **Machine Learning Models**:
  - Support Vector Classifier (SVC)
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - K-Nearest Neighbors (KNN)
  - Multinomial Naive Bayes
- **Libraries**:
  - `pandas` for data manipulation and analysis
  - `numpy` for numerical computations
  - `scikit-learn` for machine learning algorithms
  - `streamlit` for building the web application
  - `pickle` for model serialization
- **Development Tools**:
  - Google Colab for initial model development
  - Local environment for Streamlit app development

## Datasets Used

- [DataSet](https://www.kaggle.com/datasets/noorsaeed/medicine-recommendation-system-dataset?select=Symptom-severity.csv)

The following datasets are used in this project to train the models and provide recommendations:

- `Training.csv`: Dataset containing symptoms and corresponding diseases used for training the models.
- `symptoms_dict.csv`: A dictionary of symptoms for encoding input symptoms.
- `diseases_list.csv`: A list of diseases and their corresponding codes.
- `precautions_df.csv`: Precautionary measures for different diseases.
- `workout_df.csv`: Recommended workouts for various diseases.
- `description.csv`: Descriptions of diseases.
- `medications.csv`: Medications recommended for different diseases.
- `diets.csv`: Dietary recommendations for various diseases.

## Installation and Usage

To run the Medicine Recommendation System locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/medicine-recommendation-system.git
   cd medicine-recommendation-system
   ```
