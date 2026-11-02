# Diabetes Progression Predictor

A Streamlit web app that estimates diabetes disease progression from ten
patient features, using the best regression model selected in my machine
learning project.

## Overview

This project compares several regression models on scikit-learn's diabetes
dataset (442 patients, 10 features). The best-performing model was tuned
with cross-validation, saved with `joblib`, and deployed as an interactive
Streamlit dashboard.

## Features

- Interactive inputs for the ten model features, with sex selected as Male
  or Female and mapped to the dataset's standardized encoding
- A prefilled sample report dashboard with a chart of the entered standardized
  feature values
- Interactive inputs for the remaining standardized features (age, BMI,
  blood pressure, and six blood serum measurements)
- Progression-score prediction and patient report refresh after form submission
- Medical-use disclaimer included in the app

## Tech Stack

- Python
- Streamlit
- scikit-learn
- pandas, NumPy
- joblib

## Project Structure