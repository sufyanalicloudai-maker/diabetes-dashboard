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

- Interactive inputs for the ten standardized features (age, sex, BMI,
  blood pressure, and six blood serum measurements)
- Instant progression-score prediction from the saved model
- Medical-use disclaimer included in the app

## Tech Stack

- Python
- Streamlit
- scikit-learn
- pandas, NumPy
- joblib

## Project Structure