# DSAI_Group8
# AI-Based Energy Consumption Forecasting

An AI-powered forecasting system that predicts energy consumption using multiple machine learning and time-series forecasting models.

The project compares the performance of:

- Prophet
- GRU (Gated Recurrent Unit)
- CatBoost

using the World Bank Sustainable Energy for All (SE4ALL) dataset.

---

## Features

- Data preprocessing pipeline
- Exploratory Data Analysis (EDA)
- Feature engineering
- Multi-model forecasting
- Model evaluation
- Interactive Streamlit dashboard
- Model comparison visualization

---

## Project Structure

```
DSAI_Group8/
│
├── app.py
├── config.py
├── main.ipynb
│
├── pages/
│ ├── 1_Home.py
│ ├── 2_EDA.py
│ ├── 3_Model_Prediction.py
│ ├── 4_Model_Comparison.py
│ └── 5_About.py
│
├── Utils/
│
├── Output/
│ ├── models/
│ └── Model/
│
├── Source/
│ └── Dataset/
│
└── requirements.txt
```

---

## Models

### Prophet

Time-series forecasting using Facebook Prophet.

### GRU

Deep Learning sequential forecasting using PyTorch.

### CatBoost

Gradient Boosting regression with engineered lag features.

---

## Dataset

World Bank Sustainable Energy for All (SE4ALL)

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/DSAI_Group8.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

## Dashboard

The Streamlit application includes:

- Home
- Exploratory Data Analysis
- Model Prediction
- Model Comparison
- About

---

## Authors

DSAI Group 8

Evan Fanuel 

Achmad Syarif Nugraha 
