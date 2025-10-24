# IT2011 Group Assignment: Data Preprocessing & EDA for NSL-KDD

---

## 🚀 Project Overview

This project provides a comprehensive workflow for the cleaning, preprocessing, and exploratory data analysis (EDA) of the **NSL-KDD dataset**. The primary objective is to systematically transform the raw network connection data into a clean, scaled, and machine-learning-ready format suitable for a network intrusion detection system.

Our process follows a multi-stage pipeline that includes:
* Data integrity verification
* Advanced feature engineering
* Categorical data encoding
* Outlier analysis
* Numerical feature scaling

The final output is a fully preprocessed dataset, which serves as the foundational artifact for any subsequent modeling tasks.

---

## 📊 Dataset Details

* **Name:** NSL-KDD Dataset
* **Source:** Kaggle (`hassan06/nslkdd`)
* **Features:** The dataset contains 41 features that describe the properties of individual TCP connections, plus a final label indicating the connection type.
* **Description:** As a refined version of the original KDD'99 dataset, NSL-KDD is a widely-used and respected benchmark for evaluating intrusion detection models. It addresses some of the inherent problems of the original dataset, making it more suitable for modern analysis.

---

## 👥 Group Member Roles & Preprocessing Workflow

Our team approached the preprocessing task as a sequential pipeline, with each member taking ownership of a critical stage. This ensures a logical flow from raw data to a clean dataset and highlights each member's individual contribution.

| Step  | Member ID  | Assigned Technique & Responsibility                                                                                                                  |
|:------|:-----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|
| **1** | IT24100958 | **Data Loading & Verification**: Load the raw dataset and perform initial quality checks for missing values, duplicates, and data types.                |
| **2** | IT24101449 | **Feature Engineering (Binary Label)**: Create the primary binary target variable (`0` for normal, `1` for attack) to simplify the classification task. |
| **3** | IT24101454 | **Feature Engineering (Attack Categories)**: Engineer a detailed multi-class target variable by grouping specific attack types (`DoS`, `Probe`, etc.). |
| **4** | IT24101280 | **Encoding Categorical Variables**: Convert all non-numeric features (`protocol_type`, `service`) into a numerical format using Label Encoding.      |
| **5** | IT24101277 | **Outlier Analysis**: Investigate the distribution of key numerical features to identify and visualize potential outliers that could impact scaling. |
| **6** | IT24101261 | **Normalization / Scaling**: Apply Standardization (StandardScaler) to all numerical features to ensure they are on a consistent scale for modeling.      |

---

## ⚙️ How to Run the Project

### 1. Prerequisites

> **Note:** It is highly recommended to run this project within a Python virtual environment to avoid conflicts with system-wide packages.

First, ensure you have the required Python libraries installed:
```bash
pip install pandas scikit-learn matplotlib seaborn kagglehub-------++
