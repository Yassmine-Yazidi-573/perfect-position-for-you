# **CV-Based Job Recommendation System**

A machine learning project that predicts the most suitable job category for a candidate based on their CV features (experience, skills, language level, etc.).
The model uses a **Random Forest Classifier** trained on a synthetic dataset (`fake_cv_data.csv`).

---

## **📌 Project Overview**

This project demonstrates a simple end-to-end ML pipeline:

* Load and explore the dataset
* Engineer a target variable: **Job Category**
* Split data into training and testing sets
* Train a **RandomForestClassifier**
* Evaluate accuracy
* Predict recommended jobs for new users' CVs

---

## **📂 Dataset**

The dataset `fake_cv_data.csv` contains the following features:

| Feature             | Description                                           |
| ------------------- | ----------------------------------------------------- |
| Experience (years)  | Number of years of work experience                    |
| Degree              | 0 = High School, 1 = Bachelor, 2 = Master, 3 = PhD    |
| Field               | 0 = Engineering, 1 = Business, 2 = IT, 3 = Healthcare |
| Python              | 1 = knows Python                                      |
| Project Mgmt        | 1 = project management experience                     |
| Data Analysis       | 1 = data analysis experience                          |
| English             | 1 = English proficiency                               |
| Languages           | Number of spoken languages                            |
| Certification Score | Score based on certificates                           |
| Soft Skills Score   | Score for soft skills                                 |
| Job Category        | Target label (generated in notebook)                  |

The **job label** is generated automatically using:

```python
def generate_job_label(row):
    if row['Field'] == 2 and row['Python'] == 1:
        return 'Data Scientist'
    elif row['Field'] == 0 and row['Project Mgmt'] == 1:
        return 'Engineering Manager'
    elif row['Field'] == 3:
        return 'Healthcare Specialist'
    elif row['Field'] == 1:
        return 'Business Analyst'
    else:
        return 'Artist'
```

---

## **⚙️ Environment Setup**

### **Install dependencies**

```
pip install pandas numpy scikit-learn
```

## **🧠 Model Building**

### **Train/Test Split**

```python
X = cv_data.drop(['ID', 'Job Category'], axis=1)
y = cv_data['Job Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### **Train Model**

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

---

## **📊 Model Evaluation**

```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```
## **🆕 Predicting a New User's CV**

```python
new_cv_data = {
    'Experience (years)': [4],
    'Degree': [1],
    'Field': [2],
    'Python': [1],
    'Project Mgmt': [0],
    'Data Analysis': [1],
    'English': [1],
    'Languages': [2],
    'Certification Score': [7],
    'Soft Skills Score': [6]
}

new_cv_df = pd.DataFrame(new_cv_data)
model.predict(new_cv_df)
```

**Prediction:**

```
The recommended job for this CV is: Data Scientist
```

## **📥 How to Run the Project**

1. Clone the repository
2. Place `fake_cv_data.csv` in the root directory
3. Install dependencies
4. Open the notebook (`.ipynb`)
5. Run all cells
6. Modify the new CV example to test different profiles

---

## **📌 Future Improvements**

* Add more real CV features
* Use NLP to parse real CV text
* Deploy model as a web app (Flask/Streamlit)
* Add model explainability (SHAP, feature importance)

---
