# Capstone-Project-2
HR ANALYTICS PROJECT
# HR Analytics Capstone Project

## Introduction
This capstone project aims to analyze HR data to uncover insights related to employee attrition, job satisfaction, departmental trends, and other HR metrics. The project includes data loading, cleaning, statistical analysis, exploratory data analysis (EDA), and a final discussion of findings. Our objective is to support HR decision-making with data-driven recommendations.

---

## 1. Import Required Libraries
Essential Python libraries are imported to support data manipulation, visualization, and statistical analysis.
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
```

---

## 2. Load Dataset
The dataset is loaded from a CSV file.
```python
df = pd.read_csv('HR_dataset.csv')  # Replace with actual path
```

---

## 3. Initial Data Exploration
Provides an overview of the dataset, including its structure, data types, and summary statistics.
```python
print("Dataset Shape:", df.shape)
df.info()
df.describe()
df.head()
```

---

## 4. Data Cleaning and Preprocessing
### 4.1 Check for Missing Values
Identify columns with missing values.
```python
print("Missing Values:\n", df.isnull().sum())
```

### 4.2 Handle Missing Values
Missing values are forward filled as a basic imputation technique.
```python
df.fillna(method='ffill', inplace=True)
```

### 4.3 Remove Duplicates
Ensure uniqueness of records.
```python
df.drop_duplicates(inplace=True)
```

### 4.4 Correct Data Types
Correct data types where necessary.
```python
# Example: df['Age'] = df['Age'].astype(int)
```

### 4.5 Resolve Inconsistencies
Standardize categorical data entries.
```python
df['Department'] = df['Department'].str.title()
```

### 4.6 Detect and Handle Outliers
Outliers in numeric columns are capped using the IQR method.
```python
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])
```

---

## 5. Statistical Data Analysis
### 5.1 Descriptive Statistics
Basic summary statistics of the dataset.
```python
print(df.describe())
```

### 5.2 Distribution Plots
Visualize distributions of numerical features.
```python
for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
```
*Discussion:* These histograms help us identify skewness, modality, and outliers in the data. Several features show right skew, which could affect modeling.

### 5.3 Correlation Matrix
```python
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```
*Discussion:* Strong positive or negative correlations may guide feature selection. For example, `MonthlyIncome` is positively correlated with `JobLevel`.

### 5.4 Hypothesis Testing: Department vs Attrition
```python
freq_table = pd.crosstab(df['Department'], df['Attrition'])
chi2, p, dof, ex = stats.chi2_contingency(freq_table)
print(f"Chi2 Test: p-value = {p}")
```
*Discussion:* A low p-value (< 0.05) suggests a statistically significant relationship between department and attrition.

---

## 6. Exploratory Data Analysis (EDA)
### 6.1 Attrition by Department
```python
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Department', hue='Attrition')
plt.title('Attrition Count by Department')
plt.xticks(rotation=45)
plt.show()
```
*Insight:* Some departments show a higher attrition rate. This can inform where retention efforts should be focused.

### 6.2 Age vs Monthly Income
```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='MonthlyIncome', hue='Attrition')
plt.title('Age vs Monthly Income by Attrition')
plt.show()
```
*Insight:* Younger employees with lower income tend to have higher attrition, suggesting a potential area for policy intervention.

### 6.3 Job Satisfaction by Department
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Department', y='JobSatisfaction')
plt.title('Job Satisfaction by Department')
plt.xticks(rotation=45)
plt.show()
```
*Insight:* Job satisfaction differs across departments; targeted engagement programs may be needed.

---

## 7. Insights and Conclusions
```python
print("""
Insights:
- Departments with high attrition rates can be identified for HR intervention.
- Certain age and income ranges correlate with higher attrition.
- Job satisfaction varies significantly by department.

Conclusion:
- Improve workplace satisfaction in high-attrition departments.
- Consider incentives or training for younger, lower-income employees.
- Track satisfaction and retention metrics continuously.

Future Work:
- Implement classification models to predict attrition risk.
- Enhance analysis with additional features (e.g., tenure, manager satisfaction).
""")
```

---

## 8. Classification Model (To Be Added)
### 8.1 Data Preparation
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Encode categorical variables
df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Features and target
X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 8.2 Model Training
```python
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 8.3 Model Evaluation
```python
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("
Classification Report:")
print(classification_report(y_test, y_pred))
```

### 8.4 Discussion of Results
The classification report includes precision, recall, and F1-score for each class (Attrition = 0 or 1). A balanced accuracy across classes suggests the model performs reasonably well.

Visuals Discussion for Power BI Dashboard
1. Attrition by Department
What it shows: A bar/column chart with departments on the X-axis and attrition counts (Yes/No) stacked or side-by-side.

Insight: This visual helps identify which departments suffer the most from attrition. For example, high attrition in Sales or R&D might indicate poor work conditions or lack of career growth in those teams.

Actionable Recommendation: Investigate employee engagement programs or manager effectiveness in these high-attrition departments.

2. Age vs. Monthly Income by Attrition
What it shows: A scatter plot where each point represents an employee, with Age on the X-axis, Monthly Income on the Y-axis, and color representing attrition status.

Insight: If attrition is clustered among younger and lower-paid employees, it may indicate early career dissatisfaction or better job opportunities elsewhere.

Actionable Recommendation: Design retention strategies for early-career professionals, such as mentorship programs or salary benchmarking.

3. Job Satisfaction by Department
What it shows: A box plot visualizing the spread of job satisfaction scores across different departments.

Insight: Departments with low median satisfaction or wide variability in scores may be facing inconsistent leadership or unclear job roles.

Actionable Recommendation: Conduct department-specific surveys or focus groups to better understand job satisfaction drivers.

4. Overall Attrition Rate (KPI Card)
What it shows: A single number summarizing the proportion of employees who have left.

Insight: Useful for benchmarking against industry standards or internal targets.

Actionable Recommendation: Monitor this KPI quarterly and track progress against retention goals.

5. Correlation Matrix (if implemented)
What it shows: A heatmap of Pearson correlation coefficients between numeric fields like Age, Income, Job Level, etc.

Insight: Highlights strong correlations, such as between JobLevel and MonthlyIncome.

Actionable Recommendation: Use this insight when selecting features for predictive modeling.

6. Feature Importance from Classification Model (Optional)
What it shows: Bar chart showing which features (like Age, Job Role, Income) contribute most to the model's prediction of attrition.

Insight: These are the key drivers of employee turnover, as learned by your ML model.

Actionable Recommendation: Prioritize improving conditions around top-ranked drivers.

7. Slicers and Filters (Interactive Elements)
What they do: Allow users to filter data by attributes like Department, Gender, Age group, Job Role, etc.

Insight: Enables dynamic exploration—e.g., examining how attrition behaves in different groups.

Actionable Recommendation: Use slicers in presentations to showcase how patterns shift across demographics or roles.# Capstone-Project-2
HR ANALYTICS PROJECT
# HR Analytics Capstone Project

## Introduction
This capstone project aims to analyze HR data to uncover insights related to employee attrition, job satisfaction, departmental trends, and other HR metrics. The project includes data loading, cleaning, statistical analysis, exploratory data analysis (EDA), and a final discussion of findings. Our objective is to support HR decision-making with data-driven recommendations.

---

## 1. Import Required Libraries
Essential Python libraries are imported to support data manipulation, visualization, and statistical analysis.
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
```

---

## 2. Load Dataset
The dataset is loaded from a CSV file.
```python
df = pd.read_csv('HR_dataset.csv')  # Replace with actual path
```

---

## 3. Initial Data Exploration
Provides an overview of the dataset, including its structure, data types, and summary statistics.
```python
print("Dataset Shape:", df.shape)
df.info()
df.describe()
df.head()
```

---

## 4. Data Cleaning and Preprocessing
### 4.1 Check for Missing Values
Identify columns with missing values.
```python
print("Missing Values:\n", df.isnull().sum())
```

### 4.2 Handle Missing Values
Missing values are forward filled as a basic imputation technique.
```python
df.fillna(method='ffill', inplace=True)
```

### 4.3 Remove Duplicates
Ensure uniqueness of records.
```python
df.drop_duplicates(inplace=True)
```

### 4.4 Correct Data Types
Correct data types where necessary.
```python
# Example: df['Age'] = df['Age'].astype(int)
```

### 4.5 Resolve Inconsistencies
Standardize categorical data entries.
```python
df['Department'] = df['Department'].str.title()
```

### 4.6 Detect and Handle Outliers
Outliers in numeric columns are capped using the IQR method.
```python
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])
```

---

## 5. Statistical Data Analysis
### 5.1 Descriptive Statistics
Basic summary statistics of the dataset.
```python
print(df.describe())
```

### 5.2 Distribution Plots
Visualize distributions of numerical features.
```python
for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
```
*Discussion:* These histograms help us identify skewness, modality, and outliers in the data. Several features show right skew, which could affect modeling.

### 5.3 Correlation Matrix
```python
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```
*Discussion:* Strong positive or negative correlations may guide feature selection. For example, `MonthlyIncome` is positively correlated with `JobLevel`.

### 5.4 Hypothesis Testing: Department vs Attrition
```python
freq_table = pd.crosstab(df['Department'], df['Attrition'])
chi2, p, dof, ex = stats.chi2_contingency(freq_table)
print(f"Chi2 Test: p-value = {p}")
```
*Discussion:* A low p-value (< 0.05) suggests a statistically significant relationship between department and attrition.

---

## 6. Exploratory Data Analysis (EDA)
### 6.1 Attrition by Department
```python
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Department', hue='Attrition')
plt.title('Attrition Count by Department')
plt.xticks(rotation=45)
plt.show()
```
*Insight:* Some departments show a higher attrition rate. This can inform where retention efforts should be focused.

### 6.2 Age vs Monthly Income
```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='MonthlyIncome', hue='Attrition')
plt.title('Age vs Monthly Income by Attrition')
plt.show()
```
*Insight:* Younger employees with lower income tend to have higher attrition, suggesting a potential area for policy intervention.

### 6.3 Job Satisfaction by Department
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Department', y='JobSatisfaction')
plt.title('Job Satisfaction by Department')
plt.xticks(rotation=45)
plt.show()
```
*Insight:* Job satisfaction differs across departments; targeted engagement programs may be needed.

---

## 7. Insights and Conclusions
```python
print("""
Insights:
- Departments with high attrition rates can be identified for HR intervention.
- Certain age and income ranges correlate with higher attrition.
- Job satisfaction varies significantly by department.

Conclusion:
- Improve workplace satisfaction in high-attrition departments.
- Consider incentives or training for younger, lower-income employees.
- Track satisfaction and retention metrics continuously.

Future Work:
- Implement classification models to predict attrition risk.
- Enhance analysis with additional features (e.g., tenure, manager satisfaction).
""")
```

---

## 8. Classification Model (To Be Added)
### 8.1 Data Preparation
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Encode categorical variables
df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Features and target
X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 8.2 Model Training
```python
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 8.3 Model Evaluation
```python
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("
Classification Report:")
print(classification_report(y_test, y_pred))
```

### 8.4 Discussion of Results
The classification report includes precision, recall, and F1-score for each class (Attrition = 0 or 1). A balanced accuracy across classes suggests the model performs reasonably well.

Visuals Discussion for Power BI Dashboard
1. Attrition by Department
What it shows: A bar/column chart with departments on the X-axis and attrition counts (Yes/No) stacked or side-by-side.

Insight: This visual helps identify which departments suffer the most from attrition. For example, high attrition in Sales or R&D might indicate poor work conditions or lack of career growth in those teams.

Actionable Recommendation: Investigate employee engagement programs or manager effectiveness in these high-attrition departments.

2. Age vs. Monthly Income by Attrition
What it shows: A scatter plot where each point represents an employee, with Age on the X-axis, Monthly Income on the Y-axis, and color representing attrition status.

Insight: If attrition is clustered among younger and lower-paid employees, it may indicate early career dissatisfaction or better job opportunities elsewhere.

Actionable Recommendation: Design retention strategies for early-career professionals, such as mentorship programs or salary benchmarking.

3. Job Satisfaction by Department
What it shows: A box plot visualizing the spread of job satisfaction scores across different departments.

Insight: Departments with low median satisfaction or wide variability in scores may be facing inconsistent leadership or unclear job roles.

Actionable Recommendation: Conduct department-specific surveys or focus groups to better understand job satisfaction drivers.

4. Overall Attrition Rate (KPI Card)
What it shows: A single number summarizing the proportion of employees who have left.

Insight: Useful for benchmarking against industry standards or internal targets.

Actionable Recommendation: Monitor this KPI quarterly and track progress against retention goals.

5. Correlation Matrix (if implemented)
What it shows: A heatmap of Pearson correlation coefficients between numeric fields like Age, Income, Job Level, etc.

Insight: Highlights strong correlations, such as between JobLevel and MonthlyIncome.

Actionable Recommendation: Use this insight when selecting features for predictive modeling.

6. Feature Importance from Classification Model (Optional)
What it shows: Bar chart showing which features (like Age, Job Role, Income) contribute most to the model's prediction of attrition.

Insight: These are the key drivers of employee turnover, as learned by your ML model.

Actionable Recommendation: Prioritize improving conditions around top-ranked drivers.

7. Slicers and Filters (Interactive Elements)
What they do: Allow users to filter data by attributes like Department, Gender, Age group, Job Role, etc.

Insight: Enables dynamic exploration—e.g., examining how attrition behaves in different groups.

Actionable Recommendation: Use slicers in presentations to showcase how patterns shift across demographics or roles.# Capstone-Project-2
HR ANALYTICS PROJECT
# HR Analytics Capstone Project

## Introduction
This capstone project aims to analyze HR data to uncover insights related to employee attrition, job satisfaction, departmental trends, and other HR metrics. The project includes data loading, cleaning, statistical analysis, exploratory data analysis (EDA), and a final discussion of findings. Our objective is to support HR decision-making with data-driven recommendations.

---

## 1. Import Required Libraries
Essential Python libraries are imported to support data manipulation, visualization, and statistical analysis.
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
```

---

## 2. Load Dataset
The dataset is loaded from a CSV file.
```python
df = pd.read_csv('HR_dataset.csv')  # Replace with actual path
```

---

## 3. Initial Data Exploration
Provides an overview of the dataset, including its structure, data types, and summary statistics.
```python
print("Dataset Shape:", df.shape)
df.info()
df.describe()
df.head()
```

---

## 4. Data Cleaning and Preprocessing
### 4.1 Check for Missing Values
Identify columns with missing values.
```python
print("Missing Values:\n", df.isnull().sum())
```

### 4.2 Handle Missing Values
Missing values are forward filled as a basic imputation technique.
```python
df.fillna(method='ffill', inplace=True)
```

### 4.3 Remove Duplicates
Ensure uniqueness of records.
```python
df.drop_duplicates(inplace=True)
```

### 4.4 Correct Data Types
Correct data types where necessary.
```python
# Example: df['Age'] = df['Age'].astype(int)
```

### 4.5 Resolve Inconsistencies
Standardize categorical data entries.
```python
df['Department'] = df['Department'].str.title()
```

### 4.6 Detect and Handle Outliers
Outliers in numeric columns are capped using the IQR method.
```python
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])
```

---

## 5. Statistical Data Analysis
### 5.1 Descriptive Statistics
Basic summary statistics of the dataset.
```python
print(df.describe())
```

### 5.2 Distribution Plots
Visualize distributions of numerical features.
```python
for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
```
*Discussion:* These histograms help us identify skewness, modality, and outliers in the data. Several features show right skew, which could affect modeling.

### 5.3 Correlation Matrix
```python
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```
*Discussion:* Strong positive or negative correlations may guide feature selection. For example, `MonthlyIncome` is positively correlated with `JobLevel`.

### 5.4 Hypothesis Testing: Department vs Attrition
```python
freq_table = pd.crosstab(df['Department'], df['Attrition'])
chi2, p, dof, ex = stats.chi2_contingency(freq_table)
print(f"Chi2 Test: p-value = {p}")
```
*Discussion:* A low p-value (< 0.05) suggests a statistically significant relationship between department and attrition.

---

## 6. Exploratory Data Analysis (EDA)
### 6.1 Attrition by Department
```python
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Department', hue='Attrition')
plt.title('Attrition Count by Department')
plt.xticks(rotation=45)
plt.show()
```
*Insight:* Some departments show a higher attrition rate. This can inform where retention efforts should be focused.

### 6.2 Age vs Monthly Income
```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='MonthlyIncome', hue='Attrition')
plt.title('Age vs Monthly Income by Attrition')
plt.show()
```
*Insight:* Younger employees with lower income tend to have higher attrition, suggesting a potential area for policy intervention.

### 6.3 Job Satisfaction by Department
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Department', y='JobSatisfaction')
plt.title('Job Satisfaction by Department')
plt.xticks(rotation=45)
plt.show()
```
*Insight:* Job satisfaction differs across departments; targeted engagement programs may be needed.

---

## 7. Insights and Conclusions
```python
print("""
Insights:
- Departments with high attrition rates can be identified for HR intervention.
- Certain age and income ranges correlate with higher attrition.
- Job satisfaction varies significantly by department.

Conclusion:
- Improve workplace satisfaction in high-attrition departments.
- Consider incentives or training for younger, lower-income employees.
- Track satisfaction and retention metrics continuously.

Future Work:
- Implement classification models to predict attrition risk.
- Enhance analysis with additional features (e.g., tenure, manager satisfaction).
""")
```

---

## 8. Classification Model (To Be Added)
### 8.1 Data Preparation
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Encode categorical variables
df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Features and target
X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 8.2 Model Training
```python
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 8.3 Model Evaluation
```python
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("
Classification Report:")
print(classification_report(y_test, y_pred))
```

### 8.4 Discussion of Results
The classification report includes precision, recall, and F1-score for each class (Attrition = 0 or 1). A balanced accuracy across classes suggests the model performs reasonably well.

Visuals Discussion for Power BI Dashboard
1. Attrition by Department
What it shows: A bar/column chart with departments on the X-axis and attrition counts (Yes/No) stacked or side-by-side.

Insight: This visual helps identify which departments suffer the most from attrition. For example, high attrition in Sales or R&D might indicate poor work conditions or lack of career growth in those teams.

Actionable Recommendation: Investigate employee engagement programs or manager effectiveness in these high-attrition departments.

2. Age vs. Monthly Income by Attrition
What it shows: A scatter plot where each point represents an employee, with Age on the X-axis, Monthly Income on the Y-axis, and color representing attrition status.

Insight: If attrition is clustered among younger and lower-paid employees, it may indicate early career dissatisfaction or better job opportunities elsewhere.

Actionable Recommendation: Design retention strategies for early-career professionals, such as mentorship programs or salary benchmarking.

3. Job Satisfaction by Department
What it shows: A box plot visualizing the spread of job satisfaction scores across different departments.

Insight: Departments with low median satisfaction or wide variability in scores may be facing inconsistent leadership or unclear job roles.

Actionable Recommendation: Conduct department-specific surveys or focus groups to better understand job satisfaction drivers.

4. Overall Attrition Rate (KPI Card)
What it shows: A single number summarizing the proportion of employees who have left.

Insight: Useful for benchmarking against industry standards or internal targets.

Actionable Recommendation: Monitor this KPI quarterly and track progress against retention goals.

5. Correlation Matrix (if implemented)
What it shows: A heatmap of Pearson correlation coefficients between numeric fields like Age, Income, Job Level, etc.

Insight: Highlights strong correlations, such as between JobLevel and MonthlyIncome.

Actionable Recommendation: Use this insight when selecting features for predictive modeling.

6. Feature Importance from Classification Model (Optional)
What it shows: Bar chart showing which features (like Age, Job Role, Income) contribute most to the model's prediction of attrition.

Insight: These are the key drivers of employee turnover, as learned by your ML model.

Actionable Recommendation: Prioritize improving conditions around top-ranked drivers.

7. Slicers and Filters (Interactive Elements)
What they do: Allow users to filter data by attributes like Department, Gender, Age group, Job Role, etc.

Insight: Enables dynamic exploration—e.g., examining how attrition behaves in different groups.

Actionable Recommendation: Use slicers in presentations to showcase how patterns shift across demographics or roles.# Capstone-Project-2
HR ANALYTICS PROJECT
# HR Analytics Capstone Project

## Introduction
This capstone project aims to analyze HR data to uncover insights related to employee attrition, job satisfaction, departmental trends, and other HR metrics. The project includes data loading, cleaning, statistical analysis, exploratory data analysis (EDA), and a final discussion of findings. Our objective is to support HR decision-making with data-driven recommendations.

---

## 1. Import Required Libraries
Essential Python libraries are imported to support data manipulation, visualization, and statistical analysis.
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
```

---

## 2. Load Dataset
The dataset is loaded from a CSV file.
```python
df = pd.read_csv('HR_dataset.csv')  # Replace with actual path
```

---

## 3. Initial Data Exploration
Provides an overview of the dataset, including its structure, data types, and summary statistics.
```python
print("Dataset Shape:", df.shape)
df.info()
df.describe()
df.head()
```

---

## 4. Data Cleaning and Preprocessing
### 4.1 Check for Missing Values
Identify columns with missing values.
```python
print("Missing Values:\n", df.isnull().sum())
```

### 4.2 Handle Missing Values
Missing values are forward filled as a basic imputation technique.
```python
df.fillna(method='ffill', inplace=True)
```

### 4.3 Remove Duplicates
Ensure uniqueness of records.
```python
df.drop_duplicates(inplace=True)
```

### 4.4 Correct Data Types
Correct data types where necessary.
```python
# Example: df['Age'] = df['Age'].astype(int)
```

### 4.5 Resolve Inconsistencies
Standardize categorical data entries.
```python
df['Department'] = df['Department'].str.title()
```

### 4.6 Detect and Handle Outliers
Outliers in numeric columns are capped using the IQR method.
```python
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])
```

---

## 5. Statistical Data Analysis
### 5.1 Descriptive Statistics
Basic summary statistics of the dataset.
```python
print(df.describe())
```

### 5.2 Distribution Plots
Visualize distributions of numerical features.
```python
for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
```
*Discussion:* These histograms help us identify skewness, modality, and outliers in the data. Several features show right skew, which could affect modeling.

### 5.3 Correlation Matrix
```python
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```
*Discussion:* Strong positive or negative correlations may guide feature selection. For example, `MonthlyIncome` is positively correlated with `JobLevel`.

### 5.4 Hypothesis Testing: Department vs Attrition
```python
freq_table = pd.crosstab(df['Department'], df['Attrition'])
chi2, p, dof, ex = stats.chi2_contingency(freq_table)
print(f"Chi2 Test: p-value = {p}")
```
*Discussion:* A low p-value (< 0.05) suggests a statistically significant relationship between department and attrition.

---

## 6. Exploratory Data Analysis (EDA)
### 6.1 Attrition by Department
```python
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Department', hue='Attrition')
plt.title('Attrition Count by Department')
plt.xticks(rotation=45)
plt.show()
```
*Insight:* Some departments show a higher attrition rate. This can inform where retention efforts should be focused.

### 6.2 Age vs Monthly Income
```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='MonthlyIncome', hue='Attrition')
plt.title('Age vs Monthly Income by Attrition')
plt.show()
```
*Insight:* Younger employees with lower income tend to have higher attrition, suggesting a potential area for policy intervention.

### 6.3 Job Satisfaction by Department
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Department', y='JobSatisfaction')
plt.title('Job Satisfaction by Department')
plt.xticks(rotation=45)
plt.show()
```
*Insight:* Job satisfaction differs across departments; targeted engagement programs may be needed.

---

## 7. Insights and Conclusions
```python
print("""
Insights:
- Departments with high attrition rates can be identified for HR intervention.
- Certain age and income ranges correlate with higher attrition.
- Job satisfaction varies significantly by department.

Conclusion:
- Improve workplace satisfaction in high-attrition departments.
- Consider incentives or training for younger, lower-income employees.
- Track satisfaction and retention metrics continuously.

Future Work:
- Implement classification models to predict attrition risk.
- Enhance analysis with additional features (e.g., tenure, manager satisfaction).
""")
```

---

## 8. Classification Model (To Be Added)
### 8.1 Data Preparation
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Encode categorical variables
df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Features and target
X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 8.2 Model Training
```python
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 8.3 Model Evaluation
```python
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("
Classification Report:")
print(classification_report(y_test, y_pred))
```

### 8.4 Discussion of Results
The classification report includes precision, recall, and F1-score for each class (Attrition = 0 or 1). A balanced accuracy across classes suggests the model performs reasonably well.

Visuals Discussion for Power BI Dashboard
1. Attrition by Department
What it shows: A bar/column chart with departments on the X-axis and attrition counts (Yes/No) stacked or side-by-side.

Insight: This visual helps identify which departments suffer the most from attrition. For example, high attrition in Sales or R&D might indicate poor work conditions or lack of career growth in those teams.

Actionable Recommendation: Investigate employee engagement programs or manager effectiveness in these high-attrition departments.

2. Age vs. Monthly Income by Attrition
What it shows: A scatter plot where each point represents an employee, with Age on the X-axis, Monthly Income on the Y-axis, and color representing attrition status.

Insight: If attrition is clustered among younger and lower-paid employees, it may indicate early career dissatisfaction or better job opportunities elsewhere.

Actionable Recommendation: Design retention strategies for early-career professionals, such as mentorship programs or salary benchmarking.

3. Job Satisfaction by Department
What it shows: A box plot visualizing the spread of job satisfaction scores across different departments.

Insight: Departments with low median satisfaction or wide variability in scores may be facing inconsistent leadership or unclear job roles.

Actionable Recommendation: Conduct department-specific surveys or focus groups to better understand job satisfaction drivers.

4. Overall Attrition Rate (KPI Card)
What it shows: A single number summarizing the proportion of employees who have left.

Insight: Useful for benchmarking against industry standards or internal targets.

Actionable Recommendation: Monitor this KPI quarterly and track progress against retention goals.

5. Correlation Matrix (if implemented)
What it shows: A heatmap of Pearson correlation coefficients between numeric fields like Age, Income, Job Level, etc.

Insight: Highlights strong correlations, such as between JobLevel and MonthlyIncome.

Actionable Recommendation: Use this insight when selecting features for predictive modeling.

6. Feature Importance from Classification Model (Optional)
What it shows: Bar chart showing which features (like Age, Job Role, Income) contribute most to the model's prediction of attrition.

Insight: These are the key drivers of employee turnover, as learned by your ML model.

Actionable Recommendation: Prioritize improving conditions around top-ranked drivers.

7. Slicers and Filters (Interactive Elements)
What they do: Allow users to filter data by attributes like Department, Gender, Age group, Job Role, etc.

Insight: Enables dynamic exploration—e.g., examining how attrition behaves in different groups.

Actionable Recommendation: Use slicers in presentations to showcase how patterns shift across demographics or roles.# Capstone-Project-2
HR ANALYTICS PROJECT
# HR Analytics Capstone Project

## Introduction
This capstone project aims to analyze HR data to uncover insights related to employee attrition, job satisfaction, departmental trends, and other HR metrics. The project includes data loading, cleaning, statistical analysis, exploratory data analysis (EDA), and a final discussion of findings. Our objective is to support HR decision-making with data-driven recommendations.

---

## 1. Import Required Libraries
Essential Python libraries are imported to support data manipulation, visualization, and statistical analysis.
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
```

---

## 2. Load Dataset
The dataset is loaded from a CSV file.
```python
df = pd.read_csv('HR_dataset.csv')  # Replace with actual path
```

---

## 3. Initial Data Exploration
Provides an overview of the dataset, including its structure, data types, and summary statistics.
```python
print("Dataset Shape:", df.shape)
df.info()
df.describe()
df.head()
```

---

## 4. Data Cleaning and Preprocessing
### 4.1 Check for Missing Values
Identify columns with missing values.
```python
print("Missing Values:\n", df.isnull().sum())
```

### 4.2 Handle Missing Values
Missing values are forward filled as a basic imputation technique.
```python
df.fillna(method='ffill', inplace=True)
```

### 4.3 Remove Duplicates
Ensure uniqueness of records.
```python
df.drop_duplicates(inplace=True)
```

### 4.4 Correct Data Types
Correct data types where necessary.
```python
# Example: df['Age'] = df['Age'].astype(int)
```

### 4.5 Resolve Inconsistencies
Standardize categorical data entries.
```python
df['Department'] = df['Department'].str.title()
```

### 4.6 Detect and Handle Outliers
Outliers in numeric columns are capped using the IQR method.
```python
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])
```

---

## 5. Statistical Data Analysis
### 5.1 Descriptive Statistics
Basic summary statistics of the dataset.
```python
print(df.describe())
```

### 5.2 Distribution Plots
Visualize distributions of numerical features.
```python
for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
```
*Discussion:* These histograms help us identify skewness, modality, and outliers in the data. Several features show right skew, which could affect modeling.

### 5.3 Correlation Matrix
```python
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```
*Discussion:* Strong positive or negative correlations may guide feature selection. For example, `MonthlyIncome` is positively correlated with `JobLevel`.

### 5.4 Hypothesis Testing: Department vs Attrition
```python
freq_table = pd.crosstab(df['Department'], df['Attrition'])
chi2, p, dof, ex = stats.chi2_contingency(freq_table)
print(f"Chi2 Test: p-value = {p}")
```
*Discussion:* A low p-value (< 0.05) suggests a statistically significant relationship between department and attrition.

---

## 6. Exploratory Data Analysis (EDA)
### 6.1 Attrition by Department
```python
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Department', hue='Attrition')
plt.title('Attrition Count by Department')
plt.xticks(rotation=45)
plt.show()
```
*Insight:* Some departments show a higher attrition rate. This can inform where retention efforts should be focused.

### 6.2 Age vs Monthly Income
```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='MonthlyIncome', hue='Attrition')
plt.title('Age vs Monthly Income by Attrition')
plt.show()
```
*Insight:* Younger employees with lower income tend to have higher attrition, suggesting a potential area for policy intervention.

### 6.3 Job Satisfaction by Department
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Department', y='JobSatisfaction')
plt.title('Job Satisfaction by Department')
plt.xticks(rotation=45)
plt.show()
```
*Insight:* Job satisfaction differs across departments; targeted engagement programs may be needed.

---

## 7. Insights and Conclusions
```python
print("""
Insights:
- Departments with high attrition rates can be identified for HR intervention.
- Certain age and income ranges correlate with higher attrition.
- Job satisfaction varies significantly by department.

Conclusion:
- Improve workplace satisfaction in high-attrition departments.
- Consider incentives or training for younger, lower-income employees.
- Track satisfaction and retention metrics continuously.

Future Work:
- Implement classification models to predict attrition risk.
- Enhance analysis with additional features (e.g., tenure, manager satisfaction).
""")
```

---

## 8. Classification Model (To Be Added)
### 8.1 Data Preparation
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Encode categorical variables
df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Features and target
X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 8.2 Model Training
```python
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 8.3 Model Evaluation
```python
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("
Classification Report:")
print(classification_report(y_test, y_pred))
```

### 8.4 Discussion of Results
The classification report includes precision, recall, and F1-score for each class (Attrition = 0 or 1). A balanced accuracy across classes suggests the model performs reasonably well.

Visuals Discussion for Power BI Dashboard
1. Attrition by Department
What it shows: A bar/column chart with departments on the X-axis and attrition counts (Yes/No) stacked or side-by-side.

Insight: This visual helps identify which departments suffer the most from attrition. For example, high attrition in Sales or R&D might indicate poor work conditions or lack of career growth in those teams.

Actionable Recommendation: Investigate employee engagement programs or manager effectiveness in these high-attrition departments.

2. Age vs. Monthly Income by Attrition
What it shows: A scatter plot where each point represents an employee, with Age on the X-axis, Monthly Income on the Y-axis, and color representing attrition status.

Insight: If attrition is clustered among younger and lower-paid employees, it may indicate early career dissatisfaction or better job opportunities elsewhere.

Actionable Recommendation: Design retention strategies for early-career professionals, such as mentorship programs or salary benchmarking.

3. Job Satisfaction by Department
What it shows: A box plot visualizing the spread of job satisfaction scores across different departments.

Insight: Departments with low median satisfaction or wide variability in scores may be facing inconsistent leadership or unclear job roles.

Actionable Recommendation: Conduct department-specific surveys or focus groups to better understand job satisfaction drivers.

4. Overall Attrition Rate (KPI Card)
What it shows: A single number summarizing the proportion of employees who have left.

Insight: Useful for benchmarking against industry standards or internal targets.

Actionable Recommendation: Monitor this KPI quarterly and track progress against retention goals.

5. Correlation Matrix (if implemented)
What it shows: A heatmap of Pearson correlation coefficients between numeric fields like Age, Income, Job Level, etc.

Insight: Highlights strong correlations, such as between JobLevel and MonthlyIncome.

Actionable Recommendation: Use this insight when selecting features for predictive modeling.

6. Feature Importance from Classification Model (Optional)
What it shows: Bar chart showing which features (like Age, Job Role, Income) contribute most to the model's prediction of attrition.

Insight: These are the key drivers of employee turnover, as learned by your ML model.

Actionable Recommendation: Prioritize improving conditions around top-ranked drivers.

7. Slicers and Filters (Interactive Elements)
What they do: Allow users to filter data by attributes like Department, Gender, Age group, Job Role, etc.

Insight: Enables dynamic exploration—e.g., examining how attrition behaves in different groups.

Actionable Recommendation: Use slicers in presentations to showcase how patterns shift across demographics or roles.
