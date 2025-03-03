import pandas as pd
from google.colab import files
import matplotlib.pyplot as plt
import seaborn as sns

file = pd.read_excel(r"C:\Users\Cvinu\OneDrive - Amrita vishwa vidyapeetham (1)\SEM-4\ML\lab-2\Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")


print("\nMissing Values Before Imputation:\n", thyroid_data.isnull().sum())
numeric_cols = thyroid_data.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = thyroid_data.select_dtypes(include=["object"]).columns

print("\nNumeric Columns:", numeric_cols)
print("\nCategorical Columns:", categorical_cols)

plt.figure(figsize=(12, 6))
thyroid_data[numeric_cols].boxplot()
plt.xticks(rotation=45)
plt.title("Boxplot to Detect Outliers")
plt.show()

for col in numeric_cols:
    if thyroid_data[col].isnull().sum() > 0:  # Check if there are missing values
        if thyroid_data[col].skew() < 1:  # No strong skewness, assume no outliers
            thyroid_data[col].fillna(thyroid_data[col].mean(), inplace=True)  # Use Mean
        else:
            thyroid_data[col].fillna(thyroid_data[col].median(), inplace=True)  # Use Median for outliers

print("\nMissing Values After Numeric Imputation:\n", thyroid_data.isnull().sum())

for col in categorical_cols:
    if thyroid_data[col].isnull().sum() > 0:
        thyroid_data[col].fillna(thyroid_data[col].mode()[0], inplace=True)  # Use Mode

print("\nMissing Values After Categorical Imputation:\n", thyroid_data.isnull().sum())

thyroid_data.to_excel("Imputed_Thyroid_Data.xlsx", index=False)

files.download("Imputed_Thyroid_Data.xlsx")
