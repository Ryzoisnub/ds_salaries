import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('ds_salaries.csv')
df.drop_duplicates(inplace=True)
df.drop(['salary', 'salary_currency'], axis=1, inplace=True)
sns.set_style('whitegrid')
print("--- Shape of the Dataset after initial cleaning ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\n--- First 5 Rows ---")
print(df.head())
print("\n--- 1. Applying One-Hot Encoding ---")
categorical_cols = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Shape after One-Hot Encoding:", df_encoded.shape)
print("A few columns from the encoded dataframe:")
print(df_encoded[['work_year', 'salary_in_usd', 'remote_ratio', 'experience_level_EX', 'employment_type_FT']].head())
print("\n--- 2. Applying Standard Scaler ---")

X = df_encoded.drop('salary_in_usd', axis=1)
y = df_encoded['salary_in_usd']


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("Scaled features (first 5 rows):")
print(X_scaled_df.head())

print("\n--- 3. Performing Train-Test Split ---")

X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

print("Shape of the training features (X_train):", X_train.shape)
print("Shape of the testing features (X_test):", X_test.shape)
print("Shape of the training target (y_train):", y_train.shape)
print("Shape of the testing target (y_test):", y_test.shape)
