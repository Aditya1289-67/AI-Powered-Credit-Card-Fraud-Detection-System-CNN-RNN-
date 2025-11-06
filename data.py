import pandas as pd

# Path to your test CSV
test_csv_path = r"C:\Users\HP\Desktop\Fraud Detection\test_transactions_100.csv"
output_csv_path = "test_transactions_100_no_class.csv"

# Load the CSV
df = pd.read_csv(test_csv_path)

# Drop the 'Class' column
if 'Class' in df.columns:
    df = df.drop(columns=['Class'])

# Save the updated CSV
df.to_csv(output_csv_path, index=False)

print(f"'Class' column removed. New CSV saved at: {output_csv_path}")
