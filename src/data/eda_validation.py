import pandas as pd
import numpy as np

def validate_dataset(file_path: str):
    """
    Validates the dataset provenance, duplicates, and schema.
    """
    print(f"Loading dataset from: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print("--- 1. SCHEMA VALIDATION ---")
    required_columns = {'Subject', 'Body', 'Label'}
    actual_columns = set(df.columns)
    
    if not required_columns.issubset(actual_columns):
        print(f"ERROR: Missing required columns. Found {actual_columns}, Expected {required_columns}")
    else:
        print("Schema check passed.")

    print("\n--- 2. MISSING VALUES ---")
    print(df.isnull().sum())

    print("\n--- 3. DUPLICATE CHECK ---")
    total_rows = len(df)
    exact_duplicates = df.duplicated().sum()
    print(f"Total Rows: {total_rows}")
    print(f"Exact Duplicates: {exact_duplicates} ({(exact_duplicates/total_rows)*100:.2f}%)")
    
    # Check for near-duplicates or template leakage by grouping by Subject
    dup_subjects = df.duplicated(subset=['Subject']).sum()
    print(f"Duplicate Subjects (indicates template usage): {dup_subjects} ({(dup_subjects/total_rows)*100:.2f}%)")

    print("\n--- 4. LABEL DISTRIBUTION ---")
    print(df['Label'].value_counts(normalize=True))
    unique_labels = df['Label'].unique()
    if set(unique_labels) != {0, 1}:
        print(f"WARNING: Unexpected label values found: {unique_labels}")
    
    print("\n--- 5. PROVENANCE & LEAKAGE INVESTIGATION ---")
    print("Investigating potentially synthesized templates in Body...")
    # Count the most common exact bodies to see if it's heavily templated
    top_bodies = df['Body'].value_counts().head(5)
    print("Top 5 most frequent email bodies:")
    print(top_bodies)

if __name__ == "__main__":
    validate_dataset("data/raw/Zero-Day_Phishing_Emails_Corpus.csv")
