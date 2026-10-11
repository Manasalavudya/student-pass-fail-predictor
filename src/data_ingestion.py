import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, raw_data_path="data/raw/students.csv", artifacts_dir="artifacts"):
        self.raw_data_path = raw_data_path
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.train_data_path = os.path.join(self.artifacts_dir, "train.csv")
        self.test_data_path = os.path.join(self.artifacts_dir, "test.csv")

    def validate_schema(self, df: pd.DataFrame) -> bool:
        required_columns = {'StudyHours', 'Attendance', 'PreviousScore', 'Result'}
        if not required_columns.issubset(set(df.columns)):
            missing = required_columns - set(df.columns)
            raise ValueError(f"[MLOps Ingestion Error] Missing columns: {missing}")

        if df.isnull().values.any():
            raise ValueError("[MLOps Ingestion Error] Dataset contains null/NaN values.")

        # Validate numeric ranges
        if (df['StudyHours'] < 0).any() or (df['StudyHours'] > 24).any():
            raise ValueError("[MLOps Ingestion Error] StudyHours out of reasonable bounds (0-24).")
        if (df['Attendance'] < 0).any() or (df['Attendance'] > 100).any():
            raise ValueError("[MLOps Ingestion Error] Attendance out of bounds (0-100%).")
        if (df['PreviousScore'] < 0).any() or (df['PreviousScore'] > 100).any():
            raise ValueError("[MLOps Ingestion Error] PreviousScore out of bounds (0-100).")

        return True

    def initiate_data_ingestion(self):
        print("\n--- [MLOps Step 1: Data Ingestion & Schema Validation] ---")
        if not os.path.exists(self.raw_data_path):
            alt_path = "students.csv"
            if os.path.exists(alt_path):
                self.raw_data_path = alt_path
            else:
                raise FileNotFoundError(f"Raw data file not found at {self.raw_data_path}")

        print(f"Loading raw dataset from: {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)
        self.validate_schema(df)
        print(f"Data Schema Validated successfully. Total Records: {len(df)}")

        # Stratified train/test split (80% train, 20% test)
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['Result']
        )

        train_df.to_csv(self.train_data_path, index=False)
        test_df.to_csv(self.test_data_path, index=False)

        print(f"Ingested Train Set ({len(train_df)} rows) -> {self.train_data_path}")
        print(f"Ingested Test Set  ({len(test_df)} rows)  -> {self.test_data_path}")

        return self.train_data_path, self.test_data_path

if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()
