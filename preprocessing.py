import pandas as pd
import joblib

preprocessor = joblib.load("preprocessor_pipeline.pkl")

df = pd.read_csv("cdc_diabetes_health_indicators.csv")
df['Diabetes_binary'].isin([0, 1])

data_transformed = preprocessor.fit_transform(df)

joblib.dump(preprocessor, "preprocessor_pipeline_fit.pkl")

df.to_csv("test_df_after_preprocessing.csv", index=False)
