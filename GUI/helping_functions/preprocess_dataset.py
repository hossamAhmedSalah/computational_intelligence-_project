import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


# Custom LabelEncoder Transformer
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.encoders = []

	def fit(self, X, y=None):
		# Fit a LabelEncoder for each categorical column
		self.encoders = [LabelEncoder().fit(X[:, i]) for i in range(X.shape[1])]  # Assuming X is 2D
		return self

	def transform(self, X):
		X_transformed = X.copy()
		for i, encoder in enumerate(self.encoders):
			X_transformed[:, i] = encoder.transform(X[:, i])
		return X_transformed


def load_data(path):
	return pd.read_csv(path)


def build_pipeline(numerical, categorical, scaling_method="standard", imputer_type="knn"):
	scaler = StandardScaler() if scaling_method == "standard" else MinMaxScaler()
	transformers = []

	# Numerical preprocessing pipeline
	if numerical:
		num_imputer = KNNImputer() if imputer_type == "knn" else SimpleImputer(strategy='mean')
		num_pipeline = Pipeline([('imputer', num_imputer), ('scaler', scaler)])
		transformers.append(('num', num_pipeline, numerical))

	# Categorical preprocessing pipeline with Label Encoding
	if categorical:
		cat_imputer = SimpleImputer(strategy='most_frequent')
		cat_pipeline = Pipeline([('imputer', cat_imputer), ('encoder', LabelEncoderTransformer())])
		transformers.append(('cat', cat_pipeline, categorical))

	return ColumnTransformer(transformers)


def run_preprocessing(csv_path, output_path="./preprocessed_dataset/processed_data.csv",
					  scaling_method="standard", imputer_type="knn"):
	df = load_data(csv_path)
	print(f"✅ Loaded dataset with shape: {df.shape}")

	# Drop high-uniqueness columns (likely ID columns)
	df = df.loc[:, df.nunique() / len(df) < 1]

	# Identify numerical and categorical columns
	numerical, categorical = df.select_dtypes(include=['int64', 'float64']).columns.tolist(), df.select_dtypes(
		include=['object', 'category']).columns.tolist()
	print(f"Numerical columns: {numerical}")
	print(f"Categorical columns: {categorical}")

	# Build preprocessing pipeline
	pipeline = build_pipeline(numerical, categorical, scaling_method, imputer_type)
	processed = pipeline.fit_transform(df)

	# Reconstruct column names
	feature_names = list(numerical) + list(categorical)  # For label encoding, keep original column names
	processed_df = pd.DataFrame(processed, columns=feature_names)

	# Save processed data
	processed_df.to_csv(output_path, index=False)
	print(f"📁 Processed data saved to: {output_path}")

	return processed_df