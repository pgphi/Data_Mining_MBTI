import pandas as pd
from classification import train_test_split

# Variable for adjusting how many rows we work with (for testing purposes only! For production use length of dataset)
N = 1000  # len of dataset 8675

# import raw dataset
df = pd.read_csv("data/df_multi_preprocessed.csv")[0:N]

# Create Train and Test Split
X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = train_test_split(df, 0.3, 42)