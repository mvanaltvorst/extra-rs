import extra_py as ep
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

model = ep.ExtraForestRegressor(
    n_estimators = 1,
    min_samples_split = 1,
    max_depth = 2,
    n_jobs = -1
)
model

N = 100000
df = pd.read_parquet('data.parquet').head(N)
df_train, df_test = df.iloc[:int(0.8 * N)], df.iloc[int(0.8 * N):]
X_train = df_train.drop(columns=['y'])
y_train = df_train['y']
X_test = df_test.drop(columns=['y'])
y_test = df_test['y']

model.fit(X_train.values.astype(np.float32), y_train.values.astype(np.float32))

inferencer = model.get_inferencer()
inferencer



