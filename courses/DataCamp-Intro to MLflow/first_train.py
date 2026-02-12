from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.preprocessing import LabelEncoder
# mlflow.create_experiment("Unicorn Model 2")

mlflow.set_experiment("Unicorn Model 2")
enc=LabelEncoder()

df = pd.read_csv("50_Startups.csv")
df['State'] = enc.fit_transform(df[['State']])
X = df[["R&D Spend", "Administration", "Marketing Spend", "State"]]
y = df[["Profit"]]
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=232,shuffle=True)

model = LinearRegression(n_jobs=1)
model.fit(X_train,y_train)

from sklearn.metrics import r2_score
r2_score = r2_score(y_test,model.predict(X_test))
with mlflow.start_run(run_name="First Run"):
    mlflow.log_metric("r2_score",r2_score)
    mlflow.log_param("n_jobs",1)
    mlflow.log_artifact("first_train.py")