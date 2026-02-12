import mlflow
mlflow.projects.run(uri="./",entry_point="python train_model.py",experiment_name="Salary Model")
