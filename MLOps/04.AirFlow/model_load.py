from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import pandas as pd
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

client = MlflowClient()




def register_model(model_name, run_id):
    model_name = 'LogisticRegression'
    run_id = '33f5b9a0bd6c4f90b78b2fb1f36caf94'
    model_version = '1'

    try:    
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)
        model_version = mlflow.register_model(model_uri, model_name) # Version 1
        print(f"Model {model_name} registered successfully with version {model_version}")
        return model_version.version
    except Exception as e:
        print(f"Error registering model: {e}")
        raise


# Promote a model to staging
def promote_to_staging(model_name, version):
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging"
    )
    print(f'Model: {model_name}, Version: {version} promoted to Staging...')


# Promote a model to production
def promote_to_production(model_name, version):
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    print(f'Model: {model_name}, Version: {version} promoted to Production...')


# Archive a model
def archive_model(model_name, version):
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Archived"
    )
    print(f'Model: {model_name}, Version: {version} archived...')


# Load a model from MLflow Model Registry
def load_model_from_registry(model_name, stage="Production"):
    try:    
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Loaded model: {model_name} from stage: {stage}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


# Perform a prediction with the loaded model
def predict_with_model(model, input_data):

    predictions = model.predict(input_data)
    return predictions