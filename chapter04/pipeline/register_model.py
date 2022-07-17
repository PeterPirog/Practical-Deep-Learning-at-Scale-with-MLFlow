import click
import mlflow
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

import os
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://ai-machine:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://ai-machine"


@click.command(help="This program registers a trained model .")
@click.option("--mlflow_run_id", default=None,
              help="This is the mlflow run id")
@click.option("--registered_model_name", default="dl_finetuned_model", help="This is the registered model name.")
@click.option("--pipeline_run_name", default="chapter04", help="This is the mlflow run name.")
def task(mlflow_run_id, registered_model_name, pipeline_run_name):
    if mlflow_run_id is None or mlflow_run_id == 'None':
        logger.info('no model to register. exit now.')
        return
    with mlflow.start_run(run_name=pipeline_run_name) as mlrun:
        logged_model = f'runs:/{mlflow_run_id}/model'
        logger.info("logged model uri is: %s", logged_model)
        mlflow.register_model(logged_model, registered_model_name)
        mlflow.set_tag('pipeline_step', __file__)

    logger.info("finished registering model to %s", registered_model_name)


if __name__ == '__main__':
    task()
