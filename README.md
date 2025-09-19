# My first ML_PROJECTS

import dagshub
dagshub.init(repo_owner='Ahmed2797', repo_name='ml_projects', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)