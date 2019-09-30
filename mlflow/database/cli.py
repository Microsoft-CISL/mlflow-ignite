"""
CLI for database deployment module.
"""
from __future__ import print_function

import click

import mlflow.database
from mlflow.utils import cli_args, experimental


@click.group("database")
def commands():
    """
    Serve models on a SQL DB. **These commands require a onnx flavor of the model.**
    """
    pass


@commands.command("deploy")
@cli_args.MODEL_URI
@click.option("--db-uri", "-d", default=None, metavar="URI", required=True,
              help="URI to the SQL database in the form: <dialect>+<driver>://<username>:<password>@<host>:<port>/<database>")
@click.option("--flavor", "-f", default='onnx', help="The name of the flavor to use for deployment. Must be onnx.")
@click.option("--table-name", "-t", default=None,
              help="The name of the SQL table to deploy the model to. If not specified, will use 'models' table.")
@click.option("--column-name", "-c", default=None,
              help="The name of the SQL column to deploy the model to. It must support binary content. f not specified, will use 'model' column.")
@experimental
def build_image(model_uri, flavor, db_uri, table_name, column_name):
    """
    Register an MLflow model with a SQL database.
    """
    mlflow.database.deploy(
        model_uri=model_uri, db_uri=db_uri, flavor=flavor, table_name=table_name, column_name=column_name)
