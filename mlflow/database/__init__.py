"""
The ``mlflow.database`` module provides an API for deploying MLflow models to a SQL DB.
"""
from __future__ import print_function

import logging
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from distutils.version import StrictVersion

import sqlalchemy
from sqlalchemy import MetaData, create_engine

from mlflow import onnx, pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import (INTERNAL_ERROR,
                                          INVALID_PARAMETER_VALUE,
                                          RESOURCE_DOES_NOT_EXIST)
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import experimental, extract_db_type_from_uri

_logger = logging.getLogger(__name__)

SUPPORTED_DEPLOYMENT_FLAVORS = [
    onnx.FLAVOR_NAME
]


@experimental
def deploy(model_uri, db_uri, flavor, table_name=None, column_name=None):
    """
    Register an MLflow model with a SQL database.

    :param model_uri: The location, in URI format, of the MLflow model used to build the Azure
                      ML deployment image. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.
    :param db_uri: The URI of the SQL DB to deploy in the form:
                    <dialect>+<driver>://<username>:<password>@<host>:<port>/<database>
    :param flavor: The name of the flavor of the model to use for deployment. Must be ``onnx``,. If the
                   specified flavor is not present or not supported for deployment, an exception
                   will be thrown.
    :param table_name: The name of the SQL table to deploy the model to. If not specified, will use 'models' table.
    :param column_name: The name of the SQL column to deploy the model to. It must support binary content. f not specified, will use 'model' column.
    """

    model_path = _download_artifact_from_uri(model_uri)
    model_config_path = os.path.join(model_path, "MLmodel")
    if not os.path.exists(model_config_path):
        raise MlflowException(
            message=(
                "Failed to find MLmodel configuration within the specified model's"
                " root directory."),
            error_code=INVALID_PARAMETER_VALUE)
    model_config = Model.load(model_config_path)
    _validate_deployment_flavor(model_config, flavor)
    onnx_model = onnx.load_model(model_uri)

    if table_name is None:
        table_name = "models"
    if column_name is None:
        column_name = "model"
    insert_db(db_uri, table_name, column_name, onnx_model)


def _validate_deployment_flavor(model_config, flavor):
    """
    Checks that the specified flavor is a supported deployment flavor
    and is contained in the specified model. If one of these conditions
    is not met, an exception is thrown.

    :param model_config: An MLflow Model object
    :param flavor: The deployment flavor to validate
    """
    if flavor not in SUPPORTED_DEPLOYMENT_FLAVORS:
        raise MlflowException(
            message=(
                "The specified flavor: `{flavor_name}` is not supported for deployment."
                " Please use one of the supported flavors: {supported_flavor_names}".format(
                    flavor_name=flavor,
                    supported_flavor_names=SUPPORTED_DEPLOYMENT_FLAVORS)),
            error_code=INVALID_PARAMETER_VALUE)
    elif flavor not in model_config.flavors:
        raise MlflowException(
            message=("The specified model does not contain the specified deployment flavor:"
                     " `{flavor_name}`. Please use one of the following deployment flavors"
                     " that the model contains: {model_flavors}".format(
                         flavor_name=flavor, model_flavors=model_config.flavors.keys())),
            error_code=RESOURCE_DOES_NOT_EXIST)


def insert_db(db_uri, table_name, column_name, onnx_model):
    engine = sqlalchemy.create_engine(db_uri)
    # Create connection
    conn = engine.connect()
    meta = MetaData(engine, reflect=True)
    table = meta.tables[table_name]
    # insert data via insert() construct
    ins = table.insert().value(
        column_name=onnx_model)
    conn.execute(ins)
    # Close connection
    conn.close()
