import logging
import os
import posixpath
from abc import ABCMeta
import sqlalchemy
from contextlib import contextmanager

from mlflow.utils import extract_db_type_from_uri
from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import relative_path_to_artifact_path

from mlflow.exceptions import MlflowException
from mlflow.store.dbmodels.initial_artifact_store_models import Base as InitialBase
from mlflow.store.dbmodels.initial_artifact_store_models import Base, SqlArtifact
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from sqlalchemy import or_

from six.moves import urllib

_logger = logging.getLogger(__name__)


def _relative_path(base_dir, subdir_path, path_module):
    relative_path = path_module.relpath(subdir_path, base_dir)
    return relative_path if relative_path is not '.' else None


def _relative_path_local(base_dir, subdir_path):
    rel_path = _relative_path(base_dir, subdir_path, os.path)
    return relative_path_to_artifact_path(rel_path) if rel_path is not None else None


def extract_db_uri_and_path(artifact_uri):
    parsed_uri = urllib.parse.urlparse(artifact_uri)

    if parsed_uri.query == "":
        parsed_path = parsed_uri.path.split("/", 2)
        if (len(parsed_path)) == 3:
            path = parsed_uri.path.split("/", 2)[2]
            parsed_uri = parsed_uri._replace(path="/" + parsed_uri.path.split("/", 1)[1])
        else:
            path = ""
    else:
        parsed_query = parsed_uri.query.split("/", 1)
        if len(parsed_query) == 2:
            path = parsed_uri.query.split("/", 1)[1]
            parsed_uri = parsed_uri._replace(query=parsed_uri.query.split("/", 1)[0])
        else:
            path = ""

    return urllib.parse.urlunparse(parsed_uri), path


class DBArtifactRepository(ArtifactRepository):
    """
    Abstract artifact repo that defines how to upload (log) and download potentially large
    artifacts from a database backend.
    """

    __metaclass__ = ABCMeta

    def __init__(self, artifact_uri):
        self.db_uri, self.root = extract_db_uri_and_path(artifact_uri)
        self.db_type = extract_db_type_from_uri(self.db_uri)
        self.engine = sqlalchemy.create_engine(self.db_uri)
        super(DBArtifactRepository, self).__init__(self.db_uri)

        insp = sqlalchemy.inspect(self.engine)
        self.expected_tables = set([
            SqlArtifact.__tablename__,
        ])
        if len(self.expected_tables & set(insp.get_table_names())) == 0:
            DBArtifactRepository._initialize_tables(self.engine)
        Base.metadata.bind = self.engine
        SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.ManagedSessionMaker = self._get_managed_session_maker(SessionMaker)

    @staticmethod
    def _initialize_tables(engine):
        _logger.info("Creating initial MLflow database tables...")
        InitialBase.metadata.create_all(engine)

    @staticmethod
    def _get_managed_session_maker(SessionMaker):
        """
        Creates a factory for producing exception-safe SQLAlchemy sessions that are made available
        using a context manager. Any session produced by this factory is automatically committed
        if no exceptions are encountered within its associated context. If an exception is
        encountered, the session is rolled back. Finally, any session produced by this factory is
        automatically closed when the session's associated context is exited.
        """

        @contextmanager
        def make_managed_session():
            """Provide a transactional scope around a series of operations."""
            session = SessionMaker()
            try:
                yield session
                session.commit()
            except MlflowException:
                session.rollback()
                raise
            except Exception as e:
                session.rollback()
                raise MlflowException(message=e, error_code=INTERNAL_ERROR)
            finally:
                session.close()

        return make_managed_session

    def log_artifact(self, local_file, artifact_path=None):
        """
        Log a local file as an artifact, optionally taking an ``artifact_path`` to place it in
        within the run's artifacts. Run artifacts can be organized into directories, so you can
        place the artifact in a directory this way.

        :param local_file: Path to artifact to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifact.
        """

        _, file_name = os.path.split(local_file)
        with self.ManagedSessionMaker() as session:
            if artifact_path is None:
                artifact = SqlArtifact(
                    artifact_name=file_name, group_path=self.root,
                    artifact_content=open(
                        local_file, "rb").read(), artifact_initial_size=
                    os.path.getsize(local_file)
                )
            else:
                artifact = SqlArtifact(
                    artifact_name=file_name, group_path=os.path.join(self.root, artifact_path),
                    artifact_content=open(
                        local_file, "rb").read(), artifact_initial_size=
                    os.path.getsize(local_file)
                )
            session.add(artifact)
            session.flush()
            return str(artifact.artifact_id)

    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Log the files in the specified local directory as artifacts, optionally taking
        an ``artifact_path`` to place them in within the run's artifacts.

        :param local_dir: Directory of local artifacts to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifacts
        """
        with self.ManagedSessionMaker() as session:
            for subdir_path, _, files in os.walk(local_dir):

                relative_path = _relative_path_local(local_dir, subdir_path)

                if artifact_path is None:
                    db_subdir_path = relative_path if relative_path else ""
                else:
                    db_subdir_path = posixpath.join(artifact_path, relative_path) \
                        if relative_path else artifact_path

                for each_file in files:
                    source = os.path.join(subdir_path, each_file)
                    artifact = SqlArtifact(
                        artifact_name=each_file, group_path=os.path.join(self.root, db_subdir_path),
                        artifact_content=open(source, "rb").read(),
                        artifact_initial_size=os.path.getsize(source)
                    )
                    session.add(artifact)
        session.flush()

    def list_artifacts(self, path):
        """
        Return all the artifacts for this run_id directly under path. If path is a file, returns
        an empty list. Will error if path is neither a file nor directory.

        :param path: Relative source path that contains desired artifacts

        :return: List of artifacts as FileInfo listed directly under path.
        """
        with self.ManagedSessionMaker() as session:
            regex = path + '/%'
            return [artifact.to_file_info()
                    for artifact in
                    session.query(SqlArtifact).filter(
                        or_(SqlArtifact.group_path.like(os.path.join(self.root, regex)),
                            SqlArtifact.group_path == path))]

    def _download_file(self, remote_file_path, local_path):
        """
        Download the file at the specified relative remote path and saves
        it at the specified local path.

        :param remote_file_path: Source path to the remote file, relative to the root
                                 directory of the artifact repository.
        :param local_path: The path to which to save the downloaded file.
        """
        group, file_name = os.path.split(remote_file_path)

        with self.ManagedSessionMaker() as session:
            contents = [r.artifact_content for r in
                        session.query(SqlArtifact.artifact_content).filter(
                            SqlArtifact.group_path == os.path.join(self.root, group),
                            SqlArtifact.artifact_name == file_name)]
            if len(contents) == 1:
                with open(local_path, 'wb') as f:
                    f.write(contents[0])

    def clean(self):
        InitialBase.metadata.drop_all(self.engine)
