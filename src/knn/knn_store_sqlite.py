import regex as re
import sqlite3
from knn.knn_store import KNNStore
from psycopg import sql

# Allowed kwargs passed to `sqlite.connect()`. `autocommit` is omitted intentionally.
SQLITE_CONNECT_KWARGS = [
    "cached_statements",
    "check_same_thread",
    "database",
    "detect_types",
    "factory",
    "isolation_level",
    "timeout",
    "uri",
]


def extract_key_value_pairs_from_dict(original_dict, subset_keys):
    subset = {}
    keys = [key for key in original_dict.keys()]
    for key in keys:
        if key in subset_keys and original_dict.get(key, None) is not None:
            subset[key] = original_dict.pop(key)
    return subset


class KNNStoreSQLite(KNNStore):
    """KNN-MT embeddings store for SQLite.

    Attributes:
        sqlite_connect_kwargs (dict):

    """

    schema = 'public'

    def __init__(
        self,
        embedding_dim=None,
        table_prefix=None,
        configuration_table_stem=None,
        embedding_table_stem=None,
        faiss_cache_table_stem=None,
        embedding_batch_size=None,
        target_batch_size=None,
        embedding_dtype=None,
        c=None,
        **kwargs,
    ):
        """Initializes KNNStore instance.

        Passes relevant `**kwargs` to `sqlite3.connect()` for initialization or restoriation
        of the DB. To initialize in the simplest form, pass `database="your-db-path.db"` and
        a SQLite DB will be either opened or created at "your-db-path.db". See the docs for
        Python's implementation of SQLite here: https://docs.python.org/3/library/sqlite3.html

        Note: The user of `:memory:` as the `database` parameter for `sqlite3.connect()` is
        allowed but not recommended for this use case, as the size of the DB can grow large
        quite quickly when storing high-dimensionality embeddings.

        Args:
            embedding_dim (int):
            table_prefix (str):
            configuration_table_stem (str):
            embedding_table_stem (str):
            faiss_cache_table_stem (str):
            embedding_batch_size (int):
            target_batch_size (int):
            embedding_dtype (str):
            c (int):
            **kwargs (dict):
        """

        self.sqlite_connect_kwargs = extract_key_value_pairs_from_dict(
            kwargs, SQLITE_CONNECT_KWARGS
        )

        if len(self.sqlite_connect_kwargs) < 1:
            raise ValueError(
                "Please specify keyword arguments to intialize database during construction of `KNNStoreSQLite` instance."
            )

        super(KNNStoreSQLite, self).__init__(
            embedding_dim=embedding_dim,
            table_prefix=table_prefix,
            configuration_table_stem=configuration_table_stem,
            embedding_table_stem=embedding_table_stem,
            faiss_cache_table_stem=faiss_cache_table_stem,
            embedding_batch_size=embedding_batch_size,
            target_batch_size=target_batch_size,
            embedding_dtype=embedding_dtype,
            c=c,
        )

    @staticmethod
    def _validate_table_name(table_name):
        safe_table_name_pattern = r"^[\p{L}_][\p{L}\p{N}@$#_]{0,127}$"
        if not re.match(safe_table_name_pattern, table_name):
            raise ValueError(f"Invalid table name supplied: '{table_name}'.")
        return table_name

    def _get_sqlite_connection(self):
        return sqlite3.connect(**self.sqlite_connect_kwargs)

    def _initialize_database(self):
        """Initialize database for SQLite"""

        print(
            f"Creating SQLite database using configuration: {self.sqlite_connect_kwargs}."
        )

        con = self._get_sqlite_connection()
        cur = con.cursor()

        # Prevent SQL injections to table names
        valid_configuration_table_name = KNNStoreSQLite._validate_table_name(
            self.configuration_table_name
        )
        valid_embedding_table_name = KNNStoreSQLite._validate_table_name(
            self.embedding_table_name
        )
        valid_faiss_cache_table_name = KNNStoreSQLite._validate_table_name(
            self.faiss_cache_table_name
        )

        print(
            f"Creating table '{valid_configuration_table_name}' if it does not exist."
        )
        create_configuration_table_query = (
            f"create table if not exists {valid_configuration_table_name} ( "
            "   name text not null primary key, "
            "   value text "
            ");"
        )
        cur.execute(create_configuration_table_query)
        con.commit()

        print(
            f"Loading any past configurations from table '{valid_configuration_table_name}."
        )
        load_configurations_query = (
            f"select name, value from {valid_configuration_table_name};"
        )
        cur.execute(load_configurations_query)
        rows = cur.fetchall()

        for name, value in rows:
            if value != 'None':
                if name == 'embedding_dtype':
                    self.embedding_dtype = value
                elif name == 'embedding_dim':
                    self.embedding_dim = int(value)

        if self.embedding_dim is None:
            raise ValueError("Missing required parameter `embedding_dim`.")

        print(f"Upserting configurations in '{valid_configuration_table_name}'")
        upsert_embedding_dtype_query = (
            f"insert into {valid_configuration_table_name} (name, value) "
            "values ('embedding_dtype', ?) "
            "on conflict(name) do update set value = ?;"
        )
        upsert_embedding_dim_query = (
            f"insert into {valid_configuration_table_name} (name, value) "
            "values ('embedding_dim', ?) "
            "on conflict(name) do update set value = ?;"
        )

        cur.execute(
            upsert_embedding_dtype_query,
            (
                self.embedding_dtype,
                self.embedding_dtype,
            ),
        )
        cur.execute(
            upsert_embedding_dim_query,
            (
                str(self.embedding_dim),
                str(self.embedding_dim),
            ),
        )
        con.commit()

        print(f"Creating table '{valid_embedding_table_name}' if it does not exist.")
        create_embedding_table_query = (
            f"create table if not exists {valid_embedding_table_name} ( "
            "    id integer primary key autoincrement, "
            "    source_token_id integer, "
            "    target_token_id integer, "
            "    source_embedding blob, "
            "    target_embedding blob "
            ");"
        )
        cur.execute(create_embedding_table_query)
        con.commit()

        print(f"Creating table '{valid_faiss_cache_table_name}' if it does not exist.")
        create_faiss_cache_table_query = (
            f"create table if not exists {valid_faiss_cache_table_name} ( "
            "    source_token_id integer not null unique, "
            "    faiss_index blob "
            ");"
        )
        cur.execute(create_faiss_cache_table_query)
        con.commit()

        cur.execute(f"select name, value from {valid_configuration_table_name};")
        configurations = cur.fetchall()

        cur.close()
        con.close()

        print(f"Current {self.__class__.__name__} instance configurations:")
        print(configurations)

    def _store_corpus_timestep(
        self,
        source_token_id,
        target_token_id,
        source_embedding_bytestring,
        target_embedding_bytestring,
    ):
        valid_embedding_table_name = self._validate_table_name(
            self.embedding_table_name
        )

        con = self._get_sqlite_connection()
        cur = con.cursor()

        insert_embedding_query = (
            f"insert into {valid_embedding_table_name} (source_token_id, target_token_id, source_embedding, target_embedding) "
            "values (?, ?, ?, ?);"
        )

        cur.execute(
            insert_embedding_query,
            (
                source_token_id,
                target_token_id,
                source_embedding_bytestring,
                target_embedding_bytestring,
            ),
        )
        con.commit()

        cur.close()
        con.close()

    def _retrieve_all_source_token_ids(self):
        valid_embedding_table_name = self._validate_table_name(
            self.embedding_table_name
        )

        con = self._get_sqlite_connection()
        cur = con.cursor()

        # Get unique source token IDs to iterate over
        cur.execute(
            f"select distinct source_token_id from {valid_embedding_table_name};"
        )

        source_token_ids = cur.fetchall()

        cur.close()
        con.close()

        return source_token_ids

    def _retrieve_source_token_embeddings_batches(self, source_token_id):
        valid_embedding_table_name = self._validate_table_name(
            self.embedding_table_name
        )

        con = self._get_sqlite_connection()
        cur = con.cursor()

        self._reset_source_token_embeddings_offset()

        while self._embedding_table_offset == 0 or len(rows) > 0:
            valid_embedding_table_offset, valid_embedding_batch_size = (
                self._get_valid_embedding_offset_and_batch_size()
            )

            source_embedding_query = (
                "select id, source_embedding "
                f"from {valid_embedding_table_name} "
                "where source_token_id = ? and target_token_id is not null "
                "order by id "
                f"limit {valid_embedding_batch_size} "
                f"offset {valid_embedding_table_offset};"
            )

            cur.execute(
                source_embedding_query,
                (source_token_id,),
            )
            rows = cur.fetchall()
            self._increment_source_token_embeddings_offset()
            yield rows

        cur.close()
        con.close()

    def _store_source_faiss_bytestring(self, source_token_id, bytestring):
        valid_faiss_cache_table_name = self._validate_table_name(
            self.faiss_cache_table_name
        )

        con = self._get_sqlite_connection()
        cur = con.cursor()

        cur.execute(
            f"delete from {valid_faiss_cache_table_name} where source_token_id = ?;",
            (source_token_id,),
        )
        cur.execute(
            f"insert into {valid_faiss_cache_table_name} (source_token_id, faiss_index) values (?, ?);",
            (
                source_token_id,
                bytestring,
            ),
        )
        con.commit()

        cur.close()
        con.close()

    def _retrieve_source_faiss_bytestring(self, source_token_id):
        valid_faiss_cache_table_name = self._validate_table_name(
            self.faiss_cache_table_name
        )

        con = self._get_sqlite_connection()
        cur = con.cursor()

        cur.execute(
            f"select faiss_index from {valid_faiss_cache_table_name} where source_token_id = ?",
            (int(source_token_id),),
        )

        result = cur.fetchall()

        cur.close()
        con.close()

        if len(result) < 1 or len(result[0]) < 1:
            return None

        bytestring = result[0][0]

        return bytestring

    def _retrieve_target_bytestrings(self, embedding_ids):
        valid_embedding_table_name = self._validate_table_name(
            self.embedding_table_name
        )

        con = self._get_sqlite_connection()
        cur = con.cursor()

        placeholders = len(embedding_ids) * '?'

        cur.execute(
            f"select id, target_embedding from {valid_embedding_table_name} where id in ({','.join(placeholders)})",
            embedding_ids,
        )
        rows = cur.fetchall()

        cur.close()
        con.close()

        return rows

    def _retrieve_target_token_ids(self, embedding_ids):
        # TODO: ROY: Implement
        return True
