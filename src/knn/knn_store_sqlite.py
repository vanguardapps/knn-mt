import os
import sqlite3
from knn.knn_store import KNNStore


class KNNStoreSQLite(KNNStore):
    """KNN-MT embeddings store for SQLite.

    Note: Uses filepath to SQLite database to initialize / restore the KNN store.

    """

    schema = 'public'

    def __init__(
        self,
        db_filepath,
        embedding_dim=None,
        table_prefix=None,
        configuration_table_stem=None,
        embedding_table_stem=None,
        faiss_cache_table_stem=None,
        embedding_batch_size=None,
        target_batch_size=None,
        embedding_dtype=None,
    ):
        """Initializes KNNStore instance.

        Args:
            db_filepath (str):
            embedding_dim (int):
            table_prefix (str):
            configuration_table_stem (str):
            embedding_table_stem (str):
            faiss_cache_table_stem (str):
            embedding_batch_size (int):
            target_batch_size (int):
            embedding_dtype (str):

        """
        super(KNNStoreSQLite, self).__init__(
            embedding_dim=embedding_dim,
            table_prefix=table_prefix,
            configuration_table_stem=configuration_table_stem,
            embedding_table_stem=embedding_table_stem,
            faiss_cache_table_stem=faiss_cache_table_stem,
            embedding_batch_size=embedding_batch_size,
            target_batch_size=target_batch_size,
            embedding_dtype=embedding_dtype,
        )

    def _initialize_database(self):
        """Initialize database for SQLite"""
        load_dotenv()

        PGHOST = os.environ["PGHOST"]
        PGUSER = os.environ["PGUSER"]
        PGPORT = os.environ["PGPORT"]
        PGDATABASE = os.environ["PGDATABASE"]
        PGPASSWORD = os.environ["PGPASSWORD"]

        self.connection_string = f"postgresql://{PGHOST}:{PGPORT}/{PGDATABASE}?user={PGUSER}&password={PGPASSWORD}"

        with psycopg.connect(self.connection_string, autocommit=True) as conn:
            cursor = conn.cursor()

            print(
                f"Creating table '{self.configuration_table_name}' if it does not exist."
            )
            create_configuration_table_query = sql.SQL(
                """
                create table if not exists {table_name} (
                    name text not null primary key,
                    value text
                ); 
                """
            ).format(table_name=sql.Identifier(self.configuration_table_name))
            cursor.execute(create_configuration_table_query)

            print(
                f"Loading any past configurations from table '{self.configuration_table_name}."
            )
            load_configurations_query = sql.SQL(
                "select name, value from {table_name};"
            ).format(table_name=sql.Identifier(self.configuration_table_name))
            cursor.execute(load_configurations_query)
            rows = cursor.fetchall()
            for name, value in rows:
                if value != 'None':
                    if name == 'embedding_dtype':
                        self.embedding_dtype = value
                    elif name == 'embedding_dim':
                        self.embedding_dim = int(value)

            if self.embedding_dim is None:
                raise ValueError("Missing required parameter `embedding_dim`.")

            print(f"Upserting configurations in '{self.configuration_table_name}'")
            add_configurations_query = sql.SQL(
                """
                merge into {table_name} as tgt 
                using (
                    values 
                        ('embedding_dtype', %s),
                        ('embedding_dim', %s)
                ) 
                as src (
                    name,
                    value
                ) 
                on src.name = tgt.name
                when not matched then 
                insert (
                    name,
                    value
                ) 
                values (
                    name,
                    value
                );
                """
            ).format(table_name=sql.Identifier(self.configuration_table_name))
            cursor.execute(
                add_configurations_query,
                (
                    self.embedding_dtype,
                    str(self.embedding_dim),
                ),
            )

            print(f"Creating table '{self.embedding_table_name}' if it does not exist.")
            create_embedding_table_query = sql.SQL(
                """
                create table if not exists {table_name} (
                    id serial primary key,
                    source_token_id int,
                    target_token_id int,
                    source_embedding bytea,
                    target_embedding bytea
                ); 
                """
            ).format(table_name=sql.Identifier(self.embedding_table_name))
            cursor.execute(create_embedding_table_query)

            print(
                f"Creating table '{self.faiss_cache_table_name}' if it does not exist."
            )
            create_faiss_cache_table_query = sql.SQL(
                """
                create table if not exists {table_name} (
                    source_token_id int primary key,
                    faiss_index bytea
                );
                """
            ).format(table_name=sql.Identifier(self.faiss_cache_table_name))
            cursor.execute(create_faiss_cache_table_query)
            cursor.execute(
                sql.SQL("select name, value from {table_name};").format(
                    table_name=sql.Identifier(self.configuration_table_name)
                )
            )

            print(
                f"Creating table '{self.embedding_faiss_table_name}' if it does not exist."
            )
            create_embedding_faiss_table_query = sql.SQL(
                """
                create table if not exists {embedding_faiss_table_name} (
                    embedding_id int not null references {embedding_table_name} (id),
                    source_token_id int not null references {faiss_cache_table_name} (source_token_id)
                );
                """
            ).format(
                embedding_faiss_table_name=sql.Identifier(
                    self.embedding_faiss_table_name
                ),
                embedding_table_name=sql.Identifier(self.embedding_table_name),
                faiss_cache_table_name=sql.Identifier(self.faiss_cache_table_name),
            )

            cursor.execute(create_embedding_faiss_table_query)

            cursor.execute(
                sql.SQL("select name, value from {table_name};").format(
                    table_name=sql.Identifier(self.configuration_table_name)
                )
            )

            configurations = cursor.fetchall()

            print(f"Current {self.__class__.__name__} instance configurations:")
            print(configurations)

    def _store_corpus_timestep(
        self,
        source_token_id,
        target_token_id,
        source_embedding_bytestring,
        target_embedding_bytestring,
    ):
        with psycopg.connect(self.connection_string, autocommit=True) as conn:
            cursor = conn.cursor()

            insert_embedding_query = sql.SQL(
                """
                insert into {table_name} (source_token_id, target_token_id, source_embedding, target_embedding)
                values (%s, %s, %s, %s); 
                """
            ).format(table_name=sql.Identifier(self.embedding_table_name))

            cursor.execute(
                insert_embedding_query,
                (
                    source_token_id.item(),
                    target_token_id.item(),
                    source_embedding_bytestring,
                    target_embedding_bytestring,
                ),
            )

    def _retrieve_all_source_token_ids(self):
        with psycopg.connect(self.connection_string, autocommit=True) as conn:
            cursor = conn.cursor()

            # Get unique source token IDs to iterate over
            cursor.execute(
                sql.SQL("select distinct source_token_id from {table_name};").format(
                    table_name=sql.Identifier(self.embedding_table_name)
                )
            )

            source_token_ids = cursor.fetchall()
            return source_token_ids

    def _retrieve_source_token_embeddings_batch(self, source_token_id):
        with psycopg.connect(self.connection_string, autocommit=True) as conn:
            cursor = conn.cursor()
            source_embedding_query = sql.SQL(
                """
                select id, source_embedding
                from {table_name}
                where source_token_id = %s and target_token_id is not null
                order by id
                offset %s
                limit %s;
                """
            ).format(
                table_name=sql.Identifier(self.embedding_table_name),
            )

            while self._embedding_table_offset == 0 or len(rows) > 0:
                cursor.execute(
                    source_embedding_query,
                    (
                        source_token_id,
                        self._embedding_table_offset,
                        self.embedding_batch_size,
                    ),
                )
                rows = cursor.fetchall()
                self._increment_source_token_embeddings_offset()
                yield rows

    def _store_source_faiss_bytestring(self, source_token_id, bytestring):
        with psycopg.connect(self.connection_string, autocommit=True) as conn:
            with conn.transaction():
                cursor = conn.cursor()
                cursor.execute(
                    sql.SQL(
                        "delete from {table_name} where source_token_id = %s;"
                    ).format(table_name=sql.Identifier(self.faiss_cache_table_name)),
                    (source_token_id,),
                )
                cursor.execute(
                    sql.SQL(
                        "insert into {table_name} (source_token_id, faiss_index) values (%s, %s);"
                    ).format(table_name=sql.Identifier(self.faiss_cache_table_name)),
                    (
                        source_token_id,
                        bytestring,
                    ),
                )

    def _retrieve_source_faiss_bytestring(self, source_token_id):
        with psycopg.connect(self.connection_string, autocommit=True) as conn:
            cursor = conn.cursor()

            cursor.execute(
                sql.SQL(
                    "select faiss_index from {table_name} where source_token_id = %s"
                ).format(table_name=sql.Identifier(self.faiss_cache_table_name)),
                (source_token_id,),
            )

            result = cursor.fetchall()

        if len(result) < 1:
            return None

        (bytestring) = result[0]

        return bytestring

    def _retrieve_target_embeddings(self, embedding_ids):
        with psycopg.connect(self.connection_string, autocommit=True) as conn:
            cursor = conn.cursor()

            target_embeddings_query = sql.SQL(
                "select id, target_embedding from {table_name} where id in %s"
            ).format(table_name=sql.Identifier(self.embedding_table_name))

            cursor.execute(target_embeddings_query, (embedding_ids,))
            rows = cursor.fetchall()
            return rows
