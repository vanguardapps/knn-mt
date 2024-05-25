import faiss
import numpy as np
import os
import psycopg
from dotenv import load_dotenv
from itertools import islice
from psycopg import sql
from tqdm import tqdm
from utils import validate_required_params

"""
Table 0: Configuration
    name:
        String. Name of the configuration. Primary key.
    value:
        String. Value of the configuration.

Table 1: Embeddings Store
    id:
        Source-side ID. System-generated. Represents a unique source token occurrence
        within the corpus. The aligned target token ID and target embedding can be
        referenced by this universal ID as well.

    source_token_id:
        The token type ID (kind of token) at this location in the source. Depends on
        the tokenizer used upstream. Can be used as a way to group source embeddings,
        target tokens, and target embeddings by the source token type.

    target_token_id:
        The token type ID (kind of token) of the target token that was aligned to
        the source token during the alignment process.

    source_embedding:
        The embedding of the source token at this position. This assignment depends on
        the upstream embeddings model and how the output is generated, but usually it is
        the embedding created by the encoder in an encoder/decoder transformer architecture.

    target_embedding:
        The embedding of the target token at this position. Usually the embedding from
        the last hidden state of the decoder at the timestep t of this particular
        position in the corpus, taking into account the whole source, and all target
        tokens up to the point t in the sequence.
    

Table 2: FAISS Source Embeddings Index Cache
    source_token_id:
        The token type ID of the source token for which the list of embeddings are being
        pulled for vector search.
    faiss_index:
        The byte data for a serialized FAISS index using `faiss.serialize_index(index)`.

Table 3: Embedding IDs Stored in FAISS Index Cache
    embedding_id:
        The embedding ID that was added to a FAISS index cache.
    source_token_id:
        The source token ID of the FAISS index cache to which this embedding was added.

    

NOTE: Source tokens that are not able to be aligned to target tokens are ignored and their
      related embeddings are not stored.

"""


class KNNStorePG(object):
    """KNN-MT embeddings store implementation for postgres instance

    Attributes:
        default_table_prefix (str): (class attribute)
        default_configuration_table_stem (str): (class attribute)
        default_embedding_table_stem (str): (class attribute)
        default_faiss_cache_table_stem (str): (class attribute)
        default_embedding_faiss_table_stem (str): (class attribute)
        default_embedding_dtype (str): (class attribute)
        default_embedding_batch_size (int): (class attribute)
        default_target_batch_size (int): (class attribute)
        schema (str): (class attribute)
        table_prefix (str):
        configuration_table_stem (str):
        embedding_table_stem (str):
        faiss_cache_table_stem (str):
        embedding_faiss_table_stem (str):
        target_build_table_stem (str):
        configuration_table_name (str):
        embedding_table_name (str):
        faiss_cache_table_name (str):
        embedding_faiss_table_name (str):
        PGHOST (str):
        PGUSER (str):
        PGPORT (str):
        PGDATABASE (str):
        PGPASSWORD (str):
        pg_connection (object):
        embedding_dim (int):
        embedding_dtype (str):
        embedding_batch_size (int):
        target_batch_size (int):

    Note: Uses standard postgres environment variables PGHOST, PGPORT, PGUSER,
    PGPASSWORD, PGDATABASE to initialize postgres connection.
    """

    default_table_prefix = 'knn_store'
    default_configuration_table_stem = 'config'
    default_embedding_table_stem = 'embedding'
    default_faiss_cache_table_stem = 'faiss_cache'
    default_embedding_faiss_table_stem = 'embedding_faiss'
    default_embedding_batch_size = 50
    default_target_batch_size = 50
    default_embedding_dtype = 'float32'
    schema = 'public'

    def __init__(
        self,
        embedding_dim=None,
        table_prefix=None,
        configuration_table_stem=None,
        embedding_table_stem=None,
        faiss_cache_table_stem=None,
        embedding_faiss_table_stem=None,
        embedding_batch_size=None,
        target_batch_size=None,
        embedding_dtype=None,
    ):
        self.embedding_dim = embedding_dim

        self.table_prefix = (
            table_prefix
            if table_prefix is not None
            else KNNStorePG.default_table_prefix
        )

        self.configuration_table_stem = (
            configuration_table_stem
            if configuration_table_stem is not None
            else KNNStorePG.default_configuration_table_stem
        )

        self.embedding_table_stem = (
            embedding_table_stem
            if embedding_table_stem is not None
            else KNNStorePG.default_embedding_table_stem
        )

        self.faiss_cache_table_stem = (
            faiss_cache_table_stem
            if faiss_cache_table_stem is not None
            else KNNStorePG.default_faiss_cache_table_stem
        )

        self.embedding_faiss_table_stem = (
            embedding_faiss_table_stem
            if embedding_faiss_table_stem is not None
            else KNNStorePG.default_embedding_faiss_table_stem
        )

        self.embedding_batch_size = (
            embedding_batch_size
            if embedding_batch_size is not None
            else KNNStorePG.default_embedding_batch_size
        )

        self.target_batch_size = (
            target_batch_size
            if target_batch_size is not None
            else KNNStorePG.default_target_batch_size
        )

        self.embedding_dtype = (
            embedding_dtype
            if embedding_dtype is not None
            else KNNStorePG.default_embedding_dtype
        )

        self.configuration_table_name = (
            self.table_prefix + "_" + self.configuration_table_stem
        )
        self.embedding_table_name = self.table_prefix + "_" + self.embedding_table_stem
        self.faiss_cache_table_name = (
            self.table_prefix + "_" + self.faiss_cache_table_stem
        )
        self.embedding_faiss_table_name = (
            self.table_prefix + "_" + self.embedding_faiss_table_stem
        )

        load_dotenv()

        PGHOST = os.environ["PGHOST"]
        PGUSER = os.environ["PGUSER"]
        PGPORT = os.environ["PGPORT"]
        PGDATABASE = os.environ["PGDATABASE"]
        PGPASSWORD = os.environ["PGPASSWORD"]

        self.connection_string = f"postgresql://{PGHOST}:{PGPORT}/{PGDATABASE}?user={PGUSER}&password={PGPASSWORD}"
        # self.pg_connection = psycopg.connect(
        #     database=self.PGDATABASE,
        #     user=self.PGUSER,
        #     password=self.PGPASSWORD,
        #     host=self.PGHOST,
        #     port=int(self.PGPORT),
        # )
        # self.pg_connection.autocommit = True

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

            validate_required_params(dict(embedding_dim=self.embedding_dim))

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

    def insert_embeddings(
        self, source_token_id, target_token_id, source_embedding, target_embedding
    ):
        with psycopg.connect(self.connection_string, autocommit=True) as conn:
            cursor = conn.cursor()

            insert_embedding_query = sql.SQL(
                """
                insert into {table_name} (source_token_id, target_token_id, source_embedding, target_embedding)
                values (%s, %s, %s, %s); 
                """
            ).format(table_name=sql.Identifier(self.embedding_table_name))

            source_byte_string = source_embedding.numpy().tobytes()
            target_byte_string = target_embedding.numpy().tobytes()

            cursor.execute(
                insert_embedding_query,
                (
                    source_token_id.item(),
                    target_token_id.item(),
                    source_byte_string,
                    target_byte_string,
                ),
            )

    def ingest(self, batch):
        for (
            source_token_ids,
            target_ids,
            alignments,
            source_embeddings,
            target_embeddings,
        ) in zip(
            batch.input_ids_masked,
            batch.label_ids_masked,
            batch.alignments,
            batch.encoder_last_hidden_state_masked,
            batch.target_hidden_states_masked,
        ):
            for source_index, source_token_id in enumerate(source_token_ids):
                target_index = alignments.get(source_index, None)

                # Ignore any source token that was not aligned to a target token
                if target_index:
                    self.insert_embeddings(
                        source_token_id=source_token_id,
                        target_token_id=target_ids[target_index],
                        source_embedding=source_embeddings[source_index],
                        target_embedding=target_embeddings[target_index],
                    )

    def _get_new_faiss_index(self):
        return faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim))

    def _get_embedding_dtype(self):
        # TODO: add support for dtypes other than np.float32
        if self.embedding_dtype == 'float32':
            embedding_dtype = np.float32
        else:
            raise ValueError(f"Unsupported dtype used '{self.embedding_dtype}'")

        return embedding_dtype

    def _update_faiss_index_cache(self, source_token_id, batch_embeddings, batch_ids):

        faiss_index = self.get_source_faiss_index(source_token_id)

        if not faiss_index:
            faiss_index = self._get_new_faiss_index()

        batch_embeddings_np = np.array(batch_embeddings)
        batch_ids_np = np.array(batch_ids, dtype=np.int64)

        faiss.normalize_L2(batch_embeddings_np)
        faiss_index.add_with_ids(batch_embeddings_np, batch_ids_np)

        serialized_index = faiss.serialize_index(faiss_index)
        bytestring = serialized_index.tobytes()

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
                for embedding_id in batch_ids:
                    cursor.execute(
                        sql.SQL(
                            "insert into {table_name} (embedding_id, source_token_id) values (%s, %s);"
                        ).format(
                            table_name=sql.Identifier(self.embedding_faiss_table_name)
                        ),
                        (
                            embedding_id,
                            source_token_id,
                        ),
                    )

        faiss_index.reset()

    def build_source_index(self):
        with psycopg.connect(self.connection_string, autocommit=True) as conn:
            cursor = conn.cursor()

            # Get unique source token IDs to iterate over
            cursor.execute(
                sql.SQL("select distinct source_token_id from {table_name};").format(
                    table_name=sql.Identifier(self.embedding_table_name)
                )
            )

            source_token_ids = cursor.fetchall()

            # One source token ID at a time
            for (source_token_id,) in (batches := tqdm(source_token_ids)):
                batches.set_description(
                    "Building index for source token ID {source_token_id}"
                )

                source_embedding_query = sql.SQL(
                    """
                    select e.id, e.source_embedding
                    from {embedding_table_name} e
                    left join {embedding_faiss_table_name} ef on ef.embedding_id = e.id and ef.source_token_id = e.source_token_id
                    where e.source_token_id = %s and
                        e.target_token_id is not null and
                        ef.embedding_id is null
                    limit %s;
                    """
                ).format(
                    embedding_table_name=sql.Identifier(self.embedding_table_name),
                    embedding_faiss_table_name=sql.Identifier(
                        self.embedding_faiss_table_name
                    ),
                )
                cursor.execute(
                    source_embedding_query,
                    (
                        source_token_id,
                        self.embedding_batch_size,
                    ),
                )
                rows = cursor.fetchall()

                while len(rows) > 0:
                    batch_embeddings = []
                    batch_ids = []

                    for id, source_embedding in rows:
                        batch_embeddings.append(
                            np.frombuffer(
                                source_embedding, dtype=self._get_embedding_dtype()
                            )
                        )
                        batch_ids.append(id)

                    self._update_faiss_index_cache(
                        source_token_id, batch_embeddings, batch_ids
                    )

                    cursor.execute(
                        source_embedding_query,
                        (
                            source_token_id,
                            self.embedding_batch_size,
                        ),
                    )
                    rows = cursor.fetchall()

    def get_source_faiss_index(self, source_token_id):
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

        faiss_index = faiss.deserialize_index(np.frombuffer(bytestring, dtype=np.uint8))

        return faiss_index

    def knn_source_faiss_index(self, source_token_id, source_embedding, k):
        faiss_index = self.get_source_faiss_index(source_token_id)

        # TODO: Write the faiss stuff to perform the k nearest neighbor search here
        # and return the list of ids

    def knn_get_logits(self):
        # TODO: Figure out how this all comes together. Need to review math. It's something like
        # calling knn_source_faiss_index() above and then calling build_target_faiss_index(), then
        # searching that index with the target and getting the top k matching target tokens, then
        # going to the math to interpolate with existing model. that will be another class that
        # composes this probably, KNNOperator or something.
        return True

    @staticmethod
    def _batched(iterable, n):
        if n < 1:
            raise ValueError('n must be at least one')
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch

    def build_target_faiss_index(self, ids):
        faiss_index = self._get_new_faiss_index()

        with psycopg.connect(self.connection_string, autocommit=True) as conn:
            cursor = conn.cursor()

            for batch_ids in (
                batches := tqdm(KNNStorePG._batched(ids, self.target_batch_size))
            ):
                batches.set_description(
                    f"Building target index in batches of {self.target_batch_size}"
                )

                target_embeddings_query = sql.SQL(
                    "select id, target_embedding from {table_name} where id in %s"
                ).format(table_name=sql.Identifier(self.embedding_table_name))

                cursor.execute(target_embeddings_query, (batch_ids,))
                rows = cursor.fetchall()

                batch_embeddings = []
                batch_ids = []

                for id, target_embedding in rows:
                    batch_embeddings.append(
                        np.frombuffer(
                            target_embedding, dtype=self._get_embedding_dtype()
                        )
                    )
                    batch_ids.append(id)

                batch_embeddings_np = np.array(batch_embeddings)
                batch_ids_np = np.array(batch_ids, dtype=np.int64)

                faiss.normalize_L2(batch_embeddings_np)
                faiss_index.add_with_ids(batch_embeddings_np, batch_ids_np)

        return faiss_index
