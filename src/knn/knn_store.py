import faiss
import numpy as np
from itertools import islice
from tqdm import tqdm


class KNNStore(object):
    """KNN-MT embeddings store abstract class.

    Note: No database implementation takes place in the abstract class.

    Attributes:
        default_table_prefix (str): (class attribute)
        default_configuration_table_stem (str): (class attribute)
        default_embedding_table_stem (str): (class attribute)
        default_faiss_cache_table_stem (str): (class attribute)
        default_embedding_dtype (str): (class attribute)
        default_embedding_batch_size (int): (class attribute)
        default_target_batch_size (int): (class attribute)
        table_prefix (str):
        configuration_table_stem (str):
        embedding_table_stem (str):
        faiss_cache_table_stem (str):
        target_build_table_stem (str):
        configuration_table_name (str):
        embedding_table_name (str):
        faiss_cache_table_name (str):
        embedding_dim (int):
        embedding_dtype (str):
        embedding_batch_size (int):
        target_batch_size (int):
    """

    default_table_prefix = 'knn_store'
    default_configuration_table_stem = 'config'
    default_embedding_table_stem = 'embedding'
    default_faiss_cache_table_stem = 'faiss_index'
    default_embedding_batch_size = 50
    default_target_batch_size = 50
    default_embedding_dtype = 'float32'

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
        **kwargs,
    ):
        """Initializes KNNStore instance.

        Note: Subclasses must call `super().__init_()` with all constructor arguments and any `kwargs`
        needed for subclass implementation of `self._initialize_database(**kwargs)`.

        Args:
            embedding_dim (int):
            table_prefix (str):
            configuration_table_stem (str):
            embedding_table_stem (str):
            faiss_cache_table_stem (str):
            embedding_batch_size (int):
            target_batch_size (int):
            embedding_dtype (str):
            **kwargs (dict):

        """
        self.embedding_dim = embedding_dim

        self.table_prefix = (
            table_prefix if table_prefix is not None else KNNStore.default_table_prefix
        )

        self.configuration_table_stem = (
            configuration_table_stem
            if configuration_table_stem is not None
            else KNNStore.default_configuration_table_stem
        )

        self.embedding_table_stem = (
            embedding_table_stem
            if embedding_table_stem is not None
            else KNNStore.default_embedding_table_stem
        )

        self.faiss_cache_table_stem = (
            faiss_cache_table_stem
            if faiss_cache_table_stem is not None
            else KNNStore.default_faiss_cache_table_stem
        )

        self.embedding_batch_size = (
            embedding_batch_size
            if embedding_batch_size is not None
            else KNNStore.default_embedding_batch_size
        )

        self.target_batch_size = (
            target_batch_size
            if target_batch_size is not None
            else KNNStore.default_target_batch_size
        )

        self.embedding_dtype = (
            embedding_dtype
            if embedding_dtype is not None
            else KNNStore.default_embedding_dtype
        )

        self.configuration_table_name = (
            self.table_prefix + "_" + self.configuration_table_stem
        )
        self.embedding_table_name = self.table_prefix + "_" + self.embedding_table_stem
        self.faiss_cache_table_name = (
            self.table_prefix + "_" + self.faiss_cache_table_stem
        )

        self._reset_source_token_embeddings_offset()

        self._initialize_database(**kwargs)

    @staticmethod
    def _batched(iterable, n):
        if n < 1:
            raise ValueError('n must be at least one')
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch

    # TODO: Provide KNN batch that does aligning using fast_align. This is going to be kind of a
    # rough spot in the implementation. Makes me want to get back into C++ and learn fast_align
    # from scratch, make a python port of it or something. That would be a real selling point
    # for this module though, as very few people can say they have a sentence aligner in
    # code (I'm actually not sure I should look and see if someone has done this).

    #
    # Methods provided as part of base class
    #

    def ingest(self, knn_batch):
        for (
            source_token_ids,
            target_ids,
            alignments,
            source_embeddings,
            target_embeddings,
        ) in zip(
            knn_batch.input_ids_masked,
            knn_batch.label_ids_masked,
            knn_batch.alignments,
            knn_batch.encoder_last_hidden_state_masked,
            knn_batch.target_hidden_states_masked,
        ):
            for source_index, source_token_id in enumerate(source_token_ids):
                target_index = alignments.get(source_index, None)

                # Ignore any source token that was not aligned to a target token
                if target_index:
                    source_token_id = source_token_id
                    target_token_id = target_ids[target_index]
                    source_embedding = source_embeddings[source_index]
                    target_embedding = target_embeddings[target_index]
                    source_embedding_bytestring = source_embedding.numpy().tobytes()
                    target_embedding_bytestring = target_embedding.numpy().tobytes()
                    self._store_corpus_timestep(
                        source_token_id=source_token_id,
                        target_token_id=target_token_id,
                        source_embedding_bytestring=source_embedding_bytestring,
                        target_embedding_bytestring=target_embedding_bytestring,
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

    def _increment_source_token_embeddings_offset(self):
        if self.embedding_batch_size < 1:
            raise ValueError("Please ensure `embedding_batch_size` is greater than 0.")
        self._embedding_table_offset += self.embedding_batch_size

    def _reset_source_token_embeddings_offset(self):
        self._embedding_table_offset = 0

    def build_source_index(self):
        faiss_index = self._get_new_faiss_index()
        source_token_ids = self._retrieve_all_source_token_ids()

        # One source token ID at a time
        for (source_token_id,) in (batches := tqdm(source_token_ids)):
            batches.set_description(
                "Building index for source token ID {source_token_id}"
            )

            self._reset_source_token_embeddings_offset()
            embedding_batches = self._retrieve_source_token_embeddings_batch(
                source_token_id
            )

            while len(rows := next(embedding_batches)) > 0:
                batch_ids, batch_embeddings = zip(*rows)
                batch_embeddings = [
                    np.frombuffer(embedding, dtype=self._get_embedding_dtype())
                    for embedding in batch_embeddings
                ]
                batch_embeddings_np = np.array(batch_embeddings)
                batch_ids_np = np.array(batch_ids, dtype=np.int64)

                faiss.normalize_L2(batch_embeddings_np)
                faiss_index.add_with_ids(batch_embeddings_np, batch_ids_np)

            serialized_index = faiss.serialize_index(faiss_index)
            bytestring = serialized_index.tobytes()
            self._store_source_faiss_bytestring(source_token_id, bytestring)
            faiss_index.reset()

    def get_source_token_faiss_index(self, source_token_id):
        bytestring = self._retrieve_source_faiss_bytestring(source_token_id)
        faiss_index = faiss.deserialize_index(np.frombuffer(bytestring, dtype=np.uint8))
        return faiss_index

    def build_target_faiss_index(self, embedding_ids):
        faiss_index = self._get_new_faiss_index()

        for batch_ids in (
            batches := tqdm(KNNStore._batched(embedding_ids, self.target_batch_size))
        ):
            batches.set_description(
                f"Building target index in batches of {self.target_batch_size}"
            )

            rows = self._retrieve_target_embeddings(batch_ids)

            batch_ordered_ids, batch_embeddings = zip(*rows)
            batch_embeddings = [
                np.frombuffer(embedding, dtype=self._get_embedding_dtype())
                for embedding in batch_embeddings
            ]

            batch_embeddings_np = np.array(batch_embeddings)
            batch_ordered_ids_np = np.array(batch_ordered_ids, dtype=np.int64)

            faiss.normalize_L2(batch_embeddings_np)
            faiss_index.add_with_ids(batch_embeddings_np, batch_ordered_ids_np)

        return faiss_index

    def knn_source_faiss_index(self, source_token_id, source_embedding, k):
        faiss_index = self.get_source_token_faiss_index(source_token_id)

        # TODO: Write the faiss stuff to perform the k nearest neighbor search here
        # and return the list of ids

    def knn_get_logits(self):
        # TODO: Figure out how this all comes together. Need to review math. It's something like
        # calling knn_source_faiss_index() above and then calling build_target_faiss_index(), then
        # searching that index with the target and getting the top k matching target tokens, then
        # going to the math to interpolate with existing model. that will be another class that
        # composes this probably, KNNOperator or something.
        return True

    #
    # Abstract methods that must be implemented in subclass
    #

    def _initialize_database(self, **kwargs):
        """Initialize DB. This is an abstract method.

        This function initializes the DB with the tables required for the KNN store to run. This
        includes four tables:

        - Table 1: Configuration
        - Table 2: Timesteps. Source and target token IDs and embeddings per timestep
        - Table 3: Faiss indices storing encoder embeddings across each source token ID

        Table 1: Configuration key/value pairs
            >Default table name is `knn_store_config`
            >In code, known as `self.configuration_table_name`
            - name:
                String. Name of the configuration. Primary key.
            - value:
                String. Value of the configuration.

        Table 2: Source and target token IDs and embeddings per timestep
            >Default table name is `knn_store_embedding`
            >In code, known as `self.embedding_table_name`
            - id:
                Source-side ID. System-generated. Represents a unique source token occurrence
                within the corpus. The aligned target token ID and target embedding can be
                referenced by this universal ID as well.

            - source_token_id:
                The token type ID (kind of token) at this location in the source. Depends on
                the tokenizer used upstream. Can be used as a way to group source embeddings,
                target tokens, and target embeddings by the source token type.

            - target_token_id:
                The token type ID (kind of token) of the target token that was aligned to
                the source token during the alignment process.

            - source_embedding:
                The embedding of the source token at this position. This assignment depends on
                the upstream embeddings model and how the output is generated, but usually it is
                the embedding created by the encoder in an encoder/decoder transformer architecture.

            - target_embedding:
                The embedding of the target token at this position. Usually the embedding from
                the last hidden state of the decoder at the timestep t of this particular
                position in the corpus, taking into account the whole source, and all target
                tokens up to the point t in the sequence.


        Table 3: Faiss indices storing encoder embeddings across each source token ID
            >Default table name is `knn_store_faiss_index`
            >In code, known as `self.faiss_cache_table_name`
            - source_token_id:
                The token type ID of the source token for which the list of embeddings are being
                pulled for vector search.
            - faiss_index:
                The byte data for a serialized FAISS index using `faiss.serialize_index(index)`.


        Note: Each of these tables must be implemented according to the type of database chosen
              for the subclass.

        Note: Source tokens without a corresponding entry in `alignments` property of input batch
              to `KNNStore.ingest(input_batch)` (in other words, source tokens that were not able to be
              aligned to target tokens) will be ignored and their related embeddings will not be stored.

        Note: No database implementation is given. Use one of the subclasses for a particular
        type of database.

        Args:
            **kwargs:
                Keyword arguments needed to initialize the DB.
        """
        raise NotImplementedError(
            "Make sure to implement `_initialize_database` in a subclass and pass any necessary "
            "`**kwargs` for DB initialization to the base class constructor when constructing "
            "the subclass."
        )

    def _store_corpus_timestep(
        self,
        source_token_id,
        target_token_id,
        source_embedding_bytestring,
        target_embedding_bytestring,
    ):
        """Store source and target token IDs and embeddings for single timestep. This is an abstract method.

        Args:
            source_token_id (int):
            target_token_id (int):
            source_embedding_bytestring (bytes):
            target_embedding_bytestring (bytes):

        Stores the following in Table 2 in the DB:

        source_token_id (int),
        target_token_id (int),
        source_embedding (blob/bytea),
        target_embedding (blob/bytea)

        Table 2: Source and target token IDs and embeddings per timestep

        Note: No database implementation is given. Use one of the subclasses for a particular
        type of database.
        """
        raise NotImplementedError(
            "Make sure to implement `_store_corpus_timestep` in a subclass."
        )

    def _retrieve_all_source_token_ids(self):
        """Retrieve all source token IDs from Table 2. This is an abstract method.

        Retrieves `source_token_id` across all rows in Table 2 in the DB.

        Table 2: Source and target token IDs and embeddings per timestep

        Note: No database implementation is given. Use one of the subclasses for a particular
        type of database.

        Returns:
            tuple(int): All source token IDs stored in Table 2.
        """
        raise NotImplementedError(
            "Make sure to implement `_retrieve_all_source_token_ids` in a subclass."
        )

    def _retrieve_source_token_embeddings_batch(self, source_token_id):
        """Retrieves one batch of source token embeddings from the DB. This is an abstract method.

        GENERATOR FUNCTION.

        Yields a single batch of `source_embedding` fields from Table 2. Retrieves one batch
        according to self._embedding_table_offset and self.embedding_batch_size. Usually this
        will be implemented using something akin to `offset` and `limit` in the DB, and utizing
        an `order by` clause to ensure the offset and limit remain meaningful between function calls.

        Note: Must call `self._increment_source_token_embeddings_offset()` before yielding each batch.

        Note: No database implementation is given. Use one of the subclasses for a particular
        type of database.

        Args:
            source_token_id (int): Source token ID for which embeddings are retrieved.

        Yields:
            tuple((int, bytes)):
                Tuple of two-element tuples. First element in each two-element tuple is the ID of the
                individual timestep stored in Table 2. Second element in each two-element tuple is the
                bytestring corresponding to the last encoder hidden state for source token at timestep.
                When no more data exists, should yield an empty tuple.
        """
        raise NotImplementedError(
            "Make sure to implement `_retrieve_source_token_embeddings_batch` in a subclass."
        )

    def _store_source_faiss_bytestring(self, source_token_id, bytestring):
        """Stores faiss index for source embeddings across one source token ID. This is an abstract method.

        Stores the faiss index represented in the `bytestring` parameter in Table 3, overwriting any
        previous faiss index bytestring stored for this particular source_token_id. This is done using
        a DB transaction, so an index will only be removed if it is certainly being replaced by another.

        Note: No database implementation is given. Use one of the subclasses for a particular
        type of database.

        Args:
            source_token_id (int):
                Source token ID for this source embedding index.
            bytestring (bytes):
                Bytes that make up the serialized version of the faiss index for this source token ID
        """
        raise NotImplementedError(
            "Make sure to implement `_store_source_faiss_bytestring` in a subclass."
        )

    def _retrieve_source_faiss_bytestring(self, source_token_id):
        """Retrieves bytestring of serialized faiss index containing all source embeddings for
        a given source token. This is an abstract method.

        Retrieves serialized faiss index corresponding to the given `source_token_id` as a
        bytestring from the DB. Will return just the bytestring, not the row returned from
        the DB connection utility.

        Note: No database implementation is given. Use one of the subclasses for a particular
        type of database.

        Args:
            source_token_id (int):
                Source token ID for this source embedding faiss index.

        Returns:
            bytes: Bytestring of faiss index corresponding to `source_token_id`.
        """
        raise NotImplementedError(
            "Make sure to implement `_retrieve_source_faiss_bytestring` in a subclass."
        )

    def _retrieve_target_embeddings(self, embedding_ids):
        """Retrieves target token embeddings corresponding to a list of Table 2 IDs.

        Retrieves all target token embeddings according to a list of Table 2 IDs (`embedding_ids`).
        The format of the return should be a tuple of rows where each row is a two-element tuple:

        Ex: ((embedding_id1, target_embedding1), (embedding_id2, target_embedding2))

        If no data is found for the given `embedding_ids`, an empty tuple (`()`) should be returned.

        Note: No database implementation is given. Use one of the subclasses for a particular
        type of database.

        Args:
            embedding_ids (list):
                Source token ID for this source embedding faiss index.

        Returns:
            tuple(tuple(int, bytes)): A tuple of two-element tuples, each containing the timestep /
            embedding ID and the bytestring of the faiss index respectively.
        """
        raise NotImplementedError(
            "Make sure to implement `_retrieve_target_embeddings` in a subclass."
        )
