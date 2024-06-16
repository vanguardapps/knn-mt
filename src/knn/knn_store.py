import faiss
import numpy as np
import sys
import torch
from abc import ABC, abstractmethod
from itertools import islice
from tqdm import tqdm


class KNNStore(ABC):
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
        default_c (int): (class attribute)
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
        c (int):
    """

    default_table_prefix = "knn_store"
    default_configuration_table_stem = "config"
    default_embedding_table_stem = "embedding"
    default_faiss_cache_table_stem = "faiss_index"
    default_embedding_batch_size = 50
    default_target_batch_size = 50
    default_embedding_dtype = "float32"
    default_c = 5

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
            c (int):
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

        self.c = c if c is not None else KNNStore.default_c

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
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch

    @staticmethod
    def _convert_faiss_index_to_bytestring(faiss_index):
        serialized_index = faiss.serialize_index(faiss_index)
        return serialized_index.tobytes()

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
                        source_token_id=source_token_id.item(),
                        target_token_id=target_token_id.item(),
                        source_embedding_bytestring=source_embedding_bytestring,
                        target_embedding_bytestring=target_embedding_bytestring,
                    )

    def _get_new_faiss_index(self):
        return faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim))

    def _get_embedding_dtype(self):
        # TODO: add support for dtypes other than np.float32
        if self.embedding_dtype == "float32":
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

    def _get_valid_embedding_offset_and_batch_size(self):
        error_first_sentence = (
            "Please ensure you are not modifying private class members. "
        )

        if not isinstance(self._embedding_table_offset, int):
            raise ValueError(
                f"{error_first_sentence}" "`_embedding_table_offset` must be an `int`."
            )
        if isinstance(self._embedding_table_offset, bool):
            raise ValueError(
                f"{error_first_sentence}"
                "`_embedding_table_offset` must be an `int` and not a `bool`."
            )
        if self._embedding_table_offset < 0:
            raise ValueError(
                f"{error_first_sentence}"
                "`_embedding_table_offset` must be positive or zero."
            )

        if not isinstance(self.embedding_batch_size, int):
            raise ValueError(
                f"{error_first_sentence}" "`embedding_batch_size` must be an `int`."
            )
        if isinstance(self.embedding_batch_size, bool):
            raise ValueError(
                f"{error_first_sentence}"
                "`embedding_batch_size` must be an `int` and not a `bool`."
            )
        if self.embedding_batch_size < 1:
            raise ValueError(
                f"{error_first_sentence}" "`embedding_batch_size` must be positive."
            )

        return self._embedding_table_offset, self.embedding_batch_size

    def _add_bytestrings_to_faiss_index(
        self, faiss_index, batch_ids, batch_bytestrings
    ):
        batch_embeddings_np = np.array(
            [
                np.frombuffer(embedding, dtype=self._get_embedding_dtype())
                for embedding in batch_bytestrings
            ]
        )
        batch_ids_np = np.array(batch_ids, dtype=np.int64)
        faiss.normalize_L2(batch_embeddings_np)
        faiss_index.add_with_ids(batch_embeddings_np, batch_ids_np)

    def build_source_index(self):
        faiss_index = self._get_new_faiss_index()
        source_token_ids = self._retrieve_all_source_token_ids()

        # One source token ID at a time
        for (source_token_id,) in (batches := tqdm(source_token_ids)):
            batches.set_description(
                f"Building index for source token ID {source_token_id}"
            )

            embedding_batches = self._retrieve_source_token_embeddings_batches(
                source_token_id
            )

            while rows := next(embedding_batches):
                batch_ids, batch_bytestrings = zip(*rows)
                self._add_bytestrings_to_faiss_index(
                    faiss_index, batch_ids, batch_bytestrings
                )

            bytestring = KNNStore._convert_faiss_index_to_bytestring(faiss_index)
            self._store_source_faiss_bytestring(source_token_id, bytestring)
            faiss_index.reset()

    def get_source_token_faiss_index(self, source_token_id):
        bytestring = self._retrieve_source_faiss_bytestring(source_token_id)
        if bytestring is not None:
            return faiss.deserialize_index(np.frombuffer(bytestring, dtype=np.uint8))
        return None

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

    def build_target_datastore(
        self,
        encoder_input_ids,
        encoder_last_hidden_state,
        c=None,
    ):
        """Builds one target datastore faiss index for each sequence in the batch
        TODO: ROY: Finish this docstring
        """

        c = c if c is not None else self.c

        batch_size = encoder_input_ids.shape[0]
        self.target_datastore = [None] * batch_size

        for index in (batches := tqdm(range(batch_size))):
            batches.set_description(f"Building target datastore batch {index}")

            queries = {}

            # Gather faiss indices for each source_token_id in the sequence along with the queries for each
            for source_token_id, source_embedding in zip(
                encoder_input_ids[index], encoder_last_hidden_state[index]
            ):
                if queries.get(source_token_id, None) is None:
                    faiss_index = self.get_source_token_faiss_index(source_token_id)
                    if faiss_index is not None:
                        queries[source_token_id] = KNNStore.__FaissQueries__(
                            faiss_index=faiss_index,
                            embedding_dim=self.embedding_dim,
                            embedding_dtype=self._get_embedding_dtype(),
                            k=c,
                        )
                    else:
                        queries[source_token_id] = "no_index"

                if isinstance(queries[source_token_id], KNNStore.__FaissQueries__):
                    queries[source_token_id].add_query(source_embedding)

            unique_source_token_ids = queries.keys()

            if len(unique_source_token_ids) > 0:
                self.target_datastore[index] = KNNStore.__FaissQueries__(
                    faiss_index=self._get_new_faiss_index(),
                    embedding_dim=self.embedding_dim,
                    embedding_dtype=self._get_embedding_dtype(),
                )

            # Run bulk queries against faiss indices for each source token
            for source_token_id in unique_source_token_ids:
                _, embedding_ids = queries[source_token_id].run(use_gpu=True)
                unique_embedding_ids = np.unique(embedding_ids.flatten())
                rows = self._retrieve_target_bytestrings(
                    unique_embedding_ids[unique_embedding_ids > 0].tolist()
                )
                batch_ids, batch_bytestrings = zip(*rows)
                self._add_bytestrings_to_faiss_index(
                    self.target_datastore[index].faiss_index,
                    batch_ids,
                    batch_bytestrings,
                )

    def search_target_datastore(
        self,
        decoder_last_hidden_state: torch.FloatTensor,
        k: int,
        unfinished_sequences: torch.LongTensor,
        pad_token_id: int = None,
        return_probs: bool = None,
        vocab_dim: int = None,
        temperature: float = None,
    ):
        """Returns the top k target tokens from datastore.
        TODO: ROY: Finish this docstring
        """
        embedding_dtype = self._get_embedding_dtype()

        # TODO: ROY: Investigate whether the return type from faiss for distances is dependent upon the dtype of the
        # vectors between which distance is being computed.
        batch_l2_distances = np.empty((0, k), dtype=embedding_dtype)
        batch_target_token_ids = np.empty((0, k), dtype=np.int64)

        pad_token_id = pad_token_id if pad_token_id is not None else 0

        for query_embedding, faiss_queries, sequence_is_unfinished in zip(
            decoder_last_hidden_state,
            self.target_datastore,
            unfinished_sequences == 1,
        ):
            if sequence_is_unfinished and faiss_queries is not None:
                faiss_queries.add_query(query_embedding)

                # TODO: ROY: Parameterize `use_gpu` based on whether GPU is available
                l2_distances, embedding_ids = faiss_queries.run(k=k, use_gpu=True)

                target_token_ids = np.array(
                    self._retrieve_target_token_ids(tuple(embedding_ids[0])),
                    dtype=np.int64,
                ).reshape((1, -1))

                target_token_ids = np.array(
                    target_token_ids,
                    dtype=np.int64,
                ).reshape((1, -1))
            else:
                # Cut down on computational complexity for finished sequences
                l2_distances = np.zeros((1, k), dtype=embedding_dtype)
                target_token_ids = np.full(
                    (1, k), fill_value=pad_token_id, dtype=np.int64
                )

            batch_l2_distances = np.concatenate(
                (batch_l2_distances, l2_distances), axis=0
            )
            batch_target_token_ids = np.concatenate(
                (batch_target_token_ids, target_token_ids), axis=0
            )

        if not return_probs:
            return batch_l2_distances, batch_target_token_ids

        if vocab_dim is None:
            raise ValueError(
                "Missing required parameter `vocab_dim` necessary for calculating logits."
            )

        if temperature is None:
            raise ValueError(
                "Missing required parameter `temperature` necessary for calculating logits."
            )

        batch_size = batch_l2_distances.shape[0]

        # shape (batch_size, k, vocab_dim)
        one_hot_tokens = np.zeros(
            (batch_size, k, vocab_dim), dtype=self._get_embedding_dtype()
        )

        for i in range(batch_size):
            for j in range(k):
                one_hot_tokens[i, j, batch_target_token_ids[i, j]] = 1

        # TODO: ROY: Investigate whether it would work to strip away the np.exp( part here
        # and just return scores

        # shape (batch_size, k)
        exp_term = np.exp(-batch_l2_distances / temperature)

        # Replace any infinitesimal or zero values in `exp_term` with epsilon
        epsilon = 1e-7
        exp_term[exp_term < epsilon] = epsilon

        # shape (batch_size, k, vocab_dim)
        V = one_hot_tokens * exp_term.reshape(batch_size, k, 1)

        # shape (batch_size, 1, 1)
        Z = np.sum(exp_term, axis=1).reshape(batch_size, 1, 1)

        # shape (batch_size, k, vocab_dim)
        knn_probs_per_candidate = V / Z

        # `knn_probs` has shape (batch_size, vocab_dim)
        knn_probs = np.sum(knn_probs_per_candidate, axis=0)

        return knn_probs

    #
    # Abstract methods that must be implemented in subclass
    #

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def _retrieve_source_token_embeddings_batches(self, source_token_id):
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
            "Make sure to implement `_retrieve_source_token_embeddings_batches` in a subclass."
        )

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def _retrieve_target_bytestrings(self, embedding_ids):
        """Retrieves target token embeddings corresponding to a list of Table 2 IDs.

        Retrieves all target token embeddings according to a list of Table 2 IDs (`embedding_ids`).
        The format of the return should be a tuple of rows where each row is a two-element tuple:

        Ex: ((embedding_id1, target_embedding1), (embedding_id2, target_embedding2))

        If no data is found for the given `embedding_ids`, an empty tuple (`()`) should be returned.

        Note: No database implementation is given. Use one of the subclasses for a particular
        type of database.

        Args:
            embedding_ids (list):
                Table 2 row IDs for which to retrieve the `target_embedding` values.

        Returns:
            tuple(tuple(int, bytes)): A tuple of two-element tuples, each containing the timestep /
            embedding ID and the bytestring of the faiss index respectively.
        """
        raise NotImplementedError(
            "Make sure to implement `_retrieve_target_bytestrings` in a subclass."
        )

    @abstractmethod
    def _retrieve_target_token_ids(self, embedding_ids):
        """Retrieves target token IDs to a list of Table 2 IDs.

        Retrieves all target token IDs according to a list of Table 2 IDs (`embedding_ids`).
        The format of the return should be a tuple of integer target token IDs:

        Ex: (target_token_id1, target_token_id2, ...)

        If no data is found for the given `embedding_ids`, an empty tuple (`()`) should be returned.

        Note: No database implementation is given. Use one of the subclasses for a particular
        type of database.

        Args:
            embedding_ids (list):
                Table 2 row IDs for which to retrieve the `target_token_id` values.

        Returns:
            tuple(int): A tuple of integer target token IDs.
        """
        raise NotImplementedError(
            "Make sure to implement `_retrieve_target_token_ids` in a subclass."
        )

    class __FaissQueries__(dict):
        """Helper to contain one faiss index with queries.

        Attributes:
            faiss_index (faiss.swigfaiss_avx2.IndexIDMap):
                The fully initialized FAISS index stored on the CPU with vectors preloaded. Required for
                construction of `__FaissQueries__` object. Note: only CPU-stored faiss indices should be
                passed, as the index will be moved to the GPU at query time and then removed after.
            embedding_dim (int):
                The dimensionality of the query embeddings. Required for construction of `__FaissQueries__` object.
            embedding_dtype (type):
                The data type of the query embeddings. Defaults to `numpy.float32`.
            queries (ndarray(`embedding_dtype`)):
                Array of size (N, `embedding_dim`) where N is the number of query embeddings added. This will be
                directly used as input to `faiss_index.search()`.
            k (int):
                The number of nearest neighbors to return data for when running queries against the stored
                `faiss_index`. Defaults to 3.
        """

        def __init__(
            self, faiss_index=None, embedding_dim=None, embedding_dtype=None, k=None
        ):
            """Initialize a faiss index queries container.

            Args:
                faiss_index (faiss.swigfaiss_avx2.IndexIDMap):
                    The fully initialized FAISS index with vectors preloaded. Note: only CPU-stored faiss
                    indices should be passed, as the index will be moved to the GPU at query time and then
                    removed after.
                embedding_dim (int):
                    The dimensionality of the query embeddings. Required for construction of `__FaissQueries__` object.
                embedding_dtype (type):
                    The data type of the query embeddings. Defaults to `numpy.float32`.
                k (int):
                    The number of nearest neighbors to return data for when running queries against
                    the stored `faiss_index`. Defaults to 3.
            """
            if faiss_index is None:
                raise ValueError("Missing required parameter `faiss_index`.")

            if embedding_dim is None:
                raise ValueError("Missing required parameter `embedding_dim`.")

            self.faiss_index = faiss_index

            self.embedding_dim = embedding_dim
            self.embedding_dtype = (
                embedding_dtype if embedding_dtype is not None else np.float32
            )

            self.k = k if k is not None else 3

            self.queries = np.empty((0, self.embedding_dim), dtype=self.embedding_dtype)

        def add_query(self, query_embedding):
            """Add one embedding to the list of query embeddings for this faiss index.

            Args:
                query_embedding (ndarray):
                    1D array of size (`embedding_dim`) to be concatenated with existing queries. Data type is
                    determined by attribute `embedding_dtype`, set during construction.
            """
            if len(query_embedding.shape) > 1:
                raise ValueError(
                    "Parameter `query_embedding` (ndarray) must have only one dimension."
                )

            if query_embedding.shape[0] != self.embedding_dim:
                raise ValueError(
                    f"Parameter `query_embedding` (ndarray) of dimension {query_embedding.shape[0]} "
                    f"does not match configured `embedding_dim` of {self.embedding_dim}. Please "
                    f"only pass query embeddings of dimension {self.embedding_dim}."
                )

            self.queries = np.concatenate(
                (self.queries, query_embedding[np.newaxis, :])
            )

        def run(self, k=None, use_gpu=None):
            """Run all queries currently stored in the container.

            Runs all queries stored in `queries` against `faiss_index`.

            Args:
                k (int):
                    The number of nearest neighbors for which to return data when running queries against
                    the stored `faiss_index`. Defaults to `self.k` on the object, which defaults to 3
                    when not specified at construction time.
                use_gpu (bool):
                    Whether to place the index on the GPU prior to searching. This also implies that the
                    index is automatically deallocated (by calling `faiss_index.reset()`) after the search
                    is complete.

            Returns:
                tuple(ndarray, ndarray):
                    A tuple with first element containing the matrix of L2 distances from the query vector
                    to the neighbor at that column for the query at that row. The second element is the
                    matrix of IDs for the neighbor at that column for the query at that row.
            """
            k = k if k is not None else self.k
            use_gpu = use_gpu if use_gpu is not None else False

            if use_gpu:
                # TODO: ROY: Expand this to support multiple GPUs / GPU array
                # Should be something like the following:
                # gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
                res = faiss.StandardGpuResources()
                faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
            else:
                faiss_index = self.faiss_index

            distance, ids = faiss_index.search(self.queries, k)

            if use_gpu:
                faiss_index.reset()

            return distance, ids
