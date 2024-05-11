import pathlib
import pickle


class KNNStore(object):
    source_index_prefix = "source_"
    source_index_ext = ".idx"
    target_embeddings_prefix = "target_"
    target_embeddings_ext = ".emb"

    def __init__(self, store_path):
        # TODO: initialize store_path if it is not already detectably initialized.
        # store_path will just be a place to keep all the saved faiss indices and metadata.

        # create store_path directory
        path = pathlib.Path(store_path)
        path.mkdir(parents=True, exist_ok=True)

        # TODO: Replace pickling with local postgres to:
        #   - Store target embeddings with columns "source_token_id", "global_id", "embedding"
        #   - See if we can mimic storing file as blob in postgres and use that to store / load
        #     the "mini indices" we need, basically a blob for each source_token_id

        with open(path / KNNStore.primary_index, "wb") as f:
            pickle.dump(obj, f)

        def unpickle_obj(filepath):
            with open(filepath, "rb") as f:
                return pickle.load(f)

        ()

    def ingest(self, batch):
        # TODO: ingest a fully-loaded and aligned EmbeddingsBatch, saving source and target embeddings
        # to the right datastores / indices, saving metadata to the right places as appropriate.
        return False
