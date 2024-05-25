from knn.knn_dataset import KNNDataset
from knn.store_pg import KNNStorePG
from models.nllb_embeddings import NLLBEmbeddingsModel
from models.batch import EmbeddingsBatch
from torch.utils.data import DataLoader
from tqdm import tqdm

# NOTE: A PostgreSQL database must be available for this to work. Configure this application
# using the standard Postgres environment variables PGHOST, PGUSER, PGDATABASE, PGPASSWORD,
# and PGPORT.


def main():
    checkpoint = "facebook/nllb-200-distilled-600M"
    src_lang = "eng_Latn"
    tgt_lang = "deu_Latn"
    batch_size = 2

    dataset_path = "data/de-en-emea-medical-clean.csv"
    dataset = KNNDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size)

    model = NLLBEmbeddingsModel(
        checkpoint=checkpoint,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    store = KNNStorePG(embedding_dim=model.config.hidden_size)

    count = 0
    for english, german in (batches := tqdm(loader)):
        batches.set_description(f"Processing {dataset_path} in batches of {batch_size}")

        batch = EmbeddingsBatch(inputs_raw=english, labels_raw=german)
        model(batch)
        batch.postprocess()
        batch.generate_alignments(tokenizer=model.tokenizer)
        store.ingest(batch)

        if count > 10:
            break

        count += 1


if __name__ == "__main__":
    main()
