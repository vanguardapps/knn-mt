import os
import psycopg2
from dotenv import load_dotenv
from knn.knn_dataset import KNNDataset
from knn.knn_store import KNNStore
from models.nllb_embeddings import NLLBEmbeddingsModel
from models.batch import EmbeddingsBatch
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    load_dotenv()

    PGHOST = os.environ["PGHOST"]
    PGUSER = os.environ["PGUSER"]
    PGPORT = os.environ["PGPORT"]
    PGDATABASE = os.environ["PGDATABASE"]
    PGPASSWORD = os.environ["PGPASSWORD"]

    connection = psycopg2.connect(
        database=PGDATABASE,
        user=PGUSER,
        password=PGPASSWORD,
        host=PGHOST,
        port=int(PGPORT),
    )

    cursor = connection.cursor()

    cursor.execute("SELECT * from test_table;")

    # Fetch all rows from database
    record = cursor.fetchall()

    print("Data from Database: ", record)

    checkpoint = "facebook/nllb-200-distilled-600M"
    src_lang = "eng_Latn"
    tgt_lang = "deu_Latn"
    batch_size = 2
    store_path = "index_store"
    embeddings_store = []

    # TODO: write so it stores it in like 10 files, spread across. It should only be ~40GB total

    dataset_path = "data/de-en-emea-medical-clean.csv"
    dataset = KNNDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size)

    model = NLLBEmbeddingsModel(
        checkpoint=checkpoint,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )
    store = KNNStore(store_path)

    for english, german in tqdm(loader):
        batch = EmbeddingsBatch(inputs_raw=english, labels_raw=german)
        model(batch)
        batch.postprocess()
        batch.generate_alignments(tokenizer=model.tokenizer)
        store.ingest(batch)


if __name__ == "__main__":
    main()
