from knn.knn_dataset import KNNDataset
from models.nllb_embeddings import NLLBEmbeddingsModel
from models.batch import EmbeddingsBatch
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
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
        KNNStore.ingest(batch)


if __name__ == "__main__":
    main()
