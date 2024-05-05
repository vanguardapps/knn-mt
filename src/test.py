# TODO: Use NLLBEmbeddingsModel to create a knn datastore by token ID that can to kNN lookup using FAISS
#       for the novel source token, then match the h(x-1) vectorization (on same or similar model) to
#       one of several target embeddings by kNN search within that particular source group. May need to
#       read the Fast kNN-MT paper again--not sure why we would need to do kNN with the source token
#       embeddings unless for some reason the target embeddings were broken further into sub-groups.
#       This may have been the case, but it's a little weird. Maybe it just searches through all of the
#       target token embeddings for each source token ID, and that is it. I feel like I mayh have misunderstood
#       that the whole time. Anyway, let's figure out how to build this using NLLBEmbeddingsModel and FAISS.

import pandas as pd
import sys
import torch
from models.nllb_embeddings import NLLBEmbeddingsModel
from models.batch import EmbeddingsBatch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, path):
        self._df = pd.read_csv(path, dtype=str, header="infer")

    def __len__(self):
        return self._df.shape[0]

    def __getitem__(self, index):
        return self._df.iloc[index]["english"], self._df.iloc[index]["german"]


def main():
    checkpoint = "facebook/nllb-200-distilled-600M"
    src_lang = "eng_Latn"
    tgt_lang = "deu_Latn"
    batch_size = 50
    embeddings_store = []

    # TODO: write so it stores it in like 10 files, spread across. It should only be ~40GB total

    dataset_path = "data/de-en-emea-medical-clean.csv"
    dataset = CustomDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size)
    model = NLLBEmbeddingsModel(
        checkpoint=checkpoint,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    for english, german in loader:
        batch = EmbeddingsBatch(inputs_raw=english, labels_raw=german)
        model(batch)
        print(batch.token_hidden_states.size())
        print(batch.token_hidden_states.dtype)
        break


if __name__ == "__main__":
    main()
