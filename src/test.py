from models.logits_processor import CustomLogitsProcessor
from transformers.generation import LogitsProcessorList
from knn.knn_dataset import KNNDataset
from models.nllb import NLLBCheckpoint
from models.batch import LMBatch
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

    logits_processor = CustomLogitsProcessor()
    logits_processor_list = LogitsProcessorList()
    logits_processor_list.append(logits_processor)

    model = NLLBCheckpoint(
        checkpoint=checkpoint,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        logits_processor=logits_processor_list,
    )

    print(model.model.generation_config)

    # logits_processor.set_tokenizer(model.tokenizer)

    # count = 0
    # for english, _ in (batches := tqdm(loader)):
    #     batches.set_description(f"Processing {dataset_path} in batches of {batch_size}")

    #     batch = LMBatch(inputs_raw=english)
    #     print(model(batch).batch_output_text)

    #     if count > 10:
    #         break

    #     count += 1


if __name__ == "__main__":
    main()
