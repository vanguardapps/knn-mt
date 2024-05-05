from dotenv import load_dotenv
from model.embeddings_model import NLLBEmbeddingsModel
from utils import base_object


# TODO: use the below special_tokens_mask to mask out unneeded source encoder embeddings
#       from the array, as well as to do the same for the decoded embeddings / and
#       target token_ids

# en_masked = ma.array(
#         en_tokenized.input_ids, mask=en_tokenized.special_tokens_mask
#     ).compressed()


def main():
    load_dotenv()  # Sets HF_TOKEN

    bitext_output_path = "data/train.en-de"

    checkpoint = "facebook/nllb-200-distilled-600M"
    src_lang = "eng_Latn"
    tgt_lang = "deu_Latn"
    src_lang_csv = "english"
    tgt_lang_csv = "german"
    model = NLLBEmbeddingsModel(
        checkpoint, src_lang=src_lang, tgt_lang=tgt_lang, token=True
    )

    batch = base_object()

    batch.inputs = ["Hello, how are you?", "What a nice day"]
    batch.labels = ["Hallo, wie geht's?", "Was ein sehr sohn tag!"]

    outputs = model(batch)

    print(dir(outputs))


if __name__ == "__main__":
    main()
