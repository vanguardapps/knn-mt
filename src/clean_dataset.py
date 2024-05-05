# TODO: Use NLLBEmbeddingsModel to create a knn datastore by token ID that can to kNN lookup using FAISS
#       for the novel source token, then match the h(x-1) vectorization (on same or similar model) to
#       one of several target embeddings by kNN search within that particular source group. May need to
#       read the Fast kNN-MT paper again--not sure why we would need to do kNN with the source token
#       embeddings unless for some reason the target embeddings were broken further into sub-groups.
#       This may have been the case, but it's a little weird. Maybe it just searches through all of the
#       target token embeddings for each source token ID, and that is it. I feel like I mayh have misunderstood
#       that the whole time. Anyway, let's figure out how to build this using NLLBEmbeddingsModel and FAISS.
import pandas as pd
import regex as re
from tqdm import tqdm
from utils import write_csv_line


def get_quality_mask(inputs_raw):
    """Return a list of boolean values as a mask for per-sequence input quality.

    There are two criteria for a quality sequence. One is that the source text must contain
    a reasonable distribution of word characters spread across at least two words. This is
    achieved through several sequential regular expresion patterns, and is currently
    configured to test for at least 5 word characters shared between at minimum 2 words.
    Additionally, it is notable that using this logic, any sentences with only one word will
    automatically be rejected.

    The second criteria is the source text must contain less than 65% non-word characters.

    Note:
        This method assumes that `inputs_raw` attribute contains raw input sequences
        to be graded for quality.
    """
    quality_mask = []

    non_word_threshold = 0.65

    patterns = [
        r"(\pL{1,}).+(\pL{4,})",
        r"(\pL{2,}).+(\pL{3,})",
        r"(\pL{3,}).+(\pL{2,})",
        r"(\pL{4,}).+(\pL{1,})",
    ]
    non_word = r"[^\pL]"

    for sentence in inputs_raw:
        regex_results = [
            1 if bool(re.search(pattern, sentence)) else 0 for pattern in patterns
        ]
        if sum(regex_results) == 0:
            quality_mask.append(False)
            continue

        chars = list(sentence)
        punct_mask = [1 if bool(re.search(non_word, char)) else 0 for char in chars]

        if sum(punct_mask) >= non_word_threshold * len(chars):
            quality_mask.append(False)
            continue

        quality_mask.append(True)

    return quality_mask


def main():
    input_path = "data/de-en-emea-medical.csv"
    output_path = "data/de-en-emea-medical-clean.csv"

    input_df = pd.read_csv(input_path, dtype=str, header="infer")

    with open(output_path, "w") as file:
        for _, row in tqdm(input_df.iterrows()):
            english = row["english"]
            german = row["german"]
            try:
                [quality_mask] = get_quality_mask([english])
            except:
                print("skipped a line")
                continue
            if quality_mask:
                write_csv_line(file, [english, german])


if __name__ == "__main__":
    main()
