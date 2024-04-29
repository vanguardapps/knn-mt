import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# from transformers.model_output import Seq2SeqModelOutput
from dotenv import load_dotenv

load_dotenv()  # Sets HF_TOKEN

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

input_ids = tokenizer(
    "What is the best type of blue sky you can paint, sergeant?", return_tensors="pt"
).input_ids

# create decoder_input_ids with just BOS token (tokenizer.bos_token)
decoder_input_ids = tokenizer(
    tokenizer.bos_token, add_special_tokens=False, return_tensors="pt"
).input_ids

model_output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

# Tensor of shape (1, 18, 1024)
enc_hidden = model_output.encoder_last_hidden_state

vectors = torch.squeeze(enc_hidden.clone().detach()).numpy()

print(vectors.shape)

exit(0)

# Get one GPU allocation
res = faiss.StandardGpuResources()

vector_dimension = 1024
index = faiss.IndexFlatL2(vector_dimension)

# index_with_map = faiss.IndexIDMap(index)

# Place index on GPU
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
# gpu_index_with_map = faiss.index_cpu_to_gpu(res, 0, index_with_map)

faiss.normalize_L2(vectors)

index.add(vectors)
# gpu_index_with_map.add(vectors, ids)

# Finally search
k = 3
D, I = index.search(vectors[:3], k)

print(D)
print(I)
