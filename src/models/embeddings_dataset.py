from torch.utils.data import Dataset

# TODO: Use mmap.mmap to lay out various embedding data to be used directly as unpickled objects
#       that are then input to FAISS as small indices. Maybe we could store the FAISS indices
#       in a large file and mmap those instead. Not really sure waht the real situation will end
#       up being here.

# class EmbeddingsDataset(Dataset):
#     def __init__(self, )
