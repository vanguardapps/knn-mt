from knn.store_pg import KNNStorePG


def main():
    store = KNNStorePG(embedding_dim=1024)
    store.build_source_index()


if __name__ == "__main__":
    main()
