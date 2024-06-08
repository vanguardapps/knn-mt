from knn.knn_store_sqlite import KNNStoreSQLite


def main():
    store = KNNStoreSQLite(database="db/test_db.db")
    print(store.embedding_dim)
    store.build_source_index()


if __name__ == "__main__":
    main()
