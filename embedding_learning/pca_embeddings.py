from sklearn.decomposition import PCA

def generate_pca_embeddings(groups_xs, groups_ys, EMBEDDING_DIM):
    embeddings = []
    for x in groups_xs:
        # Perform PCA separately for each group
        pca = PCA(n_components=EMBEDDING_DIM)
        emb = pca.fit_transform(x.numpy())
        embeddings.append(emb)
    return embeddings