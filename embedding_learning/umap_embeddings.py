import umap
def generate_umap_embeddings(groups_xs, groups_ys, EMBEDDING_DIM):
    reducer = umap.UMAP(n_components=EMBEDDING_DIM)
    embeddings = []
    for i in range(len(groups_xs)):
        emb = reducer.fit_transform(groups_xs[i].numpy())
        embeddings.append(emb)
    return embeddings