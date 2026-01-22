import cebra

def generate_cebra_embeddings(groups_xs, groups_ys, EMBEDDING_DIM):
    multi_cebra_model = cebra.CEBRA(batch_size=512,
                                    output_dimension=EMBEDDING_DIM,
                                    max_iterations=100,
                                    max_adapt_iterations=100)
    multi_cebra_model.fit(groups_xs, groups_ys)
    embeddings = []
    for i in range(len(groups_xs)):
        emb = multi_cebra_model.transform(groups_xs[i], session_id=i)
        embeddings.append(emb)
    return embeddings