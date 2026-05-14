def test_embeddings():
    # Requires !pip install sentence-transformers
    from sentence_transformers import SentenceTransformer
    
    # TO DO authenticate requests to huggingface
    
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", 
                                          device="cpu") # choose the device to load the model to (note: GPU will often be *much* faster than CPU)
    
    # Create a list of sentences to turn into numbers
    sentences = [
        "Line 1",
        "Sentence 2",
        "Row 3"
    ]
    
    # Sentences are encoded/embedded by calling model.encode()
    embeddings = embedding_model.encode(sentences)
    embeddings_dict = dict(zip(sentences, embeddings))
    assert len(embeddings_dict['Line 1']) == 768
    assert len(embeddings_dict['Sentence 2']) == 768
    assert len(embeddings_dict['Row 3']) == 768