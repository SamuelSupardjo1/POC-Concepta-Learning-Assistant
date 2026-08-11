from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:
    """
    Wrapper untuk model embedding HuggingFace.
    """

    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def get_model(self):
        return self.model