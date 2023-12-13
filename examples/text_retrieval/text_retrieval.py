from sentence_transformers import SentenceTransformer
from datasets import load_dataset

from tinyhnsw import HNSWIndex


if __name__ == "__main__":
    dataset = load_dataset("financial_phrasebank", "sentences_allagree")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    encoded = model.encode(dataset["train"]["sentence"], show_progress_bar=True)

    index = HNSWIndex(d=384)
    index.add(encoded)

    D, I = index.search(model.encode("positive return"), k=5)

    for i in I:
        print(dataset["train"][i]["sentence"])