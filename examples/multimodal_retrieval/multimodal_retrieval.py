from typing import Iterator
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    CLIPProcessor,
)
from tinyhnsw import HNSWIndex
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import requests
import csv
import sys


class TMDBDataset:
    def __init__(self, dataset_dir="data/tmdb", batch_size=64):
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.paths = list(self.dataset_dir.glob("*.jpg"))

    def download(self):
        self.dataset_dir.mkdir(exist_ok=True, parents=True)

        with open("data/tmdb.csv") as fp:
            reader = csv.DictReader(fp)
            dataset = [row for row in reader]

        for d in tqdm(dataset):
            image_url = d["path"]
            image_data = requests.get(image_url).content
            with open(f'{self.dataset_dir}/{image_url.split("/")[-1]}', "wb") as handler:
                handler.write(image_data)

    def load_image(self, path):
        with Image.open(path) as im:
            return im.copy()

    def load(self) -> Iterator[list]:
        if not self.dataset_dir.exists():
            self.download()

        images = []
        for file in self.paths[:512]:
            try:
                images.append(self.load_image(file))
                if len(images) == self.batch_size:
                    yield images
                    images = []
            except UnidentifiedImageError:
                continue

        if images:  # yield any remaining images
            yield images


def visualize_query(query_results: list[int], query: str, dataset: TMDBDataset) -> None:
    plt.figure(figsize=(3, 7))
    plt.title(query)
    plt.axis("off")
    for i, result in enumerate(query_results):
        plt.subplot(len(query_results), 1, i + 1)
        plt.imshow(dataset.load_image(dataset.paths[result]))
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def main(query):
    dataset = TMDBDataset()

    image_model = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text_model = CLIPTextModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    text_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    index_file = Path("data/tmdb_index.pkl")
    if not index_file.exists():
        index = HNSWIndex(d=512, distance='cos')

        for images in dataset.load():
            inputs = image_processor(images=images, return_tensors="pt", padding=True)
            outputs = image_model(**inputs)
            index.add(outputs.image_embeds.detach().numpy())

        index.save(index_file)
    else:
        index = HNSWIndex.from_file(index_file)

    inputs = text_processor(text=[query], return_tensors="pt", padding=True)
    outputs = text_model(**inputs)

    q = outputs.text_embeds.detach().numpy()

    D, I = index.search(q, k=5)

    visualize_query(I, query, dataset)


if __name__ == "__main__":
    query = sys.argv[1]
    main(query)