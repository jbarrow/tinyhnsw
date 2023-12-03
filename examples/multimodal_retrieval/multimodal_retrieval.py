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
import copy
import csv


def download_tmdb():
    dataset_dir = Path("data/tmdb")
    dataset_dir.mkdir(exist_ok=True, parents=True)

    with open("data/tmdb.csv") as fp:
        reader = csv.DictReader(fp)
        dataset = [row for row in reader]

    for d in tqdm(dataset):
        image_url = d["path"]
        image_data = requests.get(image_url).content
        with open(f'data/tmdb/{image_url.split("/")[-1]}', "wb") as handler:
            handler.write(image_data)


PATHS = list(Path("data/tmdb").glob("*.jpg"))


def load_image(path):
    with Image.open(path) as im:
        return copy.deepcopy(im)


def load_tmdb(batch_size=64) -> Iterator[list]:
    if not Path("data/tmdb").exists():
        download_tmdb()

    images = []
    for file in PATHS[:512]:
        try:
            images.append(load_image(file))
            if len(images) == batch_size:
                yield images
                images = []
        except UnidentifiedImageError:
            continue

    if images:  # yield any remaining images
        yield images


def visualize_query(query_results: list[int], query: str) -> None:
    plt.figure(figsize=(3, 7))
    plt.title(query)
    plt.axis("off")
    for i, result in enumerate(query_results):
        plt.subplot(len(query_results), 1, i + 1)
        plt.imshow(load_image(PATHS[result]))
        plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_model = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text_model = CLIPTextModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    text_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if not Path("data/tmdb_index.pkl").exists():
        index = HNSWIndex(d=512, distance='cosine')

        for images in load_tmdb():
            inputs = image_processor(images=images, return_tensors="pt", padding=True)
            outputs = image_model(**inputs)
            index.add(outputs.image_embeds.detach().numpy())

        index.save("data/tmdb_index.pkl")
    else:
        index = HNSWIndex.from_file("data/tmdb_index.pkl")

    inputs = text_processor(text=["landscape"], return_tensors="pt", padding=True)
    outputs = text_model(**inputs)

    q = outputs.text_embeds.detach().numpy()

    D, I = index.search(q, k=5)

    visualize_query(I, "landscape")
