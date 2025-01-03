import json
import os
import base64


def encode_filename(string):
    return base64.urlsafe_b64encode(string.encode())


def serialize_wikipedia_embeddings(wikipedia_embeddings, output_path="./embeds"):
    for k, v in wikipedia_embeddings.items():

        with open(output_path + f"/{encode_filename(k)}", "w") as f:
            json.dump({k: v}, f, indent=4)


def deserialize_wikipedia_embeddings(input_path="./embeds"):
    wikipedia_embeddings = {}
    for file in os.listdir(input_path):
        with open(input_path + f"/{file}", "r") as f:
            wikipedia_embeddings.update(json.load(f))

    return wikipedia_embeddings
