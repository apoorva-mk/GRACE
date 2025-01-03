import redis
from typing import Dict
import faiss
from sentence_transformers import SentenceTransformer
import time
import json


class RedisGraphCache:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        decode_responses: bool = True,
        cache_size: str = "0",
        **kwargs,
    ):
        self.redis_cache = redis.Redis(
            host=host, port=port, decode_responses=decode_responses, **kwargs
        )
        self.redis_cache.config_set("maxmemory", cache_size)
        self.redis_cache.config_set("maxmemory-policy", "allkeys-lru")
        print("Cache size set to: ", cache_size)

    def write_entry_to_cache(self, key: str, value: str):
        try:
            self.redis_cache.set(name=key, value=value)
        except Exception as e:
            # TODO: Change to logger
            print(f"Failed to write record: (key: {key}, value: {value} due to ", e)

    def write_to_cache(self, entries: Dict[str, str]):
        for key, value in entries.items():
            self.write_entry_to_cache(key, value)

    def read_entry_from_cache(self, key):
        # TODO: Check that this resets cache priority for that element
        return self.redis_cache.get(key)


class SemanticCache:
    def __init__(self, json_file="cache_file.json", threshold=0.35) -> None:
        pass
        self.index = faiss.IndexFlatL2(768)
        if self.index.is_trained:
            print("Index trained")
        self.encoder = SentenceTransformer("all-mpnet-base-v2")
        self.euclidean_threshold = threshold

        self.json_file = json_file
        self.cache = {"query_str": [], "node_docs": []}
        self.store_cache(self.json_file, self.cache)

    def retrieve_cache(self, json_file):
        try:
            with open(json_file, "r") as file:
                cache = json.load(file)
        except FileNotFoundError:
            cache = {"query_str": [], "node_docs": []}

        return cache

    def store_cache(self, json_file, cache):
        with open(json_file, "w") as file:
            json.dump(cache, file)
