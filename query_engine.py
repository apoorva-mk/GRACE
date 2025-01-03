from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.core import StorageContext, KnowledgeGraphIndex


from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import KGTableRetriever, VectorIndexRetriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core import KnowledgeGraphIndex
from llama_index.core import (
    set_global_service_context,
    PromptHelper,
    ServiceContext,
    VectorStoreIndex,
)

# from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank

from pydantic import Field

from typing import List, Set, Dict, Union

from cache import RedisGraphCache, SemanticCache
from dataclasses import dataclass
import json

import time

Node = str


def transform_string(string):
    string = string.replace("_", " ")
    return string.lower()


# TODO: move to utils
def merge_node_doc_dicts(
    doc_nodes: Dict[str, NodeWithScore], new_nodes: Dict[str, NodeWithScore]
) -> Dict[str, NodeWithScore]:
    for node, node_doc in new_nodes.items():
        if node in doc_nodes:
            doc_nodes[node].score += node_doc.score
        else:
            doc_nodes[node] = node_doc

    return doc_nodes


@dataclass(frozen=True)
class ConversationalTriplet:
    subject: Node
    relation: str
    object: Node

    def __str__(self) -> str:
        return str((self.subject, self.relation, self.object))


@dataclass(frozen=True)
class Triplet:
    subject: Node
    relation: str
    object: Node

    def __str__(self) -> str:
        return str((self.subject, transform_string(self.relation), self.object))


@dataclass(frozen=True)
class SerialNodeWithScore:
    node: str
    score: float

    @property
    def node_with_score(self) -> NodeWithScore:
        return NodeWithScore(node=TextNode(text=self.node), score=self.score)

    @classmethod
    def from_dict(cls, node_dict: dict):
        return cls(**node_dict)


# Please see instructions for custom query engines at https://docs.llamaindex.ai/en/stable/examples/query_engine/custom_query_engine/
class RAGStringQueryEngine:
    """RAG String Query Engine."""

    response_synthesizer: BaseSynthesizer
    kg_index: KnowledgeGraphIndex
    doc_retriever: VectorIndexRetriever
    reranker: FlagEmbeddingReranker

    def __init__(
        self,
        llm,
        kg_index,
        doc_index,
        traverse_k,
        retrieve_k,
        return_n,
        dataset_name="wikipedia",
        prefix_dir="./data",
    ):
        super().__init__()

        self.kg_index = kg_index

        self.doc_retriever = doc_index.as_retriever(
            similarity_top_k=retrieve_k,
        )

        self.traverse_k = traverse_k
        self.retrieve_k = retrieve_k
        self.return_n = return_n

        self.response_synthesizer = get_response_synthesizer(llm)
        self.time_profile = []

        self.dataset_name = dataset_name

        self.prefix_dir = prefix_dir
        self.file_name = f"{self.prefix_dir}/{self.dataset_name}_traverse_k={self.traverse_k}_retrieve_k={self.retrieve_k}_return_n={self.return_n}_time_profile.json"

    def retrieve_kg_nodes(self, query: QueryBundle) -> Set[Triplet]:
        query_engine = self.kg_index.as_query_engine(
            include_text=False,
            response_mode="tree_summarize",
            retriever_mode="embedding",
            similarity_top_k=self.retrieve_k,
            graph_store_query_depth=self.traverse_k,
        )

        response = query_engine.query(query)

        triplets_strs = response.source_nodes[0].metadata["kg_rel_texts"]

        total_nodes = set()
        for t in triplets_strs:
            triplet_list = t[1:-1].split("'")
            total_nodes.add(Triplet(triplet_list[1], "", triplet_list[-2]))

        return total_nodes

    def retrieve_kg_documents_for_node(
        self, node: Node, k: int
    ) -> Dict[str, NodeWithScore]:

        start_time = time.time()
        triplet_list = self.traverse_from_node(node, self.traverse_k)
        curr_time = time.time()
        self.curr_time_profile["NODE TRAVERSAL"] += curr_time - start_time
        start_time = curr_time

        node_docs: Dict[str, NodeWithScore] = dict()

        for triplet in triplet_list:
            node_docs_for_triplet = self.retrieve_related_documents(triplet)
            node_docs = merge_node_doc_dicts(node_docs, node_docs_for_triplet)

        curr_time = time.time()
        self.curr_time_profile["DOCUMENT RETRIEVAL"] += curr_time - start_time

        return node_docs

    def traverse_from_node(self, node: Node, k: int) -> Set[Triplet]:
        # TODO: Implicit Caching of traversed graph triplets
        # graph_store.get_rel_map produces map of node to list of triples in for each node

        rel_map = self.kg_index.graph_store.get_rel_map([node], k)
        if isinstance(self.kg_index.graph_store, SimpleGraphStore):
            full_triplet_set = set()
            for key, triplet_list in rel_map.items():
                for trip in triplet_list:
                    full_triplet_set.add(ConversationalTriplet(*trip))
        else:
            full_triplet_set = set()
            for key, triplet_list in rel_map.items():
                for trip in triplet_list:
                    curr_subj = key
                    for i in range(len(trip) // 2):
                        full_triplet_set.add(
                            Triplet(curr_subj, trip[2 * i], trip[2 * i + 1])
                        )
                        curr_subj = trip[2 * i + 1]

        return full_triplet_set

    def retrieve_related_documents(
        self, triplet: Union[Triplet, ConversationalTriplet]
    ) -> Dict[str, NodeWithScore]:

        triplet_embedding = self.kg_index.index_struct.embedding_dict.get(str(triplet))
        if triplet_embedding is None:
            node_list = self.doc_retriever.retrieve(QueryBundle(query_str=str(triplet)))
        else:
            node_list = self.doc_retriever.retrieve(
                QueryBundle(query_str="", embedding=triplet_embedding)
            )

        return {node_doc.node.node_id: node_doc for node_doc in node_list}

    def custom_query(self, query_str: str):

        self.curr_time_profile = {}
        start_time = time.time()
        query_bundle = QueryBundle(query_str=query_str)

        # Retrieve starting nodes from KG corpus
        query_nodes: Set[Triplet] = self.retrieve_kg_nodes(query_bundle)

        curr_time = time.time()
        self.curr_time_profile["NODE RETRIEVAL"] = curr_time - start_time
        start_time = curr_time

        self.curr_time_profile["NODE TRAVERSAL"] = 0.0

        self.curr_time_profile["DOCUMENT RETRIEVAL"] = 0.0
        self.curr_time_profile["READ FROM CACHE"] = 0.0
        self.curr_time_profile["CACHE RETRIEVAL"] = 0.0
        self.curr_time_profile["WRITE TO CACHE"] = 0.0

        doc_nodes: Dict[str, NodeWithScore] = dict()

        # Retrieve documents for each node
        for triplet in query_nodes:
            doc_nodes = merge_node_doc_dicts(
                doc_nodes,
                self.retrieve_kg_documents_for_node(triplet.subject, self.traverse_k),
            )
            doc_nodes = merge_node_doc_dicts(
                doc_nodes,
                self.retrieve_kg_documents_for_node(triplet.object, self.traverse_k),
            )

        start_time = time.time()

        self.curr_time_profile["NUM TOTAL DOCUMENTS"] = len(doc_nodes)

        """# ReRank as needed
        ranked_nodes = self.reranker.postprocess_nodes(
            list(doc_nodes.values()), query_bundle
        )
        # Would sometimes result in no docs returning.
        # ranked_nodes = list(filter(lambda node: (node.score > 0), ranked_nodes))

        curr_time = time.time()
        self.curr_time_profile["NODE RANKING"] = curr_time - start_time"""

        ranked_nodes = list(doc_nodes.values())
        ranked_nodes.sort(key=lambda node: node.get_score(), reverse=True)
        ranked_nodes = ranked_nodes[: self.return_n]

        self.time_profile.append(self.curr_time_profile)

        # Respond to query
        response_obj = self.response_synthesizer.synthesize(query_bundle, ranked_nodes)
        return response_obj

    def serialize_time_profile(self):

        with open(self.file_name, "w") as f:
            json.dump(self.time_profile, f, indent=4)


class RAGCachedStringQueryEngine(RAGStringQueryEngine):

    def __init__(
        self,
        llm,
        kg_index,
        doc_index,
        traverse_k,
        retrieve_k,
        return_n,
        cache_size="0",
        cache_args={},
    ):
        super().__init__(llm, kg_index, doc_index, traverse_k, retrieve_k, return_n)
        self.cache_size = cache_size
        if cache_args == {}:
            self.cache = RedisGraphCache(cache_size=self.cache_size)
        else:
            self.cache = RedisGraphCache(cache_size=self.cache_size, **cache_args)

        self.file_name = f"{self.prefix_dir}/cache_{self.dataset_name}_traverse_k={self.traverse_k}_retrieve_k={self.retrieve_k}_return_n={self.return_n}_cache_size={self.cache_size}_time_profile.json"
        self.redis_file_name = f"{self.prefix_dir}/cache_{self.dataset_name}_traverse_k={self.traverse_k}_retrieve_k={self.retrieve_k}_return_n={self.return_n}_cache_size={self.cache_size}_redis.json"

    def retrieve_kg_documents_for_node(
        self, node: Node, k: int
    ) -> Dict[TextNode, NodeWithScore]:

        start_time = time.time()
        cache_entry = self.cache.read_entry_from_cache(node)

        curr_time = time.time()
        self.curr_time_profile["READ FROM CACHE"] += curr_time - start_time
        start_time = curr_time

        if cache_entry is not None:
            serial_node_docs = json.loads(cache_entry)
            serial_node_docs = {
                k: SerialNodeWithScore.from_dict(v) for k, v in serial_node_docs.items()
            }

            node_docs = {
                node_with_doc.node: node_with_doc.node_with_score
                for k, node_with_doc in serial_node_docs.items()
            }

            curr_time = time.time()
            self.curr_time_profile["CACHE RETRIEVAL"] += curr_time - start_time
            start_time = curr_time

            return node_docs
        else:
            node_docs = super().retrieve_kg_documents_for_node(node, k)

            start_time = time.time()
            serial_node_docs = {
                k: SerialNodeWithScore(node=node.text, score=node.score).__dict__
                for k, node in node_docs.items()
            }
            self.cache.write_entry_to_cache(node, json.dumps(serial_node_docs))

            curr_time = time.time()
            self.curr_time_profile["WRITE TO CACHE"] += curr_time - start_time
            return node_docs

    def serialize_time_profile(self):
        super().serialize_time_profile()

        with open(self.redis_file_name, "w") as f:
            json.dump(self.get_info(), f, indent=4)

    def get_info(self):
        info_dict = self.cache.redis_cache.info()

        hits = info_dict.get("keyspace_hits", 0)
        misses = info_dict.get("keyspace_misses", 0)
        if hits + misses != 0:
            hit_rate = hits / (hits + misses)
            info_dict["hit_rate"] = hit_rate
        return info_dict

    def reset_info(self):
        self.cache.redis_cache.flushall()
        self.cache.redis_cache.config_resetstat()


class SemanticCachedStringQueryEngine(RAGStringQueryEngine):
    def __init__(
        self,
        llm,
        kg_index,
        doc_index,
        traverse_k,
        retrieve_k,
        return_n,
        dataset_name="wikipedia",
        prefix_dir="./data",
        json_file="cache_file.json",
        threshold=0.35,
    ):
        super().__init__(
            llm,
            kg_index,
            doc_index,
            traverse_k,
            retrieve_k,
            return_n,
            dataset_name,
            prefix_dir,
        )
        self.threshold = threshold
        self.cache = SemanticCache(json_file=json_file, threshold=self.threshold)
        self.cache_hits = 0
        self.file_name = f"{self.prefix_dir}/semantic_cache_{self.dataset_name}_traverse_k={self.traverse_k}_retrieve_k={self.retrieve_k}_return_n={self.return_n}_threshold={self.threshold}_time_profile.json"

    def custom_query(self, query_str: str):
        query_bundle = QueryBundle(query_str=query_str)
        self.curr_time_profile = {}
        start_time = time.time()
        cache_fetch_time = 0

        try:
            # First we obtain the embeddings corresponding to the user question
            embedding = self.cache.encoder.encode([query_str])

            # Search for the nearest neighbor in the index
            self.cache.index.nprobe = 8
            D, I = self.cache.index.search(embedding, 1)

            if D[0] >= 0:
                if I[0][0] >= 0 and D[0][0] <= self.cache.euclidean_threshold:
                    print("Cache Hit!")
                    row_id = int(I[0][0])
                    self.cache_hits += 1
                    curr_time = time.time()
                    cache_fetch_time = curr_time - start_time

                    serial_node_docs = self.cache.cache["node_docs"][row_id]
                    serial_node_docs = {
                        k: SerialNodeWithScore.from_dict(v)
                        for k, v in serial_node_docs.items()
                    }

                    node_docs = {
                        node_with_doc.node: node_with_doc.node_with_score
                        for k, node_with_doc in serial_node_docs.items()
                    }

                    ranked_nodes = list(node_docs.values())
                    ranked_nodes.sort(key=lambda node: node.get_score(), reverse=True)
                    ranked_nodes = ranked_nodes[: self.return_n]

                    response_obj = self.response_synthesizer.synthesize(
                        query_bundle, ranked_nodes
                    )

                    self.curr_time_profile["SEMANTIC CACHE HITS"] = self.cache_hits
                    self.curr_time_profile["CACHE RETRIEVAL"] = cache_fetch_time
                    self.curr_time_profile["NODE RETRIEVAL"] = 0.0
                    self.curr_time_profile["NODE TRAVERSAL"] = 0.0
                    self.curr_time_profile["DOCUMENT RETRIEVAL"] = 0.0
                    self.curr_time_profile["READ FROM CACHE"] = 1.0
                    self.curr_time_profile["WRITE TO CACHE"] = 0.0
                    self.curr_time_profile["NUM TOTAL DOCUMENTS"] = len(node_docs)

                    self.time_profile.append(self.curr_time_profile)

                    return response_obj

            # Retrieve starting nodes from KG corpus
            query_nodes: Set[Triplet] = self.retrieve_kg_nodes(query_bundle)

            curr_time = time.time()
            self.curr_time_profile["NODE RETRIEVAL"] = curr_time - start_time
            start_time = curr_time

            self.curr_time_profile["NODE TRAVERSAL"] = 0.0

            self.curr_time_profile["DOCUMENT RETRIEVAL"] = 0.0
            self.curr_time_profile["READ FROM CACHE"] = 0.0
            self.curr_time_profile["CACHE RETRIEVAL"] = 0.0
            self.curr_time_profile["WRITE TO CACHE"] = 0.0

            doc_nodes: Dict[str, NodeWithScore] = dict()

            # Retrieve documents for each node
            for triplet in query_nodes:
                doc_nodes = merge_node_doc_dicts(
                    doc_nodes, self.retrieve_kg_documents_for_node(triplet.subject, 3)
                )
                doc_nodes = merge_node_doc_dicts(
                    doc_nodes, self.retrieve_kg_documents_for_node(triplet.object, 3)
                )

            start_time = time.time()

            self.curr_time_profile["NUM TOTAL DOCUMENTS"] = len(doc_nodes)

            # ReRank as needed
            # ranked_nodes = self.reranker._postprocess_nodes(
            #     list(doc_nodes.values()), query_bundle
            # )
            # Would sometimes result in no docs returning.
            # ranked_nodes = list(filter(lambda node: (node.score > 0), ranked_nodes))

            # Respond to query
            ranked_nodes = list(doc_nodes.values())
            ranked_nodes.sort(key=lambda node: node.get_score(), reverse=True)
            ranked_nodes = ranked_nodes[: self.return_n]

            self.curr_time_profile["SEMANTIC CACHE HITS"] = self.cache_hits
            self.curr_time_profile["CACHE RETRIEVAL"] = cache_fetch_time
            self.time_profile.append(self.curr_time_profile)

            # Save index embedding
            self.cache.index.add(embedding)

            self.cache.cache["query_str"].append(query_str)
            serial_node_docs = {
                k: SerialNodeWithScore(node=node.text, score=node.score).__dict__
                for k, node in doc_nodes.items()
            }
            self.cache.cache["node_docs"].append(serial_node_docs)
            self.cache.store_cache(self.cache.json_file, self.cache.cache)

            response_obj = self.response_synthesizer.synthesize(
                query_bundle, ranked_nodes
            )
            return response_obj

        except Exception as e:
            raise RuntimeError("Error during 'custom_query' method: ", e)
