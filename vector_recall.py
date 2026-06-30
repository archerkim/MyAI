"""
vector_recall.py — [research] семантический (векторный) recall концептов.

Закрывает пробел recall: символьный поиск находит концепт только по точному
имени/синониму. Векторный — по СМЫСЛУ: запрос «movement» достаёт «motion»,
«velocity», «acceleration», даже если в графе нет слова «movement».

ВАЖНО: ранее мы выяснили, что эмбеддинги nomic-embed путают синонимию и
связанность — и для СЛИЯНИЯ синонимов это плохо. Но для RECALL это РОВНО ТО, ЧТО
НУЖНО: при поиске мы хотим семантически близкие концепты, а не только точные
синонимы. Так что та же модель здесь применима по назначению.

Используется как дополнение к якорям talk.py: vector_anchors + lexical_anchors
→ окрестность графа → заземлённый ответ.
"""

import json
import math


class VectorRecall:
    def __init__(self, model="nomic-embed-text"):
        self.model = model
        self.vecs = {}                                    # concept -> embedding

    def _embed(self, texts):
        import ollama
        return ollama.Client().embed(model=self.model, input=texts)["embeddings"]

    def index(self, concepts):
        """Эмбеддит и запоминает новые концепты (идемпотентно)."""
        new = [c for c in dict.fromkeys(concepts) if c and c not in self.vecs]
        if new:
            for c, v in zip(new, self._embed([c.replace("_", " ") for c in new])):
                self.vecs[c] = v
        return len(new)

    @staticmethod
    def _cos(a, b):
        d = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
        return d / (na * nb) if na and nb else 0.0

    def search(self, query, k=5, threshold=0.0):
        """Топ-k концептов, семантически близких к запросу. Возвращает [(score, concept)]."""
        if not self.vecs:
            return []
        qv = self._embed([query.replace("_", " ")])[0]
        scored = sorted(((self._cos(qv, v), c) for c, v in self.vecs.items()),
                        reverse=True)
        return [(round(s, 3), c) for s, c in scored[:k] if s >= threshold]

    def save(self, path):
        json.dump({"model": self.model, "vecs": self.vecs}, open(path, "w"))

    def load(self, path):
        d = json.load(open(path))
        self.model = d["model"]; self.vecs = d["vecs"]
        return self


def index_brain(brain, model="nomic-embed-text"):
    """Строит VectorRecall по всем концептам смонтированного графа."""
    N = brain.get_stats()["node_count"]
    concepts = [brain.get_lemma_by_id(i) for i in range(N)]
    vr = VectorRecall(model)
    vr.index(concepts)
    return vr
