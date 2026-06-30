"""
embed_synonyms.py — АВТОМАТИЧЕСКОЕ построение языкового слоя (synonyms.py вручную)
через эмбеддинги nomic-embed-text + кластеризацию по близости.

ВАЖНЫЙ урок теста: ГОЛЫЕ слова не разделяют синонимы и просто связанные концепты
(motion~jiggling=0.58, но motion~heat=0.55, atom~molecule=0.68 — выше!). Эмбеддинг
слова меряет ТЕМАТИЧЕСКУЮ близость, не синонимию. Решение — эмбеддить концепт в
ОБЩЕМ КОНТЕКСТНОМ ШАБЛОНЕ (template): тогда общий контекст усиливает сигнал
синонимии («atoms are motion»~«atoms are jiggling»=0.85, vs «atoms are molecule»=
0.73). Порог ~0.80 разделяет чисто.

ОДНАКО даже с шаблоном кластеризация многих концептов нестабильна (общий шаблон
раздувает ВСЕ сходства → всё схлопывается). Поэтому РЕКОМЕНДУЕМЫЙ авто-метод —
group_by_llm(): LLM группирует синонимы по смыслу (точнее эмбеддингов, ~70%).
Самый надёжный — ручной словарь (synonyms.py). embed-метод оставлен как
задокументированный отрицательный результат.

Использование (для обучения с нуля — запускаешь сам):
    from embed_synonyms import group_by_llm, build_map
    groups = group_by_llm(concepts)          # рекомендуется
    syn_map = build_map(groups)              # вариант → канон, как в synonyms.py
"""

import argparse
import math
from collections import defaultdict


def _embed(concepts, template, model):
    import ollama
    client = ollama.Client()
    texts = [template.format(c.replace("_", " ")) for c in concepts]
    res = client.embed(model=model, input=texts)
    return dict(zip(concepts, res["embeddings"]))


def _cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return d / (na * nb) if na and nb else 0.0


def cluster_synonyms(concepts, threshold=0.80, template="{}",
                     model="nomic-embed-text"):
    """Кластеризует концепты по контекстным эмбеддингам. Возвращает (groups, vecs):
    groups — список групп-синонимов (≥2 элементов); vecs — карта эмбеддингов.

    template — контекст вокруг концепта (РЕКОМЕНДУЕТСЯ непустой, напр. 'atoms are {}'):
    голый '{}' плохо разделяет синонимы и связанные концепты.
    """
    concepts = list(dict.fromkeys(concepts))           # дедуп, порядок
    V = _embed(concepts, template, model)

    parent = {c: c for c in concepts}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, a in enumerate(concepts):
        for b in concepts[i + 1:]:
            if _cos(V[a], V[b]) >= threshold:
                parent[find(a)] = find(b)

    groups = defaultdict(list)
    for c in concepts:
        groups[find(c)].append(c)
    return [g for g in groups.values() if len(g) > 1], V


def group_by_llm(concepts, model="qwen2.5:7b"):
    """РЕКОМЕНДУЕМЫЙ авто-метод: LLM группирует синонимы ПО СМЫСЛУ.

    Точнее эмбеддингов: те меряют тематическую близость, а не синонимию, и при
    кластеризации либо недомёрживают (голые слова), либо схлопывают всё (общий
    шаблон). LLM понимает «одно и то же» vs «связанное». НО ~70%: может слить
    близкое (heat≈temperature) или раздробить (jiggling отдельно от motion) —
    стоит ревью или модель сильнее. Возвращает список групп (≥2 элементов).
    """
    import json
    import ollama
    prompt = (
        "Сгруппируй СИНОНИМЫ (концепты, означающие ОДНО И ТО ЖЕ) из списка. "
        "Разные, но связанные (atom vs molecule, heat vs temperature) — НЕ "
        "синонимы. Верни JSON: список групп (каждая — список концептов).\n"
        f"Концепты: {list(concepts)}"
    )
    schema = {"type": "object", "properties": {"groups": {"type": "array",
              "items": {"type": "array", "items": {"type": "string"}}}},
              "required": ["groups"]}
    r = ollama.Client().chat(model=model, options={"temperature": 0}, format=schema,
                             messages=[{"role": "user", "content": prompt}])
    return [g for g in json.loads(r.message.content)["groups"] if len(g) > 1]


def build_map(groups):
    """Группы → плоская карта вариант→канон (канон = самый короткий в группе)."""
    m = {}
    for g in groups:
        canon = min(g, key=len)
        for c in g:
            m[c] = canon
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Авто-построение синонимов через эмбеддинги.")
    ap.add_argument("module", help="путь к .brain (концепты берутся из него)")
    ap.add_argument("--method", choices=["llm", "embed"], default="llm",
                    help="llm (рекомендуется, точнее) или embed (эмбеддинги, шумно)")
    ap.add_argument("--threshold", type=float, default=0.80, help="для --method embed")
    ap.add_argument("--template", default="atoms are {}", help="для --method embed")
    ap.add_argument("--model", default=None, help="модель (по умолчанию по методу)")
    a = ap.parse_args()

    from brain_core_wrapper_local import BrainConnectionLocal
    b = BrainConnectionLocal(); b.connect(); b.mount_module(a.module, 1)
    N = b.get_stats()["node_count"]
    concepts = [b.get_lemma_by_id(i) for i in range(N)]
    b.disconnect()

    print(f"[*] {len(concepts)} концептов, метод={a.method}")
    if a.method == "llm":
        groups = group_by_llm(concepts, a.model or "qwen2.5:7b")
    else:
        groups, _ = cluster_synonyms(concepts, a.threshold, a.template,
                                     a.model or "nomic-embed-text")
    print(f"[*] найдено групп-синонимов: {len(groups)}")
    for g in sorted(groups, key=len, reverse=True):
        canon = min(g, key=len)
        print(f"  {canon}  ⇐  {', '.join(sorted(g))}")
