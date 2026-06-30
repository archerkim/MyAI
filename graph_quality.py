"""
graph_quality.py — метрика «разумности» графа знаний.

Композитный балл [0..1] из пяти ИНТРИНСИВНЫХ под-метрик (эталон не нужен — всё
считается из самого графа) плюс опциональный LLM-судья на фактическую точность.

Под-метрики (каждая 0..1):
  1. consistency   — логическая согласованность: доля рёбер, НЕ участвующих в
                     нарушениях (циклы is_a/part_of, самоссылки, opposite_of-
                     противоречия). Считается через reasoner.validate/contradictions.
  2. connectivity  — доля узлов в крупнейшей компоненте связности (граф знаний
                     должен быть связной паутиной, а не россыпью островов).
  3. atomicity     — доля «чистых» атомарных концептов (1–3 слова, не boilerplate),
                     а не фраз/мусора. Фразы-узлы не склеиваются → рвут граф.
  4. relational    — нормированная энтропия распределения отношений: используется
                     ли весь словарь сбалансированно, или всё схлопнулось в
                     related_to (тогда энтропия низкая).
  5. nonredundancy — 1 − доля узлов-дублей (ед./мн. и т.п., по канонизации).

Композит = взвешенная сумма (веса см. _WEIGHTS). Это НЕ абсолютная истина, а
воспроизводимый индикатор: позволяет сравнивать версии пайплайна и ловить
регрессии. Фактическую достоверность измеряет отдельный LLM-судья (--judge N).
"""

import argparse
import math
import sys
from collections import Counter, defaultdict

from brain_core_wrapper_local import INT_TO_LABEL, ALLOWED_RELATIONS
from extractor import _canonicalize_concept, MAX_CONCEPT_WORDS
from reasoner import validate, find_contradictions

_WEIGHTS = {  # сумма = 1.0
    "consistency": 0.25,
    "connectivity": 0.20,
    "atomicity": 0.25,
    "relational": 0.15,
    "nonredundancy": 0.15,
}
_JUNK_MARKERS = ("javascript", "console", "browser", "cookie", "helping", "http",
                 "message", "error", "page", "consideration", "information")


def _violating_edges(facts):
    """Множество рёбер, участвующих в логических нарушениях."""
    bad = set()
    for kind, who in validate(facts):
        if kind.startswith("self_reference"):
            bad.add(tuple(who[0]))
        elif kind.startswith("cycle"):
            rel = kind[kind.index("[") + 1:-1]
            for a, b in zip(who, who[1:]):       # рёбра вдоль цикла
                bad.add((a, rel, b))
    for kind, who in find_contradictions(facts):
        for t in who:
            if isinstance(t, tuple) and len(t) == 3:
                bad.add(t)
    return bad


def _largest_component_fraction(nodes, facts):
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for (s, _, o) in facts:
        ra, rb = find(s), find(o)
        if ra != rb:
            parent[ra] = rb
    sizes = Counter(find(n) for n in nodes)
    return (max(sizes.values()) / len(nodes)) if nodes else 0.0


def _is_clean_concept(c):
    w = c.split("_")
    if not (1 <= len(w) <= MAX_CONCEPT_WORDS):
        return False
    return not any(m in c for m in _JUNK_MARKERS)


def evaluate(facts):
    """facts — итерируемое (s,r,o). Возвращает отчёт-словарь с под-метриками,
    композитом и сырыми диагностиками."""
    facts = list(facts)
    nodes = {x for (s, _, o) in facts for x in (s, o)}
    E, N = len(facts), len(nodes)
    if E == 0 or N == 0:
        return {"composite": 0.0, "error": "пустой граф"}

    # 1. consistency
    bad = _violating_edges(facts)
    consistency = 1.0 - len(bad) / E

    # 2. connectivity
    connectivity = _largest_component_fraction(nodes, facts)

    # 3. atomicity
    clean = sum(_is_clean_concept(n) for n in nodes)
    atomicity = clean / N

    # 4. relational (нормированная энтропия)
    rc = Counter(r for (_, r, _) in facts)
    H = -sum((c / E) * math.log(c / E) for c in rc.values())
    relational = H / math.log(len(ALLOWED_RELATIONS))   # норм. на максимум словаря
    related_frac = rc.get("related_to", 0) / E

    # 5. nonredundancy
    canon = Counter(_canonicalize_concept(n) for n in nodes)
    collisions = sum(c - 1 for c in canon.values() if c > 1)
    nonredundancy = 1.0 - collisions / N

    subs = {"consistency": consistency, "connectivity": connectivity,
            "atomicity": atomicity, "relational": relational,
            "nonredundancy": nonredundancy}
    composite = sum(_WEIGHTS[k] * subs[k] for k in _WEIGHTS)

    # диагностики
    avg_deg = 2 * E / N
    deg = defaultdict(int)
    for (s, _, o) in facts:
        deg[s] += 1; deg[o] += 1
    leaf_frac = sum(1 for n in nodes if deg[n] == 1) / N

    return {
        "composite": round(composite, 3),
        "subscores": {k: round(v, 3) for k, v in subs.items()},
        "diagnostics": {
            "nodes": N, "edges": E, "avg_degree": round(avg_deg, 2),
            "leaf_fraction": round(leaf_frac, 2),
            "related_to_fraction": round(related_frac, 2),
            "violating_edges": len(bad),
            "duplicate_collisions": collisions,
        },
    }


def evaluate_brain(brain):
    from predictor import facts_from_brain
    return evaluate(facts_from_brain(brain))


def judge_precision(facts, sample=20, model="qwen2.5:7b", seed=0):
    """LLM-судья: доля СЛУЧАЙНЫХ триплетов, признанных истинными и корректными.
    Грубая оценка фактической точности. ВНИМАНИЕ: судит LLM — оценка приблизительна
    и смещена, если судья = генератор. Лучше брать судью сильнее генератора."""
    import random
    import ollama
    facts = list(facts)
    random.Random(seed).shuffle(facts)
    facts = facts[:sample]
    client = ollama.Client()
    ok = 0
    verdicts = []
    for (s, r, o) in facts:
        prompt = (f"Факт-триплет из графа знаний по физике: ({s}) --{r}--> ({o}).\n"
                  "Это утверждение ИСТИННО и корректно сформулировано? Ответь одним "
                  "словом: YES или NO.")
        resp = client.chat(model=model, options={"temperature": 0},
                           messages=[{"role": "user", "content": prompt}])
        v = "YES" in resp.message.content.strip().upper()[:5]
        ok += v
        verdicts.append(((s, r, o), "YES" if v else "NO"))
    return ok / len(facts) if facts else 0.0, verdicts


def _print_report(rep):
    print(f"=== КОМПОЗИТНЫЙ БАЛЛ: {rep['composite']:.3f} / 1.000 ===")
    print("под-метрики (вес):")
    for k, w in _WEIGHTS.items():
        print(f"  {k:14} {rep['subscores'][k]:.3f}   (вес {w})")
    print("диагностики:")
    for k, v in rep["diagnostics"].items():
        print(f"  {k:22} {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Метрика качества графа знаний.")
    ap.add_argument("module", help="путь к .brain")
    ap.add_argument("--judge", type=int, default=0,
                    help="оценить фактическую точность LLM-судьёй на N случайных рёбрах")
    ap.add_argument("--model", default="qwen2.5:7b")
    a = ap.parse_args()

    from brain_core_wrapper_local import BrainConnectionLocal
    from predictor import facts_from_brain
    b = BrainConnectionLocal(); b.connect(); b.mount_module(a.module, 1)
    facts = facts_from_brain(b)
    _print_report(evaluate(facts))
    if a.judge:
        prec, verdicts = judge_precision(facts, sample=a.judge, model=a.model)
        print(f"\n=== LLM-СУДЬЯ ({a.model}): фактическая точность ≈ {prec:.0%} "
              f"на {a.judge} рёбрах ===")
        for tri, v in verdicts:
            if v == "NO":
                print(f"  ✗ ({tri[0]}) --{tri[1]}--> ({tri[2]})")
    b.disconnect()
