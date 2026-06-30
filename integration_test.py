"""
integration_test.py — сквозная проверка нейро-символической архитектуры на
готовом .brain-модуле: FOL-валидация + вывод + структурное предсказание +
двунаправленная активация + заземлённый запрос.

Это НЕ unittest, а демонстрация-прогон (нужны Ollama + собранный модуль).
Юнит-тесты логики — в test_brain_modules.py.

Запуск:
    python3 integration_test.py <module.brain> [--query "..."] [--model qwen2.5:7b]
"""

import argparse
import sys

from brain_core_wrapper_local import BrainConnectionLocal
from reasoner import (forward_chain, proof_tree, validate, find_contradictions,
                      GraphReasoner, DERIVED_MODULE_ID)
from predictor import StructuralPredictor, facts_from_brain
from verbalizer import format_subgraph_for_prompt


def _bidirectional_neighborhood(brain, edges, anchor, depth=2):
    adj = {}
    for e in edges:
        adj.setdefault(e["from_id"], set()).add(e["to_id"])
        adj.setdefault(e["to_id"], set()).add(e["from_id"])
    seen, frontier = {anchor}, {anchor}
    for _ in range(depth):
        nxt = set().union(*[adj.get(x, set()) for x in frontier]) - seen
        seen |= nxt; frontier = nxt
    return seen


def run(path, query, model):
    b = BrainConnectionLocal(); b.connect(); b.mount_module(path, 1)
    facts = facts_from_brain(b)
    print(f"### ГРАФ: {b.get_stats()}  (фактов-предикатов: {len(facts)})")

    print("\n### 1. ВАЛИДАЦИЯ СТРУКТУРЫ ###")
    issues = validate(facts)
    print(f"нарушений: {len(issues)}")
    for kind, who in issues[:6]:
        print(f"  ⚠ {kind}: {who}")

    print("\n### 2. ПРОТИВОРЕЧИЯ (opposite_of) ###")
    contra = find_contradictions(facts)
    print(f"противоречий: {len(contra)}")
    for kind, who in contra[:6]:
        print(f"  ⚠ {kind}: {who}")

    print("\n### 3. FOL FORWARD-CHAINING ###")
    derived, proofs = forward_chain(facts)
    print(f"выведено новых фактов: {len(derived)}")
    for f in derived[:3]:
        print(proof_tree(proofs, f)); print()
    n = GraphReasoner(b).materialize(derived)
    print(f"материализовано {n} рёбер (module_id={DERIVED_MODULE_ID}); граф теперь {b.get_stats()}")

    print("\n### 4. СТРУКТУРНЫЙ ПРЕДИКТОР (гипотезы, не в графе) ###")
    conj = StructuralPredictor(facts).conjectures(min_shared=1, min_sim=0.2, top_k=6)
    print(f"гипотез: {len(conj)}")
    for c in conj:
        ev = ", ".join(f"{nn}~{s}" for nn, s, _ in c.evidence[:2])
        print(f"  ? ({c.triple[0]}) --{c.triple[1]}--> ({c.triple[2]})  score={c.score}  [≈ {ev}]")

    print("\n### 5. ДВУНАПРАВЛЕННАЯ АКТИВАЦИЯ ###")
    qword = next((w for w in query.lower().replace("?", "").split()
                  if b.get_node_id(w) != -1), None)
    anchor = b.get_node_id(qword) if qword else 0
    b.reset_activations(); b.set_activation(anchor, 1.0)
    for _ in range(3):
        b.spread_activation_step([anchor])
    sig = b.get_significant_nodes()
    print(f"из '{b.get_lemma_by_id(anchor)}': "
          f"{[(s['lemma'], round(s['activation'],2)) for s in sig[:8]]}")

    print("\n### 6. ЗАЗЕМЛЁННЫЙ ЗАПРОС (включает выведенные рёбра) ###")
    import ollama
    all_e = b.get_subgraph_edges(list(range(b.get_stats()["node_count"])))
    nbrs = _bidirectional_neighborhood(b, all_e, anchor, depth=2)
    edges = sorted([e for e in all_e if e["from_id"] in nbrs and e["to_id"] in nbrs],
                   key=lambda e: -e["conductance"])[:16]
    sign = [{"id": x, "lemma": b.get_lemma_by_id(x)} for x in nbrs]
    print(f"вопрос: {query}\nдоказательная база: {len(edges)} рёбер")
    ev = format_subgraph_for_prompt(sign, edges)
    r = ollama.Client().chat(model=model, options={"temperature": 0}, messages=[
        {"role": "system", "content":
         "Объясняй СТРОГО по фрагменту графа ниже, только его факты, без внешних "
         "знаний; если пусто — 'В графе нет данных'. 2-3 предложения."},
        {"role": "user", "content": f"{ev}\n\nВопрос: {query}"}])
    print("ОТВЕТ:", r.message.content.strip())
    b.disconnect()
    print("\n### ИНТЕГРАЦИЯ ПРОЙДЕНА ###")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Сквозной прогон архитектуры на .brain-модуле.")
    ap.add_argument("module", help="путь к .brain")
    ap.add_argument("--query", default="How are atoms related to molecules and motion?")
    ap.add_argument("--model", default="qwen2.5:7b")
    a = ap.parse_args()
    run(a.module, a.query, a.model)
