"""
predictor.py — структурное предсказание рёбер (аналогия по окрестностям).

Идея из структурной теории графов (ср. Diestel, понятия окрестности вершины и
структурной эквивалентности): если две вершины имеют ПОХОЖИЕ типизированные
окрестности, то связь, присутствующая у одной, ВЕРОЯТНА и у другой.

Пример: motorcycle --has_property--> wheels. car и motorcycle разделяют
{(has_property,engine),(used_for,transport)} → структурно похожи → граф ВЫДВИГАЕТ
ГИПОТЕЗУ car --has_property--> wheels. Но ребро НЕ создаётся: гипотеза висит в
HypothesisStore и ПРОДВИГАЕТСЯ в граф только когда поглощаемая информация
подтвердит её (confirm). Так граф «предвосхищает», но не выдумывает факты.

Отличие от reasoner.py (FOL): там вывод ДЕДУКТИВНЫЙ и истинный (если посылки
истинны). Здесь — ИНДУКТИВНАЯ догадка по аналогии: она лишь правдоподобна,
поэтому требует внешнего подтверждения перед попаданием в граф.
"""

import json
from collections import defaultdict, namedtuple

from brain_core_wrapper_local import INT_TO_LABEL, ALLOWED_RELATIONS

# triple — кортеж (s,r,o); evidence — список (похожий_узел, similarity, его_ребро)
Conjecture = namedtuple("Conjecture", "triple score evidence")


class StructuralPredictor:
    """Считает структурное сходство узлов и выдвигает гипотезы-рёбра."""

    def __init__(self, facts):
        self.facts = set(facts)
        self.out = defaultdict(set)   # узел -> {(relation, object)} исходящих
        self.nodes = set()
        for (s, r, o) in self.facts:
            self.out[s].add((r, o))
            self.nodes.update((s, o))

    def similarity(self, a, b):
        """Жаккар по типизированным исходящим рёбрам. Возвращает (sim, shared)."""
        sa, sb = self.out.get(a, set()), self.out.get(b, set())
        if not sa or not sb:
            return 0.0, 0
        shared = len(sa & sb)
        return shared / len(sa | sb), shared

    def similar_nodes(self, node, min_shared=2, min_sim=0.0, top=10):
        """Узлы, структурно похожие на node (по общим типизированным рёбрам)."""
        res = []
        for other in self.nodes:
            if other == node:
                continue
            sim, shared = self.similarity(node, other)
            if shared >= min_shared and sim >= min_sim:
                res.append((other, sim, shared))
        return sorted(res, key=lambda x: -x[1])[:top]

    def conjectures(self, min_shared=2, min_sim=0.34, top_k=None):
        """Гипотезы-рёбра: для каждого узла A берём структурно похожие B и
        переносим рёбра B, которых нет у A. Скор = сумма сходств «поручителей».
        """
        cand = {}
        for a in self.nodes:
            if not self.out.get(a):
                continue
            sa = self.out[a]
            for b, sim, _ in self.similar_nodes(a, min_shared, min_sim):
                for (rel, obj) in self.out.get(b, set()) - sa:
                    if obj == a:
                        continue                       # без петель
                    tri = (a, rel, obj)
                    if tri in self.facts:
                        continue
                    d = cand.setdefault(tri, {"score": 0.0, "evidence": []})
                    d["score"] += sim
                    d["evidence"].append((b, round(sim, 2), (b, rel, obj)))
        out = [Conjecture(t, round(d["score"], 3), d["evidence"]) for t, d in cand.items()]
        out.sort(key=lambda c: -c.score)
        return out[:top_k] if top_k else out


class HypothesisStore:
    """Хранит ОТЛОЖЕННЫЕ гипотезы; продвигает их при подтверждении данными.

    Гипотезы НЕ в графе — это «ожидания». confirm(новые_триплеты) переводит
    совпавшие гипотезы в confirmed (их можно затем добавить в граф как обычные
    утверждённые рёбра).
    """

    def __init__(self):
        self.pending = {}     # triple -> Conjecture
        self.confirmed = []   # список triple

    def add(self, conjectures):
        for c in conjectures:
            # держим максимум скора, если гипотеза приходила повторно
            cur = self.pending.get(c.triple)
            if cur is None or c.score > cur.score:
                self.pending[c.triple] = c

    def confirm(self, new_triples):
        """Поглощаем новые факты: совпавшие с гипотезами — продвигаем.
        Возвращает список продвинутых Conjecture."""
        promoted = []
        for t in new_triples:
            t = tuple(t)
            if t in self.pending:
                promoted.append(self.pending.pop(t))
                self.confirmed.append(t)
        return promoted

    def top(self, n=10):
        return sorted(self.pending.values(), key=lambda c: -c.score)[:n]

    def save(self, path):
        json.dump({
            "pending": [[list(c.triple), c.score, c.evidence] for c in self.pending.values()],
            "confirmed": [list(t) for t in self.confirmed],
        }, open(path, "w"), ensure_ascii=False, indent=2)

    def load(self, path):
        d = json.load(open(path))
        self.pending = {tuple(t): Conjecture(tuple(t), s, ev) for t, s, ev in d["pending"]}
        self.confirmed = [tuple(t) for t in d["confirmed"]]
        return self


def facts_from_brain(brain):
    """Считывает факты (s,r,o) из смонтированного графа."""
    N = brain.get_stats()["node_count"]
    facts = set()
    for e in brain.get_subgraph_edges(list(range(N))):
        s = brain.get_lemma_by_id(e["from_id"]); o = brain.get_lemma_by_id(e["to_id"])
        r = INT_TO_LABEL.get(e["dep_type"])
        if s and o and r in ALLOWED_RELATIONS:
            facts.add((s, r, o))
    return facts


if __name__ == "__main__":
    # --- демонстрация на игрушечном примере «car / motorcycle / bicycle» ---
    facts = [
        ("motorcycle", "has_property", "wheels"),
        ("motorcycle", "has_property", "engine"),
        ("motorcycle", "used_for", "transport"),
        ("car", "has_property", "engine"),
        ("car", "has_property", "doors"),
        ("car", "used_for", "transport"),
        ("bicycle", "has_property", "wheels"),
        ("bicycle", "used_for", "transport"),
    ]
    pred = StructuralPredictor(facts)
    store = HypothesisStore()
    store.add(pred.conjectures(min_shared=1, min_sim=0.2))

    print("=== ГИПОТЕЗЫ (не в графе, ждут подтверждения) ===")
    for c in store.top():
        ev = ", ".join(f"{n}~{sim}" for n, sim, _ in c.evidence)
        print(f"  ? ({c.triple[0]}) --{c.triple[1]}--> ({c.triple[2]})  "
              f"score={c.score}  [по аналогии с: {ev}]")

    print("\n=== ПОГЛОЩАЕМ ФАКТ: (car, has_property, wheels) ===")
    promoted = store.confirm([("car", "has_property", "wheels")])
    for c in promoted:
        print(f"  ✓ ПОДТВЕРЖДЕНО и продвинуто: {c.triple}")
    print(f"  осталось гипотез: {len(store.pending)}")
