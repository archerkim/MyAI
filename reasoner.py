"""
reasoner.py — first-order logic над графом знаний (forward-chaining).

Каждый триплет (subject, relation, object) — это БИНАРНЫЙ ПРЕДИКАТ:
    causes(force, acceleration),  part_of(nucleus, atom),  is_a(gravity, force).
Это уже атомарные факты фрагмента FOL. Здесь добавляется ВЫВОД — порождение
новых фактов из имеющихся по универсально-квантифицированным правилам (клаузам
Хорна / Datalog). Это разрешимый, завершимый фрагмент FOL: без кванторов
существования, дизъюнкции в голове и отрицания — то, что корректно
материализовать прямо в графе.

Правила (∀X,Y,Z,P — переменные):
  ТРАНЗИТИВНОСТЬ   is_a(X,Y) ∧ is_a(Y,Z)        → is_a(X,Z)
                   part_of(X,Y) ∧ part_of(Y,Z)  → part_of(X,Z)
                   causes(X,Y) ∧ causes(Y,Z)    → causes(X,Z)
  НАСЛЕДОВАНИЕ     is_a(X,Y) ∧ has_property(Y,P) → has_property(X,P)
                   example_of(X,Y) ∧ has_property(Y,P) → has_property(X,P)
  СИММЕТРИЯ        opposite_of(X,Y)             → opposite_of(Y,X)
                   related_to(X,Y)              → related_to(Y,X)
  СУБСУМПЦИЯ       example_of(X,Y)              → is_a(X,Y)

Вывод — forward chaining до НЕПОДВИЖНОЙ ТОЧКИ (пока появляются новые факты).
Каждый производный факт хранит ДОКАЗАТЕЛЬСТВО (правило + посылки), поэтому
любой вывод можно развернуть в дерево и проверить.

ВАЖНО: вывод СОХРАНЯЕТ ИСТИННОСТЬ, но не создаёт её — «мусор на входе → мусор на
выходе». Если в графе ошибочный факт (atom is_a molecule), транзитивность
размножит ошибку. Поэтому производные рёбра помечаются отдельным module_id и
отделимы (unmount).
"""

from brain_core_wrapper_local import INT_TO_LABEL, ALLOWED_RELATIONS

# module_id, которым помечаются ВЫВЕДЕННЫЕ рёбра — чтобы отличать их от
# утверждённых и снимать одним brain.unmount_module(DERIVED_MODULE_ID).
DERIVED_MODULE_ID = 60000

TRANSITIVE = ("is_a", "part_of", "causes")
SYMMETRIC = ("opposite_of", "related_to")
INHERIT_VIA = ("is_a", "example_of")  # (X via Y) ∧ has_property(Y,P) → has_property(X,P)


def forward_chain(facts, max_iters=10, max_derived=20000):
    """Прямой вывод до неподвижной точки.

    facts — итерируемое из троек (subject, relation, object).
    Возвращает (derived, proofs): derived — список НОВЫХ троек в порядке вывода;
    proofs — dict tройка → (имя_правила, [посылки]).
    """
    facts = set(facts)
    proofs, derived = {}, []

    for _ in range(max_iters):
        # индекс (subject, relation) → множество objects
        sr = {}
        for (s, r, o) in facts:
            sr.setdefault((s, r), set()).add(o)

        new = {}

        def add(cand, rule, premises):
            s, r, o = cand
            if s == o or cand in facts or cand in new:
                return  # без тривиальных петель и дублей
            new[cand] = (rule, premises)

        # транзитивность
        for r in TRANSITIVE:
            for (s, rr), objs in sr.items():
                if rr != r:
                    continue
                for o in objs:
                    for o2 in sr.get((o, r), ()):
                        add((s, r, o2), f"transitivity[{r}]", [(s, r, o), (o, r, o2)])

        # симметрия
        for r in SYMMETRIC:
            for (s, rr), objs in sr.items():
                if rr != r:
                    continue
                for o in objs:
                    add((o, r, s), f"symmetry[{r}]", [(s, r, o)])

        # наследование свойств
        for via in INHERIT_VIA:
            for (x, rr), ys in sr.items():
                if rr != via:
                    continue
                for y in ys:
                    for p in sr.get((y, "has_property"), ()):
                        add((x, "has_property", p), f"inheritance[{via}]",
                            [(x, via, y), (y, "has_property", p)])

        # субсумпция: example_of(X,Y) → is_a(X,Y)
        for (x, rr), ys in sr.items():
            if rr != "example_of":
                continue
            for y in ys:
                add((x, "is_a", y), "subsumption[example_of→is_a]", [(x, "example_of", y)])

        if not new:
            break  # неподвижная точка
        for cand, (rule, prem) in new.items():
            facts.add(cand)
            proofs[cand] = (rule, prem)
            derived.append(cand)
            if len(derived) >= max_derived:
                return derived, proofs

    return derived, proofs


def proof_tree(proofs, fact, _depth=0, _seen=None):
    """Разворачивает доказательство факта в текстовое дерево (для проверки)."""
    _seen = _seen or set()
    pad = "  " * _depth
    s, r, o = fact
    line = f"{pad}({s}) --{r}--> ({o})"
    if fact not in proofs or fact in _seen:
        return line + ("  [факт из графа]" if fact not in proofs else "  …")
    _seen = _seen | {fact}
    rule, premises = proofs[fact]
    out = [line + f"   ⇐ {rule}"]
    for p in premises:
        out.append(proof_tree(proofs, p, _depth + 1, _seen))
    return "\n".join(out)


class GraphReasoner:
    """Применяет FOL-вывод к смонтированному графу и материализует результат."""

    def __init__(self, brain):
        self.brain = brain

    def load_facts(self):
        N = self.brain.get_stats()["node_count"]
        edges = self.brain.get_subgraph_edges(list(range(N)))
        facts = set()
        for e in edges:
            s = self.brain.get_lemma_by_id(e["from_id"])
            o = self.brain.get_lemma_by_id(e["to_id"])
            r = INT_TO_LABEL.get(e["dep_type"])
            if s and o and r in ALLOWED_RELATIONS:
                facts.add((s, r, o))
        return facts

    def infer(self, **kw):
        return forward_chain(self.load_facts(), **kw)

    def materialize(self, derived):
        """Добавляет выведенные рёбра в граф под DERIVED_MODULE_ID.
        Их можно снять: brain.unmount_module(DERIVED_MODULE_ID)."""
        self.brain.set_module_id(DERIVED_MODULE_ID)
        try:
            for (s, r, o) in derived:
                self.brain.add_triple(s, r, o)
        finally:
            self.brain.set_module_id(0)
        return self.brain.count_module_edges(DERIVED_MODULE_ID)


if __name__ == "__main__":
    import sys
    from brain_core_wrapper_local import BrainConnectionLocal

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("usage: python3 reasoner.py <module.brain>")
        sys.exit(1)

    brain = BrainConnectionLocal(); brain.connect(); brain.mount_module(path, 1)
    r = GraphReasoner(brain)
    before = brain.get_stats()
    derived, proofs = r.infer()
    print(f"[reasoner] факты: {before['total_edge_count']} → выведено {len(derived)} новых")
    print("\n=== примеры выводов с доказательствами ===")
    for fact in derived[:8]:
        print(proof_tree(proofs, fact))
        print()
    n = r.materialize(derived)
    print(f"[reasoner] материализовано {n} рёбер под module_id={DERIVED_MODULE_ID}")
    print(f"[reasoner] граф теперь: {brain.get_stats()}")
    brain.disconnect()
