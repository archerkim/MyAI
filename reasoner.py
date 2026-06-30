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


# ============================================================================
#  Расширения FOL: backward chaining, валидация, противоречия
# ============================================================================

def prove(goal, facts, max_depth=8):
    """Backward chaining: ДОКАЗАТЬ конкретную цель goal=(s,r,o), не материализуя
    весь вывод. Возвращает дерево-доказательство (вложенные кортежи) или None.

    Узел дерева: (fact, rule_or_None, [подцели]). rule=None — факт прямо в графе.
    Поддерживает те же правила, что и forward_chain.
    """
    factset = set(facts)
    index = {}
    for (s, r, o) in factset:
        index.setdefault((s, r), set()).add(o)
        index.setdefault((r, o), set()).add(s)  # обратный индекс по (relation, object)

    def _prove(g, depth, visiting):
        if g in factset:
            return (g, None, [])             # лист: факт уже в графе
        if depth <= 0 or g in visiting:
            return None                      # обрыв рекурсии / цикл
        s, r, o = g
        visiting = visiting | {g}

        # симметрия: r(s,o) ⇐ r(o,s)
        if r in SYMMETRIC:
            sub = _prove((o, r, s), depth - 1, visiting)
            if sub:
                return (g, f"symmetry[{r}]", [sub])

        # субсумпция: is_a(s,o) ⇐ example_of(s,o)
        if r == "is_a":
            sub = _prove((s, "example_of", o), depth - 1, visiting)
            if sub:
                return (g, "subsumption[example_of→is_a]", [sub])

        # транзитивность: r(s,o) ⇐ r(s,m) ∧ r(m,o)
        if r in TRANSITIVE:
            for m in index.get((s, r), ()):          # известные s --r--> m
                if m == o:
                    continue
                sub2 = _prove((m, r, o), depth - 1, visiting)
                if sub2:
                    return (g, f"transitivity[{r}]", [((s, r, m), None, []), sub2])

        # наследование: has_property(s,o) ⇐ (s via y) ∧ has_property(y,o)
        if r == "has_property":
            for via in INHERIT_VIA:
                for y in index.get((s, via), ()):
                    sub2 = _prove((y, "has_property", o), depth - 1, visiting)
                    if sub2:
                        return (g, f"inheritance[{via}]", [((s, via, y), None, []), sub2])
        return None

    return _prove(goal, max_depth, frozenset())


def render_proof(node, depth=0):
    """Текстовое дерево backward-доказательства."""
    if node is None:
        return "  " * depth + "НЕ ДОКАЗАНО"
    fact, rule, subs = node
    pad = "  " * depth
    line = f"{pad}({fact[0]}) --{fact[1]}--> ({fact[2]})"
    line += "  [факт из графа]" if rule is None else f"   ⇐ {rule}"
    return "\n".join([line] + [render_proof(s, depth + 1) for s in subs])


def _find_cycle(adj):
    """Возвращает один цикл (список узлов) в ориентированном adj, или None."""
    WHITE, GREY, BLACK = 0, 1, 2
    color, stack = {}, []

    def dfs(u):
        color[u] = GREY; stack.append(u)
        for v in adj.get(u, ()):  # noqa
            if color.get(v, WHITE) == GREY:
                return stack[stack.index(v):] + [v]
            if color.get(v, WHITE) == WHITE:
                c = dfs(v)
                if c:
                    return c
        color[u] = BLACK; stack.pop()
        return None

    for n in list(adj):
        if color.get(n, WHITE) == WHITE:
            c = dfs(n)
            if c:
                return c
    return None


def validate(facts):
    """Структурные нарушения логической согласованности графа.
    Возвращает список (тип, [причастные факты/узлы])."""
    facts = set(facts)
    issues = []
    # самоссылки в иерархических отношениях
    for (s, r, o) in facts:
        if s == o and r in ("is_a", "part_of", "causes"):
            issues.append((f"self_reference[{r}]", [(s, r, o)]))
    # циклы в is_a (таксономия) и part_of (мереология) — должны быть DAG
    for r in ("is_a", "part_of"):
        adj = {}
        for (s, rr, o) in facts:
            if rr == r and s != o:
                adj.setdefault(s, set()).add(o)
        cyc = _find_cycle(adj)
        if cyc:
            issues.append((f"cycle[{r}]", cyc))
    return issues


def find_contradictions(facts):
    """Противоречия через семантику opposite_of (взаимное исключение).
    Возвращает список (тип, [причастные факты])."""
    facts = set(facts)
    opp = {}
    for (s, r, o) in facts:
        if r == "opposite_of":
            opp.setdefault(s, set()).add(o)
            opp.setdefault(o, set()).add(s)
    issues, seen = [], set()

    def emit(kind, fs):
        key = (kind, frozenset(fs))
        if key not in seen:
            seen.add(key); issues.append((kind, list(fs)))

    # X противоположно самому себе
    for (s, r, o) in facts:
        if r == "opposite_of" and s == o:
            emit("self_opposite", [(s, r, o)])
    # is_a(X,Y) И opposite_of(X,Y) — нельзя быть видом того, чему противоположен
    for (s, r, o) in facts:
        if r == "is_a" and o in opp.get(s, ()):
            emit("is_a_vs_opposite", [(s, "is_a", o), (s, "opposite_of", o)])
    # Z обладает двумя ВЗАИМНО ПРОТИВОПОЛОЖНЫМИ свойствами
    props = {}
    for (s, r, o) in facts:
        if r == "has_property":
            props.setdefault(s, set()).add(o)
    for z, ps in props.items():
        for p1 in ps:
            for p2 in opp.get(p1, ()):
                if p2 in ps:
                    a, b = sorted((p1, p2))  # нормализуем порядок, чтобы не дублировать
                    emit("opposite_properties",
                         [(z, "has_property", a), (z, "has_property", b), (a, "opposite_of", b)])
    return issues


def sanitize(triples):
    """Отбрасывает рёбра, нарушающие логическую согласованность, ДО добавления в
    граф. Инкрементально держит is_a/part_of ацикличными (DAG) и не допускает
    прямых противоречий opposite_of. Возвращает (clean, dropped):
      clean   — список допущенных троек (порядок сохранён);
      dropped — список (тройка, причина).

    Порядок входа важен: при конфликте побеждает РАНЕЕ принятое ребро.
    """
    clean, dropped = [], []
    accepted = set()
    hier = {"is_a": {}, "part_of": {}}   # инкрементальные adjacency для DAG-проверки
    opp = set()                          # frozenset({s,o}) для opposite_of

    def reachable(adj, src, dst):
        """Достижим ли dst из src в adj (есть ли уже путь dst..src — тогда новое
        ребро src->dst замкнёт цикл)."""
        stack, seen = [src], set()
        while stack:
            x = stack.pop()
            if x == dst:
                return True
            if x in seen:
                continue
            seen.add(x)
            stack.extend(adj.get(x, ()))
        return False

    for (s, r, o) in triples:
        if s == o and r in ("is_a", "part_of", "causes"):
            dropped.append(((s, r, o), f"self_reference[{r}]")); continue
        if r in ("is_a", "part_of"):
            if reachable(hier[r], o, s):      # путь o..s уже есть → s->o замкнёт цикл
                dropped.append(((s, r, o), f"cycle[{r}]")); continue
        if r == "is_a" and frozenset((s, o)) in opp:
            dropped.append(((s, r, o), "is_a_vs_opposite")); continue
        if r == "opposite_of" and (s, "is_a", o) in accepted:
            dropped.append(((s, r, o), "opposite_vs_is_a")); continue

        # ребро принято — обновляем структуры
        if r in ("is_a", "part_of"):
            hier[r].setdefault(s, set()).add(o)
        if r == "opposite_of":
            opp.add(frozenset((s, o)))
        clean.append((s, r, o)); accepted.add((s, r, o))

    return clean, dropped


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
