"""
reflection.py — РАЗМЫШЛЕНИЕ над уже существующим графом.

Ингест собирает факты. reasoner.py выводит из них истинные следствия, predictor.py
угадывает по аналогии. Этот слой отвечает на другой вопрос: ЧТО ГРАФ ПОНИМАЕТ
ТЕПЕРЬ, чего не содержал НИ ОДИН отдельный источник? Может ли сумма поглощённой
информации дать понимание БОЛЬШЕ, чем сумма источников по раздельности?

Формализация «целое больше суммы частей» (строгая и проверяемая):

    ЭМЕРДЖЕНТНЫЙ ВЫВОД — заключение C, которое выводимо из ОБЪЕДИНЕНИЯ фактов
    графа, но НЕ выводимо из фактов НИ ОДНОГО источника по отдельности.

Каждое ребро несёт провенанс (module_id), поэтому у каждого факта есть источник(и).
Вывод C эмерджентен ⟺ листья его доказательства (факты прямо в графе) охватывают
≥2 разных источника И C нет в дедуктивном замыкании ни одного отдельного источника.
Это ровно «1+1 > 2»: знание родилось НА СТЫКЕ документов, ни в одном не лежало.

Заземление сохранено: у каждого инсайта — дерево доказательства (reasoner.proofs)
и список источников-посылок. Рефлексия НИЧЕГО не выдумывает: она лишь делает явным
то, что объединение УЖЕ влечёт, но что было невидимо внутри отдельных источников.

Три оператора:
  1. synergistic_deductions — кросс-источниковые выводы (проверка неразложимости).
  2. bridge_concepts       — узлы на стыке ≥2 источников (точки спайки понимания).
  3. convergent_conclusions — выводы с ≥2 независимыми доказательствами (подтверждение).

Отличие от соседей: reasoner.py выводит ВСЕ следствия (не различая, синергетичны ли
они); predictor.py угадывает индуктивно. Здесь — ДЕДУКТИВНО (истинно), но фильтр на
ЭМЕРДЖЕНТНОСТЬ: оставляем только то, чего по отдельности не было.
"""

from collections import defaultdict, namedtuple

from brain_core_wrapper_local import INT_TO_LABEL, ALLOWED_RELATIONS
from reasoner import forward_chain, proof_tree, TRANSITIVE

# инсайт: derived — выведенная тройка; sources — множество module_id посылок;
# leaves — факты-листья доказательства; depth — глубина цепочки; score — синергия.
Insight = namedtuple("Insight", "triple sources leaves depth score")
Bridge = namedtuple("Bridge", "node sources fan score")
Convergence = namedtuple("Convergence", "triple witnesses sources score")


def _leaves(proofs, fact, _seen=None):
    """Множество фактов-ЛИСТЬЕВ доказательства fact — это факты прямо в графе
    (их нет в proofs). Именно у листьев есть источник-провенанс."""
    _seen = _seen or set()
    if fact in _seen:
        return set()
    if fact not in proofs:
        return {fact}            # лист: утверждённый факт графа
    _seen = _seen | {fact}
    out = set()
    _, premises = proofs[fact]
    for p in premises:
        out |= _leaves(proofs, p, _seen)
    return out


def _depth(proofs, fact, _seen=None):
    """Глубина доказательного дерева (длина самой длинной цепочки правил)."""
    _seen = _seen or set()
    if fact not in proofs or fact in _seen:
        return 0
    _seen = _seen | {fact}
    _, premises = proofs[fact]
    return 1 + max((_depth(proofs, p, _seen) for p in premises), default=0)


def synergistic_deductions(facts, fact_sources, max_iters=10):
    """Кросс-источниковые выводы: следствия объединения, которых нет ни в одном
    источнике по отдельности. Возвращает список Insight, отсортированный по синергии.

    facts        — множество троек (s,r,o) всего графа.
    fact_sources — dict тройка → set(module_id): откуда пришёл факт.

    Алгоритм:
      1. Полное замыкание forward_chain(все факты) → выводы + доказательства.
      2. Для каждого вывода берём источники листьев. Кандидат, если источников ≥2.
      3. ПРОВЕРКА НЕРАЗЛОЖИМОСТИ: считаем замыкание КАЖДОГО отдельного источника;
         если вывод попал хоть в одно одиночное замыкание — он НЕ эмерджентен (его
         даёт один источник), отбрасываем. Остаётся только истинно кросс-источниковое.
    """
    derived, proofs = forward_chain(facts, max_iters=max_iters)

    # дедуктивное замыкание каждого источника по отдельности — для проверки «1+1>2».
    per_source = defaultdict(set)
    for f, mods in fact_sources.items():
        for m in mods:
            per_source[m].add(f)
    single_closures = {}
    for m, fs in per_source.items():
        d, _ = forward_chain(fs, max_iters=max_iters)
        single_closures[m] = set(fs) | set(d)

    insights = []
    for fact in derived:
        leaves = _leaves(proofs, fact)
        sources = set()
        for lf in leaves:
            sources |= fact_sources.get(lf, set())
        if len(sources) < 2:
            continue                              # вывод внутри одного источника
        # неразложимость: вывода нет ни в одном одиночном замыкании
        if any(fact in single_closures.get(m, ()) for m in sources):
            continue
        depth = _depth(proofs, fact)
        # синергия: чем больше источников слито и длиннее цепочка — тем сильнее
        score = len(sources) * 10 + len(leaves) + depth
        insights.append(Insight(fact, sources, leaves, depth, score))

    insights.sort(key=lambda i: (-i.score, i.triple))
    return insights, proofs


def bridge_concepts(facts, fact_sources, min_sources=2):
    """Узлы-МОСТЫ: концепты, инцидентные рёбрам из ≥2 разных источников. Это точки,
    где знание из разных документов сходится на одной сущности — физические места
    спайки понимания. Скор = (число источников, разнообразие соседей).

    fan — сколько РАЗНЫХ соседей у узла суммарно (ширина стыка).
    """
    node_sources = defaultdict(set)   # узел → {module_id инцидентных рёбер}
    node_fan = defaultdict(set)       # узел → {соседи}
    for (s, r, o) in facts:
        mods = fact_sources.get((s, r, o), set())
        node_sources[s] |= mods
        node_sources[o] |= mods
        node_fan[s].add(o)
        node_fan[o].add(s)

    bridges = []
    for n, mods in node_sources.items():
        if len(mods) < min_sources:
            continue
        fan = len(node_fan[n])
        bridges.append(Bridge(n, mods, fan, len(mods) * 10 + fan))
    bridges.sort(key=lambda b: (-b.score, b.node))
    return bridges


def convergent_conclusions(facts, fact_sources):
    """Выводы, к которым ведёт ≥2 НЕЗАВИСИМЫХ доказательства. Несколько независимых
    свидетельств одного заключения — это подтверждение: уверенность растёт НЕ
    аддитивно (каждый независимый путь умножает правдоподобие). Здесь — для
    транзитивных отношений: is_a(s,o) подтверждается каждым промежуточным m, через
    которого s --r--> m --r--> o. Несколько разных m (особенно из разных источников)
    = конвергентное, устойчивое к ошибке одного источника заключение.
    """
    # индекс s --r--> o
    sr = defaultdict(set)
    for (s, r, o) in facts:
        sr[(s, r)].add(o)

    out = []
    for r in TRANSITIVE:
        # для каждой пары (s,o) собираем промежуточные свидетели m
        witnesses = defaultdict(set)
        for (s, rr), objs in sr.items():
            if rr != r:
                continue
            for m in objs:
                for o in sr.get((m, r), ()):
                    if o == s or o in objs:       # o уже прямой сосед s → не вывод
                        continue
                    witnesses[(s, r, o)].add(m)
        for tri, ms in witnesses.items():
            if len(ms) < 2:
                continue                          # один свидетель — не конвергенция
            s, r, o = tri
            srcs = set()
            for m in ms:
                srcs |= fact_sources.get((s, r, m), set())
                srcs |= fact_sources.get((m, r, o), set())
            # больше независимых свидетелей и источников → выше устойчивость
            score = len(ms) * 10 + len(srcs)
            out.append(Convergence(tri, frozenset(ms), srcs, score))
    out.sort(key=lambda c: (-c.score, c.triple))
    return out


class GraphReflector:
    """Размышление над смонтированным графом с учётом провенанса источников."""

    def __init__(self, brain):
        self.brain = brain

    def load(self):
        """Читает граф как (facts, fact_sources). fact_sources хранит ВСЕ module_id,
        под которыми встретилось ребро (одно и то же утверждение из разных
        источников = корроборация)."""
        N = self.brain.get_stats()["node_count"]
        edges = self.brain.get_subgraph_edges(list(range(N)))
        facts = set()
        fact_sources = defaultdict(set)
        for e in edges:
            s = self.brain.get_lemma_by_id(e["from_id"])
            o = self.brain.get_lemma_by_id(e["to_id"])
            r = INT_TO_LABEL.get(e["dep_type"])
            if s and o and r in ALLOWED_RELATIONS:
                tri = (s, r, o)
                facts.add(tri)
                fact_sources[tri].add(e["module_id"])
        return facts, fact_sources

    def reflect(self, max_iters=10):
        """Полный проход рефлексии. Возвращает dict с тремя видами инсайтов."""
        facts, fact_sources = self.load()
        insights, proofs = synergistic_deductions(facts, fact_sources, max_iters)
        return {
            "synergistic": insights,
            "proofs": proofs,
            "bridges": bridge_concepts(facts, fact_sources),
            "convergent": convergent_conclusions(facts, fact_sources),
            "n_facts": len(facts),
            "n_sources": len({m for ms in fact_sources.values() for m in ms}),
        }


def report(result, top=8):
    """Человекочитаемый отчёт рефлексии."""
    lines = [
        f"[reflection] фактов: {result['n_facts']}  источников: {result['n_sources']}",
        "",
        f"=== ЭМЕРДЖЕНТНЫЕ ВЫВОДЫ (нет ни в одном источнике по отдельности) "
        f"— {len(result['synergistic'])} ===",
    ]
    proofs = result["proofs"]
    for ins in result["synergistic"][:top]:
        s, r, o = ins.triple
        lines.append(f"\n  ⊕ ({s}) --{r}--> ({o})   синергия={ins.score}  "
                     f"источников={len(ins.sources)} {sorted(ins.sources)}")
        for ln in proof_tree(proofs, ins.triple).splitlines():
            lines.append("      " + ln)
    if not result["synergistic"]:
        lines.append("  (пока нет — нужно ≥2 источника с пересекающимися концептами)")

    lines.append(f"\n=== МОСТЫ-КОНЦЕПТЫ (стыки источников) — {len(result['bridges'])} ===")
    for b in result["bridges"][:top]:
        lines.append(f"  ◇ {b.node:24s} источники={sorted(b.sources)}  соседей={b.fan}")

    lines.append(f"\n=== КОНВЕРГЕНТНЫЕ ВЫВОДЫ (≥2 независимых пути) "
                 f"— {len(result['convergent'])} ===")
    for c in result["convergent"][:top]:
        s, r, o = c.triple
        lines.append(f"  ≡ ({s}) --{r}--> ({o})   через {len(c.witnesses)} свидетелей: "
                     f"{sorted(c.witnesses)}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1].endswith(".brain"):
        from brain_core_wrapper_local import BrainConnectionLocal
        brain = BrainConnectionLocal(); brain.connect()
        # можно смонтировать несколько модулей под разными id для кросс-источника
        for i, path in enumerate(sys.argv[1:], start=1):
            brain.mount_module(path, i)
        result = GraphReflector(brain).reflect()
        print(report(result))
        brain.disconnect()
    else:
        # --- демонстрация на игрушке: ДВА источника, инсайт рождается на стыке ---
        # Источник 1 (учебник биологии):   whale is_a mammal,  mammal is_a animal
        # Источник 2 (заметка про дыхание): animal has_property needs_oxygen
        # Ни один источник по отдельности не знает, что КИТУ нужен кислород.
        facts = {
            ("whale", "is_a", "mammal"),
            ("mammal", "is_a", "animal"),
            ("animal", "has_property", "needs_oxygen"),
            ("mammal", "has_property", "warm_blooded"),
        }
        fact_sources = {
            ("whale", "is_a", "mammal"): {1},
            ("mammal", "is_a", "animal"): {1},
            ("mammal", "has_property", "warm_blooded"): {1},
            ("animal", "has_property", "needs_oxygen"): {2},   # ← другой источник
        }
        result_insights, proofs = synergistic_deductions(facts, fact_sources)
        print("=== ЭМЕРДЖЕНТНЫЕ ВЫВОДЫ (1+1 > 2) ===")
        for ins in result_insights:
            s, r, o = ins.triple
            print(f"\n  ⊕ ({s}) --{r}--> ({o})   синергия={ins.score}  "
                  f"источников={sorted(ins.sources)}")
            print(proof_tree(proofs, ins.triple))
        print("\nЗаметь: 'warm_blooded' НЕ всплыло как инсайт — оно наследуется внутри "
              "\nодного источника (1), это не эмерджентность. 'needs_oxygen' для кита — "
              "\nэмерджентно: его даёт ТОЛЬКО стык источников 1 и 2.")
        print("\n=== МОСТЫ ===")
        for b in bridge_concepts(facts, fact_sources):
            print(f"  ◇ {b.node}: источники={sorted(b.sources)} соседей={b.fan}")
