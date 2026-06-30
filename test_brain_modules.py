"""
test_brain_modules.py — тесты персистентности, модулей знаний и устойчивости.

Запуск (внешних зависимостей не требует):
    python3 -m unittest test_brain_modules -v
или просто:
    python3 test_brain_modules.py

Перед запуском должна быть собрана brain_core.so:
    cc -shared -fPIC -O2 -o brain_core.so myAI.c
"""

import os
import struct
import tempfile
import unittest

from brain_core_wrapper_local import (
    BrainConnectionLocal,
    build_module_from_triples,
    relation_label_to_enum,
    INT_TO_LABEL,
    RELATION_TYPE_MAP,
    ALLOWED_RELATIONS,
)
from verbalizer import dep_type_to_string

# --- общие тестовые данные ---
PHYSICS = [
    ("force", "causes", "acceleration"),
    ("acceleration", "measured_in", "meter_per_second_squared"),
    ("mass_physics", "has_property", "inertia"),
]
BIOLOGY = [
    ("mass_biology", "is_a", "tissue"),
    ("force", "related_to", "muscle"),  # узел 'force' общий с PHYSICS
]


class TempBrainMixin(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="brain_test_")

    def tearDown(self):
        for name in os.listdir(self.tmp):
            os.remove(os.path.join(self.tmp, name))
        os.rmdir(self.tmp)

    def path(self, name):
        return os.path.join(self.tmp, name)

    def brain(self):
        b = BrainConnectionLocal()
        b.connect()
        self.addCleanup(b.disconnect)
        return b


class TestPersistence(TempBrainMixin):
    def test_round_trip_preserves_counts_and_weights(self):
        p = self.path("phys.brain")
        stats = build_module_from_triples(PHYSICS, p, module_id=1)
        self.assertEqual(stats["total_edge_count"], len(PHYSICS))

        # снимем эталонные веса из исходного мозга
        src = self.brain()
        src.set_module_id(1)
        for s, r, o in PHYSICS:
            src.add_triple(s, r, o)
        src.save(self.path("phys2.brain"))

        # загрузим в пустой мозг и сверим
        dst = self.brain()
        dst.load(p)
        self.assertEqual(dst.get_stats()["node_count"], stats["node_count"])
        self.assertEqual(dst.get_stats()["total_edge_count"], stats["total_edge_count"])

        fid = dst.get_node_id("force")
        self.assertNotEqual(fid, -1)
        edges = dst.get_subgraph_edges([fid, dst.get_node_id("acceleration")])
        causes = [e for e in edges if e["from_id"] == fid]
        self.assertEqual(len(causes), 1)
        # одно вхождение факта -> conductance == CONDUCTANCE_INCREMENT (0.1)
        self.assertAlmostEqual(causes[0]["conductance"], 0.1, places=4)
        self.assertEqual(causes[0]["dep_type"], relation_label_to_enum("causes"))

    def test_repeated_fact_accumulates_conductance(self):
        b = self.brain()
        b.set_module_id(1)
        b.add_triple("force", "causes", "acceleration")
        b.add_triple("force", "causes", "acceleration")  # повтор в том же модуле
        fid = b.get_node_id("force")
        edges = b.get_subgraph_edges([fid, b.get_node_id("acceleration")])
        e = [e for e in edges if e["from_id"] == fid][0]
        self.assertGreater(e["conductance"], 0.1)  # накопилось больше одного инкремента


class TestModules(TempBrainMixin):
    def setUp(self):
        super().setUp()
        self.p_phys = self.path("phys.brain")
        self.p_bio = self.path("bio.brain")
        build_module_from_triples(PHYSICS, self.p_phys, module_id=1)
        build_module_from_triples(BIOLOGY, self.p_bio, module_id=2)

    def test_mount_glues_shared_nodes_by_lemma(self):
        b = self.brain()
        b.mount_module(self.p_phys, 1)
        b.mount_module(self.p_bio, 2)
        # PHYSICS: force,acceleration,m/s^2,mass_physics,inertia = 5 узлов
        # BIOLOGY: mass_biology,tissue,force,muscle = 4, из них force общий
        self.assertEqual(b.get_stats()["node_count"], 8)
        self.assertEqual(b.get_stats()["total_edge_count"], 5)

    def test_count_module_edges(self):
        b = self.brain()
        b.mount_module(self.p_phys, 1)
        b.mount_module(self.p_bio, 2)
        self.assertEqual(b.count_module_edges(1), len(PHYSICS))
        self.assertEqual(b.count_module_edges(2), len(BIOLOGY))
        self.assertEqual(b.count_module_edges(999), 0)

    def test_unmount_removes_only_its_module(self):
        b = self.brain()
        b.mount_module(self.p_phys, 1)
        b.mount_module(self.p_bio, 2)
        removed = b.unmount_module(2)
        self.assertEqual(removed, len(BIOLOGY))
        self.assertEqual(b.count_module_edges(2), 0)
        self.assertEqual(b.count_module_edges(1), len(PHYSICS))  # модуль 1 цел

    def test_remount_is_idempotent(self):
        b = self.brain()
        b.mount_module(self.p_phys, 1)
        b.mount_module(self.p_phys, 1)  # повторное монтирование под тем же id
        self.assertEqual(b.count_module_edges(1), len(PHYSICS))  # не задвоилось


class TestEnumSync(TempBrainMixin):
    """Гарантирует, что словарь отношений Python синхронен с enum C-ядра:
    add_triple(rel) -> ребро с тем же int -> обратно та же строка."""

    def test_every_relation_round_trips_through_c_core(self):
        b = self.brain()
        b.set_module_id(1)
        for rel in ALLOWED_RELATIONS:
            subj, obj = f"s_{rel}", f"o_{rel}"
            b.add_triple(subj, rel, obj)
            sid, oid = b.get_node_id(subj), b.get_node_id(obj)
            self.assertNotEqual(sid, -1, f"узел {subj} не создан")
            edges = b.get_subgraph_edges([sid, oid])
            edge = [e for e in edges if e["from_id"] == sid][0]
            self.assertEqual(
                edge["dep_type"], RELATION_TYPE_MAP[rel],
                f"C-ядро вернуло другой int для отношения {rel}",
            )
            self.assertEqual(
                dep_type_to_string(edge["dep_type"]), rel,
                f"обратная карта INT_TO_LABEL рассинхронизирована для {rel}",
            )

    def test_int_to_label_covers_all_relations(self):
        for rel, code in RELATION_TYPE_MAP.items():
            self.assertEqual(INT_TO_LABEL.get(code), rel)


class TestCorruptFiles(TempBrainMixin):
    def test_bad_magic_raises_not_crashes(self):
        bad = self.path("bad.brain")
        with open(bad, "wb") as f:
            f.write(b"NOPE" + b"\x00" * 16)
        b = self.brain()
        with self.assertRaises(RuntimeError):
            b.load(bad)

    def test_oversized_lemma_length_rejected(self):
        bad = self.path("huge_lemma.brain")
        with open(bad, "wb") as f:
            f.write(b"BRN1")                       # magic
            f.write(struct.pack("<I", 1))          # version
            f.write(struct.pack("<I", 1))          # node_count = 1
            f.write(struct.pack("<Q", 0))          # edge_total = 0
            f.write(struct.pack("<I", 9_999_999))  # lemma len > BRAIN_MAX_LEMMA_LEN
        b = self.brain()
        with self.assertRaises(RuntimeError):
            b.load(bad)

    def test_truncated_edge_section_rejected(self):
        bad = self.path("truncated.brain")
        with open(bad, "wb") as f:
            f.write(b"BRN1")
            f.write(struct.pack("<I", 1))          # version
            f.write(struct.pack("<I", 1))          # node_count = 1
            f.write(struct.pack("<Q", 5))          # обещаем 5 рёбер...
            f.write(struct.pack("<I", 4))          # lemma len = 4
            f.write(b"node")                       # lemma
            # ...а рёбер не пишем вовсе -> fread должен упасть на первом
        b = self.brain()
        with self.assertRaises(RuntimeError):
            b.load(bad)


class TestReasoner(unittest.TestCase):
    """FOL forward-chaining (reasoner.py) — чистая логика, без графа."""

    def test_transitivity_multi_hop(self):
        from reasoner import forward_chain
        facts = [("a", "part_of", "b"), ("b", "part_of", "c"), ("c", "part_of", "d")]
        derived, _ = forward_chain(facts)
        ds = set(derived)
        self.assertIn(("a", "part_of", "c"), ds)
        self.assertIn(("b", "part_of", "d"), ds)
        self.assertIn(("a", "part_of", "d"), ds)  # вывод из выведенного (fixpoint)

    def test_property_inheritance(self):
        from reasoner import forward_chain
        facts = [("hydrogen", "is_a", "element"), ("element", "has_property", "mass")]
        derived, _ = forward_chain(facts)
        self.assertIn(("hydrogen", "has_property", "mass"), set(derived))

    def test_symmetry_and_subsumption(self):
        from reasoner import forward_chain
        facts = [("hot", "opposite_of", "cold"), ("water", "example_of", "liquid")]
        ds = set(forward_chain(facts)[0])
        self.assertIn(("cold", "opposite_of", "hot"), ds)   # симметрия
        self.assertIn(("water", "is_a", "liquid"), ds)      # субсумпция

    def test_no_self_loops_and_terminates(self):
        from reasoner import forward_chain
        # цикл a->b->a не должен порождать (a,part_of,a) и должен завершиться
        derived, _ = forward_chain([("a", "part_of", "b"), ("b", "part_of", "a")])
        self.assertNotIn(("a", "part_of", "a"), set(derived))
        self.assertNotIn(("b", "part_of", "b"), set(derived))

    def test_proof_is_traceable(self):
        from reasoner import forward_chain, proof_tree
        facts = [("a", "is_a", "b"), ("b", "is_a", "c")]
        _, proofs = forward_chain(facts)
        tree = proof_tree(proofs, ("a", "is_a", "c"))
        self.assertIn("transitivity[is_a]", tree)
        self.assertIn("факт из графа", tree)  # посылки — исходные факты


class TestFOLExtensions(unittest.TestCase):
    """Backward chaining, валидация, противоречия."""

    def test_backward_chaining_proves_and_fails(self):
        from reasoner import prove
        facts = [("a", "part_of", "b"), ("b", "part_of", "c")]
        self.assertIsNotNone(prove(("a", "part_of", "c"), facts))   # доказуемо
        self.assertIsNone(prove(("a", "part_of", "z"), facts))      # нет

    def test_validate_catches_cycle_and_self_ref(self):
        from reasoner import validate
        kinds = {k for k, _ in validate(
            [("a", "is_a", "b"), ("b", "is_a", "a"), ("x", "part_of", "x")])}
        self.assertIn("cycle[is_a]", kinds)
        self.assertIn("self_reference[part_of]", kinds)

    def test_contradiction_opposite_properties(self):
        from reasoner import find_contradictions
        kinds = {k for k, _ in find_contradictions(
            [("hot", "opposite_of", "cold"),
             ("stove", "has_property", "hot"), ("stove", "has_property", "cold")])}
        self.assertIn("opposite_properties", kinds)

    def test_sanitize_drops_cycle_keeps_first(self):
        from reasoner import sanitize
        # force is_a pressure (принято), pressure is_a force (замкнёт цикл → drop)
        clean, dropped = sanitize([("force", "is_a", "pressure"),
                                   ("pressure", "is_a", "force"),
                                   ("a", "part_of", "a")])
        self.assertIn(("force", "is_a", "pressure"), clean)
        self.assertNotIn(("pressure", "is_a", "force"), clean)  # цикл отброшен
        reasons = {r for _, r in dropped}
        self.assertIn("cycle[is_a]", reasons)
        self.assertIn("self_reference[part_of]", reasons)

    def test_sanitize_preserves_duplicates(self):
        from reasoner import sanitize
        # повторный факт усиливает проводимость — не должен отбрасываться
        clean, _ = sanitize([("x", "causes", "y"), ("x", "causes", "y")])
        self.assertEqual(clean.count(("x", "causes", "y")), 2)


class TestGraphQuality(unittest.TestCase):
    """Метрика качества графа должна различать чистый и мусорный граф."""

    def test_clean_scores_higher_than_messy(self):
        from graph_quality import evaluate
        clean = [("force", "causes", "acceleration"), ("acceleration", "is_a", "change"),
                 ("mass", "has_property", "inertia"), ("force", "measured_in", "newton"),
                 ("energy", "is_a", "quantity"), ("energy", "measured_in", "joule")]
        messy = [("a_very_long_phrase_concept_here", "related_to", "another_long_phrase_thing"),
                 ("javascript_console", "has_property", "messages"),
                 ("atoms", "is_a", "thing"), ("atom", "is_a", "thing"),  # дубль ед./мн.
                 ("x", "is_a", "y"), ("y", "is_a", "x")]                  # цикл
        sc_clean = evaluate(clean)["composite"]
        sc_messy = evaluate(messy)["composite"]
        self.assertGreater(sc_clean, sc_messy)

    def test_metrics_in_unit_range(self):
        from graph_quality import evaluate
        rep = evaluate([("a", "is_a", "b"), ("b", "part_of", "c")])
        for v in rep["subscores"].values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        self.assertGreaterEqual(rep["composite"], 0.0)
        self.assertLessEqual(rep["composite"], 1.0)


class TestCurriculum(unittest.TestCase):
    """Планировщик должен ставить основы перед зависимым материалом."""

    def test_orders_prerequisites_first(self):
        from curriculum import schedule
        # eng использует force (из physics), physics использует vector (из math)
        profiles = {
            "eng": {"defines": {"beam"}, "uses": {"beam", "force"}},
            "physics": {"defines": {"force"}, "uses": {"force", "vector"}},
            "math": {"defines": {"vector"}, "uses": {"vector"}},
        }
        order, _ = schedule(profiles)
        self.assertEqual(order, ["math", "physics", "eng"])

    def test_independent_docs_terminate(self):
        from curriculum import schedule
        order, _ = schedule({"a": {"defines": {"x"}, "uses": {"x"}},
                             "b": {"defines": {"y"}, "uses": {"y"}}})
        self.assertEqual(set(order), {"a", "b"})


class TestSynonyms(unittest.TestCase):
    """Языковой слой: синонимы схлопываются к канону → факты выравниваются."""

    def test_synonyms_map_to_canon(self):
        from synonyms import normalize_concept
        self.assertEqual(normalize_concept("jiggling"), "motion")
        self.assertEqual(normalize_concept("atomic_motion"), "motion")
        self.assertEqual(normalize_concept("everything"), "matter")

    def test_singular_plural_still_applies(self):
        from synonyms import normalize_concept
        self.assertEqual(normalize_concept("atoms"), "atom")  # дедуп до синонимов

    def test_equivalent_facts_align(self):
        from synonyms import normalize_triple
        a = normalize_triple(("atom", "has_property", "jiggling"))
        b = normalize_triple(("atom", "has_property", "motion"))
        self.assertEqual(a, b)  # разные формулировки → один факт → корроборация

    def test_build_map_picks_shortest_canon(self):
        from embed_synonyms import build_map
        m = build_map([["motion", "moving", "movement"], ["matter", "everything"]])
        self.assertEqual(m["moving"], "motion")     # канон = самый короткий
        self.assertEqual(m["everything"], "matter")


class TestStructuralPredictor(unittest.TestCase):
    """Аналогия по структуре + подтверждение перед материализацией."""

    FACTS = [
        ("motorcycle", "has_property", "wheels"),
        ("motorcycle", "has_property", "engine"),
        ("motorcycle", "used_for", "transport"),
        ("car", "has_property", "engine"),
        ("car", "used_for", "transport"),
    ]

    def test_conjectures_car_has_wheels(self):
        from predictor import StructuralPredictor
        triples = {c.triple for c in StructuralPredictor(self.FACTS)
                   .conjectures(min_shared=1, min_sim=0.2)}
        self.assertIn(("car", "has_property", "wheels"), triples)

    def test_confirmation_promotes_only_matching(self):
        from predictor import StructuralPredictor, HypothesisStore
        store = HypothesisStore()
        store.add(StructuralPredictor(self.FACTS).conjectures(min_shared=1, min_sim=0.2))
        self.assertIn(("car", "has_property", "wheels"), store.pending)
        promoted = store.confirm([("car", "has_property", "wheels"),
                                  ("car", "has_property", "nonexistent")])
        self.assertEqual(len(promoted), 1)
        self.assertNotIn(("car", "has_property", "wheels"), store.pending)  # продвинуто
        self.assertEqual(store.confirmed, [("car", "has_property", "wheels")])


class TestReflection(unittest.TestCase):
    """Рефлексия: эмерджентные выводы на стыке источников («1+1 > 2»)."""

    # Два источника. По отдельности ни один не выводит whale→needs_oxygen.
    FACTS = {
        ("whale", "is_a", "mammal"),
        ("mammal", "is_a", "animal"),
        ("mammal", "has_property", "warm_blooded"),   # целиком в источнике 1
        ("animal", "has_property", "needs_oxygen"),    # источник 2
    }
    SRC = {
        ("whale", "is_a", "mammal"): {1},
        ("mammal", "is_a", "animal"): {1},
        ("mammal", "has_property", "warm_blooded"): {1},
        ("animal", "has_property", "needs_oxygen"): {2},
    }

    def test_cross_source_insight_emerges(self):
        from reflection import synergistic_deductions
        ins, _ = synergistic_deductions(self.FACTS, self.SRC)
        triples = {i.triple for i in ins}
        # выводимо ТОЛЬКО объединением источников 1 и 2:
        self.assertIn(("whale", "has_property", "needs_oxygen"), triples)

    def test_intra_source_deduction_is_not_emergent(self):
        # warm_blooded целиком из источника 1 → НЕ эмерджентно, не должно всплыть
        from reflection import synergistic_deductions
        ins, _ = synergistic_deductions(self.FACTS, self.SRC)
        triples = {i.triple for i in ins}
        self.assertNotIn(("whale", "has_property", "warm_blooded"), triples)
        for i in ins:
            self.assertGreaterEqual(len(i.sources), 2)   # инвариант: всегда ≥2 источника

    def test_non_decomposability_single_source_yields_nothing(self):
        # Если весь граф из одного источника, эмерджентных выводов нет по определению
        from reflection import synergistic_deductions
        one_src = {t: {1} for t in self.FACTS}
        ins, _ = synergistic_deductions(self.FACTS, one_src)
        self.assertEqual(ins, [])

    def test_bridge_concept_is_the_junction(self):
        from reflection import bridge_concepts
        bridges = {b.node for b in bridge_concepts(self.FACTS, self.SRC)}
        self.assertIn("animal", bridges)   # узел, где сходятся источники 1 и 2
        self.assertNotIn("whale", bridges)  # whale инцидентен только источнику 1

    def test_convergent_needs_two_independent_witnesses(self):
        from reflection import convergent_conclusions
        # два независимых пути a→d: через b и через c
        facts = {("a", "is_a", "b"), ("b", "is_a", "d"),
                 ("a", "is_a", "c"), ("c", "is_a", "d")}
        src = {t: {1} for t in facts}
        conv = {c.triple: c for c in convergent_conclusions(facts, src)}
        self.assertIn(("a", "is_a", "d"), conv)
        self.assertEqual(conv[("a", "is_a", "d")].witnesses, frozenset({"b", "c"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
