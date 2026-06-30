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


if __name__ == "__main__":
    unittest.main(verbosity=2)
