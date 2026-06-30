import ctypes
import os
from pathlib import Path

# --- 1. настройка и загрузка библиотеки ---

# предполагаем, что .so файл находится в той же директории, что и этот wrapper
# или в корне проекта.
# этот код делает поиск более надежным.
try:
    # __file__ - это путь к текущему файлу (wrapper.py)
    _lib_path = Path(__file__).parent.resolve() / "brain_core.so"
    if not _lib_path.exists():
        # если не нашли, пробуем поискать на уровень выше (в корне проекта)
        _lib_path = Path(__file__).parent.parent.resolve() / "brain_core.so"
        if not _lib_path.exists():
            raise FileNotFoundError("could not find brain_core.so")

    brain_lib = ctypes.CDLL(str(_lib_path))
    print(f"[*] successfully loaded c-library from: {_lib_path}")

except (OSError, FileNotFoundError) as e:
    print(f"[!] critical error: could not load brain_core.so. please compile it first.")
    print(f"[!] error details: {e}")
    # если библиотека не найдена, дальнейшая работа бессмысленна.
    # в реальном приложении здесь был бы выход или более сложная обработка.
    brain_lib = None


# --- 2. воссоздание enum и структур из c ---

DEP_TYPE_MAP = {
    # Special
    "ROOT": 1,
    # Core Arguments
    "nsubj": 2, "dobj": 3, "iobj": 4, "csubj": 5, "ccomp": 6, "xcomp": 7,
    # Nominal Dependents
    "obl": 8, "vocative": 9, "expl": 10, "dislocated": 11, "nmod": 12, "appos": 13, "nummod": 14,
    # Non-core
    "advcl": 15, "advmod": 16, "discourse": 17,
    # Compounding
    "compound": 18, "fixed": 19, "flat": 20, "goeswith": 21,
    # Case/Prepositions
    "case": 22, "acl": 23, "amod": 24,
    # Coordination/Aux
    "aux": 25, "cop": 26, "conj": 27, "cc": 28,
    # Punctuation/Determiners
    "det": 29, "mark": 30, "punct": 31,
    # Special/Other
    "agent": 32, "attr": 33, "dative": 34, "oprd": 35, "predet": 36, "prep": 37,
    "nsubjpass": 38, "csubjpass": 39, "relcl": 40, "prt": 41, "intj": 42,
    "meta": 43, "neg": 44, "poss": 45, "pcomp": 46, "quantmod": 47,
    # Default
    "UNKNOWN": 0
}

def dep_label_to_enum(label):
    """конвертирует строку-метку зависимости spaCy в ее целочисленное значение enum."""
    return DEP_TYPE_MAP.get(label, 0)

# --- семантический словарь отношений (нейро-символический ингест) ---
# ДОЛЖЕН быть синхронизирован с enum REL_* в myAI.c (значения с 100).
RELATION_TYPE_MAP = {
    "is_a": 100,
    "part_of": 101,
    "has_property": 102,
    "causes": 103,
    "requires": 104,
    "enables": 105,
    "used_for": 106,
    "defined_as": 107,
    "measured_in": 108,
    "example_of": 109,
    "opposite_of": 110,
    "related_to": 111,
}

# список разрешённых отношений для промпта LLM-экстрактора (единый источник правды)
ALLOWED_RELATIONS = list(RELATION_TYPE_MAP.keys())

def relation_label_to_enum(label):
    """строка семантического отношения -> int enum (related_to как fallback)."""
    return RELATION_TYPE_MAP.get(label, RELATION_TYPE_MAP["related_to"])

# объединённая обратная карта int -> строка (для вербализации и инспекции)
INT_TO_LABEL = {v: k for k, v in DEP_TYPE_MAP.items()}
INT_TO_LABEL.update({v: k for k, v in RELATION_TYPE_MAP.items()})

# c-шная структура ParsedToken, воссозданная в ctypes
class ParsedTokenC(ctypes.Structure):
    _fields_ = [
        ("lemma", ctypes.c_char_p),
        ("head_offset", ctypes.c_int),
        ("dep_type", ctypes.c_int) # enum в c - это int в python
    ]

class ActivatedNodeC(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("activation", ctypes.c_float)
    ]

# c-шная структура GraphStats
class GraphStatsC(ctypes.Structure):
    _fields_ = [
        ("node_count", ctypes.c_uint32),
        ("total_edge_count", ctypes.c_uint64)
    ]

class EdgeInfoC(ctypes.Structure):
    _fields_ = [
        ("from_id", ctypes.c_uint32),
        ("to_id", ctypes.c_uint32),
        ("dep_type", ctypes.c_int),
        ("conductance", ctypes.c_float),
        ("module_id", ctypes.c_uint16)
    ]

# --- 3. объявление прототипов c-функций ---
if brain_lib:
    # handle (void*)
    brain_lib.brain_create.restype = ctypes.c_void_p

    brain_lib.brain_free.argtypes = [ctypes.c_void_p]

    brain_lib.brain_ingest_sentence.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ParsedTokenC),
        ctypes.c_size_t
    ]
    brain_lib.brain_ingest_sentence.restype = ctypes.c_int

    brain_lib.brain_get_stats.argtypes = [ctypes.c_void_p]
    brain_lib.brain_get_stats.restype = GraphStatsC

    brain_lib.brain_get_node_id.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    brain_lib.brain_get_node_id.restype = ctypes.c_int32

    brain_lib.brain_get_significant_nodes.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ActivatedNodeC)]
    brain_lib.brain_get_significant_nodes.restype = ctypes.c_size_t
    brain_lib.brain_spread_activation_step.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32), # указатель на массив uint32_t
        ctypes.c_size_t
    ]
    brain_lib.brain_spread_activation_step.restype = ctypes.c_int

    brain_lib.brain_get_lemma_by_id.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    brain_lib.brain_get_lemma_by_id.restype = ctypes.c_char_p

    brain_lib.brain_get_subgraph_edges.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(EdgeInfoC)]
    brain_lib.brain_get_subgraph_edges.restype = ctypes.c_size_t

    brain_lib.brain_reset_activations.argtypes = [ctypes.c_void_p]
    brain_lib.brain_reset_activations.restype = None

    brain_lib.brain_set_activation.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_float
    ]
    brain_lib.brain_set_activation.restype = ctypes.c_int

    # --- персистентность и модули знаний ---
    brain_lib.brain_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    brain_lib.brain_save.restype = ctypes.c_int

    brain_lib.brain_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    brain_lib.brain_load.restype = ctypes.c_int

    brain_lib.brain_merge_from_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint16]
    brain_lib.brain_merge_from_file.restype = ctypes.c_int

    brain_lib.brain_set_module_id.argtypes = [ctypes.c_void_p, ctypes.c_uint16]
    brain_lib.brain_set_module_id.restype = None

    brain_lib.brain_add_triple.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
    ]
    brain_lib.brain_add_triple.restype = ctypes.c_int

    brain_lib.brain_unmount_module.argtypes = [ctypes.c_void_p, ctypes.c_uint16]
    brain_lib.brain_unmount_module.restype = ctypes.c_size_t

    brain_lib.brain_count_module_edges.argtypes = [ctypes.c_void_p, ctypes.c_uint16]
    brain_lib.brain_count_module_edges.restype = ctypes.c_size_t
# --- 4. класс-обертка ---

class BrainConnectionLocal:
    """
    python-интерфейс для локальной c-библиотеки brain_core.
    управляет жизненным циклом указателя на граф.
    """
    def __init__(self):
        if not brain_lib:
            raise RuntimeError("c-library brain_core.so is not loaded. cannot create connection.")
        self.graph_ptr = None
    def is_connected(self):
        if not self.graph_ptr:
            return False
        return True
    def connect(self):
        """создает новый граф в памяти и сохраняет указатель на него."""
        if self.is_connected():
            print("[warning] already connected. disconnecting first.")
            self.disconnect()

        print("[wrapper] creating new brain instance via c-library...")
        self.graph_ptr = brain_lib.brain_create()
        if not self.graph_ptr:
            raise MemoryError("c-function brain_create() returned a null pointer.")

    def disconnect(self):
        """освобождает память, занятую графом."""
        if self.is_connected():
            print("[wrapper] freeing brain instance via c-library...")
            brain_lib.brain_free(self.graph_ptr)
            self.graph_ptr = None

    def ingest_sentence(self, tokens_data):
        """
        конвертирует python-список словарей в c-массив структур и передает в c-функцию.
        """
        if not self.is_connected():
            raise ConnectionError("not connected to brain. call connect() first.")

        num_tokens = len(tokens_data)
        if num_tokens == 0:
            return

        # создаем массив c-структур
        c_tokens_array = (ParsedTokenC * num_tokens)()

        # чтобы gc не удалил наши байтовые строки, пока c-код их читает,
        # мы должны держать их в живых в python-области видимости.
        string_buffers = []

        for i, token in enumerate(tokens_data):
            # конвертируем python str в c-шные `char*` (bytes)
            lemma_bytes = token['lemma'].encode('utf-8')
            string_buffers.append(lemma_bytes)

            c_tokens_array[i].lemma = lemma_bytes
            c_tokens_array[i].head_offset = token['head_offset']
            c_tokens_array[i].dep_type = dep_label_to_enum(token['dep'])

        # вызываем c-функцию
        result = brain_lib.brain_ingest_sentence(self.graph_ptr, c_tokens_array, num_tokens)
        if result != 0:
            raise RuntimeError(f"c-function brain_ingest_sentence() returned error code {result}")

    def get_stats(self):
        """вызывает c-функцию для получения статистики и возвращает ее как python-словарь."""
        if not self.is_connected():
            raise ConnectionError("not connected to brain. call connect() first.")

        stats_c = brain_lib.brain_get_stats(self.graph_ptr)

        return {
            "node_count": stats_c.node_count,
            "total_edge_count": stats_c.total_edge_count
        }

    def get_node_id(self, lemma):
        """находит id узла по лемме."""
        if not self.is_connected():
            raise ConnectionError("not connected to brain. call connect() first.")

        lemma_bytes = lemma.encode('utf-8')
        node_id = brain_lib.brain_get_node_id(self.graph_ptr, lemma_bytes)
        return node_id # вернет -1, если не найдено

    def __enter__(self):
        """позволяет использовать 'with' синтаксис."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """гарантирует, что disconnect будет вызван."""
        self.disconnect()

    def get_significant_nodes(self):
        if not self.is_connected():
            raise ConnectionError("not connected")
        max_nodes = self.get_stats()["node_count"]
        # создаем буфер для результатов
        results_array_type = ActivatedNodeC * max_nodes
        results_buffer = results_array_type()

        # вызываем c-функцию
        count = brain_lib.brain_get_significant_nodes(self.graph_ptr, max_nodes, results_buffer)

        # конвертируем c-структуры обратно в python-словари
        py_results = []
        for i in range(count):
            node_c = results_buffer[i]
            # здесь нам понадобится get_lemma_by_id, который мы уже спроектировали
            lemma = self.get_lemma_by_id(node_c.id)

            py_results.append({
                "id": node_c.id,
                "lemma": lemma,
                "activation": node_c.activation
            })

        return py_results

    def get_lemma_by_id(self, node_id):
        if not self.is_connected():
            raise ConnectionError("not connected")

        lemma_bytes = brain_lib.brain_get_lemma_by_id(self.graph_ptr, node_id)
        if lemma_bytes:
            return lemma_bytes.decode('utf-8')
        return None

    def spread_activation_step(self, anchor_ids):
        """
        вызывает c-функцию для выполнения одного шага распространения активации.
        """
        if not self.is_connected():
            raise ConnectionError("not connected")

        num_anchors = len(anchor_ids)
        if num_anchors == 0:
            # можно просто ничего не делать, или передать пустой массив
            # для чистоты вызовем с пустым массивом
            anchor_array = (ctypes.c_uint32 * 0)()
        else:
            anchor_array = (ctypes.c_uint32 * num_anchors)(*anchor_ids)

        result = brain_lib.brain_spread_activation_step(self.graph_ptr, anchor_array, num_anchors)

        if result != 0:
            raise RuntimeError(f"c-function brain_spread_activation_step() returned error code {result}")

    def get_subgraph_edges(self, node_ids):
        if not self.is_connected():
            raise ConnectionError("not connected")
        max_edges = self.get_stats()["total_edge_count"]
        num_nodes = len(node_ids)
        if num_nodes == 0: return []

        # передаем массив id в c
        nodes_array = (ctypes.c_uint32 * num_nodes)(*node_ids)

        # создаем буфер для результатов
        results_buffer = (EdgeInfoC * max_edges)()

        count = brain_lib.brain_get_subgraph_edges(self.graph_ptr, nodes_array, num_nodes, max_edges, results_buffer)

        # конвертируем в python dict'ы
        py_results = []
        for i in range(count):
            edge_c = results_buffer[i]
            py_results.append({
                "from_id": edge_c.from_id,
                "to_id": edge_c.to_id,
                "dep_type": edge_c.dep_type, # можно конвертировать в строку здесь
                "conductance": edge_c.conductance,
                "module_id": edge_c.module_id
            })
        return py_results
    def set_activation(self, node_id, value):
        if not self.is_connected():
            raise ConnectionError("not connected")
        return brain_lib.brain_set_activation(self.graph_ptr, node_id, value)
    def reset_activations(self):
        if not self.is_connected():
            raise ConnectionError("not connected")
        brain_lib.brain_reset_activations(self.graph_ptr)

    # --- персистентность и модули знаний ---
    def set_module_id(self, module_id):
        """тег, которым помечаются рёбра при последующем ингесте (0 = ядро)."""
        if not self.is_connected():
            raise ConnectionError("not connected")
        brain_lib.brain_set_module_id(self.graph_ptr, module_id)

    def save(self, path):
        """сохраняет весь текущий граф в бинарный файл модуля (.brain)."""
        if not self.is_connected():
            raise ConnectionError("not connected")
        rc = brain_lib.brain_save(self.graph_ptr, str(path).encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"brain_save() returned error code {rc}")

    def load(self, path):
        """загружает граф из файла В ПУСТОЙ мозг (точное восстановление весов)."""
        if not self.is_connected():
            raise ConnectionError("not connected")
        rc = brain_lib.brain_load(self.graph_ptr, str(path).encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"brain_load() returned error code {rc}")

    def merge_from_file(self, path, module_id):
        """монтирует модуль поверх текущего графа (склейка по леммам)."""
        if not self.is_connected():
            raise ConnectionError("not connected")
        rc = brain_lib.brain_merge_from_file(self.graph_ptr, str(path).encode("utf-8"), module_id)
        if rc != 0:
            raise RuntimeError(f"brain_merge_from_file() returned error code {rc}")

    def mount_module(self, path, module_id):
        """СОЕДИНЕНИЕ (идемпотентно): подключает модуль из файла под данным
        module_id. Если модуль с этим id уже смонтирован — сначала снимает его,
        чтобы повторное монтирование не накапливало проводимость."""
        if not self.is_connected():
            raise ConnectionError("not connected")
        self.unmount_module(module_id)
        self.merge_from_file(path, module_id)

    def unmount_module(self, module_id):
        """РАЗЪЕДИНЕНИЕ: снимает модуль (удаляет все его рёбра).
        Возвращает число удалённых рёбер."""
        if not self.is_connected():
            raise ConnectionError("not connected")
        return brain_lib.brain_unmount_module(self.graph_ptr, module_id)

    def count_module_edges(self, module_id):
        """сколько рёбер сейчас принадлежит модулю (0 = не смонтирован)."""
        if not self.is_connected():
            raise ConnectionError("not connected")
        return brain_lib.brain_count_module_edges(self.graph_ptr, module_id)

    def add_triple(self, subject, relation, obj):
        """нейро-символический ингест: (subject) --relation--> (object).
        relation — строка из ALLOWED_RELATIONS (или int enum)."""
        if not self.is_connected():
            raise ConnectionError("not connected")
        rel_int = relation if isinstance(relation, int) else relation_label_to_enum(relation)
        rc = brain_lib.brain_add_triple(
            self.graph_ptr,
            subject.encode("utf-8"),
            rel_int,
            obj.encode("utf-8"),
        )
        if rc != 0:
            raise RuntimeError(f"brain_add_triple() returned error code {rc}")


# --- утилиты уровня пакета знаний (без сервера и без ИИ) ---

def build_module_from_triples(triples, out_path, module_id=1):
    """Собирает пакет знаний из списка триплетов и сохраняет в .brain.

    triples — итерируемое из (subject, relation, object) либо dict с теми же
    ключами. relation — строка из ALLOWED_RELATIONS. Возвращает статистику.
    """
    brain = BrainConnectionLocal()
    brain.connect()
    try:
        brain.set_module_id(module_id)
        for t in triples:
            if isinstance(t, dict):
                subj, rel, obj = t["subject"], t["relation"], t["object"]
            else:
                subj, rel, obj = t
            brain.add_triple(subj, rel, obj)
        stats = brain.get_stats()
        brain.save(out_path)
        return stats
    finally:
        brain.disconnect()


def merge_module_files(input_paths, out_path):
    """СЛИЯНИЕ: объединяет несколько .brain-пакетов в один файл.

    Узлы склеиваются по каноническому концепту; рёбра и их веса сохраняются
    точно, исходные module_id каждого пакета остаются (так объединённый пакет
    можно потом разъединить обратно). Возвращает статистику итогового графа.
    """
    brain = BrainConnectionLocal()
    brain.connect()
    try:
        for path in input_paths:
            brain.load(path)  # load умеет дозагружать в непустой граф (append)
        stats = brain.get_stats()
        brain.save(out_path)
        return stats
    finally:
        brain.disconnect()
