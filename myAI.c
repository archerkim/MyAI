/*
 * brain_core_v2.c
 * a refactored, library-ready core for the gsr knowledge graph.
 *
 * features:
 * - typed edges (dependency types).
 * - functions designed for external library calls (ffi).
 * - robust memory management and error handling.
 *
 * compile as a shared library:
 *   gcc -shared -o brain_core.so -fPIC brain_core_v2.c -lm
 *
 * requires uthash.h
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "uthash.h"

// --- tunable constants ---
#define INITIAL_NODE_CAPACITY 30000
#define INITIAL_EDGE_CAPACITY 450000
#define CONDUCTANCE_INCREMENT 0.1f

// --- data structures ---

// dependency types enum - this must be kept in sync with the python wrapper!
typedef enum DependencyType {
    // --- Special Types ---
    DEP_UNKNOWN = 0, // Default/error value
    DEP_ROOT,        // The root of the sentence

    // --- Core Arguments ---
    DEP_NSUBJ,       // Nominal subject
    DEP_DOBJ,        // Direct object
    DEP_IOBJ,        // Indirect object
    DEP_CSUBJ,       // Clausal subject
    DEP_CCOMP,       // Clausal complement
    DEP_XCOMP,       // Open clausal complement

    // --- Nominal Dependents ---
    DEP_OBL,         // Oblique nominal
    DEP_VOCATIVE,    // Vocative
    DEP_EXPL,        // Expletive
    DEP_DISLOCATED,  // Dislocated element
    DEP_NMOD,        // Nominal modifier
    DEP_APPOS,       // Appositional modifier
    DEP_NUMMOD,      // Numeric modifier

    // --- Non-core Dependents ---
    DEP_ADVCL,       // Adverbial clause modifier
    DEP_ADVMOD,      // Adverbial modifier
    DEP_DISCOURSE,   // Discourse element

    // --- Compounding and Unclassified ---
    DEP_COMPOUND,    // Compound
    DEP_FIXED,       // Fixed multiword expression
    DEP_FLAT,        // Flat multiword expression
    DEP_GOESWITH,    // Goes with

    // --- Case and Prepositions ---
    DEP_CASE,        // Case marker
    DEP_ACL,         // Adjectival clause
    DEP_AMOD,        // Adjectival modifier

    // --- Coordination and Auxiliaries ---
    DEP_AUX,         // Auxiliary
    DEP_COP,         // Copula
    DEP_CONJ,        // Conjunct
    DEP_CC,          // Coordinating conjunction

    // --- Punctuation and Determiners ---
    DEP_DET,         // Determiner
    DEP_MARK,        // Marker
    DEP_PUNCT,       // Punctuation

    // --- Special Relations (often from specific parsers) ---
    DEP_AGENT,       // Agent (semantic role)
    DEP_ATTR,        // Attribute
    DEP_DATIVE,      // Dative
    DEP_OPRD,        // Object predicate
    DEP_PREDET,      // Predeterminer
    DEP_PREP,        // Prepositional modifier

    // --- Relations for passives and relatives ---
    DEP_NSUBJPASS,   // Passive nominal subject
    DEP_CSUBJPASS,   // Passive clausal subject
    DEP_RELCL,       // Relative clause modifier

    // --- Other common ones from spaCy's English model ---
    DEP_PRT,         // Particle (e.g., "put UP the book")
    DEP_INTJ,        // Interjection
    DEP_META,        // Meta modifier
    DEP_NEG,         // Negation modifier
    DEP_POSS,        // Possession modifier
    DEP_PCOMP,       // Prepositional complement
    DEP_QUANTMOD,    // Quantifier phrase modifier

    // --- Our Custom Types (for future expansion) ---
    DEP_MATH_EQUALS,
    DEP_MATH_OPERAND,
    DEP_LOGIC_CAUSES,
    DEP_LOGIC_IMPLIES,

    // --- Semantic relation vocabulary (нейро-символический ингест) ---
    // Эти типы заполняет LLM-экстрактор триплетов (см. extractor.py + RELATION_TYPE_MAP).
    // Значения начинаются со 100, чтобы не пересекаться с синтаксическим диапазоном.
    REL_IS_A = 100,       // X является разновидностью Y (taxonomy)
    REL_PART_OF,          // X — часть Y (meronymy)
    REL_HAS_PROPERTY,     // X обладает свойством Y
    REL_CAUSES,           // X вызывает Y
    REL_REQUIRES,         // X требует Y
    REL_ENABLES,          // X делает возможным Y
    REL_USED_FOR,         // X используется для Y
    REL_DEFINED_AS,       // X определяется как Y
    REL_MEASURED_IN,      // X измеряется в Y
    REL_EXAMPLE_OF,       // X — пример Y
    REL_OPPOSITE_OF,      // X противоположно Y
    REL_RELATED_TO,       // X связано с Y (общая связь)

    // Sentinel value to know the total count if needed
    DEP_COUNT

} DependencyType;

typedef struct Edge {
    uint32_t to_node_id;
    float conductance;
    DependencyType dep_type;
    uint64_t last_activated_op_count;
    uint32_t repetition_counter;
    uint16_t module_id; // provenance: из какого модуля знаний пришло ребро (0 = ядро)
} Edge;

typedef struct Node {
    uint32_t id;
    char* lemma;
    float activation;
    Edge* edges;
    uint32_t edge_count;
    uint32_t edge_capacity;

    UT_hash_handle hh;
} Node;

typedef struct Graph {
    Node* nodes;
    uint32_t node_count;
    uint32_t node_capacity;

    Node* node_lookup_table;
    uint64_t global_op_counter;
    uint16_t current_module_id; // тег, которым помечаются новые рёбра при ингесте
} Graph;

// input structure for a single token from python
typedef struct ParsedToken {
    const char* lemma;
    int head_offset;
    DependencyType dep_type;
} ParsedToken;

typedef struct GraphStats {
    uint32_t node_count;
    uint64_t total_edge_count;
} GraphStats;

typedef struct ActivatedNode {
    uint32_t id;
    float activation;
} ActivatedNode;

typedef struct EdgeInfo {
    uint32_t from_id;
    uint32_t to_id;
    DependencyType dep_type;
    float conductance;
    uint16_t module_id;
} EdgeInfo;

uint32_t get_or_create_node(Graph* graph, const char* lemma);
void add_or_update_edge(Node* from_node, uint32_t to_id, DependencyType type, uint64_t op_counter, uint16_t module_id);

// --- public api functions (for ffi) ---

// we use a void* pointer to the Graph to hide implementation details from the caller (opaque pointer)
typedef void* BrainHandle;

BrainHandle brain_create() {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    if (!graph) return NULL;

    graph->node_capacity = INITIAL_NODE_CAPACITY;
    graph->nodes = (Node*)calloc(graph->node_capacity, sizeof(Node)); // calloc zeroes memory
    if (!graph->nodes) {
        free(graph);
        return NULL;
    }

    graph->node_count = 0;
    graph->node_lookup_table = NULL;
    graph->global_op_counter = 0;
    graph->current_module_id = 0;

    printf("[brain_core] new brain created.\n");
    return (BrainHandle)graph;
}

void brain_free(BrainHandle handle) {
    if (!handle) return;
    Graph* graph = (Graph*)handle;

    for (uint32_t i = 0; i < graph->node_count; ++i) {
        free(graph->nodes[i].lemma);
        free(graph->nodes[i].edges);
    }

    HASH_CLEAR(hh, graph->node_lookup_table);
    free(graph->nodes);
    free(graph);
    printf("[brain_core] brain has been freed.\n");
}

int brain_ingest_sentence(BrainHandle handle, const ParsedToken* sentence, size_t sentence_len) {
    if (!handle || !sentence || sentence_len == 0) return -1; // error code
    Graph* graph = (Graph*)handle;

    graph->global_op_counter++;

    // using a variable-length array (vla) - c99 feature
    uint32_t node_ids[sentence_len];

    // pass 1: get node ids
    for (size_t i = 0; i < sentence_len; ++i) {
        node_ids[i] = get_or_create_node(graph, sentence[i].lemma);
    }

    // pass 2: create typed, directional edges
    for (size_t i = 0; i < sentence_len; ++i) {
        if (sentence[i].head_offset != 0) {
            int head_idx = i + sentence[i].head_offset;
            if (head_idx >= 0 && head_idx < sentence_len) {
                uint32_t child_id = node_ids[i];
                uint32_t head_id = node_ids[head_idx];

                // dependency points from child to head
                Node* child_node = &graph->nodes[child_id];
                add_or_update_edge(child_node, head_id, sentence[i].dep_type, graph->global_op_counter, graph->current_module_id);

                // optionally add a reverse edge with a different type if needed
                // for now, we keep it directional to represent the dependency tree
            }
        }
    }

    return 0; // success
}

// --- internal (static) function implementations ---

uint32_t get_or_create_node(Graph* graph, const char* lemma) {
    Node* found_node;
    HASH_FIND_STR(graph->node_lookup_table, lemma, found_node);
    if (found_node) return found_node->id;

    if (graph->node_count >= graph->node_capacity) {
        graph->node_capacity *= 2;
        Node* new_nodes_ptr = (Node*)realloc(graph->nodes, sizeof(Node) * graph->node_capacity);
        if(!new_nodes_ptr) {
            // handle realloc failure, maybe exit or return an error
            perror("failed to reallocate nodes array");
            exit(EXIT_FAILURE);
        }
        graph->nodes = new_nodes_ptr;

        // rebuild hash table as realloc might have moved the memory block
        graph->node_lookup_table = NULL;
        for (uint32_t i = 0; i < graph->node_count; ++i) {
            HASH_ADD_KEYPTR(hh, graph->node_lookup_table, graph->nodes[i].lemma, strlen(graph->nodes[i].lemma), &graph->nodes[i]);
        }
    }

    uint32_t new_id = graph->node_count;
    Node* new_node = &graph->nodes[new_id];

    new_node->id = new_id;
    new_node->lemma = strdup(lemma);
    new_node->edges = NULL;
    new_node->edge_count = 0;
    new_node->activation = 0.0f;
    new_node->edge_capacity = 0;

    HASH_ADD_KEYPTR(hh, graph->node_lookup_table, new_node->lemma, strlen(new_node->lemma), new_node);

    graph->node_count++;
    return new_id;
}

void add_or_update_edge(Node* from_node, uint32_t to_id, DependencyType type, uint64_t op_counter, uint16_t module_id) {
    // Ребро принадлежит модулю: ключ дедупликации включает module_id.
    // Это значит, что одинаковый факт из РАЗНЫХ модулей хранится как
    // отдельные рёбра — каждый модуль можно снять (unmount) независимо.
    // Повтор факта ВНУТРИ одного модуля по-прежнему накапливает проводимость.
    for (uint32_t i = 0; i < from_node->edge_count; ++i) {
        if (from_node->edges[i].to_node_id == to_id &&
            from_node->edges[i].dep_type == type &&
            from_node->edges[i].module_id == module_id) {
            float increment = CONDUCTANCE_INCREMENT / (1.0f + from_node->edges[i].conductance);
            from_node->edges[i].conductance += increment;
            from_node->edges[i].repetition_counter++;
            from_node->edges[i].last_activated_op_count = op_counter;
            return;
        }
    }

    // if not, create a new edge
    if (from_node->edge_count >= from_node->edge_capacity) {
        from_node->edge_capacity = (from_node->edge_capacity == 0) ? INITIAL_EDGE_CAPACITY : from_node->edge_capacity * 2;
        Edge* new_edges_ptr = (Edge*)realloc(from_node->edges, sizeof(Edge) * from_node->edge_capacity);
         if(!new_edges_ptr) {
            perror("failed to reallocate edges array");
            exit(EXIT_FAILURE);
        }
        from_node->edges = new_edges_ptr;
    }

    Edge* new_edge = &from_node->edges[from_node->edge_count];
    new_edge->to_node_id = to_id;
    new_edge->dep_type = type;
    new_edge->conductance = CONDUCTANCE_INCREMENT;
    new_edge->repetition_counter = 1;
    new_edge->last_activated_op_count = op_counter;
    new_edge->module_id = module_id;

    from_node->edge_count++;
}

// устанавливает тег модуля для последующих операций ингеста.
void brain_set_module_id(BrainHandle handle, uint16_t module_id) {
    if (!handle) return;
    ((Graph*)handle)->current_module_id = module_id;
}

// нейро-символический ингест: добавляет один семантический триплет
// (subject) --rel--> (object). узлы создаются по канонической лемме-концепту.
// ребро помечается текущим module_id. возвращает 0 при успехе.
int brain_add_triple(BrainHandle handle, const char* subject,
                     DependencyType rel, const char* object) {
    if (!handle || !subject || !object) return -1;
    Graph* graph = (Graph*)handle;

    graph->global_op_counter++;
    uint32_t subj_id = get_or_create_node(graph, subject);
    uint32_t obj_id  = get_or_create_node(graph, object);

    add_or_update_edge(&graph->nodes[subj_id], obj_id, rel,
                       graph->global_op_counter, graph->current_module_id);
    return 0;
}

GraphStats brain_get_stats(BrainHandle handle) {
    GraphStats stats = {0, 0};
    if (!handle) return stats;
    Graph* graph = (Graph*)handle;

    stats.node_count = graph->node_count;
    for (uint32_t i = 0; i < graph->node_count; ++i) {
        stats.total_edge_count += graph->nodes[i].edge_count;
    }
    return stats;
}

int32_t brain_get_node_id(BrainHandle handle, const char* lemma) {
    if (!handle) return -1;
    Graph* graph = (Graph*)handle;
    Node* found_node;
    HASH_FIND_STR(graph->node_lookup_table, lemma, found_node);
    return found_node ? (int32_t)found_node->id : -1;
}

// вспомогательная функция для печати (не часть public api)
// этот switch должен быть синхронизирован с enum DependencyType и
// с DEP_TYPE_MAP в brain_core_wrapper_local.py
const char* dep_type_to_string(DependencyType type) {
    switch(type) {
        case DEP_ROOT:       return "ROOT";
        case DEP_NSUBJ:      return "nsubj";
        case DEP_DOBJ:       return "dobj";
        case DEP_IOBJ:       return "iobj";
        case DEP_CSUBJ:      return "csubj";
        case DEP_CCOMP:      return "ccomp";
        case DEP_XCOMP:      return "xcomp";
        case DEP_OBL:        return "obl";
        case DEP_VOCATIVE:   return "vocative";
        case DEP_EXPL:       return "expl";
        case DEP_DISLOCATED: return "dislocated";
        case DEP_NMOD:       return "nmod";
        case DEP_APPOS:      return "appos";
        case DEP_NUMMOD:     return "nummod";
        case DEP_ADVCL:      return "advcl";
        case DEP_ADVMOD:     return "advmod";
        case DEP_DISCOURSE:  return "discourse";
        case DEP_COMPOUND:   return "compound";
        case DEP_FIXED:      return "fixed";
        case DEP_FLAT:       return "flat";
        case DEP_GOESWITH:   return "goeswith";
        case DEP_CASE:       return "case";
        case DEP_ACL:        return "acl";
        case DEP_AMOD:       return "amod";
        case DEP_AUX:        return "aux";
        case DEP_COP:        return "cop";
        case DEP_CONJ:       return "conj";
        case DEP_CC:         return "cc";
        case DEP_DET:        return "det";
        case DEP_MARK:       return "mark";
        case DEP_PUNCT:      return "punct";
        case DEP_AGENT:      return "agent";
        case DEP_ATTR:       return "attr";
        case DEP_DATIVE:     return "dative";
        case DEP_OPRD:       return "oprd";
        case DEP_PREDET:     return "predet";
        case DEP_PREP:       return "prep";
        case DEP_NSUBJPASS:  return "nsubjpass";
        case DEP_CSUBJPASS:  return "csubjpass";
        case DEP_RELCL:      return "relcl";
        case DEP_PRT:        return "prt";
        case DEP_INTJ:       return "intj";
        case DEP_META:       return "meta";
        case DEP_NEG:        return "neg";
        case DEP_POSS:       return "poss";
        case DEP_PCOMP:      return "pcomp";
        case DEP_QUANTMOD:   return "quantmod";
        case DEP_MATH_EQUALS:  return "math_equals";
        case DEP_MATH_OPERAND: return "math_operand";
        case DEP_LOGIC_CAUSES: return "logic_causes";
        case DEP_LOGIC_IMPLIES:return "logic_implies";
        case REL_IS_A:         return "is_a";
        case REL_PART_OF:      return "part_of";
        case REL_HAS_PROPERTY: return "has_property";
        case REL_CAUSES:       return "causes";
        case REL_REQUIRES:     return "requires";
        case REL_ENABLES:      return "enables";
        case REL_USED_FOR:     return "used_for";
        case REL_DEFINED_AS:   return "defined_as";
        case REL_MEASURED_IN:  return "measured_in";
        case REL_EXAMPLE_OF:   return "example_of";
        case REL_OPPOSITE_OF:  return "opposite_of";
        case REL_RELATED_TO:   return "related_to";
        default: return "UNKNOWN";
    }
}

void brain_reset_activations(BrainHandle handle) {
    if (!handle) return;
    Graph* graph = (Graph*)handle;
    for (uint32_t i = 0; i < graph->node_count; ++i) {
        graph->nodes[i].activation = 0.0f;
    }
}

int brain_set_activation(BrainHandle handle, uint32_t node_id, float value) {
    if (!handle) return -1;
    Graph* graph = (Graph*)handle;
    if (node_id >= graph->node_count) return -2; // id вне диапазона

    graph->nodes[node_id].activation = value;
    return 0; // успех
}

int brain_spread_activation_step(BrainHandle handle, const uint32_t* anchor_ids, size_t num_anchors) {
    if (!handle) return -1;
    Graph* graph = (Graph*)handle;
    float* next_activations = (float*)calloc(graph->node_count, sizeof(float));
    if (!next_activations) return -2;

    char* is_anchor_node = (char*)calloc(graph->node_count, sizeof(char));
    if (!is_anchor_node) { free(next_activations); return -2; }
    for (size_t i = 0; i < num_anchors; ++i) {
        is_anchor_node[anchor_ids[i]] = 1;
    }

    for (uint32_t i = 0; i < graph->node_count; ++i) {
        Node* current_node = &graph->nodes[i];

        // если узел "спит", он ничего не излучает
        if (current_node->activation == 0.0f) {
            continue;
        }

        // посчитаем сумму проводимостей всех исходящих ребер
        float total_conductance = 0.0f;
        for (uint32_t j = 0; j < current_node->edge_count; ++j) {
            total_conductance += current_node->edges[j].conductance;
        }

        if (total_conductance == 0.0f) {
            continue; // узел-тупик
        }

        // теперь "делим" активацию
        for (uint32_t j = 0; j < current_node->edge_count; ++j) {
            Edge* edge = &current_node->edges[j];
            Node* neighbor_node = &graph->nodes[edge->to_node_id];

            float activation_to_send = current_node->activation * (edge->conductance / total_conductance);

            // накапливаем "приходящую" активацию во временном массиве
            next_activations[neighbor_node->id] += activation_to_send;
        }
    }
    for (uint32_t i = 0; i < graph->node_count; ++i) {
        // здесь можно добавить "затухание" - чтобы энергия не накапливалась вечно
        // и "внешний приток" - чтобы якоря оставались активными
        float decay_factor = 0.5f;
        float anchor_boost = (is_anchor_node[i] == 1) ? 0.5f : 0.0f;

        // новая активация = (старая * затухание) + приток от соседей + подпитка якоря
        graph->nodes[i].activation = (graph->nodes[i].activation * decay_factor) + next_activations[i] + anchor_boost;
    }

    free(next_activations);
    free(is_anchor_node);
    return 0;
}

int compare_activated_nodes(const void* a, const void* b) {
    ActivatedNode* nodeA = (ActivatedNode*)a;
    ActivatedNode* nodeB = (ActivatedNode*)b;
    if (nodeA->activation < nodeB->activation) return 1;
    if (nodeA->activation > nodeB->activation) return -1;
    return 0;
}

const char* brain_get_lemma_by_id(BrainHandle handle, uint32_t node_id) {
    if (!handle) return NULL;
    Graph* graph = (Graph*)handle;
    if (node_id >= graph->node_count) return NULL;
    return graph->nodes[node_id].lemma;
}

size_t brain_get_significant_nodes(BrainHandle handle, size_t max_results_size, ActivatedNode* results) {
    if (!handle || max_results_size == 0 || !results) return 0;
    Graph* graph = (Graph*)handle;

    // --- шаг 1: собираем все активные узлы во временный массив ---
    ActivatedNode* active_nodes = (ActivatedNode*)malloc(sizeof(ActivatedNode) * graph->node_count);
    if (!active_nodes) {
        perror("malloc failed for active_nodes");
        return 0;
    }

    size_t active_count = 0;
    for (uint32_t i = 0; i < graph->node_count; ++i) {
        if (graph->nodes[i].activation > 1e-6) { // используем небольшой эпсилон для точности float
            active_nodes[active_count].id = graph->nodes[i].id;
            active_nodes[active_count].activation = graph->nodes[i].activation;
            active_count++;
        }
    }

    // если активных узлов нет или он всего один, "обрыва" быть не может
    if (active_count <= 1) {
        for (size_t i = 0; i < active_count; ++i) {
            results[i] = active_nodes[i];
        }
        free(active_nodes);
        return active_count;
    }

    // --- шаг 2: сортируем активные узлы по убыванию активации ---
    qsort(active_nodes, active_count, sizeof(ActivatedNode), compare_activated_nodes);

    // --- шаг 3: ищем самый большой "обрыв" (максимальную разницу) В ХВОСТЕ ---
    // ВАЖНО: поиск начинается с index 1, а не 0. Узлы-якоря держатся на ~1.0
    // (anchor_boost), а распространённая активация соседей на порядок меньше
    // (~0.1), поэтому самый большой перепад ВСЕГДА между якорем и остальными —
    // и наивный поиск с index 0 возвращал бы только сам якорь. Пропуская верхний
    // узел(ы), мы находим естественный обрыв среди РАСПРОСТРАНЁННОГО облака и
    // всегда включаем якорь + значимых соседей.
    float max_delta = -1.0f;
    size_t cut_off_index = active_count - 1; // по умолчанию — оставить все активные

    for (size_t i = 1; i + 1 < active_count; ++i) {
        float current_delta = active_nodes[i].activation - active_nodes[i+1].activation;
        if (current_delta > max_delta) {
            max_delta = current_delta;
            cut_off_index = i;
        }
    }

    // количество узлов, которые мы считаем значимыми
    size_t significant_count = cut_off_index + 1;

    // --- (опциональная защита) если "обрыв" слишком маленький, возможно, все узлы - шум ---
    // например, если самый большой перепад всего 0.001
    // для mvp мы можем пропустить этот шаг, но для v1.0 он важен
    // if (max_delta < MINIMUM_SIGNIFICANT_DROP) { significant_count = 0; }


    // --- шаг 4: копируем "значимые" узлы в выходной массив ---
    size_t num_to_copy = (significant_count < max_results_size) ? significant_count : max_results_size;

    for (size_t i = 0; i < num_to_copy; ++i) {
        results[i] = active_nodes[i];
    }

    free(active_nodes);
    return num_to_copy;
}

size_t brain_get_subgraph_edges(BrainHandle handle,
                                const uint32_t* node_ids, size_t num_nodes,
                                size_t max_edges, EdgeInfo* results) {
    if (!handle || !node_ids || num_nodes == 0) return 0;
    Graph* graph = (Graph*)handle;

    // шаг 1: создаем быструю lookup-таблицу, чтобы проверять
    // принадлежность id к "значимому" списку за O(1)
    char* is_significant = (char*)calloc(graph->node_count, sizeof(char));
    if (!is_significant) return 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        if (node_ids[i] < graph->node_count) {
            is_significant[node_ids[i]] = 1;
        }
    }

    size_t edge_count = 0;

    // шаг 2: главный двойной цикл в C
    for (size_t i = 0; i < num_nodes; ++i) {
        uint32_t from_id = node_ids[i];
        Node* from_node = &graph->nodes[from_id];

        for (uint32_t j = 0; j < from_node->edge_count; ++j) {
            Edge* edge = &from_node->edges[j];

            // проверяем, что to_id тоже в списке
            if (is_significant[edge->to_node_id] == 1) {
                if (edge_count < max_edges) {
                    // заполняем нашу "контрактную" структуру
                    results[edge_count].from_id = from_id;
                    results[edge_count].to_id = edge->to_node_id;
                    results[edge_count].dep_type = edge->dep_type;
                    results[edge_count].conductance = edge->conductance;
                    results[edge_count].module_id = edge->module_id;
                    edge_count++;
                } else {
                    // буфер результатов переполнен, выходим
                    free(is_significant);
                    return edge_count;
                }
            }
        }
    }
    free(is_significant);
    return edge_count;
}

// --- персистентность (сериализация модулей знаний) ---
//
// бинарный формат файла модуля ".brain":
//   [magic   : 4 bytes = "BRN1"]
//   [version : uint32]
//   [node_count : uint32]
//   [edge_total : uint64]
//   node section (в порядке id 0..node_count-1):
//     [lemma_len : uint32][lemma bytes (без \0)]
//   edge section:
//     [from_id : uint32][to_id : uint32][dep_type : uint32]
//     [conductance : float][repetition_counter : uint32][module_id : uint16]

#define BRAIN_FILE_MAGIC "BRN1"
#define BRAIN_FILE_VERSION 1u
#define BRAIN_MAX_LEMMA_LEN 4096u // санитарный предел длины леммы при загрузке (защита от битого файла)

// сырое добавление ребра с точными значениями (для load, без аккумуляции проводимости)
static void append_edge_raw(Node* from_node, uint32_t to_id, DependencyType type,
                            float conductance, uint32_t reps, uint16_t module_id) {
    if (from_node->edge_count >= from_node->edge_capacity) {
        from_node->edge_capacity = (from_node->edge_capacity == 0) ? 8 : from_node->edge_capacity * 2;
        Edge* p = (Edge*)realloc(from_node->edges, sizeof(Edge) * from_node->edge_capacity);
        if (!p) { perror("realloc edges (load)"); exit(EXIT_FAILURE); }
        from_node->edges = p;
    }
    Edge* e = &from_node->edges[from_node->edge_count++];
    e->to_node_id = to_id;
    e->dep_type = type;
    e->conductance = conductance;
    e->repetition_counter = reps;
    e->last_activated_op_count = 0;
    e->module_id = module_id;
}

// сохраняет весь текущий граф в файл. возвращает 0 при успехе.
int brain_save(BrainHandle handle, const char* path) {
    if (!handle || !path) return -1;
    Graph* graph = (Graph*)handle;

    FILE* f = fopen(path, "wb");
    if (!f) { perror("fopen (save)"); return -2; }

    uint64_t edge_total = 0;
    for (uint32_t i = 0; i < graph->node_count; ++i) edge_total += graph->nodes[i].edge_count;

    uint32_t version = BRAIN_FILE_VERSION;
    if (fwrite(BRAIN_FILE_MAGIC, 1, 4, f) != 4) goto write_err;
    if (fwrite(&version, sizeof(version), 1, f) != 1) goto write_err;
    if (fwrite(&graph->node_count, sizeof(graph->node_count), 1, f) != 1) goto write_err;
    if (fwrite(&edge_total, sizeof(edge_total), 1, f) != 1) goto write_err;

    // node section
    for (uint32_t i = 0; i < graph->node_count; ++i) {
        Node* n = &graph->nodes[i];
        uint32_t len = (uint32_t)strlen(n->lemma);
        if (fwrite(&len, sizeof(len), 1, f) != 1) goto write_err;
        if (len && fwrite(n->lemma, 1, len, f) != len) goto write_err;
    }

    // edge section
    for (uint32_t i = 0; i < graph->node_count; ++i) {
        Node* n = &graph->nodes[i];
        for (uint32_t j = 0; j < n->edge_count; ++j) {
            Edge* e = &n->edges[j];
            uint32_t from_id = i;
            uint32_t dep = (uint32_t)e->dep_type;
            if (fwrite(&from_id, sizeof(from_id), 1, f) != 1) goto write_err;
            if (fwrite(&e->to_node_id, sizeof(e->to_node_id), 1, f) != 1) goto write_err;
            if (fwrite(&dep, sizeof(dep), 1, f) != 1) goto write_err;
            if (fwrite(&e->conductance, sizeof(e->conductance), 1, f) != 1) goto write_err;
            if (fwrite(&e->repetition_counter, sizeof(e->repetition_counter), 1, f) != 1) goto write_err;
            if (fwrite(&e->module_id, sizeof(e->module_id), 1, f) != 1) goto write_err;
        }
    }

    fclose(f);
    printf("[brain_core] saved %u nodes, %llu edges -> %s\n",
           graph->node_count, (unsigned long long)edge_total, path);
    return 0;

write_err:
    perror("fwrite (save)");
    fclose(f);
    return -3;
}

// читает заголовок + node section, создавая узлы и возвращая массив
// "локальный id модуля -> id в целевом графе". вызывающий освобождает мапу.
// возвращает node_count модуля, либо 0 при ошибке (мапа = NULL).
static uint32_t read_header_and_nodes(Graph* graph, FILE* f, uint32_t** out_map, uint64_t* out_edge_total) {
    char magic[4];
    uint32_t version = 0, node_count = 0;
    uint64_t edge_total = 0;
    *out_map = NULL;

    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, BRAIN_FILE_MAGIC, 4) != 0) {
        fprintf(stderr, "[brain_core] bad magic / not a .brain file\n");
        return 0;
    }
    if (fread(&version, sizeof(version), 1, f) != 1 || version != BRAIN_FILE_VERSION) {
        fprintf(stderr, "[brain_core] unsupported file version: %u\n", version);
        return 0;
    }
    if (fread(&node_count, sizeof(node_count), 1, f) != 1) return 0;
    if (fread(&edge_total, sizeof(edge_total), 1, f) != 1) return 0;

    uint32_t* map = (uint32_t*)malloc(sizeof(uint32_t) * (node_count ? node_count : 1));
    if (!map) { perror("malloc id map"); return 0; }

    for (uint32_t i = 0; i < node_count; ++i) {
        uint32_t len = 0;
        if (fread(&len, sizeof(len), 1, f) != 1) { free(map); return 0; }
        if (len > BRAIN_MAX_LEMMA_LEN) { // битый/враждебный файл: не доверяем длине
            fprintf(stderr, "[brain_core] lemma length %u exceeds limit %u\n", len, BRAIN_MAX_LEMMA_LEN);
            free(map); return 0;
        }
        char* lemma = (char*)malloc(len + 1);
        if (!lemma) { perror("malloc lemma"); free(map); return 0; }
        if (len && fread(lemma, 1, len, f) != len) { free(lemma); free(map); return 0; }
        lemma[len] = '\0';
        map[i] = get_or_create_node(graph, lemma); // ключ — лемма (клей между модулями)
        free(lemma);
    }

    *out_map = map;
    *out_edge_total = edge_total;
    return node_count;
}

// загружает файл В ПУСТОЙ граф, восстанавливая точные значения рёбер.
int brain_load(BrainHandle handle, const char* path) {
    if (!handle || !path) return -1;
    Graph* graph = (Graph*)handle;

    FILE* f = fopen(path, "rb");
    if (!f) { perror("fopen (load)"); return -2; }

    uint32_t* map = NULL;
    uint64_t edge_total = 0;
    uint32_t node_count = read_header_and_nodes(graph, f, &map, &edge_total);
    if (!map) { fclose(f); return -3; }

    for (uint64_t k = 0; k < edge_total; ++k) {
        uint32_t from_id, to_id, dep, reps;
        float cond; uint16_t module_id;
        if (fread(&from_id, sizeof(from_id), 1, f) != 1 ||
            fread(&to_id,   sizeof(to_id),   1, f) != 1 ||
            fread(&dep,     sizeof(dep),     1, f) != 1 ||
            fread(&cond,    sizeof(cond),    1, f) != 1 ||
            fread(&reps,    sizeof(reps),    1, f) != 1 ||
            fread(&module_id, sizeof(module_id), 1, f) != 1) {
            free(map); fclose(f); return -4;
        }
        if (from_id >= node_count || to_id >= node_count) continue; // защита от битого файла
        append_edge_raw(&graph->nodes[map[from_id]], map[to_id],
                        (DependencyType)dep, cond, reps, module_id);
    }

    free(map);
    fclose(f);
    printf("[brain_core] loaded %u nodes, %llu edges from %s\n",
           node_count, (unsigned long long)edge_total, path);
    return 0;
}

// монтирует модуль ПОВЕРХ существующего графа: узлы склеиваются по леммам,
// рёбра аккумулируются (как при повторном ингесте) и помечаются module_id.
int brain_merge_from_file(BrainHandle handle, const char* path, uint16_t module_id) {
    if (!handle || !path) return -1;
    Graph* graph = (Graph*)handle;

    FILE* f = fopen(path, "rb");
    if (!f) { perror("fopen (merge)"); return -2; }

    uint32_t* map = NULL;
    uint64_t edge_total = 0;
    uint32_t node_count = read_header_and_nodes(graph, f, &map, &edge_total);
    if (!map) { fclose(f); return -3; }

    graph->global_op_counter++;
    for (uint64_t k = 0; k < edge_total; ++k) {
        uint32_t from_id, to_id, dep, reps;
        float cond; uint16_t file_module_id;
        if (fread(&from_id, sizeof(from_id), 1, f) != 1 ||
            fread(&to_id,   sizeof(to_id),   1, f) != 1 ||
            fread(&dep,     sizeof(dep),     1, f) != 1 ||
            fread(&cond,    sizeof(cond),    1, f) != 1 ||
            fread(&reps,    sizeof(reps),    1, f) != 1 ||
            fread(&file_module_id, sizeof(file_module_id), 1, f) != 1) {
            free(map); fclose(f); return -4;
        }
        if (from_id >= node_count || to_id >= node_count) continue;
        (void)reps; (void)cond; (void)file_module_id;
        add_or_update_edge(&graph->nodes[map[from_id]], map[to_id],
                           (DependencyType)dep, graph->global_op_counter, module_id);
    }

    free(map);
    fclose(f);
    printf("[brain_core] merged module %u (%u nodes) from %s\n",
           module_id, node_count, path);
    return 0;
}

// --- соединение / разъединение модулей ---

// считает, сколько рёбер принадлежит модулю (для статистики/манифеста).
size_t brain_count_module_edges(BrainHandle handle, uint16_t module_id) {
    if (!handle) return 0;
    Graph* graph = (Graph*)handle;
    size_t count = 0;
    for (uint32_t i = 0; i < graph->node_count; ++i) {
        Node* n = &graph->nodes[i];
        for (uint32_t j = 0; j < n->edge_count; ++j)
            if (n->edges[j].module_id == module_id) count++;
    }
    return count;
}

// РАЗЪЕДИНЕНИЕ: удаляет все рёбра, принадлежащие модулю, и возвращает их число.
// Узлы не удаляются (id = индекс массива используется рёбрами); осиротевшие узлы
// остаются инертными — без исходящих рёбер они не участвуют в распространении
// активации и переиспользуются при повторном монтировании того же модуля.
size_t brain_unmount_module(BrainHandle handle, uint16_t module_id) {
    if (!handle) return 0;
    Graph* graph = (Graph*)handle;
    size_t removed = 0;
    for (uint32_t i = 0; i < graph->node_count; ++i) {
        Node* n = &graph->nodes[i];
        uint32_t w = 0; // позиция записи (compaction на месте)
        for (uint32_t r = 0; r < n->edge_count; ++r) {
            if (n->edges[r].module_id == module_id) { removed++; continue; }
            if (w != r) n->edges[w] = n->edges[r];
            w++;
        }
        n->edge_count = w;
    }
    return removed;
}

int main(){
    printf("--- starting brain core v2 simulation ---\n");
    BrainHandle brain = brain_create();

    // --- тестовые данные ---
    ParsedToken sentence1[] = {
        {"boy", 1, DEP_NSUBJ},
        {"sleeps", 0, DEP_ROOT} // boy -> sleeps
    };
    size_t len1 = sizeof(sentence1) / sizeof(sentence1[0]);

    ParsedToken sentence2[] = {
        {"tired", 1, DEP_AMOD},
        {"boy", 0, DEP_NSUBJ}, // tired -> boy
    };
    size_t len2 = sizeof(sentence2) / sizeof(sentence2[0]);

    // --- этап 1: ингест ---
    printf("\n[ingest] processing 'boy sleeps'...\n");
    brain_ingest_sentence(brain, sentence1, len1);

    printf("[ingest] processing 'tired boy'...\n");
    brain_ingest_sentence(brain, sentence2, len2);

    printf("[ingest] processing 'boy sleeps' a second time to test conductance update...\n");
    brain_ingest_sentence(brain, sentence1, len1);


    // --- этап 2: инспекция ---
    printf("\n--- brain state inspection ---\n");

    // 2a: общая статистика
    GraphStats stats = brain_get_stats(brain);
    printf("[stats] total nodes: %u\n", stats.node_count);
    printf("[stats] total edges: %llu\n", stats.total_edge_count);

    // 2b: инспекция конкретных узлов
    const char* lemmas_to_check[] = {"boy", "sleeps", "tired", "nonexistent"};
    int num_lemmas = sizeof(lemmas_to_check) / sizeof(lemmas_to_check[0]);

    for (int i = 0; i < num_lemmas; ++i) {
        const char* lemma = lemmas_to_check[i];
        printf("\n[inspect] checking node for lemma: '%s'\n", lemma);

        int32_t node_id = brain_get_node_id(brain, lemma);

        if (node_id == -1) {
            printf("  -> node not found in graph.\n");
            continue;
        }

        // грязный, но рабочий способ получить указатель на узел для чтения
        Graph* graph = (Graph*)brain;
        Node* node = &graph->nodes[node_id];

        printf("  -> node id: %d, total outgoing edges: %u\n", node->id, node->edge_count);
        for (uint32_t j = 0; j < node->edge_count; ++j) {
            Edge* edge = &node->edges[j];
            Node* target_node = &graph->nodes[edge->to_node_id];

            printf("    - edge to '%s'(id:%u) | type: %s | g: %.4f | reps: %u\n",
                   target_node->lemma,
                   edge->to_node_id,
                   dep_type_to_string(edge->dep_type),
                   edge->conductance,
                   edge->repetition_counter);
        }
    }


    // --- этап 2.5: проверка персистентности (save -> load) ---
    printf("\n--- testing persistence ---\n");
    const char* test_path = "/tmp/selftest.brain";
    brain_save(brain, test_path);

    BrainHandle reloaded = brain_create();
    brain_load(reloaded, test_path);
    GraphStats rstats = brain_get_stats(reloaded);
    printf("[verify] reloaded brain: %u nodes, %llu edges (expected %u / %llu)\n",
           rstats.node_count, (unsigned long long)rstats.total_edge_count,
           stats.node_count, (unsigned long long)stats.total_edge_count);

    // проверим, что проводимость восстановилась точно
    int32_t boy_id = brain_get_node_id(reloaded, "boy");
    if (boy_id != -1) {
        Graph* rg = (Graph*)reloaded;
        Node* boy = &rg->nodes[boy_id];
        for (uint32_t j = 0; j < boy->edge_count; ++j)
            printf("[verify] boy --(%s)--> %s | g=%.4f reps=%u\n",
                   dep_type_to_string(boy->edges[j].dep_type),
                   rg->nodes[boy->edges[j].to_node_id].lemma,
                   boy->edges[j].conductance, boy->edges[j].repetition_counter);
    }
    brain_free(reloaded);

    // --- этап 3: очистка ---
    brain_free(brain);
    printf("\n--- simulation complete, memory freed ---\n");
    return 0;
}