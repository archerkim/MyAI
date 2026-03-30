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

    // Sentinel value to know the total count if needed
    DEP_COUNT

} DependencyType;

typedef struct Edge {
    uint32_t to_node_id;
    float conductance;
    DependencyType dep_type;
    uint64_t last_activated_op_count;
    uint32_t repetition_counter;
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
} EdgeInfo;

uint32_t get_or_create_node(Graph* graph, const char* lemma);
void add_or_update_edge(Node* from_node, uint32_t to_id, DependencyType type, uint64_t op_counter);

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
                add_or_update_edge(child_node, head_id, sentence[i].dep_type, graph->global_op_counter);

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

void add_or_update_edge(Node* from_node, uint32_t to_id, DependencyType type, uint64_t op_counter) {
    // check if edge to `to_id` with the same type already exists
    for (uint32_t i = 0; i < from_node->edge_count; ++i) {
        if (from_node->edges[i].to_node_id == to_id && from_node->edges[i].dep_type == type) {
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

    from_node->edge_count++;
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
const char* dep_type_to_string(DependencyType type) {
    // этот switch должен быть синхронизирован с enum
    switch(type) {
        case DEP_NSUBJ: return "NSUBJ";
        case DEP_AMOD: return "AMOD";
        case DEP_ROOT: return "ROOT";
        case DEP_NSUBJ: return "NSUBJ";
    DEP_DOBJ: return "";
    DEP_IOBJ: return "";
    DEP_CSUBJ: return "";
    DEP_CCOMP: return "";
    DEP_XCOMP: return "";
    // --- Nominal Dependents ---
    DEP_OBL: return "";
    DEP_VOCATIVE: return "";
    DEP_EXPL: return "";
    DEP_DISLOCATED: return "";
    DEP_NMOD: return "";
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

    // Sentinel value to know the total count if needed
    DEP_COUNT
        // ... добавь другие по мере необходимости
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
    printf("Spreading activation");
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

    // --- шаг 3: ищем самый большой "обрыв" (максимальную разницу) ---
    float max_delta = -1.0f;
    size_t cut_off_index = 0; // индекс, ПОСЛЕ которого происходит обрыв

    for (size_t i = 0; i < active_count - 1; ++i) {
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
    return active_count;
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


    // --- этап 3: очистка ---
    brain_free(brain);
    printf("\n--- simulation complete, memory freed ---\n");
    return 0;
}