# MyAI — a neuro-symbolic brain that has to prove what it learns

A knowledge graph in C with a Python reasoning stack on top, in which **a local LLM is
never trusted**. The model is allowed to *propose* — read text, name concepts, guess a
parametrization, read a worked solution — but nothing enters the system's memory until a
**deterministic judge** certifies it. What survives the judge becomes symbolic knowledge:
typed edges, formulas with provenance, geometry with a verification trace.

Everything runs offline on a laptop (Ollama + `qwen2.5:7b` / `llama3.1:8b`, Apple M4, 16 GB).
No API keys, no cloud inference.

```
                  PROPOSE (LLM, fallible)          CERTIFY (deterministic)       PERSIST
text / web page ──► triplet extraction        ──►  concept filter + FOL check ──► .brain graph
physics problem ──► "what is given / found"   ──►  units + planner + residual ──► numeric answer
worked solution ──► "which formulas are used" ──►  must reproduce the printed  ──► formula registry
                                                    answer within 2%
"a cylinder"    ──► parametric surface guess  ──►  Gauss's law: Φ=P inside, 0   ──► geometry library
                                                    outside — cannot be gamed
```

---

## Why

Retrieval-augmented systems check a generation against **text**. Nothing checks it against
**the world**. When an LLM imagines a scene — a geometry, a configuration — the result is
accepted because it looks plausible, and plausibility is exactly the failure mode
neuro-symbolic AI exists to remove.

So this repo asks a narrow, decidable question: *can a symbolic knowledge base be grown from
a model that hallucinates, if every candidate must pass a label-free oracle before admission?*

Geometry is the clean testbed, because such an oracle exists and cannot be argued with. For a
closed surface and a point source, total flux is exactly `P` inside and exactly `0` outside.
A wrong radius, a wrong orientation, a degenerate face, a missing cap — all fail numerically.
There is no way to satisfy the judge except by being right.

---

## The stack

### 1. The core — a knowledge graph in C (`myAI.c` → `brain_core.so`)

Concept nodes and typed, weighted edges, `uthash`-indexed, exposed to Python over `ctypes`
(`brain_core_wrapper_local.py`).

* **Semantic relations** (`REL_IS_A`, `PART_OF`, `CAUSES`, `REQUIRES`, `ENABLES`, `USED_FOR`,
  `DEFINED_AS`, `MEASURED_IN`, `EXAMPLE_OF`, `OPPOSITE_OF`, `RELATED_TO`) plus quantitative
  dependency edges (`INCREASES_WITH`, `DECREASES_WITH`, `DEPENDS_ON`) that carry the *shape*
  of a formula, not just its text.
* **Bidirectional spreading activation** with per-edge conductance — the "thinking" primitive
  that finds what is near a question.
* **Modules with provenance**: each topic is a separate `.brain` file with its own `module_id`,
  mounted on demand. Derived facts get their own module id, so inference is separable and can
  be unmounted in one call.

### 2. Reasoning over the graph

| Module | What it does |
|---|---|
| `reasoner.py` | Forward-chaining FOL (Horn/Datalog fragment): transitivity, inheritance, symmetry, subsumption — to a fixed point. Every derived fact keeps its **proof tree**. |
| `predictor.py` | Inductive analogy: structurally similar neighborhoods suggest a missing edge. The guess goes to a `HypothesisStore` and is only promoted once ingested text confirms it — the graph anticipates without inventing. |
| `reflection.py` | Emergence: conclusions derivable from the *union* of sources but from no single source's own closure. "1 + 1 > 2", made checkable via per-source forward chaining. |
| `temporal_memory.py` | Bi-temporal facts (`valid_from`/`valid_to` + `recorded_at`), functional-relation supersession, `invalidate` as soft-delete, corroboration by repetition. |
| `vector_recall.py` | Semantic recall of concepts (`nomic-embed-text`) — "rubbing surfaces" finds `friction` when lexical anchors fail. |

### 3. Math and physics that actually compute

| Module | What it does |
|---|---|
| `math_core.py` | Symbolic core: parse to AST, `solve_for`, `diff`, `simplify`, definite integration, `compute`. Also projects a formula into **typed dependency edges** — `v = d/t` becomes *v increases with d, decreases with t* — so the graph learns the qualitative meaning of an equation. |
| `planner.py` | Deterministic search over core operations to build a multi-step solution — no LLM in the loop. Every step is independently checked (algebraic steps by residual substitution, amplitude steps by numeric probe). |
| `physics_solver.py` | The language boundary: the LLM only classifies the problem and extracts *given / find* with units. Units → SI, formula from the graph or from the problem text, arithmetic by `math_core`. If no formula is known, it **refuses** rather than inventing one. |
| `geometry_core.py` | Self-teaching geometry. A surface is a parametric embedding `(u,v)→(x,y,z)` as expression strings; meshing, finite-difference normals, auto-orientation, point-source flux. `learn_body()` lets the LLM propose and **Gauss's law judge** it; failures go back for repair. Survivors persist to `mind/geometry.json` with their verification trace and are then used with no LLM at all. CSG (`substitute_params`, `transform_spec`, `clip_spec`, `compose_body`) grows new bodies from certified ones. |
| `imagine.py` | Text → scene → answer. The problem is matched against a library of primitives, then grounded three times: every slot has an `"other"` escape hatch, a deterministic lint, and a **second-pass reverse verification** where each spec value becomes a claim about the problem text that must be confirmed — otherwise an honest refusal. |

### 4. Learning without a human writing formulas

| Module | What it does |
|---|---|
| `extractor.py` | Text → `(subject, relation, object)` triplets via Ollama structured outputs (Pydantic JSON-schema, `temperature=0`), with canonical concept names, a controlled relation vocabulary, and a deterministic post-filter that drops phrase-nodes and leaked negations. `StreamingExtractor` carries concept and pronoun context across chunks. |
| `ingest_pipeline.py` | End-to-end CLI: document → clean text → chunks → triplets → `.brain` module → mount check. |
| `learn_solutions.py` | Learns **from other people's worked solutions**. The LLM only reads which formulas a solution used; the judge then requires the planner to *reproduce the printed answer from the printed data* (2% relative tolerance). An invented formula cannot reproduce it. Survivors land in the module's formula registry with provenance. |
| `web_learn.py` | The same loop pointed at an allowlisted set of URLs (`physics.info`, `phys.libretexts.org`, Wikipedia). A human chooses the pages; the judge decides what is kept. |
| `curriculum.py` | Decides *what to study next* from concept dependencies — definitions before use, foundations before superstructure — because streaming ingest is order-sensitive. |
| `knowledge_manager.py` | Topic router that creates modules by itself and keeps only the relevant ones mounted. |
| `synonyms.py` | The language layer: collapse "atoms in motion" / "atoms jiggling" into one canonical node, so three sources about one fact corroborate instead of forking. |

### 5. Talking to it

`talk.py` is the single entry point. It routes a question across all three components at once:
**qwen** reads the language, the **graph** supplies the formula and justifies it with its own
edges, and **math_core** does the arithmetic. Conversational answers are strictly grounded in a
retrieved subgraph — if the neighborhood does not support an answer, the required reply is
"there is no data in the graph", not a guess.

```bash
python3 talk.py mind/waves.brain                      # REPL
python3 talk.py mind/waves.brain -q "what is a standing wave?"
```

Questions may be asked in any language: when lexical anchors miss, the model picks anchors
*from the graph's actual node list* (never inventing one), which bridges the question's language
to the graph's English snake_case vocabulary. Prose answers are currently phrased in Russian.

### 6. Measuring itself

* `graph_quality.py` — substrate quality: consistency, connectivity, atomicity, relation
  entropy, non-redundancy.
* `effectiveness.py` — a six-axis scorecard for the *whole system*: substrate, amplification,
  soundness, groundedness, **emergence**, **corroboration**. On the live graph today: composite
  **0.729**, substrate 0.941, soundness 0.999, groundedness 1.000, amplification 0.635
  (4671 derived over 2684 asserted), emergence 0.009, corroboration 0.002.
* `honest_eval.py` / `honest_test_compare.py` — LLM-alone vs. system, with **randomized numbers**
  (seeded) so recall is excluded, and Python arithmetic as the independent reference. Includes an
  explicit out-of-scope block, so real coverage is visible rather than hidden.

---

## Quickstart

Requires Python 3.11, a C compiler, and [Ollama](https://ollama.com) running locally.

```bash
# 1. build the C core
gcc -shared -o brain_core.so -fPIC myAI.c -lm

# 2. python deps
pip install ollama pydantic

# 3. models (local, free)
ollama pull qwen2.5:7b        # default: extraction, language, proposals
ollama pull nomic-embed-text  # optional: vector recall

# 4. tests — no network, no LLM
python3 -m unittest test_math_core test_geometry_core test_planner \
                    test_brain_modules test_learn_solutions

# 5. ingest a document into a knowledge module
python3 ingest_pipeline.py Feynman_Lectures_vol1ch1.html -o mind/physics.brain --mount

# 6. ask it something
python3 talk.py mind/waves.brain -q "how are wavelength and frequency related?"
```

Demos worth running: `python3 math_core.py`, `python3 geometry_core.py` (learns a cylinder and
checks it against a textbook), `python3 imagine.py`.

---

## Verified status

Measured on this machine, not asserted:

* **130 unit tests green** — 45 `math_core`, 50 `brain_modules`, 13 `geometry_core`,
  13 `planner`, 9 `learn_solutions`. No network, no LLM.
* **Live graph**: 1416 nodes / 2684 edges across two modules (`physics` 47 concepts,
  `waves` 228 concepts + ingested chapters).
* **Certified geometry library**: 4 bodies (sphere, cylinder, hemisphere, ellipsoid), each
  stored with its own Gauss-law verification trace.
* **External check** (Irodov 1.4): source at cylinder centre → 70.712 vs 70.711 analytic;
  source at base centre → 44.721 = `(P/2)h/√(R²+h²)`. Derived from the learned geometry, not
  hardcoded.
* **Extraction quality** (24-sentence generalization benchmark, English physics): main-fact
  recall 88%, precision ~84% after the strengthened few-shot prompt and deterministic filter.

---

## Honest limitations

* **Coverage equals library coverage.** The honest end-to-end baseline on 70 unmodified
  textbook problems is 1/70 — with zero false positives. The bottleneck is *scene
  formalization*, not the math engine: adding a symbolic integrator unlocked nothing.
* **The proposer is the weak link, and it fails interestingly.** `qwen2.5:7b` invents a correct
  sphere parametrization unaided, but cannot produce a correct disk — not by invention, not from
  few-shot, and *not even by copying a correct parametrization placed directly in its context*.
  Shown `x = v·cos(u)`, it writes `R·cos(u)`: the parametric prior overrides the retrieved
  source. `llama3.1` fails identically. This is contextual entrainment observed in **formal
  output where correctness is decidable by an oracle rather than by annotation** — a cleaner
  measurement instrument than open-ended QA.
* **Inference preserves truth, it does not create it.** Garbage in, amplified garbage out — one
  wrong `is_a` edge and transitivity multiplies it. Hence separable provenance on derived edges.
* **Emergence, as currently defined, is partly an artifact of how a corpus is cut into sources**
  (0.494 on ten balanced sources; 0.009 on this lopsided pair). A partition-invariant measure is
  an open problem, not a solved one.
* Ceiling is a 7B model on 16 GB of laptop RAM. Scale questions are unrunnable here.
* Code comments and docstrings are in Russian; identifiers and the graph vocabulary are English.

---

## Research direction

`proposal_grounded_imagination.md` — *Verifier-Grounded Imagination: learning grounded symbolic
representations under a non-LLM oracle*. Open questions: acquisition rates at scale, where the
invent/copy crossover sits across model sizes, whether the trust guarantee survives a
**probabilistic** oracle instead of a conservation law, and whether a certified representation
library improves downstream solving or only its trustworthiness.

`RESEARCH.md` — the temporal / updatable / vector-recall memory branch: conflict-resolution
policy, temporal FOL, a hybrid ranker fusing vector + spreading activation + conductance, and
making the temporal store the source of truth with `.brain` as a derived current slice.

---

## Repository map

```
myAI.c, uthash.h, CMakeLists.txt     C knowledge-graph core → brain_core.so
brain_core_wrapper_local.py          ctypes FFI, modules, triples, persistence
extractor.py ingest_pipeline.py      text → semantic triplets → .brain
reasoner.py predictor.py reflection.py   deduction, analogy, emergence
temporal_memory.py vector_recall.py  time-aware facts, semantic recall
math_core.py planner.py physics_solver.py    symbolic math, plan search, solving
geometry_core.py imagine.py scene.py      certified geometry, text → scene
learn_solutions.py web_learn.py autolearn.py curriculum.py   learning loops
graph_quality.py effectiveness.py honest_eval.py   self-measurement
talk.py cli.py brain_server.py       interfaces
mind/                                the knowledge workspace (gitignored)
```

---

Author: **Artur Kim**.
