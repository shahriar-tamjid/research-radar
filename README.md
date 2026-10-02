# ResearchRadar

### Academic Paper Semantic Search & Recommendation System

> **Discover scientific literature using semantic similarity.**

ResearchRadar is a compact, end-to-end **Information Retrieval (IR) and NLP** project implemented entirely in **Google Colab**. It uses the **BEIR SciFact** benchmark to build a scientific literature discovery system that combines lexical retrieval, semantic embeddings, content-based recommendations, interactive visualisation, and quantitative retrieval evaluation.

The project is intentionally designed around a simple pipeline:

```text
Text → Representation → Similarity → Ranking → Recommendation → Evaluation
```

Rather than hiding the fundamentals behind a large framework, ResearchRadar keeps the architecture small enough to understand, run, and explain in an interview or technical discussion.

---

## Overview

Scientific search is more than matching the exact words in a query. The same idea can be expressed using different terminology, abbreviations, or scientific language. ResearchRadar explores this problem by comparing two retrieval approaches:

- **TF-IDF retrieval** for transparent lexical matching.
- **Sentence embeddings** for semantic similarity.

A simple **hybrid retriever** then combines the two signals. The same embedding space is also reused to provide **content-based paper recommendations**.

The notebook evaluates all three retrieval methods against the **official SciFact test relevance judgements (qrels)**, so the project includes both a working demonstration and measurable IR evaluation.

ResearchRadar is **not** a chatbot or generative-AI application. It retrieves documents from the benchmark corpus and ranks them according to similarity.

---

## Key Features

- Semantic search over the BEIR SciFact scientific corpus
- TF-IDF lexical retrieval baseline
- Sentence-transformer document and query embeddings
- Cosine-similarity ranking
- Content-based academic paper recommendation
- Simple hybrid lexical + semantic retrieval
- Precision@5, Recall@5, MRR@10, and NDCG@10 evaluation
- Qualitative error analysis using the SciFact qrels
- Document/query text exploration
- PCA-based embedding-space visualisation
- Interactive Plotly embedding explorer
- Interactive Google Colab search and recommendation demo using `ipywidgets`
- Embedding caching to avoid unnecessary regeneration
- Reproducible random seed and a clean, notebook-first implementation

---

## System Architecture

The system is deliberately small and has two primary user-facing paths: **semantic search** and **paper recommendation**.

```text
                              RESEARCHRADAR
                                   |
                    +--------------+--------------+
                    |                             |
              SEMANTIC SEARCH              PAPER RECOMMENDER
                    |                             |
               User Query                    Selected Paper
                    |                             |
            Sentence Embedding            Sentence Embedding
                    |                             |
            Cosine Similarity              Cosine Similarity
                    |                             |
               Top Papers                 Similar Papers
                    +--------------+--------------+
                                   |
                          Visual Exploration
                                   |
                            IR Evaluation
                       TF-IDF | Semantic | Hybrid
```

At a high level:

1. Scientific documents are loaded from SciFact.
2. Titles and abstracts are combined into the retrieval text.
3. TF-IDF provides a lexical representation.
4. Sentence Transformers provide dense semantic embeddings.
5. Cosine similarity is used to score query/document or document/document pairs.
6. Documents are ranked and returned to the user.
7. The same document embeddings drive content-based recommendations.
8. Retrieval methods are evaluated using SciFact's relevance judgements.
9. The embedding space and retrieval behaviour are explored visually.

---

## NLP & Information Retrieval Concepts

### Information Retrieval

Information Retrieval is the task of finding and ranking useful documents from a collection in response to a user's information need. ResearchRadar demonstrates this through query-to-document scoring and ranking.

### TF-IDF

TF-IDF is used as the project's transparent lexical baseline. It gives higher importance to terms that are useful for distinguishing documents while reducing the influence of terms that occur widely across the corpus.

The notebook uses:

- English stop-word removal
- unigrams and bigrams (`ngram_range=(1, 2)`)
- sublinear term frequency
- a maximum vocabulary size of 50,000 features
- cosine similarity for retrieval scoring

### Embeddings

A sentence embedding converts text into a dense numerical vector. Texts with related meaning tend to occupy nearby locations in the embedding space.

ResearchRadar uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The notebook produces **384-dimensional** document embeddings and normalises them before similarity calculations.

The model is used as a pretrained general-purpose encoder. ResearchRadar does **not** train or fine-tune a transformer on SciFact.

### Cosine Similarity

The similarity between two vectors is computed as:

```text
cos(A, B) = (A · B) / (||A|| ||B||)
```

Because the notebook stores normalised embeddings, semantic similarity can be calculated efficiently with a dot product.

A higher cosine similarity means that two vectors are more aligned in the embedding space. It is **not** a probability of relevance and it does **not** establish scientific correctness.

### Content-Based Recommendation

The recommendation component uses the content of a selected paper rather than user behaviour:

```text
Selected paper
      ↓
Paper embedding
      ↓
Compare with corpus embeddings
      ↓
Rank by cosine similarity
      ↓
Recommend top-k similar papers
```

This is a **content-based recommender**. It is not collaborative filtering and it is not personalised, because the SciFact benchmark does not provide user-rating or interaction history.

### Hybrid Retrieval

The hybrid retriever combines semantic and lexical scores after per-query min-max normalisation:

```text
hybrid_score = α × semantic_score + (1 − α) × lexical_score
```

The notebook uses:

```text
α = 0.7
```

This value is used as a transparent demonstration setting rather than a claimed globally optimal hyperparameter.

---

## Dataset

ResearchRadar uses **SciFact** through the **BEIR** benchmark framework.

The notebook loads the official test retrieval split with BEIR's `GenericDataLoader`, producing three core structures:

| Structure | Meaning |
|---|---|
| `corpus` | Document ID → title and abstract/text |
| `queries` | Query ID → scientific claim/query |
| `qrels` | Query ID → relevant document ID → relevance grade |

The recorded notebook run reports:

| Dataset component | Count |
|---|---:|
| Corpus documents | **5,183** |
| Test queries | **300** |
| Queries with qrels | **300** |

The benchmark corpus is represented as a DataFrame with:

```text
doc_id
title
text
combined_text
```

where:

```text
combined_text = title + ". " + abstract/text
```

The repository does not claim ownership of the SciFact dataset. It is used as a benchmark for the experiment.

---

## Methodology

### 1. Dataset Loading

The notebook downloads SciFact automatically into the Colab runtime and loads the test split using BEIR.

### 2. Text Preparation

The title and abstract are combined because the title provides a strong topical signal while the abstract provides the surrounding scientific context.

No aggressive stemming or lemmatisation pipeline is applied to the semantic retrieval text.

### 3. Lexical Retrieval

A TF-IDF matrix is built for the full corpus. A query is transformed into the same sparse feature space and ranked by cosine similarity.

### 4. Semantic Retrieval

Each document is encoded once using `all-MiniLM-L6-v2`. Query text is encoded at search time and compared against the cached document embedding matrix.

The notebook uses batched embedding generation and stores the document embeddings under:

```text
/content/researchradar_cache/
```

### 5. Document Similarity

A selected document's embedding is compared against every other document embedding. The selected document itself is removed from the candidate set before ranking.

### 6. Content-Based Recommendation

The document-similarity function is reused directly as the recommendation mechanism. This keeps the recommender transparent and avoids introducing a separate recommendation framework.

### 7. Hybrid Retrieval

The TF-IDF and semantic scores are independently min-max normalised for each query and combined with `alpha = 0.7`.

### 8. Retrieval Evaluation

All three methods are scored over the full SciFact test query set and compared against the official qrels.

### 9. Error Analysis

The notebook categorises queries according to whether the top TF-IDF and semantic result is relevant according to the qrels:

- TF-IDF relevant / Semantic not relevant
- Semantic relevant / TF-IDF not relevant
- Both not relevant
- Both relevant

The recorded run found examples in all four categories.

---

## Evaluation

The notebook evaluates:

- **Precision@5** — the proportion of retrieved top-5 results that are relevant, averaged across queries.
- **Recall@5** — how much of the known relevant set is retrieved within the top five.
- **MRR@10** — rewards placing the first relevant result as high as possible within the top ten.
- **NDCG@10** — rewards relevant results appearing near the top and accounts for graded relevance.

The measurements below are copied from the notebook's recorded SciFact evaluation run:

| Method | Precision@5 | Recall@5 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.1460 | 0.6758 | 0.5478 | 0.5890 |
| Semantic | 0.1647 | 0.7413 | 0.6068 | 0.6484 |
| Hybrid | 0.1720 | 0.7766 | 0.6569 | 0.6990 |

For this recorded run, the hybrid configuration produced the highest measured value for each of the four reported metrics among the three tested methods. These numbers should be understood as **results of this particular SciFact experiment**, not as a universal claim about retrieval methods.

The notebook also generates a grouped evaluation bar chart and a box plot showing the distribution of top-10 retrieval scores across queries.

---

## Error Analysis

The project goes beyond a single metric table by inspecting where retrieval strategies behave differently.

Examples from the recorded run include:

| Case | Query | Observed behaviour |
|---|---|---|
| TF-IDF relevant / Semantic not relevant | `Activation of PPM1D suppresses p53 function.` | TF-IDF retrieved a qrel-relevant PPM1D paper at rank 1, while semantic retrieval selected a different p53-related paper. |
| Semantic relevant / TF-IDF not relevant | `ADAR1 binds to Dicer to cleave pre-miRNA.` | Semantic retrieval returned the qrel-relevant ADAR1/Dicer paper at rank 1, while TF-IDF selected another Dicer-related paper. |
| Both not relevant | `0-dimensional biomaterials show inductive properties.` | Neither method returned the known relevant document at rank 1. |
| Both relevant | `AIRE is expressed in some skin tumors.` | Both methods returned the same qrel-relevant paper at rank 1. |

The recorded query counts were:

| Category | Available examples |
|---|---:|
| TF-IDF relevant / Semantic not relevant | 30 |
| Semantic relevant / TF-IDF not relevant | 48 |
| Both not relevant | 119 |
| Both relevant | 103 |

This analysis helps illustrate that lexical overlap and semantic similarity can fail or succeed for different reasons. The notebook deliberately treats these as observations from the benchmark rather than causal explanations.

---

## Visualisations

ResearchRadar contains several visual components designed to make retrieval behaviour easier to inspect.

### Document Length Distribution

Shows the distribution of word counts across the scientific documents.

### Query Length Distribution

Shows how long the benchmark queries are compared with the much larger document texts.

### Frequent-Term Analysis

A simple corpus-level term-frequency view highlights common vocabulary. The recorded run includes terms such as `cells`, `cell`, `patients`, `expression`, and `cancer`.

### Semantic Search Score Chart

A horizontal bar chart shows the top five semantic-search results and their cosine similarity scores.

### Retrieval Score Distribution

A box plot compares the distribution of top-10 retrieval scores produced by TF-IDF, semantic retrieval, and hybrid retrieval.

### Evaluation Comparison

A grouped Plotly bar chart compares Precision@5, Recall@5, MRR@10, and NDCG@10 across retrieval methods.

### PCA Embedding Explorer

Up to **750** document embeddings are projected from 384 dimensions into two principal components using PCA and displayed in an interactive Plotly scatter plot.

In the recorded run:

```text
PC1 explained variance: 0.0804
PC2 explained variance: 0.0397
PC1 + PC2:              0.1202
```

This is a 2D projection for exploration. It should not be interpreted as preserving the full structure of the original 384-dimensional space.

---

## Interactive ResearchRadar Demo

The notebook includes a lightweight interactive interface built with `ipywidgets` inside Google Colab.

The demo supports:

```text
Research question
      ↓
Search
      ↓
Top 5 semantic results
      ↓
Select a paper
      ↓
Recommend similar papers
```

The interface uses the same retrieval and recommendation functions introduced earlier in the notebook rather than implementing a separate hidden system.

The default example query is:

> What is the relationship between sleep deprivation and cognitive performance?

---

## Example Queries

The notebook includes eight demonstration queries:

1. What is the relationship between sleep deprivation and cognitive performance?
2. Does physical activity reduce the risk of cardiovascular disease?
3. How does inflammation affect Alzheimer's disease?
4. What factors influence cancer progression?
5. Can machine learning improve medical diagnosis?
6. How are biomarkers used in disease detection?
7. What is the role of oxidative stress in human disease?
8. Does diet influence the development of cardiovascular disease?

These are used as search inputs against the actual SciFact corpus. Results are not hard-coded or fabricated.

---

## Example Search Result

For the example query:

```text
What is the relationship between inflammation and Alzheimer's disease?
```

the recorded semantic-search run returned:

| Rank | Paper | Similarity |
|---:|---|---:|
| 1 | Local neuroinflammation and the progression of ... | 0.6987 |
| 2 | Chronic inflammation (inflammaging) and its potential contribution to age-associated diseases. | 0.5992 |
| 3 | Alzheimer's disease: the two-hit hypothesis. | 0.5694 |
| 4 | Immunoproteasome and LMP2 polymorphism in aged ... | 0.5598 |
| 5 | Alzheimer’s Disease Risk Gene CD33 Inhibits Mi... | 0.5581 |

The titles above are shown as recorded in the notebook output; the full results and abstract previews are available when the notebook is executed.

---

## Example Recommendation

The notebook also demonstrates recommendation from a selected paper. For the sleep-related demo, the selected paper was:

> **Cognitive behavioral therapy vs zopiclone for treatment of chronic primary insomnia in older adults: a randomized controlled trial.**

The content-based recommender returned five nearest documents, including:

- Long-term, nightly benzodiazepine treatment of ...
- Role of common hypnotics on the phenotypic causes of obstructive sleep apnoea: paradoxical effects of zolpidem.
- Mindfulness meditation and improvement in sleep quality and daytime impairment among older adults with sleep disturbances: a randomized clinical trial.
- Treatments for somnambulism in adults: assessing ...
- A parasomnia overlap disorder involving sleepwalking, sleep terrors, and REM sleep behavior disorder in 33 polysomnographically confirmed cases.

These recommendations represent embedding-space similarity, not scientific equivalence or evidence that one paper supports another.

---

## Technologies

### Core

- Python
- Google Colab
- NumPy
- Pandas
- scikit-learn
- SciPy

### NLP / IR

- BEIR
- SciFact
- Sentence Transformers
- PyTorch
- TF-IDF
- Cosine similarity

### Visualisation / Interaction

- Matplotlib
- Seaborn
- Plotly
- `ipywidgets`

### Utility

- tqdm
- JSON/path-based local caching

---

## How to Run in Google Colab

The notebook is designed to run from top to bottom in a fresh Colab runtime.

### 1. Clone or download the repository

Place these files in a GitHub repository:

```text
ResearchRadar/
├── ResearchRadar.ipynb
├── README.md
├── requirements.txt
├── .gitignore
```

### 2. Open the notebook in Google Colab

Open `ResearchRadar.ipynb` with Google Colab.

### 3. Run the notebook from the beginning

The setup cell installs the required packages. No manually uploaded dataset is required.

### 4. Let the notebook download SciFact

The notebook checks for the dataset in the Colab runtime and downloads it automatically when required.

### 5. Wait for the embedding stage to finish

The first run downloads `all-MiniLM-L6-v2` and encodes the corpus in batches. The notebook then caches the document embeddings.

### 6. Explore the retrieval and recommendation sections

After setup, you can:

- run TF-IDF search,
- run semantic search,
- compare hybrid results,
- inspect similar papers,
- use the interactive search interface,
- explore the PCA embedding space,
- and review the benchmark evaluation.

For repeated experimentation in the same Colab runtime, cached embeddings are reused when the model name and corpus document IDs still match the cache metadata.

---

## Project Structure

```text
ResearchRadar/
│
├── ResearchRadar.ipynb      # Main end-to-end Google Colab project
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignores datasets, cached embeddings and local artefacts
```

The notebook is the **primary project artefact**. The implementation intentionally does not split the project into a large collection of Python modules.

---

## Reproducibility & Performance

The notebook includes a few simple measures to keep execution practical and repeatable in Colab.

### Reproducibility

- Random seed fixed at `42`
- NumPy, Python `random`, and PyTorch seeds set where available
- Deterministic document/query ordering derived from the loaded corpus structures
- Benchmark qrels are used directly for evaluation

### Performance

- Lightweight pretrained embedding model
- Batch embedding generation with batch size `64`
- Normalised embeddings for efficient dot-product similarity
- Vectorised NumPy/scikit-learn similarity calculations
- Local `.npy` embedding cache
- Cache metadata checks for model name, corpus size, embedding dimension, and document IDs
- PCA visualisation limited to at most `750` sampled documents

The project does **not** require model training, transformer fine-tuning, an external vector database, or a paid API.

---

## Limitations

ResearchRadar is intentionally small and educational, so its limitations are important to understand.

1. **SciFact is a relatively small benchmark** compared with modern scholarly search indexes.
2. **The embedding model is not fine-tuned on SciFact**, so it is used as a general-purpose semantic encoder.
3. **Semantic similarity does not guarantee scientific correctness.** Similar vectors are not proof that a claim is true.
4. **Recommendations are content-based, not personalised.** There is no user history, rating model, or collaborative filtering component.
5. **The corpus is a benchmark collection of scientific texts**, not a complete modern academic search engine with continuously updated metadata, citations, authors, publication years, and indexing infrastructure.
6. **A retrieval ranking is not an evidence verdict.** A highly ranked abstract should not be treated as proof that a scientific claim is correct.
7. **The reported metrics are benchmark-specific.** Results from this experiment should not be generalised automatically to other datasets or search domains.

---

## Future Improvements

The current design intentionally stops before adding unnecessary complexity. Natural next steps include:

- BM25 as an additional lexical baseline
- Scientific-domain embedding models
- Cross-encoder reranking
- Citation-graph-based recommendations
- Metadata filtering by year, topic, author, or venue
- Personalised recommendation using user interaction history
- Larger scientific retrieval datasets
- Multimodal paper retrieval
- Explainable retrieval and recommendation
- More systematic hyperparameter tuning for the hybrid retriever

These are possible extensions rather than components of the current implementation.

---

## Repository Requirements

The repository includes a `requirements.txt` file with the packages used by the notebook.

Current dependency ranges are:

```text
beir>=2.2,<3
sentence-transformers>=3,<6
scikit-learn>=1.5,<2
pandas>=2.0,<3
numpy>=1.24,<3
matplotlib>=3.8,<4
seaborn>=0.13,<1
plotly>=5,<7
scipy>=1.10,<2
tqdm>=4.66,<5
ipywidgets>=8,<9
torch>=2.0,<3
```

---

## References & Attribution

### BEIR

Thakur, N. et al. **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.** NeurIPS Datasets and Benchmarks Track.

<https://github.com/beir-cellar/beir>

### SciFact

SciFact is a scientific claim verification / evidence retrieval benchmark included in BEIR. ResearchRadar uses the benchmark's corpus, queries, and test relevance judgements for retrieval experimentation and evaluation.

BEIR dataset information:

<https://github.com/beir-cellar/beir/wiki/Datasets-available>

### Sentence Transformers

ResearchRadar uses the pretrained model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Sentence Transformers:

<https://www.sbert.net/>

Model card:

<https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>

---

## Conclusion

ResearchRadar demonstrates a complete small-scale Information Retrieval workflow using a real scientific benchmark:

```text
SciFact dataset
     ↓
Text preparation
     ↓
TF-IDF baseline ──────────┐
                          ├──→ Ranked retrieval
Sentence embeddings ─────┤
                          └──→ Content-based recommendations
          ↓
     Hybrid retrieval
          ↓
 Visual exploration
          ↓
Quantitative evaluation
          ↓
   Error analysis
```

The main goal is not to build the most complicated search system. It is to demonstrate, clearly and concretely, how we can move from **raw scientific text to representations, similarity scores, ranked retrieval, recommendations, visualisation, and empirical evaluation** in a single reproducible Colab project.

---

## Project Checklist

| Capability | Implemented |
|---|:---:|
| Information Retrieval | ✅ |
| NLP Embeddings | ✅ |
| Document Similarity | ✅ |
| Recommendation System | ✅ |
| Visualisation | ✅ |
| Quantitative Evaluation | ✅ |
| Error Analysis | ✅ |
| Interactive Demo | ✅ |
| Reproducibility | ✅ |
| Google Colab Compatibility | ✅ |

---

**ResearchRadar** — *Discover scientific literature using semantic similarity.*
