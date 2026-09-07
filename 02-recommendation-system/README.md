# Hybrid Recommendation System: Anime Recommendation Engine

**Author:** Krisna Rhesa  
**Track:** Applied Machine Learning - Dicoding Indonesia  
**Dataset:** [Anime Recommendations Database (Kaggle / MyAnimeList)](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)

---

## Project Domain

In digital entertainment and multimedia streaming platforms, the exponential growth of content catalogs creates a persistent challenge known as **information overload**. Users frequently struggle to discover relevant titles among tens of thousands of available media entries, while niche productions struggle for visibility (the long-tail distribution problem) (Ricci et al., 2015).

Recommendation systems have emerged as critical infrastructure to navigate high-dimensional catalogs. Content-Based Filtering (CBF) techniques infer similarity directly from intrinsic item metadata (such as genres, themes, and narrative tags), providing transparent recommendations that completely bypass the new-user cold-start obstacle (Harper & Konstan, 2015). Conversely, Collaborative Filtering (CF) architectures - specifically modern deep learning implementations leveraging neural embeddings - excel at capturing non-linear, latent preference associations directly from historical user-item interactions (He et al., 2017; Koren et al., 2009).

This project implements a **Hybrid Recommender System** that integrates both paradigms:
1. **Content-Based Filtering**: Driven by TF-IDF vectorization and Cosine Similarity over anime genre metadata.
2. **Collaborative Filtering**: Powered by an end-to-end deep neural network (`RecommenderNet`) implemented in PyTorch with user and item embedding representations.

### References
- Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), 1-19. https://doi.org/10.1145/2827872
- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Tat-Seng, C. (2017). Neural Collaborative Filtering. *Proceedings of the 26th International Conference on World Wide Web*, 173-182. https://doi.org/10.1145/3038912.3052569
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *Computer*, 42(8), 30-37. https://doi.org/10.1109/MC.2009.263
- Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook* (2nd ed.). Springer. https://doi.org/10.1007/978-1-4899-7637-6

---

## Business Understanding

The commercial objective is to maximize user retention and engagement on a digital streaming catalog by generating personalized, highly relevant recommendation feeds.

### Problem Statements
1. How can intrinsic genre metadata be leveraged to produce accurate "Similar Anime" recommendations without relying on external user collaborative signals?
2. How can a scalable deep neural collaborative filtering architecture be engineered to predict individual user ratings and surface latent preference matches?
3. How do content-based and collaborative algorithms compare across recommendation accuracy, computational overhead, and cold-start resilience?

### Goals
1. Construct a Content-Based Filtering pipeline using TF-IDF and Cosine Similarity to attain out-of-sample **Precision@5 above 80%**.
2. Build and train a PyTorch `RecommenderNet` Collaborative Filtering model to predict user ratings with a **Test RMSE below 1.50** on the 1-10 rating scale.
3. Conduct comprehensive multi-metric benchmarking (Precision@K, RMSE, MSE) and design a unified hybrid deployment strategy.

---

## Data Understanding

The project utilizes the verified **Anime Recommendations Database** from Kaggle, compiled by Cooper Union from MyAnimeList:
- `anime.csv`: **12,294 titles** with 7 metadata attributes (`anime_id`, `name`, `genre`, `type`, `episodes`, `rating`, `members`).
- `rating.csv`: **7,813,737 user-item ratings** across 3 columns (`user_id`, `anime_id`, `rating`), including 1,476,496 implicit ratings (`rating = -1`).

### Data Quality Audit & Duplicate Handling

A comprehensive quality audit revealed:
- **`anime.csv`**: 0 duplicates. Missing values detected on `rating` (230 records), `genre` (62 records), and `type` (25 records).
- **`rating.csv`**: Exactly **1 duplicate row** was identified via `df_rating.duplicated()`. The duplicated record represents an identical interaction pair: `user_id = 42653`, `anime_id = 16498`, and `rating = 8` (recorded at indices 4499258 and 4499316).

#### Technical Justification for Duplicate Sanitation
In high-throughput logging systems, duplicate records commonly stem from network latency retries, duplicate HTTP form submissions, or lack of database idempotency constraints.

Sanitation was performed via `df_rating.drop_duplicates(keep='first')`, calibrating the interaction corpus from 7,813,737 down to **7,813,736 rows**.

**Engineering Rationale:**
1. **Relational Data Integrity**: Collaborative filtering models assume unique binary coordinates $(u, i)$ in the interaction space. Redundant pairs violate matrix consistency.
2. **Gradient Loss Regularization**: Retaining duplicate rows injects artificial loss contributions during backpropagation, leading to gradient distortion and over-representing specific embedding weights in `RecommenderNet`.
3. **Data Leakage Mitigation**: Deduplication prior to train-test splitting guarantees that identical samples do not concurrently leak across train and evaluation sets, preventing optimistically biased metrics.

#### Missing Value Remediation Strategy
1. **62 Records with Missing `genre` (Dropped)**: Genre strings are mandatory for TF-IDF feature extraction.
2. **230 Records with Missing `rating` (Median Imputation: 6.57)**: Retains catalog diversity. The distribution is negatively skewed (*skewness* = -0.54), making the robust median (6.57) far superior to the outlier-sensitive mean (6.47).
3. **25 Records with Missing `type` (Filled as 'Unknown')**: Maintains metadata shape without fabricating format categories.

---

## Data Preparation Pipeline

1. **Duplicate Elimination**: `drop_duplicates()` sanitizes the 1 redundant interaction pair.
2. **Filtering Implicit Ratings (-1)**: 1,476,496 unrated entries filtered, leaving **6,337,240 explicit ratings** (1-10).
3. **Active User Extraction (Top-500 Users)**: To ensure computational feasibility and avoid GPU memory overflow during deep learning, a dense interaction subset of 500 top-rating users was extracted, containing **497,207 ratings** across **9,429 unique anime**.
4. **HTML Entity Normalization**: 292 anime titles containing raw HTML escape sequences (`&quot;`, `&#039;`, `&amp;`) were unescaped into standard characters via `html.unescape()`.
5. **Categorical Entity Encoding**: User and anime IDs mapped to zero-indexed contiguous integer arrays for PyTorch `nn.Embedding` lookup layers.
6. **Target Min-Max Normalization**: Explicit 1-10 ratings scaled to the $[0, 1]$ interval to match the output dynamic range of the Sigmoid activation.
7. **Train-Test Split**: 80% train (**397,765 ratings**) and 20% test (**99,442 ratings**) split under `random_state=42`.
8. **TF-IDF & Cosine Similarity Matrix**: `TfidfVectorizer` extracted 47 unique genre tokens, followed by pairwise Cosine Similarity calculation producing a $(12,232 \times 12,232)$ dense matrix.

---

## Modeling

### 1. Content-Based Filtering (CBF)
Computes angular distance between TF-IDF feature vectors:

$$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$

Top-5 recommendations for benchmark title **"Kimi no Na wa."** (Genres: *Drama, Romance, School, Supernatural*):
1. *Wind: A Breath of Heart (TV)* (Sim: 0.871)
2. *Wind: A Breath of Heart (OVA)* (Sim: 0.871)
3. *Aura: Maryuuinkouga Saigo no Tatakai* (Sim: 0.852)
4. *Angel Beats!* (Sim: 0.778)
5. *Kokoro Connect* (Sim: 0.771)

### 2. Collaborative Filtering (PyTorch RecommenderNet)
An end-to-end deep learning recommender architecture:
- **Embedding Layers**: `nn.Embedding(n_users, 50)` and `nn.Embedding(n_items, 50)`.
- **Bias Embeddings**: `nn.Embedding(n_users, 1)` and `nn.Embedding(n_items, 1)`.
- **Forward Operation**: Computes the dot product of latent vectors, incorporates user/item bias terms, and maps through a Sigmoid activation:

$$\hat{y} = \sigma \left( \mathbf{P}_u \cdot \mathbf{Q}_i + b_u + b_i \right)$$

$$\hat{r} = \hat{y} \times (r_{\max} - r_{\min}) + r_{\min}$$

- **Optimization**: MSE Loss, AdamW optimizer ($\text{lr} = 0.001$, $\text{weight\_decay} = 10^{-5}$), batch size 1,024, trained across 10 epochs.

---

## Evaluation

### Content-Based Filtering Evaluation
Precision at K measures the fraction of recommended items sharing relevant genre tags with the reference item:

$$\text{Precision@K} = \frac{\text{Number of Relevant Recommendations in Top-K}}{K}$$

All 5 recommended titles for *Kimi no Na wa.* shared dominant target genres (Drama/Romance/Supernatural), yielding:
$$\text{Precision@5} = \frac{5}{5} = \mathbf{100.00\%}$$

### Collaborative Filtering Evaluation
Validation performance across 10 training epochs:

*Table 1. Epoch-by-Epoch Convergence of PyTorch RecommenderNet*

| Epoch | Train Loss (MSE) | Val Loss (MSE) | Train RMSE | Test RMSE |
| :---: | :---: | :---: | :---: | :---: |
| **01** | 0.0270 | 0.0245 | 1.4793 | 1.4093 |
| **03** | 0.0205 | 0.0223 | 1.2882 | 1.3432 |
| **05** | 0.0152 | 0.0215 | 1.1098 | 1.3197 |
| **07** | 0.0104 | 0.0214 | 0.9168 | **1.3159** |
| **10** | 0.0074 | 0.0218 | 0.7737 | **1.3296** |

**Final Evaluation Metrics:**
- **Final Test RMSE:** **1.3296** (well below the project threshold target of < 1.50).
- **Final Test MSE:** **1.7678**.

![Training Curves](figures/training_loss_rmse_curve.png)
*Figure 1. Loss (MSE) and RMSE Curves for PyTorch RecommenderNet across 10 Epochs*

---

## Production Deployment Recommendation: Hybrid Architecture

For production streaming services, an ensemble **Hybrid Recommender System** delivers the optimal balance:
1. **New Users (Cold-Start)**: Served by **Content-Based Filtering** to display similar items immediately upon clicking any title.
2. **Active Users (Warm Profiles)**: Scored by **Neural Collaborative Filtering** to uncover serendipitous personal preferences across the broader catalog.
3. **Ensemble Reranking**: Weighted fusion of CBF similarity scores and CF predicted ratings for home feed generation.
