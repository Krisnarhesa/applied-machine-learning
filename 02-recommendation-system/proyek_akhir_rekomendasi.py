from __future__ import annotations
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from typing import Any, Dict, List, Tuple
import html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def load_original_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Memuat dataset ORIGINAL dari Kaggle (Anime Recommendations Database).
    
    Dataset original:
 - anime.csv: 12.294 baris, 7 kolom (230 missing rating, 62 missing genre, 25 missing type)
 - rating.csv: 7.813.737 baris, 3 kolom (termasuk 1.476.496 baris rating=-1)
    
    Returns:
        Tuple: (df_anime, df_rating) dalam kondisi asli tanpa modifikasi
    """
    if os.path.exists("anime.csv") and os.path.exists("rating.csv"):
        anime_path = "anime.csv"
        rating_path = "rating.csv"
    else:
        cache_path = os.path.join(
            os.path.expanduser("~"),
            ".cache", "kagglehub", "datasets",
            "CooperUnion", "anime-recommendations-database",
            "versions", "1"
        )
        if os.path.exists(os.path.join(cache_path, "anime.csv")) and os.path.exists(os.path.join(cache_path, "rating.csv")):
            anime_path = os.path.join(cache_path, "anime.csv")
            rating_path = os.path.join(cache_path, "rating.csv")
        else:
            import kagglehub
            dataset_dir = kagglehub.dataset_download("CooperUnion/anime-recommendations-database")
            anime_path = os.path.join(dataset_dir, "anime.csv")
            rating_path = os.path.join(dataset_dir, "rating.csv")
    
    df_anime = pd.read_csv(anime_path)
    df_rating = pd.read_csv(rating_path)
    
    return df_anime, df_rating


def prepare_data(
    df_anime: pd.DataFrame,
    df_rating: pd.DataFrame,
    n_top_users: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Melakukan filtering dan pembersihan data secara eksplisit.
    
    Langkah-langkah:
    1. Hapus data duplikat pada rating.csv
    2. Filter rating=-1 dari rating.csv (pengguna menonton tanpa memberi nilai)
    3. Seleksi N pengguna paling aktif
    4. Drop anime tanpa genre (62 baris) - wajib untuk CBF
    5. Imputasi median untuk missing rating di anime.csv
    6. Isi missing type dengan 'Unknown'
    7. Normalisasi HTML entity pada nama anime (292 judul)
    
    Returns:
        Tuple: (df_anime_clean, df_rating_clean) data siap proses
    """
    n_dupes = df_rating.duplicated().sum()
    if n_dupes > 0:
        print(f"[PREP] Menghapus data duplikat pada rating.csv: {n_dupes} baris")
        df_rating = df_rating.drop_duplicates().copy()
        print(f"[PREP] Sisa baris rating setelah drop duplikat: {len(df_rating):,}")
    
    n_neg1 = (df_rating["rating"] == -1).sum()
    print(f"[PREP] Baris rating=-1 (menonton tanpa menilai): {n_neg1:,}")
    df_rating = df_rating[df_rating["rating"] != -1].copy()
    print(f"[PREP] Sisa baris setelah filter rating=-1: {len(df_rating):,}")
    
    user_counts = df_rating["user_id"].value_counts()
    top_users = user_counts.head(n_top_users).index
    df_rating = df_rating[df_rating["user_id"].isin(top_users)].copy()
    print(f"[PREP] Sisa baris setelah seleksi {n_top_users} pengguna aktif: {len(df_rating):,}")
    print(f"[PREP] Pengguna unik: {df_rating['user_id'].nunique()}")
    
    n_no_genre = df_anime["genre"].isnull().sum()
    print(f"[PREP] Anime tanpa genre: {n_no_genre} (akan di-drop)")
    df_anime = df_anime.dropna(subset=["genre"]).copy()
    
    n_missing_rating = df_anime["rating"].isnull().sum()
    median_rating = df_anime["rating"].median()
    print(f"[PREP] Missing rating setelah drop genre: {n_missing_rating}, median: {median_rating:.2f}")
    df_anime["rating"] = df_anime["rating"].fillna(median_rating)
    
    n_missing_type = df_anime["type"].isnull().sum()
    if n_missing_type > 0:
        print(f"[PREP] Missing type: {n_missing_type} (diisi 'Unknown')")
        df_anime["type"] = df_anime["type"].fillna("Unknown")
    
    n_html = df_anime["name"].apply(lambda x: html.unescape(str(x)) != str(x)).sum()
    print(f"[PREP] Nama anime dengan HTML entity: {n_html} (akan dinormalisasi)")
    df_anime["name"] = df_anime["name"].apply(lambda x: html.unescape(str(x)))
    
    df_anime = df_anime.reset_index(drop=True)
    df_rating = df_rating.reset_index(drop=True)
    
    return df_anime, df_rating


def export_eda_visualizations(
    df_anime: pd.DataFrame,
    df_rating: pd.DataFrame,
    output_dir: str = "figures",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 11})

    all_genres: List[str] = []
    for g_str in df_anime["genre"].dropna():
        for g in str(g_str).split(", "):
            all_genres.append(g.strip())
    top_genres = pd.Series(all_genres).value_counts().head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_genres.values, y=top_genres.index, hue=top_genres.index, palette="mako", legend=False)
    plt.title("Top 10 Genre Anime Terpopuler", fontweight="bold", pad=12)
    plt.xlabel("Jumlah Anime")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_top_genres.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_rating, x="rating", hue="rating", palette="viridis", legend=False)
    plt.title("Distribusi Rating Pengguna (Skala 1 - 10)", fontweight="bold", pad=12)
    plt.xlabel("Rating Pengguna")
    plt.ylabel("Frekuensi Interaksi")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_distribusi_rating.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    type_order = df_anime["type"].value_counts().index
    sns.countplot(data=df_anime, y="type", order=type_order, hue="type", palette="crest", legend=False)
    plt.title("Distribusi Format Penayangan Anime", fontweight="bold", pad=12)
    plt.xlabel("Jumlah Anime")
    plt.ylabel("Tipe Format")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_tipe_anime.png"), dpi=300)
    plt.close()


def build_content_based_model(
    df_anime: pd.DataFrame,
) -> Tuple[np.ndarray, TfidfVectorizer]:
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_anime["genre"].fillna(""))
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix, tfidf


def get_content_based_recommendations(
    anime_title: str,
    df_anime: pd.DataFrame,
    similarity_matrix: np.ndarray,
    top_n: int = 5,
) -> Tuple[pd.Series, pd.DataFrame]:
    idx = df_anime[df_anime["name"] == anime_title].index[0]
    target_anime = df_anime.iloc[idx]
    target_genres = set(str(target_anime["genre"]).split(", "))

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    recommendations = []
    for i, score in sim_scores:
        rec_item = df_anime.iloc[i]
        rec_genres = set(str(rec_item["genre"]).split(", "))
        overlap = len(target_genres.intersection(rec_genres)) / len(target_genres)
        recommendations.append({
            "anime_id": rec_item["anime_id"],
            "name": rec_item["name"],
            "genre": rec_item["genre"],
            "type": rec_item["type"],
            "similarity_score": round(float(score), 4),
            "relevan": overlap >= 0.5,
        })

    return target_anime, pd.DataFrame(recommendations)


class RecommenderNet(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_size: int = 50) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.sigmoid = nn.Sigmoid()

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user_idx)
        user_b = self.user_bias(user_idx).squeeze(-1)
        item_emb = self.item_embedding(item_idx)
        item_b = self.item_bias(item_idx).squeeze(-1)

        interaction = (user_emb * item_emb).sum(dim=-1)
        output = torch.sigmoid(interaction + user_b + item_b)
        return output


def prepare_collaborative_data(
    df_rating: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    user_ids = sorted(df_rating["user_id"].unique())
    item_ids = sorted(df_rating["anime_id"].unique())

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    idx_to_user = {i: u for i, u in enumerate(user_ids)}
    item_to_idx = {it: i for i, it in enumerate(item_ids)}
    idx_to_item = {i: it for i, it in enumerate(item_ids)}

    df_prepared = df_rating.copy()
    df_prepared["user_idx"] = df_prepared["user_id"].map(user_to_idx)
    df_prepared["item_idx"] = df_prepared["anime_id"].map(item_to_idx)

    min_rating = 1.0
    max_rating = 10.0
    df_prepared["norm_rating"] = (df_prepared["rating"] - min_rating) / (max_rating - min_rating)

    train_df, test_df = train_test_split(df_prepared, test_size=test_size, random_state=random_state)

    train_dataset = TensorDataset(
        torch.tensor(train_df["user_idx"].values, dtype=torch.long),
        torch.tensor(train_df["item_idx"].values, dtype=torch.long),
        torch.tensor(train_df["norm_rating"].values, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    test_users = torch.tensor(test_df["user_idx"].values, dtype=torch.long)
    test_items = torch.tensor(test_df["item_idx"].values, dtype=torch.long)
    test_labels = torch.tensor(test_df["norm_rating"].values, dtype=torch.float32)

    metadata = {
        "num_users": len(user_ids),
        "num_items": len(item_ids),
        "user_to_idx": user_to_idx,
        "idx_to_user": idx_to_user,
        "item_to_idx": item_to_idx,
        "idx_to_item": idx_to_item,
        "min_rating": min_rating,
        "max_rating": max_rating,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
    }

    return train_loader, test_users, test_items, test_labels, metadata


def train_collaborative_model(
    train_loader: DataLoader,
    test_users: torch.Tensor,
    test_items: torch.Tensor,
    test_labels: torch.Tensor,
    metadata: Dict[str, Any],
    epochs: int = 10,
    learning_rate: float = 0.005,
    output_dir: str = "figures",
) -> Tuple[RecommenderNet, Dict[str, List[float]]]:
    torch.manual_seed(42)
    model = RecommenderNet(metadata["num_users"], metadata["num_items"], embedding_size=50)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()

    history: Dict[str, List[float]] = {
        "train_loss": [], "test_loss": [], "train_rmse": [], "test_rmse": []
    }

    min_r, max_r = metadata["min_rating"], metadata["max_rating"]

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_batches = 0

        for batch_u, batch_i, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_u, batch_i)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        avg_train_loss = running_loss / total_batches
        train_rmse = float(np.sqrt(avg_train_loss) * (max_r - min_r))

        model.eval()
        with torch.no_grad():
            test_preds = model(test_users, test_items)
            test_loss = criterion(test_preds, test_labels).item()
            test_rmse = float(np.sqrt(test_loss) * (max_r - min_r))

        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(test_loss)
        history["train_rmse"].append(train_rmse)
        history["test_rmse"].append(test_rmse)

        print(f"[INFO] Epoch {epoch:02d}/{epochs:02d} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(1, epochs + 1), history["train_loss"], label="Train Loss (MSE)", marker="o")
    axes[0].plot(range(1, epochs + 1), history["test_loss"], label="Validation Loss (MSE)", marker="s")
    axes[0].set_title("Kurva Loss (MSE)", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(range(1, epochs + 1), history["train_rmse"], label="Train RMSE", marker="o", color="green")
    axes[1].plot(range(1, epochs + 1), history["test_rmse"], label="Validation RMSE", marker="s", color="red")
    axes[1].set_title("Kurva RMSE (Skala 1-10)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss_rmse_curve.png"), dpi=300)
    plt.close()

    return model, history


def get_collaborative_recommendations(
    user_id: int,
    model: RecommenderNet,
    df_anime: pd.DataFrame,
    df_rating: pd.DataFrame,
    metadata: Dict[str, Any],
    top_n: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    u_idx = metadata["user_to_idx"][user_id]
    past_ratings = df_rating[df_rating["user_id"] == user_id]
    rated_anime_ids = set(past_ratings["anime_id"].values)

    all_item_ids = list(metadata["item_to_idx"].keys())
    unvisited_ids = [aid for aid in all_item_ids if aid not in rated_anime_ids]
    unvisited_indices = [metadata["item_to_idx"][aid] for aid in unvisited_ids]

    user_tensor = torch.tensor([u_idx] * len(unvisited_indices), dtype=torch.long)
    item_tensor = torch.tensor(unvisited_indices, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        preds_norm = model(user_tensor, item_tensor).numpy()
        preds_rating = preds_norm * (metadata["max_rating"] - metadata["min_rating"]) + metadata["min_rating"]

    top_idx = np.argsort(preds_rating)[::-1][:top_n]
    top_anime_ids = [unvisited_ids[i] for i in top_idx]
    top_predicted_ratings = [preds_rating[i] for i in top_idx]

    rec_df = df_anime[df_anime["anime_id"].isin(top_anime_ids)].copy()
    rec_df["predicted_rating"] = rec_df["anime_id"].map(dict(zip(top_anime_ids, top_predicted_ratings)))
    rec_df = rec_df.sort_values(by="predicted_rating", ascending=False)

    top_history = past_ratings.rename(columns={"rating": "user_rating"}).merge(
        df_anime[["anime_id", "name", "genre", "type"]], on="anime_id"
    ).sort_values(by="user_rating", ascending=False).head(5)
    return top_history, rec_df


def main() -> None:
    print("=" * 60)
    print("[INFO] Memuat dataset ORIGINAL dari Kaggle...")
    print("=" * 60)
    df_anime_orig, df_rating_orig = load_original_datasets()
    print(f"[INFO] Dataset Original - Anime: {len(df_anime_orig):,} baris, Rating: {len(df_rating_orig):,} baris")
    print(f"[INFO] Missing values anime.csv: rating={df_anime_orig['rating'].isnull().sum()}, "
          f"genre={df_anime_orig['genre'].isnull().sum()}, type={df_anime_orig['type'].isnull().sum()}")
    print(f"[INFO] Rating -1 pada rating.csv: {(df_rating_orig['rating'] == -1).sum():,}")
    
    print("\n" + "=" * 60)
    print("[INFO] Data Preparation - Filtering & Pembersihan...")
    print("=" * 60)
    df_anime, df_rating = prepare_data(df_anime_orig, df_rating_orig, n_top_users=500)
    print(f"[INFO] Data Akhir - Anime: {len(df_anime):,}, Rating: {len(df_rating):,}")

    print("\n[INFO] Mengekspor visualisasi EDA ke 'figures/'...")
    export_eda_visualizations(df_anime, df_rating)

    print("\n[INFO] Membangun model Content-Based Filtering...")
    sim_matrix, tfidf = build_content_based_model(df_anime)
    print(f"[INFO] Dimensi TF-IDF: {tfidf.get_feature_names_out().shape[0]} genre unik")
    print(f"[INFO] Matriks Cosine Similarity: {sim_matrix.shape}")

    sample_query = "Kimi no Na wa."
    target_anime, cbf_recs = get_content_based_recommendations(sample_query, df_anime, sim_matrix, top_n=5)
    print(f"\n[DEMO CBF] Rekomendasi Top-5 untuk: '{target_anime['name']}' [{target_anime['genre']}]")
    print(cbf_recs[["anime_id", "name", "genre", "similarity_score", "relevan"]].to_string(index=False))

    precision_cbf = (cbf_recs["relevan"].sum() / len(cbf_recs)) * 100
    print(f"[EVAL CBF] Precision@5: {precision_cbf:.2f}%")

    print("\n[INFO] Menyiapkan data Collaborative Filtering...")
    train_loader, test_users, test_items, test_labels, metadata = prepare_collaborative_data(df_rating)
    print(f"[INFO] Data latih: {metadata['train_samples']:,}, Data uji: {metadata['test_samples']:,}")

    print("\n[INFO] Melatih model Deep Learning RecommenderNet (PyTorch)...")
    model, history = train_collaborative_model(train_loader, test_users, test_items, test_labels, metadata, epochs=10)

    final_test_rmse = history["test_rmse"][-1]
    final_test_mse = history["test_loss"][-1] * ((metadata["max_rating"] - metadata["min_rating"]) ** 2)
    print(f"\n[EVAL CF] Final Test RMSE: {final_test_rmse:.4f}, Test MSE: {final_test_mse:.4f}")

    sample_user_id = 226
    past_top_df, cf_recs = get_collaborative_recommendations(sample_user_id, model, df_anime, df_rating, metadata, top_n=5)
    print(f"\n[DEMO CF] Riwayat Anime Favorit Pengguna #{sample_user_id}:")
    print(past_top_df[["anime_id", "name", "genre", "user_rating"]].to_string(index=False))
    print(f"\n[DEMO CF] Top-5 Rekomendasi Anime Baru untuk Pengguna #{sample_user_id}:")
    print(cf_recs[["anime_id", "name", "genre", "predicted_rating"]].to_string(index=False))

    print("\n[INFO] Eksekusi selesai. Grafik tersimpan di folder 'figures/'.")


if __name__ == "__main__":
    main()
