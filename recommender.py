from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
USER_ARTISTS_PATH = DATA_DIR / "user_artists.dat"
ARTISTS_PATH = DATA_DIR / "artists.dat"


class LastFMRecommender:
    def __init__(
        self,
        user_artists_path: Path = USER_ARTISTS_PATH,
        artists_path: Path = ARTISTS_PATH,
        n_components: int = 50,
        neighbor_count: int = 20,
        random_state: int = 42,
    ) -> None:
        self.user_artists_path = Path(user_artists_path)
        self.artists_path = Path(artists_path)
        self.n_components = n_components
        self.neighbor_count = neighbor_count
        self.random_state = random_state

        print("[Recommender] Loading Last.FM dataset...", flush=True)
        self.interactions = pd.read_csv(self.user_artists_path, sep="\t")
        self.artists = pd.read_csv(self.artists_path, sep="\t")
        self.interactions["rating"] = np.log1p(self.interactions["weight"]).astype(np.float32)

        self.artist_lookup = (
            self.artists[["id", "name", "url", "pictureURL"]]
            .drop_duplicates(subset=["id"])
            .set_index("id")
        )
        self.artist_id_to_name = self.artist_lookup["name"].to_dict()
        self.artist_name_to_id = self._build_artist_name_index()

        print("[Recommender] Building full user-item matrix...", flush=True)
        self.user_item_full = self._build_user_item_matrix(self.interactions)
        self.full_user_ids = self.user_item_full.index.to_numpy()
        self.full_artist_ids = self.user_item_full.columns.to_numpy()
        self.full_user_index = {user_id: idx for idx, user_id in enumerate(self.full_user_ids)}
        self.full_artist_index = {artist_id: idx for idx, artist_id in enumerate(self.full_artist_ids)}
        self.user_item_full_values = self.user_item_full.to_numpy(dtype=np.float32, copy=True)
        self.artist_user_matrix = csr_matrix(self.user_item_full_values.T)

        print("[Recommender] Computing user-user cosine similarity...", flush=True)
        self.user_similarity_full = cosine_similarity(self.user_item_full_values)
        np.fill_diagonal(self.user_similarity_full, 0.0)

        print("[Recommender] Fitting Truncated SVD model...", flush=True)
        self.svd_full = self._fit_svd(self.user_item_full_values)

        print("[Recommender] Preparing evaluation split...", flush=True)
        train_interactions, test_interactions = self._split_train_test(self.interactions)
        self.eval_test_interactions = test_interactions
        self.eval_test_items = (
            test_interactions.groupby("userID")["artistID"].apply(set).to_dict()
            if not test_interactions.empty
            else {}
        )

        print("[Recommender] Building evaluation matrices...", flush=True)
        self.user_item_train = self._build_user_item_matrix(
            train_interactions,
            user_ids=self.user_item_full.index,
            artist_ids=self.user_item_full.columns,
        )
        self.user_item_train_values = self.user_item_train.to_numpy(dtype=np.float32, copy=True)
        self.train_user_ids = self.user_item_train.index.to_numpy()
        self.train_artist_ids = self.user_item_train.columns.to_numpy()
        self.train_user_index = {user_id: idx for idx, user_id in enumerate(self.train_user_ids)}
        self.train_artist_index = {artist_id: idx for idx, artist_id in enumerate(self.train_artist_ids)}

        print("[Recommender] Computing evaluation user-user similarity...", flush=True)
        self.user_similarity_train = cosine_similarity(self.user_item_train_values)
        np.fill_diagonal(self.user_similarity_train, 0.0)

        print("[Recommender] Fitting evaluation Truncated SVD model...", flush=True)
        self.svd_train = self._fit_svd(self.user_item_train_values)

        print("[Recommender] Evaluating recommendation quality...", flush=True)
        self.metrics = self.evaluate_models()
        self.sample_users = [int(user_id) for user_id in self.full_user_ids[:50]]
        self.top_users = self._compute_top_users()
        self.top_artists = self._compute_top_artists()
        print("[Recommender] Initialization complete.", flush=True)

    def _build_artist_name_index(self) -> Dict[str, int]:
        name_index: Dict[str, int] = {}
        for row in self.artists.itertuples(index=False):
            normalized_name = self._normalize_artist_name(str(row.name))
            name_index.setdefault(normalized_name, int(row.id))
        return name_index

    @staticmethod
    def _normalize_artist_name(name: str) -> str:
        return " ".join(name.strip().lower().split())

    def _build_user_item_matrix(
        self,
        interactions: pd.DataFrame,
        user_ids: Optional[pd.Index] = None,
        artist_ids: Optional[pd.Index] = None,
    ) -> pd.DataFrame:
        matrix = interactions.pivot_table(
            index="userID",
            columns="artistID",
            values="rating",
            aggfunc="sum",
            fill_value=0.0,
        )
        if user_ids is not None:
            matrix = matrix.reindex(index=user_ids, fill_value=0.0)
        if artist_ids is not None:
            matrix = matrix.reindex(columns=artist_ids, fill_value=0.0)
        return matrix.astype(np.float32)

    def _split_train_test(
        self,
        interactions: pd.DataFrame,
        test_ratio: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(self.random_state)
        train_parts: List[pd.DataFrame] = []
        test_parts: List[pd.DataFrame] = []

        for _, user_frame in interactions.groupby("userID", sort=False):
            if len(user_frame) <= 1:
                train_parts.append(user_frame)
                continue

            desired_test = max(1, int(np.ceil(len(user_frame) * test_ratio)))
            test_size = min(desired_test, len(user_frame) - 1)
            chosen_indices = rng.choice(user_frame.index.to_numpy(), size=test_size, replace=False)
            mask = user_frame.index.isin(chosen_indices)
            test_parts.append(user_frame.loc[mask])
            train_parts.append(user_frame.loc[~mask])

        train_df = pd.concat(train_parts, ignore_index=True)
        test_df = pd.concat(test_parts, ignore_index=True) if test_parts else interactions.iloc[0:0].copy()
        return train_df, test_df

    def _fit_svd(self, matrix_values: np.ndarray) -> Dict[str, np.ndarray]:
        max_components = max(1, min(self.n_components, min(matrix_values.shape) - 1))
        sparse_matrix = csr_matrix(matrix_values)
        svd = TruncatedSVD(n_components=max_components, random_state=self.random_state)
        transformed = svd.fit_transform(sparse_matrix).astype(np.float32)
        singular_values = svd.singular_values_.astype(np.float32)
        safe_singular_values = np.where(singular_values == 0, 1e-8, singular_values)
        u = transformed / safe_singular_values
        sigma = np.diag(singular_values)
        vt = svd.components_.astype(np.float32)
        user_features = (u * singular_values).astype(np.float32)
        return {
            "model": svd,
            "u": u.astype(np.float32),
            "sigma": sigma.astype(np.float32),
            "vt": vt,
            "user_features": user_features,
        }

    def _resolve_artist_id(self, artist_name: str) -> int:
        normalized_name = self._normalize_artist_name(artist_name)
        if normalized_name in self.artist_name_to_id:
            return self.artist_name_to_id[normalized_name]

        partial_match = self.artists[self.artists["name"].str.lower().str.contains(normalized_name, na=False)]
        if not partial_match.empty:
            return int(partial_match.iloc[0]["id"])

        raise ValueError(f"Artist '{artist_name}' was not found in the catalog.")

    def _artist_payload(self, artist_id: int) -> Dict[str, Optional[str]]:
        artist_row = self.artist_lookup.loc[artist_id] if artist_id in self.artist_lookup.index else None
        if artist_row is None:
            return {
                "artist_id": int(artist_id),
                "artist_name": f"Artist {artist_id}",
                "url": None,
                "picture_url": None,
            }
        return {
            "artist_id": int(artist_id),
            "artist_name": str(artist_row["name"]),
            "url": None if pd.isna(artist_row["url"]) else str(artist_row["url"]),
            "picture_url": None if pd.isna(artist_row["pictureURL"]) else str(artist_row["pictureURL"]),
        }

    def _recommend_with_user_cf(
        self,
        user_id: int,
        n: int,
        matrix_values: np.ndarray,
        user_index: Dict[int, int],
        artist_ids: np.ndarray,
        similarity_matrix: np.ndarray,
    ) -> List[Dict[str, float]]:
        if user_id not in user_index:
            raise ValueError(f"User '{user_id}' does not exist in the interaction matrix.")

        user_idx = user_index[user_id]
        similarity_vector = similarity_matrix[user_idx].copy()
        neighbor_indices = np.argsort(similarity_vector)[::-1]
        neighbor_indices = neighbor_indices[neighbor_indices != user_idx][: self.neighbor_count]

        if neighbor_indices.size == 0:
            return []

        neighbor_similarities = similarity_vector[neighbor_indices].astype(np.float32)
        neighbor_ratings = matrix_values[neighbor_indices]
        rated_mask = (neighbor_ratings > 0).astype(np.float32)

        weighted_sum = (neighbor_ratings * neighbor_similarities[:, None] * rated_mask).sum(axis=0)
        similarity_sum = (np.abs(neighbor_similarities)[:, None] * rated_mask).sum(axis=0)
        scores = np.divide(
            weighted_sum,
            similarity_sum,
            out=np.zeros(matrix_values.shape[1], dtype=np.float32),
            where=similarity_sum > 0,
        )

        seen_mask = matrix_values[user_idx] > 0
        scores[seen_mask] = -np.inf

        candidate_indices = np.where(np.isfinite(scores))[0]
        if candidate_indices.size == 0:
            return []

        top_n = min(n, candidate_indices.size)
        ranked_candidate_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1][:top_n]]

        recommendations: List[Dict[str, float]] = []
        for artist_idx in ranked_candidate_indices:
            artist_id = int(artist_ids[artist_idx])
            payload = self._artist_payload(artist_id)
            payload["score"] = float(scores[artist_idx])
            recommendations.append(payload)
        return recommendations

    def _recommend_with_svd(
        self,
        user_id: int,
        n: int,
        matrix_values: np.ndarray,
        user_index: Dict[int, int],
        artist_ids: np.ndarray,
        svd_model: Dict[str, np.ndarray],
    ) -> List[Dict[str, float]]:
        if user_id not in user_index:
            raise ValueError(f"User '{user_id}' does not exist in the interaction matrix.")

        user_idx = user_index[user_id]
        reconstructed_scores = svd_model["user_features"][user_idx] @ svd_model["vt"]
        reconstructed_scores = reconstructed_scores.astype(np.float32, copy=False)

        seen_mask = matrix_values[user_idx] > 0
        reconstructed_scores[seen_mask] = -np.inf

        candidate_indices = np.where(np.isfinite(reconstructed_scores))[0]
        if candidate_indices.size == 0:
            return []

        top_n = min(n, candidate_indices.size)
        ranked_candidate_indices = candidate_indices[
            np.argsort(reconstructed_scores[candidate_indices])[::-1][:top_n]
        ]

        recommendations: List[Dict[str, float]] = []
        for artist_idx in ranked_candidate_indices:
            artist_id = int(artist_ids[artist_idx])
            payload = self._artist_payload(artist_id)
            payload["score"] = float(reconstructed_scores[artist_idx])
            recommendations.append(payload)
        return recommendations

    def get_user_cf_recommendations(self, user_id: int, n: int = 10) -> List[Dict[str, float]]:
        return self._recommend_with_user_cf(
            user_id=user_id,
            n=n,
            matrix_values=self.user_item_full_values,
            user_index=self.full_user_index,
            artist_ids=self.full_artist_ids,
            similarity_matrix=self.user_similarity_full,
        )

    def get_svd_recommendations(self, user_id: int, n: int = 10) -> List[Dict[str, float]]:
        return self._recommend_with_svd(
            user_id=user_id,
            n=n,
            matrix_values=self.user_item_full_values,
            user_index=self.full_user_index,
            artist_ids=self.full_artist_ids,
            svd_model=self.svd_full,
        )

    def get_similar_artists(self, artist_name: str, n: int = 10) -> List[Dict[str, float]]:
        artist_id = self._resolve_artist_id(artist_name)
        if artist_id not in self.full_artist_index:
            raise ValueError(f"Artist '{artist_name}' has no interactions in the dataset.")

        artist_idx = self.full_artist_index[artist_id]
        similarity_scores = cosine_similarity(
            self.artist_user_matrix[artist_idx],
            self.artist_user_matrix,
        ).ravel()
        similarity_scores[artist_idx] = -np.inf

        candidate_indices = np.where(np.isfinite(similarity_scores))[0]
        top_n = min(n, candidate_indices.size)
        ranked_indices = candidate_indices[np.argsort(similarity_scores[candidate_indices])[::-1][:top_n]]

        similar_artists: List[Dict[str, float]] = []
        for candidate_idx in ranked_indices:
            similar_artist_id = int(self.full_artist_ids[candidate_idx])
            payload = self._artist_payload(similar_artist_id)
            payload["score"] = float(similarity_scores[candidate_idx])
            similar_artists.append(payload)
        return similar_artists

    def _compute_top_users(self) -> List[Dict[str, int]]:
        top_users = (
            self.interactions.groupby("userID")
            .agg(total_plays=("weight", "sum"), unique_artists=("artistID", "nunique"))
            .reset_index()
            .sort_values("total_plays", ascending=False)
            .head(10)
        )
        return [
            {
                "user_id": int(row.userID),
                "total_plays": int(row.total_plays),
                "unique_artists": int(row.unique_artists),
            }
            for row in top_users.itertuples(index=False)
        ]

    def _compute_top_artists(self) -> List[Dict[str, int]]:
        top_artists = (
            self.interactions.groupby("artistID")
            .agg(total_plays=("weight", "sum"), listener_count=("userID", "nunique"))
            .reset_index()
            .sort_values("total_plays", ascending=False)
            .head(10)
        )
        results: List[Dict[str, int]] = []
        for row in top_artists.itertuples(index=False):
            payload = self._artist_payload(int(row.artistID))
            payload["total_plays"] = int(row.total_plays)
            payload["listener_count"] = int(row.listener_count)
            results.append(payload)
        return results

    def _evaluate_single_model(self, model_name: str, ks: tuple[int, ...] = (5, 10)) -> Dict[str, float]:
        scores = {k: {"precision": [], "recall": []} for k in ks}
        max_k = max(ks)
        eligible_users = [user_id for user_id in self.eval_test_items if user_id in self.train_user_index]

        for user_id in eligible_users:
            test_items = self.eval_test_items.get(user_id, set())
            if not test_items:
                continue

            if model_name == "user_cf":
                recommendations = self._recommend_with_user_cf(
                    user_id=user_id,
                    n=max_k,
                    matrix_values=self.user_item_train_values,
                    user_index=self.train_user_index,
                    artist_ids=self.train_artist_ids,
                    similarity_matrix=self.user_similarity_train,
                )
            else:
                recommendations = self._recommend_with_svd(
                    user_id=user_id,
                    n=max_k,
                    matrix_values=self.user_item_train_values,
                    user_index=self.train_user_index,
                    artist_ids=self.train_artist_ids,
                    svd_model=self.svd_train,
                )

            recommended_ids = [item["artist_id"] for item in recommendations]
            if not recommended_ids:
                for k in ks:
                    scores[k]["precision"].append(0.0)
                    scores[k]["recall"].append(0.0)
                continue

            for k in ks:
                hits = len(set(recommended_ids[:k]) & test_items)
                scores[k]["precision"].append(hits / k)
                scores[k]["recall"].append(hits / len(test_items))

        model_metrics: Dict[str, float] = {}
        for k in ks:
            precision_key = f"precision@{k}"
            recall_key = f"recall@{k}"
            model_metrics[precision_key] = float(np.mean(scores[k]["precision"])) if scores[k]["precision"] else 0.0
            model_metrics[recall_key] = float(np.mean(scores[k]["recall"])) if scores[k]["recall"] else 0.0
        return model_metrics

    def _print_metrics_table(self, metrics: Dict[str, Dict[str, float]]) -> None:
        print("\n[Recommender] Evaluation Results", flush=True)
        print(f"{'Model':<12} {'P@5':>10} {'P@10':>10} {'R@5':>10} {'R@10':>10}", flush=True)
        for model_name, model_metrics in metrics.items():
            print(
                f"{model_name:<12} "
                f"{model_metrics['precision@5']:>10.4f} "
                f"{model_metrics['precision@10']:>10.4f} "
                f"{model_metrics['recall@5']:>10.4f} "
                f"{model_metrics['recall@10']:>10.4f}",
                flush=True,
            )

    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        metrics = {
            "user_cf": self._evaluate_single_model("user_cf"),
            "svd": self._evaluate_single_model("svd"),
        }
        self._print_metrics_table(metrics)
        return metrics

    def get_sample_users(self) -> List[int]:
        return self.sample_users

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        return self.metrics

    def get_top_users(self) -> List[Dict[str, int]]:
        return self.top_users

    def get_top_artists(self) -> List[Dict[str, int]]:
        return self.top_artists


_DEFAULT_RECOMMENDER: Optional[LastFMRecommender] = None


def get_default_recommender() -> LastFMRecommender:
    global _DEFAULT_RECOMMENDER
    if _DEFAULT_RECOMMENDER is None:
        _DEFAULT_RECOMMENDER = LastFMRecommender()
    return _DEFAULT_RECOMMENDER


def get_user_cf_recommendations(user_id: int, N: int = 10) -> List[Dict[str, float]]:
    return get_default_recommender().get_user_cf_recommendations(user_id, N)


def get_svd_recommendations(user_id: int, N: int = 10) -> List[Dict[str, float]]:
    return get_default_recommender().get_svd_recommendations(user_id, N)


def get_similar_artists(artist_name: str, N: int = 10) -> List[Dict[str, float]]:
    return get_default_recommender().get_similar_artists(artist_name, N)


def evaluate_models() -> Dict[str, Dict[str, float]]:
    return get_default_recommender().evaluate_models()
