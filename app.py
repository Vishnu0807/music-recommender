from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from recommender import get_default_recommender


print("[App] Initializing recommender engine...", flush=True)
recommender = get_default_recommender()
print("[App] Recommender engine is ready.", flush=True)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


def _parse_positive_int(value: str, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must be an integer.") from exc

    if parsed <= 0:
        raise ValueError(f"'{field_name}' must be greater than zero.")
    return parsed


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/users")
def api_users():
    return jsonify({"users": recommender.get_sample_users()})


@app.route("/api/recommend/usercf")
def api_user_cf_recommendations():
    try:
        user_id = _parse_positive_int(request.args.get("user_id"), "user_id")
        n = _parse_positive_int(request.args.get("n", 10), "n")
        recommendations = recommender.get_user_cf_recommendations(user_id, n)
        return jsonify({"user_id": user_id, "model": "user_cf", "recommendations": recommendations})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/recommend/svd")
def api_svd_recommendations():
    try:
        user_id = _parse_positive_int(request.args.get("user_id"), "user_id")
        n = _parse_positive_int(request.args.get("n", 10), "n")
        recommendations = recommender.get_svd_recommendations(user_id, n)
        return jsonify({"user_id": user_id, "model": "svd", "recommendations": recommendations})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/similar")
def api_similar_artists():
    try:
        artist_name = request.args.get("artist", "").strip()
        if not artist_name:
            raise ValueError("'artist' query parameter is required.")
        n = _parse_positive_int(request.args.get("n", 10), "n")
        results = recommender.get_similar_artists(artist_name, n)
        return jsonify({"artist": artist_name, "results": results})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/metrics")
def api_metrics():
    return jsonify(recommender.get_metrics())


@app.route("/api/topusers")
def api_top_users():
    return jsonify({"top_users": recommender.get_top_users()})


@app.route("/api/topartists")
def api_top_artists():
    return jsonify({"top_artists": recommender.get_top_artists()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
