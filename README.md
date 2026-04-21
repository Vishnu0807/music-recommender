# Personalized Music Recommendation System

This project builds a complete personalized music recommendation system on top of the Last.FM HetRec 2011 2K-user dataset. It exposes a Flask API, a single-file React dashboard, collaborative filtering recommenders, Truncated SVD matrix factorization, and offline evaluation using Precision@K and Recall@K.

## Project Structure

```text
music_recommendation/
├── app.py
├── recommender.py
├── requirements.txt
├── README.md
├── data/
│   ├── user_artists.dat
│   └── artists.dat
└── templates/
    └── index.html
```

## Dataset

- Source: [GroupLens HetRec 2011 Last.FM Dataset](https://grouplens.org/datasets/hetrec-2011/)
- Version used: `hetrec2011-lastfm-2k.zip`
- Core files:
  - `data/user_artists.dat`: user to artist play counts
  - `data/artists.dat`: artist metadata

## Modeling Pipeline

### 1. Data Preparation

- Reads both `.dat` files with `pandas.read_csv(..., sep="\t")`
- Applies log normalization to play counts:

```python
rating = log(1 + weight)
```

- Builds a user-item matrix with users on rows, artists on columns, and missing values filled with zero

### 2. User-Based Collaborative Filtering

- Computes cosine similarity between user vectors
- Uses the top 20 nearest neighbors to estimate unseen artist scores
- Returns the top-N unrated artists for a selected user

### 3. Item-Based Similar Artist Search

- Uses artist listener vectors from the transposed user-item matrix
- Computes cosine similarity for artist-to-artist lookup
- Returns the most similar artists for a given artist name

### 4. Truncated SVD Matrix Factorization

- Uses `sklearn.decomposition.TruncatedSVD`
- Default latent dimension: `50`
- Reconstructs scores from latent user and item factors
- Recommends unseen artists from the reconstructed preference surface

### 5. Evaluation

- Splits 20% of each user's known interactions into a test set
- Trains User-CF and Truncated SVD on the remaining 80%
- Reports:
  - Precision@5
  - Precision@10
  - Recall@5
  - Recall@10

## API Endpoints

### `GET /`

Serves the dashboard UI.

### `GET /api/users`

Returns 50 sample user IDs.

### `GET /api/recommend/usercf?user_id=X&n=10`

Returns User-CF recommendations.

### `GET /api/recommend/svd?user_id=X&n=10`

Returns Truncated SVD recommendations.

### `GET /api/similar?artist=NAME&n=10`

Returns the top similar artists for the provided artist name.

### `GET /api/metrics`

Returns offline evaluation metrics for both models.

### `GET /api/topusers`

Returns the 10 most active users by total play count.

### `GET /api/topartists`

Returns the 10 most played artists globally.

## Frontend Dashboard

The dashboard is implemented in a single `templates/index.html` file using:

- React via CDN
- Babel Standalone for JSX
- Chart.js for charts
- Native `fetch()` for API calls

Tabs included:

- User Recommendations
- Similar Artists
- Model Metrics
- Top Charts

## Setup

### 1. Activate the virtual environment

PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Run the application

```powershell
.\venv\Scripts\python.exe app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Example API Calls

```powershell
Invoke-RestMethod "http://127.0.0.1:5000/api/users"
Invoke-RestMethod "http://127.0.0.1:5000/api/recommend/usercf?user_id=2&n=5"
Invoke-RestMethod "http://127.0.0.1:5000/api/recommend/svd?user_id=2&n=5"
Invoke-RestMethod "http://127.0.0.1:5000/api/similar?artist=Radiohead&n=5"
Invoke-RestMethod "http://127.0.0.1:5000/api/metrics"
```

## Notes

- The app prints loading progress to the console while it initializes models and computes metrics.
- Truncated SVD is used instead of dense full SVD because it is much better suited to sparse recommendation data.
- Artist-name lookup supports exact matches first, then a fallback partial-name search.

## Allowed Python Packages

This project stays within the requested package set:

- `flask`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`

## Troubleshooting

- If package installation fails, confirm the virtual environment exists and has internet access for `pip`.
- If the app starts slowly, that is expected on first launch because similarity matrices and evaluation metrics are computed at startup.
- If an artist search returns an error, try a more exact artist name from `artists.dat`.
