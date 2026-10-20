
from flask import Flask, jsonify, request

from recommendation.recommender import recommend_for_user

app = Flask(__name__)


@app.get("/")
def home():
    return jsonify({
        "status": "success",
        "message": "LibSphere Recommendation API is running"
    })


@app.get("/api/recommendations")
def get_recommendations():
    # Temporary test data — replace with real data later.
    borrowed_books = []

    recommendations = recommend_for_user(
        borrowed_books,
        programme=None,
        subject_area=None,
        limit=min(max(request.args.get("limit", 8, type=int), 1), 20),
    )

    return jsonify({
        "recommendations": recommendations
    })


if __name__ == "__main__":
    app.run(debug=True)
