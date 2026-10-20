
from recommendation.recommender import recommend_for_user

# Example borrowing history
borrowed_books = [
    {
        "title": "Introduction to Computer Science",
        "isbn": ""
    }
]

# Generate recommendations
try:
    recommendations = recommend_for_user(
        borrowed_books=borrowed_books,
        limit=5
    )

    print("\nRecommended books:\n")

    for i, book in enumerate(recommendations, start=1):
        print(f"{i}. {book.get('title', 'Unknown title')}")
        print(f"   Author: {book.get('authors', 'Unknown')}")
        print(f"   Reason: {book.get('reason', 'Similar to your interests')}")
        print()

except Exception as e:
    print("Recommendation test failed:")
    print(e)
