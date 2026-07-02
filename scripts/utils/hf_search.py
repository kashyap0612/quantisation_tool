# scripts/utils/hf_search.py

from huggingface_hub import list_models


def search_models(query: str, limit: int = 20):
    """
    Search Hugging Face models and return
    useful metadata for UI display.
    """

    models = list_models(
        search=query,
        limit=limit,
        sort="downloads",
        direction=-1
    )

    results = []

    for model in models:
        results.append({
            "id": model.id,
            "downloads": getattr(model, "downloads", 0) or 0,
            "likes": getattr(model, "likes", 0) or 0,
        })

    return results


if __name__ == "__main__":

    results = search_models("mobilenet")

    for i, model in enumerate(results, start=1):
        print(
            f"{i}. {model['id']} | "
            f"Downloads: {model['downloads']} | "
            f"Likes: {model['likes']}"
        )