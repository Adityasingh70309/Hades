import argparse
from inference import predict, LABEL_NAMES
from explain import explain as shap_explain


def classify_text(text: str, explain: bool = False) -> dict:
    result = predict(text)

    output = {
        "text": text,
        "label": result["label"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "is_unsure": False,
    }

    if explain:
        explanation = shap_explain(text)
        output["top_contributing_words"] = explanation["top_contributing_words"]

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="text to classify")
    parser.add_argument("--explain", action="store_true", help="run SHAP explanation")
    args = parser.parse_args()

    r = classify_text(args.text, explain=args.explain)

    print(f"\nText     : {r['text']}")
    print(f"Label    : {r['label']}")
    print(f"Confidence: {r['confidence']:.3f}")
    print(f"Probs    : {r['probabilities']}")

    if args.explain and "top_contributing_words" in r:
        print(f"SHAP     : {r['top_contributing_words']}")