

import shap
import numpy as np
import torch
from inference import model, tokenizer, device, LABEL_NAMES, predict



def _predict_proba(texts: list[str]) -> np.ndarray:
    results = []
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=192
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
        results.append(probs)
    return np.array(results)


# SHAP's Text masker handles tokenization/perturbation for us
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(_predict_proba, masker, output_names=LABEL_NAMES)


def explain(text: str, top_k: int = 5) -> dict:
   
    pred = predict(text)
    predicted_label = pred["label"]
    label_idx = LABEL_NAMES.index(predicted_label)

    shap_values = explainer([text])

    # shap_values[0, :, label_idx] gives per-token contribution to the predicted class
    tokens = shap_values.data[0]
    values = shap_values.values[0, :, label_idx]

    # pair tokens with their shap values, sort by absolute impact
    word_impact = list(zip(tokens, values))
    word_impact_sorted = sorted(word_impact, key=lambda x: abs(x[1]), reverse=True)

    top_words = [(str(w).strip(), float(v)) for w, v in word_impact_sorted[:top_k] if str(w).strip()]

    return {
        "text": text,
        "predicted_label": predicted_label,
        "confidence": pred["confidence"],
        "top_contributing_words": top_words,
    }


def print_explanation(text: str, top_k: int = 5):
    """Pretty-print an explanation to the console (useful for quick testing)."""
    result = explain(text, top_k=top_k)
    print(f"\nText: {result['text']}")
    print(f"Predicted: {result['predicted_label']} (confidence: {result['confidence']:.3f})")
    print("Top contributing words (word: shap_value, positive = pushed toward this label):")
    for word, value in result["top_contributing_words"]:
        direction = "+" if value > 0 else "-"
        print(f"  {direction} {word:20s} {value:+.4f}")


if __name__ == "__main__":
    test_examples = [
        "You people don't belong here, go back to where you came from.",
        "This is such a stupid take, you idiot.",
        "I had a great day at the park with my friends.",
    ]

    for ex in test_examples:
        print_explanation(ex)
