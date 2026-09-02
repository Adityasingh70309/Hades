
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# ---- CONFIG ----
MODEL_NAME = "Noveau/hades-hatexplain-deberta-v3"  
LABEL_NAMES = ["hatespeech", "normal", "offensive"]   


print(f"Loading model from {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()  # inference mode, disables dropout etc.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on {device}.")


def predict(text: str) -> dict:
  
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=192,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    pred_idx = int(probs.argmax())
    label = LABEL_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    prob_dict = {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}

    return {
        "text": text,
        "label": label,
        "confidence": confidence,
        "probabilities": prob_dict,
    }


def predict_batch(texts: list[str]) -> list[dict]:
    """Run inference on a list of texts. Returns a list of result dicts."""
    return [predict(t) for t in texts]


def predict_with_unsure_threshold(text: str, threshold: float = 0.6) -> dict:
   
    result = predict(text)
    if result["confidence"] < threshold:
        result["label"] = "unsure"
    return result


if __name__ == "__main__":
    # Quick manual test
    test_examples = [
        "I had a great day at the park with my friends.",
        "You people don't belong here, go back to where you came from.",
        "This is such a stupid take, you idiot.",
    ]

    for ex in test_examples:
        result = predict(ex)
        print(f"\nText: {result['text']}")
        print(f"Predicted: {result['label']} (confidence: {result['confidence']:.3f})")
        print(f"Probabilities: {result['probabilities']}")

    print("\n--- With Unsure threshold (0.6) ---")
    for ex in test_examples:
        result = predict_with_unsure_threshold(ex, threshold=0.5)
        print(f"{ex[:50]:50s} -> {result['label']} ({result['confidence']:.3f})")
