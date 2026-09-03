import argparse
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from text_pipeline import classify_text
from slang_module import get_slang_context
from offensive_lexicon import lexicon_boost, lexicon_hits


class HADESState(TypedDict):
    text: str
    text_result: Optional[dict]
    slang_context: Optional[dict]
    final_label: Optional[str]
    confidence_summary: Optional[dict]
    reasoning: Optional[str]
    explanation: Optional[str]


def classifier_agent(state: HADESState) -> HADESState:
    print("\n[ClassifierAgent] Running ...")

    text_result = classify_text(state["text"], explain=False)
    print(f"  Text -> {text_result['label']} ({text_result['confidence']:.3f})")

    # run slang lookup if model isnt very confident
    slang_context = None
    if text_result["confidence"] < 0.75:
        slang_context = get_slang_context(state["text"])
        if slang_context["oov_tokens"]:
            print(f"  Slang detected: {slang_context['oov_tokens']}")

    return {
        **state,
        "text_result": text_result,
        "slang_context": slang_context,
    }


def fusion_agent(state: HADESState) -> HADESState:
    print("\n[FusionAgent] Running ...")

    text_result = state["text_result"]
    slang_context = state.get("slang_context")

    probs = dict(text_result["probabilities"])
    raw_label = text_result["label"]
    raw_conf = text_result["confidence"]

    # offensive-lexicon soft boost: only for borderline cases
    hits = lexicon_hits(state["text"])
    boosted = False
    if hits and (raw_conf < 0.75 or probs.get("offensive", 0) > 0.25):
        probs = lexicon_boost(probs, state["text"])
        boosted = True

    # re-derive label from (possibly boosted) probs
    final_label = max(probs, key=probs.get)
    best_conf = float(probs[final_label])

    slang_note = ""
    if slang_context and slang_context["oov_tokens"]:
        resolved = [t for t, d in slang_context["definitions"].items() if d]
        slang_note = f" slang detected: {slang_context['oov_tokens']} resolved: {resolved}."

    lex_note = f" lexicon_hits: {hits} boost: {boosted}." if hits else ""
    # keep raw for debugging
    reasoning = (
        f"text={raw_label}({raw_conf:.2f})->fusion={final_label}({best_conf:.2f}).{slang_note}{lex_note} "
        f"probs: {probs} raw: {text_result['probabilities']}"
    )

    print(f"  Final label: {final_label} ({best_conf:.3f}){' [lexicon boosted]' if boosted else ''}")

    return {
        **state,
        "final_label": final_label,
        "confidence_summary": probs,
        "reasoning": reasoning,
    }


def explanation_agent(state: HADESState) -> HADESState:
    print("\n[ExplanationAgent] Running ...")

    final_label = state.get("final_label")
    text_result = state.get("text_result")
    slang_context = state.get("slang_context")
    reasoning = state.get("reasoning", "")

    lines = []
    lines.append(f"HADES Decision: {final_label.upper()}")
    lines.append("=" * 40)
    lines.append(f"Text signal: {text_result['label']} ({text_result['confidence']:.3f})")

    # shap only fires on flagged content, its slow on cpu
    if final_label in ["hatespeech", "offensive"]:
        try:
            from explain import explain as shap_explain
            shap_result = shap_explain(state["text"])
            top = shap_result["top_contributing_words"][:3]
            word_str = ", ".join(f"'{w}' ({v:+.3f})" for w, v in top)
            lines.append(f"  -> SHAP top words: {word_str}")
        except Exception:
            pass

    if slang_context and slang_context["oov_tokens"]:
        lines.append(f"OOV/slang: {slang_context['oov_tokens']}")
        for term, defn in slang_context["definitions"].items():
            if defn and term.lower() not in defn.lower()[:50]:
                lines.append(f"  -> {term}: {defn[:80]}...")

    lines.append("")
    lines.append(f"Reasoning: {reasoning}")

    explanation = "\n".join(lines)
    print(f"\n{explanation}")

    return {**state, "explanation": explanation}


def build_graph():
    g = StateGraph(HADESState)

    g.add_node("classifier_agent", classifier_agent)
    g.add_node("fusion_agent", fusion_agent)
    g.add_node("explanation_agent", explanation_agent)

    g.add_edge(START, "classifier_agent")
    g.add_edge("classifier_agent", "fusion_agent")
    g.add_edge("fusion_agent", "explanation_agent")
    g.add_edge("explanation_agent", END)

    return g.compile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    app = build_graph()

    state: HADESState = {
        "text": args.text,
        "text_result": None,
        "slang_context": None,
        "final_label": None,
        "confidence_summary": None,
        "reasoning": None,
        "explanation": None,
    }

    result = app.invoke(state)

    print("\n" + "=" * 40)
    print(f"Label    : {result['final_label']}")
    print(f"Confidence: {result['confidence_summary']}")