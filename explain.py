# """
# explain.py — Generates a natural-language explanation for why a product was recommended.
# """


# def build_explanation(pid: str, c_bonus: float, e_bonus: float, s_score: float) -> str:
#     """
#     pid     : product ID
#     c_bonus : context/budget relevance score (0–1)
#     e_bonus : emotion/mood match score (0–1)
#     s_score : content similarity score (0–1)
#     Returns : markdown-formatted explanation string
#     """

#     lines = []

#     # Similarity
#     if s_score >= 0.7:
#         lines.append(f"🔗 **High similarity** to your selected product (score: `{s_score:.2f}`). Very closely related features and category.")
#     elif s_score >= 0.4:
#         lines.append(f"🔗 **Moderate similarity** to your selection (score: `{s_score:.2f}`). Shares several characteristics.")
#     else:
#         lines.append(f"🔗 **Complementary match** (similarity: `{s_score:.2f}`). Different but relevant to your search context.")

#     # Context / Budget
#     if c_bonus >= 0.8:
#         lines.append(f"💰 **Premium budget match** — this product aligns with your high-end preference (context score: `{c_bonus:.2f}`).")
#     elif c_bonus >= 0.5:
#         lines.append(f"💰 **Balanced value** — fits well within your stated budget range (context score: `{c_bonus:.2f}`).")
#     else:
#         lines.append(f"💰 **Budget-friendly pick** — recommended for cost-conscious shoppers (context score: `{c_bonus:.2f}`).")

#     # Emotion / Mood
#     if e_bonus >= 0.7:
#         lines.append(f"😊 **Mood alignment is high** — your positive intent boosted this recommendation (emotion score: `{e_bonus:.2f}`).")
#     elif e_bonus >= 0.4:
#         lines.append(f"😐 **Neutral mood match** — a reliable, objective recommendation (emotion score: `{e_bonus:.2f}`).")
#     else:
#         lines.append(f"🤔 **Exploratory pick** — suggested to broaden your options despite cautious mood signal (emotion score: `{e_bonus:.2f}`).")

#     # Overall composite
#     composite = round((s_score + c_bonus + e_bonus) / 3, 2)
#     lines.append(f"\n**Overall AI Confidence Score:** `{composite}/1.0`")

#     return "\n\n".join(lines)



# updated code
"""
explain.py — Generates a natural-language explanation for why a product was recommended.
"""


def build_explanation(pid: str, c_bonus: float, e_bonus: float, s_score: float) -> str:
    """
    pid     : product ID
    c_bonus : context/budget relevance score (0-1)
    e_bonus : emotion/mood match score (0-1)
    s_score : content similarity score (0-1)
    Returns : markdown-formatted explanation string
    """

    lines = []

    # Similarity
    if s_score >= 0.7:
        lines.append(f"🔗 **High similarity** to your selected product (score: `{s_score:.2f}`)")
    elif s_score >= 0.4:
        lines.append(f"🔗 **Moderate similarity** to your selection (score: `{s_score:.2f}`)")
    else:
        lines.append(f"🔗 **Complementary match** (similarity: `{s_score:.2f}`)")

    # Context / Budget
    if c_bonus >= 0.8:
        lines.append(f"💰 **Premium budget match** (context score: `{c_bonus:.2f}`)")
    elif c_bonus >= 0.5:
        lines.append(f"💰 **Balanced value** (context score: `{c_bonus:.2f}`)")
    else:
        lines.append(f"💰 **Budget-friendly pick** (context score: `{c_bonus:.2f}`)")

    # Emotion / Mood
    if e_bonus >= 0.7:
        lines.append(f"😊 **Mood alignment is high** (emotion score: `{e_bonus:.2f}`)")
    elif e_bonus >= 0.4:
        lines.append(f"😐 **Neutral mood match** (emotion score: `{e_bonus:.2f}`)")
    else:
        lines.append(f"🤔 **Exploratory pick** (emotion score: `{e_bonus:.2f}`)")

    # Overall composite
    composite = round((s_score + c_bonus + e_bonus) / 3, 2)
    lines.append(f"\n**Overall AI Confidence Score:** `{composite}/1.0`")

    return "\n\n".join(lines)