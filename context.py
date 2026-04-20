# """
# context.py — Applies a budget-context bonus to recommendation scores.
# Input:  list of (pid, score) tuples
# Output: list of (pid, score, context_bonus) tuples
# """


# # Budget multiplier map — higher budget = favour higher-priced items
# BUDGET_MAP = {
#     "Budget":   0.2,
#     "Balanced": 0.5,
#     "Premium":  0.8,
#     "Luxury":   1.0,
# }


# def apply_context_adjustment(recs: list, context: dict) -> list:
#     """
#     recs    : list of (pid, score) tuples
#     context : dict with key "budget" → one of Budget/Balanced/Premium/Luxury
#     Returns : list of (pid, score, context_bonus) tuples
#     """
#     budget_label = context.get("budget", "Balanced")
#     bonus = BUDGET_MAP.get(budget_label, 0.5)

#     adjusted = []
#     for rec in recs:
#         pid, score = rec[0], rec[1]
#         adjusted.append((pid, float(score), float(bonus)))

#     return adjusted



# updated code


"""
context.py — Applies a budget-context bonus to recommendation scores.
"""

# Budget multiplier map — higher budget = favour higher-priced items
BUDGET_MAP = {
    "Budget": 0.2,
    "Balanced": 0.5,
    "Premium": 0.8,
    "Luxury": 1.0,
    "Low": 0.2,      # For compatibility
    "Medium": 0.5,   # For compatibility
    "High": 0.8      # For compatibility
}


def apply_context_adjustment(recs: list, context: dict) -> list:
    """
    recs    : list of (pid, score) tuples
    context : dict with key "budget" → one of Budget/Balanced/Premium/Luxury
    Returns : list of (pid, score, context_bonus) tuples
    """
    budget_label = context.get("budget", "Balanced")
    bonus = BUDGET_MAP.get(budget_label, 0.5)

    adjusted = []
    for rec in recs:
        if len(rec) == 2:
            pid, score = rec
            adjusted.append((pid, float(score), float(bonus)))
        elif len(rec) >= 3:
            pid, score = rec[0], rec[1]
            adjusted.append((pid, float(score), float(bonus)))

    return adjusted