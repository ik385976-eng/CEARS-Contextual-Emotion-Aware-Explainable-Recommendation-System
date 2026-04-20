# """
# emotion.py — Applies a mood/emotion bonus to recommendation scores.
# Input:  list of (pid, score, c_bonus) tuples  +  mood string  +  id→name map
# Output: list of (pid, score, c_bonus, emotion_bonus) tuples
# """

# from textblob import TextBlob


# def _mood_score(mood_text: str) -> float:
#     """
#     Convert a free-text mood description into a 0–1 emotion bonus.
#     Positive mood → higher bonus (user is more engaged).
#     Negative mood → slight dampening.
#     """
#     if not mood_text or not isinstance(mood_text, str):
#         return 0.5
#     polarity = TextBlob(mood_text).sentiment.polarity  # -1 to +1
#     # Scale to 0–1
#     return round((polarity + 1) / 2, 4)


# def apply_emotion_adjustment(recs: list, mood_input: str, id_to_name: dict) -> list:
#     """
#     recs        : list of (pid, score, c_bonus) tuples
#     mood_input  : user's free-text mood description
#     id_to_name  : dict mapping product_id → product_name (for future keyword matching)
#     Returns     : list of (pid, score, c_bonus, emotion_bonus) tuples
#     """
#     e_bonus = _mood_score(mood_input)

#     adjusted = []
#     for rec in recs:
#         if len(rec) == 3:
#             pid, score, c_bonus = rec
#         elif len(rec) == 2:
#             pid, score = rec
#             c_bonus = 0.0
#         else:
#             pid, score, c_bonus = rec[0], rec[1], rec[2]

#         adjusted.append((pid, float(score), float(c_bonus), float(e_bonus)))

#     return adjusted



# updated code

"""
emotion.py — Applies a mood/emotion bonus to recommendation scores.
"""

from textblob import TextBlob


def _mood_score(mood_text: str) -> float:
    """
    Convert a free-text mood description into a 0–1 emotion bonus.
    Positive mood → higher bonus (user is more engaged).
    """
    if not mood_text or not isinstance(mood_text, str):
        return 0.5
    try:
        polarity = TextBlob(mood_text).sentiment.polarity  # -1 to +1
        # Scale to 0-1
        return round((polarity + 1) / 2, 4)
    except:
        return 0.5


def apply_emotion_adjustment(recs: list, mood_input: str, id_to_name: dict = None) -> list:
    """
    recs        : list of (pid, score, c_bonus) tuples
    mood_input  : user's free-text mood description
    id_to_name  : dict mapping product_id → product_name (optional)
    Returns     : list of (pid, score, c_bonus, emotion_bonus) tuples
    """
    e_bonus = _mood_score(mood_input)

    adjusted = []
    for rec in recs:
        if len(rec) == 3:
            pid, score, c_bonus = rec
        elif len(rec) == 2:
            pid, score = rec
            c_bonus = 0.0
        else:
            pid, score, c_bonus = rec[0], rec[1], rec[2]

        adjusted.append((pid, float(score), float(c_bonus), float(e_bonus)))

    return adjusted