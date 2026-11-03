"""Post-native-decoding votes, with deterministic first-frame final ties."""
from collections import defaultdict
import math


def vote_strings(strings, confidences, rule, length=7):
    if not strings or len(strings) != len(confidences):
        raise ValueError('Expected matching nonempty predictions and confidences')
    if rule not in {'majority', 'char_majority'}:
        raise ValueError('Unsupported voting rule')
    if any(not math.isfinite(c) for c in confidences):
        raise ValueError('Nonfinite confidence')
    def winner(pairs):
        scores = defaultdict(list)
        for value, confidence in pairs:
            scores[value].append(confidence)
        best = max(scores, key=lambda v: (len(scores[v]), sum(scores[v]) / len(scores[v])))
        return best, scores[best]
    if rule == 'majority':
        best, scores = winner(zip(strings, confidences))
        return best, sum(scores) / len(scores)
    chars, scores = [], []
    for pos in range(length):
        pairs = [(s[pos], c) for s, c in zip(strings, confidences) if pos < len(s)]
        if not pairs:
            break
        char, values = winner(pairs)
        chars.append(char)
        scores.extend(values)
    return ''.join(chars), sum(scores) / len(scores) if scores else 0.0
