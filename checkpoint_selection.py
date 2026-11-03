"""Validation-score selection, with an explicit alphabetical filename tie-break."""
import re


def checkpoint_accuracy(path):
    match = re.search(r'(?:^|_)acc_([0-9]+(?:\.[0-9]+)?)', path.stem)
    if not match:
        raise ValueError(f'Invalid scored checkpoint filename: {path.name}')
    return float(match.group(1))


def ranked_checkpoints(directory):
    return sorted(directory.glob('*acc_*.pth'), key=lambda p: (-checkpoint_accuracy(p), p.name))


def select_checkpoint(directory):
    ranked = ranked_checkpoints(directory)
    selected = ranked[0] if ranked else directory / 'last.pth'
    if not selected.is_file():
        raise FileNotFoundError(selected)
    return selected
