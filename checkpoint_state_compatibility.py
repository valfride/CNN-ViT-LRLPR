"""Strict checkpoint loading without corrupting internal module names.

Strip DDP wrapper prefixes only, preserving legitimate internal module names.
Never change tensor objects/values, and reject cleanup collisions or partial loads.
"""


def clean_wrapper_prefixes(state):
    cleaned = {}
    for key, value in state.items():
        name = str(key)
        while name.startswith('module.'):
            name = name[len('module.'):]
        if name in cleaned:
            raise ValueError('Checkpoint wrapper-prefix cleanup collision')
        cleaned[name] = value
    return cleaned


def compatibility_report(expected, candidate):
    return dict(missing=sorted(set(expected)-set(candidate)),
                unexpected=sorted(set(candidate)-set(expected)),
                shape_mismatches=sorted(k for k in set(expected)&set(candidate)
                                        if tuple(expected[k].shape)!=tuple(candidate[k].shape)))


def require_complete_state(expected, candidate):
    report = compatibility_report(expected,candidate)
    if any(report.values()):
        raise ValueError('Checkpoint/model state is incomplete or shape-incompatible: '
                         + ', '.join(f'{k}={len(v)}' for k,v in report.items()))
    return candidate
