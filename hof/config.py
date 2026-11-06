"""Load config.yaml into attribute-access dicts: cfg.render.flame.glow."""
import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Cfg(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def _wrap(x):
    if isinstance(x, dict):
        return Cfg({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def path(rel):
    """Resolve a repo-relative path from config."""
    return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)


def load_config(p="config.yaml"):
    with open(path(p)) as f:
        return _wrap(yaml.safe_load(f))
