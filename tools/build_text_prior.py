"""Embed the public item wording once, offline, into submission/text_emb.npz.

The three schemas in the official starter kit are public; they hold the
question and option labels and nothing about respondents. Each (item, option)
pair becomes one string, built by `_option_texts()` in submission/main.py
(the single definition, imported here), and is stored under the SHA-256 of
that string. At run time main.py looks every string up by its hash; if any is
missing it embeds the whole instrument with its own fallback instead, so one
instrument never mixes two embedding spaces. The encoder and its commit are
pinned once, as TEXT_MODEL in main.py; main.py ignores a bundle whose
recorded model differs.

    python tools/build_text_prior.py <starter-kit>/data

Needs sentence-transformers and network access to fetch the model once.
"""

import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np

SCHEMAS = ("unicef", "world_bank", "unhcr")
GATED_VALUE = "NA_GATED"                               # config.yml gated_value

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "submission" / "text_emb.npz"


def main(data_dir):
    sys.dont_write_bytecode = True     # no .pyc (it records local paths)
    spec = importlib.util.spec_from_file_location(
        "submission_main", ROOT / "submission" / "main.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    texts = []
    for name in SCHEMAS:
        schema = json.loads((pathlib.Path(data_dir) / f"{name}.json").read_text())
        schema["gated_value"] = GATED_VALUE
        for item, rec in schema["items"].items():
            if rec["class"] == "PREDICT":
                texts.extend(m._option_texts(schema, item))
    texts = sorted(set(texts))
    model_id, revision = m.TEXT_MODEL.split("@")   # pinned in main.py

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, revision=revision, device="cpu")
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=False)
    keys = np.array([hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts])
    np.savez_compressed(OUT, keys=keys, emb=emb.astype(np.float16),
                        model=np.array(m.TEXT_MODEL))
    print(f"{len(texts)} strings, dim {emb.shape[1]} -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
