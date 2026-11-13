# SimulacraBench entry: per-item model selection by run-time cross-validation

Team `franciseliang`'s submission to the SimulacraBench competition (NeurIPS 2026
Competition Track). The task: given each respondent's visible (GIVEN) survey answers,
predict a probability vector for every held-out answer. Each blank cell is scored
independently with a floored log score.

The method is described in full in [`writeup/report.tex`](writeup/report.tex), the
4-page technical report. In short:

- The real survey data are never visible offline, so **every model is fitted inside
  `predict()`**, on the visible rows of the frame the grader passes in.
- 3-fold cross-validation over those visible rows, scored with the grader's own floored
  log score, picks per-item blend weights over four candidates:
  - **M0**: smoothed crowd marginal;
  - **M1**: per-item logistic regression on the GIVEN answers;
  - **M2**: a multi-task network, one per instrument, with one softmax head per item;
  - **M3**: for gated items, `P(asked) · P(answer | asked)`.
- **M2t** adds a prior from the question/option wording: frozen sentence embeddings,
  shared through the output layer. It replaces M2 on an instrument only if it wins out
  of fold by two standard errors.
- The marginal is always on the blend grid, so no item does worse than the baseline in
  validation. Failures and time-outs fall back to the marginal.

## Files

| Path | What it is |
|---|---|
| `submission/main.py` | The submission: defines `predict(frame, schema)` |
| `submission/text_emb.npz` | Precomputed wording embeddings (1.2 MB), keyed by SHA-256 of each string |
| `tools/build_text_prior.py` | Rebuilds `text_emb.npz` from the public schemas in the official starter kit |
| `writeup/report.tex`, `writeup/refs.bib` | Technical report (NeurIPS 2026 style) |

## Reproduce the submission file

```bash
zip -j -X submission.zip submission/main.py submission/text_emb.npz
```

The archive holds exactly these two files at its root. Check it with the official
starter kit (`github.com/SituatedEvals/public`):

```bash
python <starter-kit>/tools/check_submission_zip.py submission.zip   # prints OK
```

Score it locally on the kit's synthetic sandbox data with the kit's `score.py`. Real
scores only come from the hosted grader, because the real data are not public.

Runtime needs only what the organizers' evaluation image provides: Python 3.13, numpy,
pandas, scipy, scikit-learn, joblib, and PyTorch (the network uses the GPU if one is
present). There is no `requirements.txt`, and nothing is downloaded at run time.

## Rebuild the wording embeddings (optional)

```bash
pip install sentence-transformers
python tools/build_text_prior.py <starter-kit>/data
```

This downloads `sentence-transformers/all-MiniLM-L6-v2` once, at the pinned commit
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (Apache-2.0). It embeds only the question
and option labels from the three public schemas; no respondent data are involved. If a
schema string is missing from the bundle, `main.py` switches that whole instrument to a
character n-gram TF-IDF + SVD embedding.

## Notes

- `main.py` refers to a separate design document that is not part of this repository.
  `writeup/report.tex` covers the same material.
- Predictions are deterministic given the input frame, except for GPU nondeterminism in
  network training.

## License

MIT (see `LICENSE`). The bundled embeddings are outputs of all-MiniLM-L6-v2, which is
licensed Apache-2.0 by its authors.
