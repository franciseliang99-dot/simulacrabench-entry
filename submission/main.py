"""SimulacraBench submission: per-item conditional models chosen by run-time
self-validation, falling back to the crowd marginal.

See ../DESIGN.md. Everything is fitted inside predict() on the visible rows of
the frame it is given; nothing about the hidden data is bundled.

Candidates, all scored on the same K folds of the visible rows:
  M0  smoothed marginal (smoothing constant chosen per item)
  M1  per-item logistic regression on the GIVEN block (C chosen per item)
  M2  one multi-task network per instrument, a softmax head per item; the
      hidden rows get the average of its K fold networks
  M2t the same network whose output weights also have a part shared through
      the wording: logit(item, option) gets h(x) . P t(item, option), where t
      is a fixed sentence embedding of "question + option label" (bundled in
      text_emb.npz, built offline from the public schemas by
      tools/build_text_prior.py) and P is learned. The free per-item weights
      carry heavier weight decay, so thin items lean on items worded alike.
      Once per instrument, M2t takes M2's place in the blend only if its
      out-of-fold score per row is higher by at least TEXT_MIN_Z standard
      errors.
  M3  gated items only: P(asked) x P(answer | asked), where P(asked) is a
      logistic regression that also sees how likely the gate parent is to
      hold an opening value, and the answer model is fitted on asked rows only
Per item, the blend weights over (M0, M1, M2, M3) that maximise the held-out
graded log score are kept; (1, 0, 0, 0) is always on the grid, so an item
never does worse in validation than the marginal.
"""

import hashlib
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

FLOOR = 1e-3          # the grader's floor (config.yml scoring.floor)
ALPHAS = (0.1, 0.5, 2.0)
CS = (0.1, 0.5, 2.0)
FOLDS = 3
W_STEP = 0.1
SEED = 0
C3 = 0.5              # M3's two logistic regressions
BUDGET_S = 240.0      # per instrument; the phase-1 budget is 900 s for all three

# M2 (multi-task network)
NN_HIDDEN = 256
NN_DROPOUT = 0.2
NN_LR = 3e-3
NN_WD = 1e-4
NN_BATCH = 256
NN_MAX_EPOCHS = 60
NN_PATIENCE = 5
NN_INNER_VAL = 0.1

# M2t (text prior)
TEXT_PRIOR = True
TEXT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "text_emb.npz")
NN_WD_FREE = 1.0      # decoupled decay on M2t's free output weights
TEXT_MODEL = ("sentence-transformers/all-MiniLM-L6-v2"
              "@1110a243fdf4706b3f48f1d95db1a4f5529b4d41")   # Apache-2.0
TEXT_FALLBACK_DIM = 64
TEXT_MIN_Z = 2.0     # M2t must beat M2 by this many s.e. out of fold

_STEPS = np.round(np.arange(0, 1 + 1e-9, W_STEP), 10)
BLENDS = [(round(1.0 - a - b - c, 10), a, b, c)
          for a in _STEPS for b in _STEPS for c in _STEPS
          if a + b + c <= 1.0 + 1e-9]


def _options(schema, item):
    record = schema["items"][item]
    opts = list(record["values"])
    if record.get("gate"):
        opts.append(schema["gated_value"])
    return opts


def _option_texts(schema, item):
    """One string per option slot of an item, aligned with _options().

    tools/build_text_prior.py imports this to build the bundle, so the
    strings looked up at run time are the ones that were embedded.
    """
    record = schema["items"][item]
    q = " ".join(str(record.get("question") or item).split())
    texts = [f"Question: {q}\nAnswer: {' '.join(str(v).split())}"
             for v in record["values"]]
    if record.get("gate"):
        texts.append(f"Question: {q}\nAnswer: (was not asked this question)")
    return texts


_BUNDLE = None


def _bundle():
    """{sha256: unit vector} from text_emb.npz, or {} if it cannot be read."""
    global _BUNDLE
    if _BUNDLE is None:
        try:
            z = np.load(TEXT_FILE)
            if str(z["model"]) != TEXT_MODEL:
                raise ValueError("bundle built by another encoder")
            _BUNDLE = dict(zip(z["keys"].tolist(),
                               z["emb"].astype(np.float32)))
        except Exception:
            _BUNDLE = {}
    return _BUNDLE


def _text_matrix(schema, target):
    """(n_items, kmax, d) embeddings of every option slot, zero-padded.

    Bundled vectors if every string of this instrument is in the bundle;
    otherwise the whole instrument is embedded by TF-IDF + SVD fitted on its
    own strings, so the two spaces are never mixed. None if neither works.
    """
    per_item = [_option_texts(schema, i) for i in target]
    flat = [t for ts in per_item for t in ts]
    if not flat:
        return None
    b = _bundle()
    keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in flat]
    if b and all(k in b for k in keys):
        vecs = np.stack([b[k] for k in keys])
    else:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                             sublinear_tf=True).fit_transform(flat)
        dim = min(TEXT_FALLBACK_DIM, tf.shape[0] - 1, tf.shape[1] - 1)
        if dim < 2:
            return None
        vecs = TruncatedSVD(dim, random_state=SEED).fit_transform(tf)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    kmax = max(len(ts) for ts in per_item)
    out = np.zeros((len(target), kmax, vecs.shape[1]), dtype=np.float32)
    pos = 0
    for j, ts in enumerate(per_item):
        out[j, :len(ts)] = vecs[pos:pos + len(ts)]
        pos += len(ts)
    return out


def _graded(p):
    """The grader's renormalise-and-mix, applied row-wise."""
    p = p / p.sum(axis=1, keepdims=True)
    k = p.shape[1]
    return FLOOR + (1.0 - k * FLOOR) * p


def _rowlog(p, y):
    return np.log(_graded(p)[np.arange(y.size), y])


def _logscore(p, y):
    return float(_rowlog(p, y).sum())


def _marginal(y, k, alpha):
    w = np.bincount(y, minlength=k).astype(float) + alpha
    return w / w.sum()


def _design(frame, schema, given):
    """One-hot design matrix over the GIVEN block, levels from the schema."""
    blocks = []
    for item in given:
        levels = {v: i for i, v in enumerate(_options(schema, item))}
        col = frame[item].map(levels).to_numpy()
        ok = ~pd.isna(col)
        rows = np.flatnonzero(ok)
        blocks.append(sparse.csr_matrix(
            (np.ones(rows.size), (rows, col[ok].astype(int))),
            shape=(len(frame), len(levels))))
    if not blocks:
        return sparse.csr_matrix((len(frame), 0))
    return sparse.hstack(blocks, format="csr")


def _lr_proba(X_tr, y_tr, X_te, k, c):
    classes = np.unique(y_tr)
    out = np.zeros((X_te.shape[0], k))
    if classes.size < 2 or X_tr.shape[1] == 0:
        out[:, classes] = 1.0 / max(classes.size, 1)
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LogisticRegression(C=c, max_iter=200)
        model.fit(X_tr, y_tr)
    out[:, model.classes_] = model.predict_proba(X_te)
    return out


def _m0(y, k, folds, n_hid):
    """Out-of-fold and final marginal, smoothing constant chosen on the folds."""
    best = None
    for alpha in ALPHAS:
        oof = np.zeros((y.size, k))
        for tr, te in folds:
            oof[te] = _marginal(y[tr], k, alpha)
        s = _logscore(oof, y)
        if best is None or s > best[0]:
            best = (s, alpha, oof)
    _, alpha, oof = best
    return oof, np.tile(_marginal(y, k, alpha), (n_hid, 1))


def _m1(X_vis, y, X_hid, k, folds, deadline):
    """Out-of-fold and final logistic regression, C chosen on the folds.

    Also returns the fold models' mean prediction on the hidden rows, which is
    calibrated like the out-of-fold one (M3 uses it as a feature).
    """
    X_both = sparse.vstack([X_vis, X_hid], format="csr")
    n = y.size
    best = None
    for c in CS:
        oof = np.zeros((n, k))
        hid = np.zeros((X_hid.shape[0], k))
        for tr, te in folds:
            if time.monotonic() > deadline:
                raise TimeoutError
            p = _lr_proba(X_vis[tr], y[tr],
                          X_both[np.r_[te, n + np.arange(X_hid.shape[0])]], k, c)
            oof[te], hid = p[:te.size], hid + p[te.size:] / len(folds)
        s = _logscore(oof, y)
        if best is None or s > best[0]:
            best = (s, c, oof, hid)
    _, c, oof, hid_folds = best
    return oof, _lr_proba(X_vis, y, X_hid, k, c), hid_folds


# ---------------------------------------------------------------- M3 ----------

def _parent_fold_preds(X_vis, y, k, folds, deadline):
    """Parent predictions for M3's fold models, free of the held-out fold.

    For outer fold f, rows in its training part get out-of-fold predictions
    from a 2-fold split *inside* that part, and its held-out rows get a model
    fitted on the whole training part. A plain out-of-fold vector would not
    do: its values on the training rows come from models that saw the
    held-out rows' parent answers, and the child's gate is almost a function
    of those, so the child's validation score comes out optimistic (measured:
    +0.011 log score per row on the sandbox, enough to win the blend).
    The held-out rows get the mean of the same two half models, so their
    feature is as confident as the one the child was trained on.
    """
    out = []
    for tr, te in folds:
        if time.monotonic() > deadline:
            raise TimeoutError
        p = np.zeros((y.size, k))
        half = (tr[0::2], tr[1::2])
        for a, b in (half, half[::-1]):
            p[b] = _lr_proba(X_vis[a], y[a], X_vis[b], k, C3)
            p[te] += 0.5 * _lr_proba(X_vis[a], y[a], X_vis[te], k, C3)
        out.append(p)
    return out


def _m3(X_vis, y, X_hid, k, folds, mass_folds, mass_vis, mass_hid, deadline):
    """Gate-factored model for a gated item; the last option is the sentinel.

    P(sentinel) = 1 - P(asked); P(v) = P(asked) * P(v | asked). The answer
    model only sees rows where the item was asked. The gate is never assumed
    to hold: P(asked) is learned, the parent's mass is only a feature.
    """
    g = k - 1
    asked = (y != g).astype(int)

    def feats(X, m):
        m = np.clip(m, 1e-3, 1.0 - 1e-3)
        extra = np.c_[np.log(m / (1.0 - m)), m]
        return sparse.hstack([X, sparse.csr_matrix(extra)], format="csr")

    def fit(tr, Fv, F_te, X_te):
        if time.monotonic() > deadline:
            raise TimeoutError
        pa = _lr_proba(Fv[tr], asked[tr], F_te, 2, C3)[:, 1]
        a = tr[asked[tr] == 1]
        if a.size == 0:
            pv = np.full((X_te.shape[0], g), 1.0 / g)
        else:
            pv = _lr_proba(X_vis[a], y[a], X_te, g, C3)
        return np.c_[pa[:, None] * pv, 1.0 - pa]

    oof = np.zeros((y.size, k))
    for (tr, te), m in zip(folds, mass_folds):
        Fv = feats(X_vis, m)
        oof[te] = fit(tr, Fv, Fv[te], X_vis[te])
    # Final fit: plain out-of-fold feature on the visible rows, fold-model
    # mean on the hidden rows, so both are calibrated alike.
    return oof, fit(np.arange(y.size), feats(X_vis, mass_vis),
                    feats(X_hid, mass_hid), X_hid)


# ---------------------------------------------------------------- M2 ----------

def _m2(X_vis, Y, ks, X_hid, folds, deadline, text=None):
    """Multi-task network: out-of-fold and final probabilities per item.

    Y is (n_vis, n_items) of class indices; ks the option count per item.
    text, if given, is the (n_items, kmax, d) output of _text_matrix: the
    output layer then gains the shared text part (M2t). Without it the
    network and its initialisation are exactly M2's.
    Returns (out-of-fold, hidden) lists of per-item probability arrays, the
    hidden ones averaged over the fold networks; raises TimeoutError past
    the deadline.
    """
    import torch

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    n_items, kmax = len(ks), max(ks)
    mask = torch.full((n_items, kmax), -1e9)
    for j, k in enumerate(ks):
        mask[j, :k] = 0.0
    mask = mask.to(dev)
    Xv = torch.tensor(X_vis.toarray(), dtype=torch.float32, device=dev)
    Xh = torch.tensor(X_hid.toarray(), dtype=torch.float32, device=dev)
    Yt = torch.tensor(Y, dtype=torch.long, device=dev)
    d = Xv.shape[1]

    T = None
    if text is not None:
        T = torch.tensor(text[:, :kmax].reshape(n_items * kmax, -1),
                         dtype=torch.float32, device=dev)

    def make():
        # Same layers, created in the same order as M2's, so with text=None
        # the random initialisation is identical.
        net = torch.nn.Sequential(
            torch.nn.Linear(d, NN_HIDDEN), torch.nn.ReLU(),
            torch.nn.Dropout(NN_DROPOUT),
            torch.nn.Linear(NN_HIDDEN, NN_HIDDEN), torch.nn.ReLU(),
            torch.nn.Dropout(NN_DROPOUT),
            torch.nn.Linear(NN_HIDDEN, n_items * kmax))
        if T is not None:
            net = torch.nn.ModuleDict({
                "seq": net,
                "proj": torch.nn.Linear(T.shape[1], NN_HIDDEN, bias=False)})
        return net.to(dev)

    def logp(net, X):
        if T is None:
            z = net(X)
        else:
            h = net["seq"][:-1](X)
            z = net["seq"][-1](h) + h @ net["proj"](T).T
        z = z.view(-1, n_items, kmax) + mask
        return torch.log_softmax(z, dim=-1)

    def optimiser(net):
        if T is None:
            return torch.optim.AdamW(net.parameters(), lr=NN_LR,
                                     weight_decay=NN_WD)
        free = net["seq"][-1].weight
        rest = [p for p in net.parameters() if p is not free]
        return torch.optim.AdamW(
            [{"params": rest, "weight_decay": NN_WD},
             {"params": [free], "weight_decay": NN_WD_FREE}], lr=NN_LR)

    def nll(net, X, Yb):
        lp = logp(net, X)
        return -lp.gather(2, Yb.unsqueeze(2)).squeeze(2).sum(1).mean()

    def train(idx):
        """Train on rows idx, early-stopping on an inner split."""
        g = torch.Generator().manual_seed(SEED)
        idx = torch.as_tensor(idx)
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        n_val = max(1, int(NN_INNER_VAL * idx.numel()))
        val, fit = perm[:n_val].to(dev), perm[n_val:]
        torch.manual_seed(SEED)
        net = make()
        opt = optimiser(net)
        best, best_state, bad = np.inf, None, 0
        for ep in range(1, NN_MAX_EPOCHS + 1):
            if time.monotonic() > deadline:
                raise TimeoutError
            net.train()
            order = fit[torch.randperm(fit.numel(), generator=g)].to(dev)
            for s in range(0, order.numel(), NN_BATCH):
                b = order[s:s + NN_BATCH]
                opt.zero_grad()
                nll(net, Xv[b], Yt[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(nll(net, Xv[val], Yt[val]))
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= NN_PATIENCE:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net

    def proba(net, X):
        with torch.no_grad():
            return torch.exp(logp(net, X)).cpu().numpy().astype(float)

    # The hidden rows get the average of the fold networks, not a refit on
    # all rows: a refit with an extrapolated epoch count scored worse on the
    # control sandboxes (0.003-0.007 per cell) and once reversed an M2/M2t
    # choice. The average is still a better predictor than any single fold
    # network, and it helps M2 more than M2t, so out-of-fold scores
    # overstate M2t's gain a little; the selection below needs a margin.
    oof = np.zeros((Xv.shape[0], n_items, kmax))
    hid = np.zeros((Xh.shape[0], n_items, kmax))
    for tr, te in folds:
        net = train(tr)
        oof[te] = proba(net, Xv[torch.as_tensor(te).to(dev)])
        hid += proba(net, Xh) / len(folds)
    return ([oof[:, j, :k] for j, k in enumerate(ks)],
            [hid[:, j, :k] for j, k in enumerate(ks)])


# ---------------------------------------------------------------- blend -------

def _blend(y, oofs, hids):
    """Pick blend weights on the out-of-fold predictions, apply to the final.

    oofs/hids are (M0, M1, M2, M3), M0 and M1 always present, M2/M3 possibly
    None. Weights (1, 0, 0, 0) are always allowed.
    """
    best_s, best_w = -np.inf, (1.0, 0.0, 0.0, 0.0)
    for w in BLENDS:
        if any(wi > 0 and m is None for wi, m in zip(w, oofs)):
            continue
        p = sum(wi * m for wi, m in zip(w, oofs) if wi > 0)
        s = _logscore(p, y)
        if s > best_s + 1e-9:
            best_s, best_w = s, w
    out = sum(wi * h for wi, h in zip(best_w, hids) if wi > 0)
    return out, best_w


def predict(frame, schema):
    t0 = time.monotonic()
    items = [n for n, r in schema["items"].items()
             if r["class"] in ("GIVEN", "PREDICT")]
    given = [n for n in items if schema["items"][n]["class"] == "GIVEN"]
    target = [n for n in items if schema["items"][n]["class"] == "PREDICT"]

    blank = frame[items].isna().to_numpy()
    hidden_rows = blank.any(axis=1)
    vis = frame.loc[~hidden_rows]
    n_hid = int(hidden_rows.sum())

    X = _design(frame, schema, given)
    X_vis, X_hid = X[~hidden_rows], X[hidden_rows]

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(vis))
    folds = []
    for f in range(FOLDS):
        te = np.sort(order[f::FOLDS])
        tr = np.setdiff1d(np.arange(len(vis)), te)
        folds.append((tr, te))

    ys, ks, odd = {}, {}, set()
    for item in target:
        opts = _options(schema, item)
        levels = {v: i for i, v in enumerate(opts)}
        y = vis[item].map(levels)
        if y.isna().any():
            # Visible values outside the schema: model this item by the
            # marginal of the values that do map, never by a guessed label.
            odd.add(item)
            y = y.fillna(0)
            ok = vis[item].map(levels).dropna().to_numpy().astype(int)
            ys[item + "\0ok"] = ok
        ys[item], ks[item] = y.to_numpy().astype(int), len(opts)

    def fallback(item):
        y = ys.get(item + "\0ok", ys[item])
        return np.tile(_marginal(y, ks[item], 0.5), (n_hid, 1))

    # M0 + M1 per item, in parallel processes.
    def job(item):
        y, k = ys[item], ks[item]
        try:
            o0, h0 = _m0(y, k, folds, n_hid)
        except Exception:
            return item, None
        if time.monotonic() - t0 > BUDGET_S:
            return item, (o0, h0, None, None, None)
        try:
            o1, h1, f1 = _m1(X_vis, y, X_hid, k, folds, t0 + BUDGET_S)
        except Exception:
            o1 = h1 = f1 = None
        return item, (o0, h0, o1, h1, f1)

    try:
        parts = dict(Parallel(n_jobs=-1, prefer="processes", max_nbytes=None)(
            delayed(job)(i) for i in target))
    except Exception:
        # Worker pool unusable on this host (import, pickling, OOM-kill,
        # temp dir): same work in this process.
        parts = dict(job(i) for i in target)

    # M2 over all items at once, in this process.
    nn = None
    if target and len(vis) >= 50 and time.monotonic() - t0 < 0.5 * BUDGET_S:
        try:
            Y = np.stack([ys[i] for i in target], axis=1)
            nn = _m2(X_vis, Y, [ks[i] for i in target], X_hid, folds,
                     deadline=t0 + BUDGET_S)
        except Exception:
            nn = None

    # M2t: the same network with the text-shared output part.
    nnt = None
    if (TEXT_PRIOR and nn is not None
            and time.monotonic() - t0 < 0.5 * BUDGET_S):
        try:
            text = _text_matrix(schema, target)
            if text is not None:
                nnt = _m2(X_vis, Y, [ks[i] for i in target], X_hid, folds,
                          deadline=t0 + 0.7 * BUDGET_S, text=text)
        except Exception:
            nnt = None
    # One choice per instrument: M2t replaces M2 only if its out-of-fold
    # score per row is higher by TEXT_MIN_Z standard errors (rows are the
    # independent unit; items with values outside the schema are left out).
    # Choosing per item, or on any positive gain, let fold noise pick M2t
    # where the wording carried nothing (control sandbox with random
    # "wording": up to -0.005 per cell).
    if nnt is not None:
        try:
            d = sum(_rowlog(nnt[0][j], ys[i]) - _rowlog(nn[0][j], ys[i])
                    for j, i in enumerate(target) if i not in odd)
            if (np.ndim(d) == 1 and d.size > 1 and d.std(ddof=1) > 0
                    and d.mean() > TEXT_MIN_Z * d.std(ddof=1) / np.sqrt(d.size)):
                nn = nnt
        except Exception:
            pass

    # M3 for gated items, in parallel processes. The parent's chance of
    # opening the gate comes from its own visible value (GIVEN parent) or
    # from its M1 prediction (PREDICT parent; no M3 if M1 is missing). On the
    # hidden rows that is the fold models' mean, so the feature is calibrated
    # the same way as the out-of-fold one the M3 models were trained on.
    def parent_pred(p):
        opts = _options(schema, p)
        if schema["items"][p]["class"] == "GIVEN":
            lv = {v: i for i, v in enumerate(opts)}
            col = frame[p].map(lv).to_numpy()
            onehot = np.zeros((len(frame), len(opts)))
            ok = ~pd.isna(col)
            onehot[np.flatnonzero(ok), col[ok].astype(int)] = 1.0
            return onehot[~hidden_rows], onehot[hidden_rows], opts
        part = parts.get(p)
        if part is None or p in odd:
            return None
        _, _, o1, _, f1 = part
        return None if o1 is None else (o1, f1, opts)

    # Fold-clean parent predictions, once per PREDICT parent.
    # A malformed gate only costs M3, never the instrument.
    try:
        pparents = sorted({schema["items"][i]["gate"]["parent"] for i in target
                           if schema["items"][i].get("gate")
                           and schema["items"][i]["gate"]["parent"] in ys
                           and schema["items"][i]["gate"]["parent"] not in odd})
    except Exception:
        pparents = []

    def jobp(p):
        try:
            return p, _parent_fold_preds(X_vis, ys[p], ks[p], folds,
                                         t0 + BUDGET_S)
        except Exception:
            return p, None

    pfold = {}
    if pparents and time.monotonic() - t0 < 0.7 * BUDGET_S:
        try:
            pfold = dict(Parallel(n_jobs=-1, prefer="processes", max_nbytes=None)(
                delayed(jobp)(p) for p in pparents))
        except Exception:
            pfold = dict(jobp(p) for p in pparents)

    gated = []
    for item in target:
        try:
            gate = schema["items"][item].get("gate")
            if not gate or item in odd or parts.get(item) is None:
                continue
            pp = (parent_pred(gate["parent"])
                  if gate["parent"] in schema["items"] else None)
            if pp is None:
                continue
            o, h, opts = pp
            if schema["items"][gate["parent"]]["class"] == "GIVEN":
                pf = [o] * len(folds)     # a visible answer, nothing to leak
            elif pfold.get(gate["parent"]) is not None:
                pf = pfold[gate["parent"]]
            else:
                continue
            open_idx = [i for i, v in enumerate(opts)
                        if v in set(gate["observed_if"])]
            gated.append((item, [f[:, open_idx].sum(1) for f in pf],
                          o[:, open_idx].sum(1), h[:, open_idx].sum(1)))
        except Exception:
            continue

    def job3(item, mf, mv, mh):
        try:
            return item, _m3(X_vis, ys[item], X_hid, ks[item], folds, mf, mv, mh,
                             t0 + BUDGET_S)
        except Exception:
            return item, None

    m3 = {}
    if gated and time.monotonic() - t0 < 0.7 * BUDGET_S:
        try:
            m3 = dict(Parallel(n_jobs=-1, prefer="processes", max_nbytes=None)(
                delayed(job3)(*a) for a in gated))
        except Exception:
            m3 = dict(job3(*a) for a in gated)

    preds = {}
    for j, item in enumerate(target):
        part = parts.get(item)
        if part is None or item in odd:
            preds[item] = fallback(item)
            continue
        o0, h0, o1, h1, _ = part
        o2, h2 = (nn[0][j], nn[1][j]) if nn is not None else (None, None)
        o3, h3 = m3.get(item) or (None, None)
        if o1 is None:
            o1, h1 = o0, h0          # no M1: let its weight collapse onto M0
        try:
            preds[item], _ = _blend(ys[item], (o0, o1, o2, o3),
                                    (h0, h1, h2, h3))
        except Exception:
            preds[item] = fallback(item)

    # Assemble in canonical order: rows top to bottom, items in schema order.
    hid_pos = np.cumsum(hidden_rows) - 1
    col = {n: j for j, n in enumerate(items)}
    out = []
    for r in np.flatnonzero(hidden_rows):
        h = hid_pos[r]
        for n in items:
            if blank[r, col[n]]:
                if n in preds:
                    v = preds[n][h]
                else:   # a blank GIVEN cell should not happen; stay legal
                    v = np.full(len(_options(schema, n)), 1.0)
                v = np.clip(np.nan_to_num(v, nan=0.0), 0.0, None) + 1e-9
                out.append(v.tolist())
    return out
