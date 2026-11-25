"""Fingerprinting spec v2 for the Denwix plagiarism index.

The browser port (fp.js on denwix.com) MUST reproduce this exactly:
  1. NFKC, lowercase, drop combining marks (Unicode category M) and the Arabic
     tatweel (U+0640), then tokens = runs of Unicode letters/digits.
  2. word hash = FNV-1a 64 over the token's UTF-8 bytes.
  3. shingle hash: h = h*P + w (mod 2^64) over K words, then the splitmix64 mix.
  4. winnowing: window W, keep the minimum (rightmost on ties), unique positions.
Index key: bucket = fp & (2^BB - 1), key = (fp >> BB) & 0xffffffff. (Low bits: winnowing
keeps window minimums, so the HIGH bits of kept hashes are skewed toward 0.)
"""
import unicodedata
import numpy as np
import regex

K = 5           # words per shingle
W = 4           # winnowing window
BB = 20         # bucket bits
P = np.uint64(1099511628211)
MASK = (1 << 64) - 1
_TOK = regex.compile(r"[\p{L}\p{N}]+")
_DROP = regex.compile("[\\p{M}" + chr(0x640) + "]+")
_wcache = {}


def tokens(text):
    s = _DROP.sub("", unicodedata.normalize("NFKC", text).lower())
    return _TOK.findall(s)


def word_hash(w):
    h = _wcache.get(w)
    if h is None:
        h = 0xCBF29CE484222325
        for b in w.encode("utf-8"):
            h = ((h ^ b) * 0x100000001B3) & MASK
        if len(_wcache) < 4_000_000:
            _wcache[w] = h
    return h


def _mix(z):
    z = z ^ (z >> np.uint64(30))
    z = z * np.uint64(0xBF58476D1CE4E5B9)
    z = z ^ (z >> np.uint64(27))
    z = z * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def fingerprints(text, return_pos=False):
    toks = tokens(text)
    n = len(toks)
    empty = np.empty(0, np.uint64)
    if n < K:
        return (empty, np.empty(0, np.int64)) if return_pos else empty
    wh = np.fromiter((word_hash(t) for t in toks), np.uint64, n)
    m = n - K + 1
    h = np.zeros(m, np.uint64)
    with np.errstate(over="ignore"):
        for j in range(K):
            h = h * P + wh[j:j + m]
        h = _mix(h)
    if m <= W:
        pos = np.array([m - 1 - int(np.argmin(h[::-1]))])
    else:
        win = np.lib.stride_tricks.sliding_window_view(h, W)
        r = W - 1 - np.argmin(win[:, ::-1], axis=1)
        pos = np.unique(np.arange(len(win)) + r)
    fp = h[pos]
    return (fp, pos) if return_pos else fp
