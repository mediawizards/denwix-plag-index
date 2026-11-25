# Denwix plagiarism index builder

Builds the fingerprint index behind the free plagiarism checker on
[denwix.com/plagiarism-checker/](https://denwix.com/plagiarism-checker/) and
publishes it to the Hugging Face dataset `mediawizards01/denwix-plag-index`.
Visitors' browsers read the index with HTTP range requests, so checks cost the
site nothing and have no daily cap.

## Run

Actions → **Build plagiarism index** → Run workflow → pick `lang`
(en, es, de, fr, it, ar). `max_shards` = 2 builds a quick test index.

1. `plan` lists the language's parquet shards in `wikimedia/wikipedia` (dump 20231101).
2. `shard` (one job per shard, 20 at a time) fingerprints every article
   (5-word shingles, winnowing window 4, spec in `fp.py`) and stages the records
   plus the title list on Hugging Face.
3. `merge` sorts each of the 16 parts, drops shingles shared by more than 50
   articles, writes `v2/<lang>/fp-NN.bin`, `offsets.bin`, `meta.json` and removes
   the staging folder.

Secret needed: `HF_TOKEN` (Hugging Face write token).

## Files per language (`v2/<lang>/`)

| File | Content |
|---|---|
| `meta.json` | parameters (k, w, bucket bits, record count, shards) |
| `offsets.bin` | uint32 × (2^20 + 1): global record index where each bucket starts |
| `fp-NN.bin` | records (uint32 key, uint32 docid) for buckets NN×65536 … |
| `docs/SSS.txt` + `.idx` | article titles per shard; docid = shard << 20 \| line |

Fingerprint spec changes must be made in `fp.py` here and in the site's `plag-fp.js` together.
Text source: Wikipedia (CC BY-SA 4.0); only hashes and titles are stored.
