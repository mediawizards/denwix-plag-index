"""Fingerprint one Wikipedia parquet shard and stage it on Hugging Face.

usage: python build_shard.py LANG SHARD_NO PARQUET_PATH_IN_WIKIMEDIA [--no-upload] [--max-docs N]

Writes (local ./out, mirrored to the dataset repo):
  v2/<lang>/docs/<shard>.txt   one title per line, local doc number = line number
  v2/<lang>/docs/<shard>.idx   uint32 byte offsets into the .txt (n + 1 entries)
  staging/<lang>/<shard>/pNN.bin  records (uint64 fp, uint32 docid), NN = bucket >> 16
docid = shard << 20 | local doc number.
"""
import os, sys, time, argparse
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfApi
from fp import fingerprints, BB

REPO = os.environ.get("INDEX_REPO", "mediawizards01/denwix-plag-index")
PARTS = 16
REC = np.dtype([("fp", "<u8"), ("doc", "<u4")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lang"); ap.add_argument("shard", type=int); ap.add_argument("file")
    ap.add_argument("--no-upload", action="store_true"); ap.add_argument("--max-docs", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    path = hf_hub_download("wikimedia/wikipedia", a.file, repo_type="dataset", local_dir="raw")
    out = "out"
    sd = f"{a.shard:03d}"
    os.makedirs(f"{out}/v2/{a.lang}/docs", exist_ok=True)
    os.makedirs(f"{out}/staging/{a.lang}/{sd}", exist_ok=True)
    titles, fps, docs = [], [], []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=["title", "text"]):
        for title, text in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if a.max_docs and len(titles) >= a.max_docs:
                break
            f = np.unique(fingerprints(text or ""))
            local = len(titles)
            if local >= (1 << 20):
                raise SystemExit("shard has more than 2^20 docs")
            titles.append((title or "").replace("\n", " "))
            if len(f):
                fps.append(f)
                docs.append(np.full(len(f), (a.shard << 20) | local, np.uint32))
        print(f"{len(titles)} docs {time.time()-t0:.0f}s", flush=True)
        if a.max_docs and len(titles) >= a.max_docs:
            break
    fp = np.concatenate(fps) if fps else np.empty(0, np.uint64)
    dc = np.concatenate(docs) if docs else np.empty(0, np.uint32)
    part = ((fp & np.uint64((1 << BB) - 1)) >> np.uint64(BB - 4)).astype(np.int64)
    for p in range(PARTS):
        sel = part == p
        rec = np.empty(int(sel.sum()), REC)
        rec["fp"] = fp[sel]; rec["doc"] = dc[sel]
        rec.tofile(f"{out}/staging/{a.lang}/{sd}/p{p:02d}.bin")
    blob = ("\n".join(titles) + "\n").encode("utf-8")
    open(f"{out}/v2/{a.lang}/docs/{sd}.txt", "wb").write(blob)
    lens = np.array([len(t.encode("utf-8")) + 1 for t in titles], np.int64)
    np.concatenate([[0], np.cumsum(lens)]).astype("<u4").tofile(f"{out}/v2/{a.lang}/docs/{sd}.idx")
    print(f"shard {sd}: {len(titles)} docs, {len(fp)} fingerprints, {time.time()-t0:.0f}s", flush=True)
    if not a.no_upload:
        api = HfApi()
        for attempt in range(6):
            try:
                api.upload_folder(folder_path=out, repo_id=REPO, repo_type="dataset",
                                  commit_message=f"{a.lang} shard {sd}")
                break
            except Exception as e:  # concurrent commits from other jobs: back off and retry
                print("upload retry", attempt, e, flush=True)
                time.sleep(20 * (attempt + 1))
        else:
            raise SystemExit("upload failed")


if __name__ == "__main__":
    main()
