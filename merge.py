"""Merge the staged shards of one language into the final index and publish it.

usage: python merge.py LANG [--local DIR] [--no-upload]

Final layout in the dataset repo (v2/<lang>/):
  fp-NN.bin    records (uint32 key, uint32 docid) sorted by (bucket, key); part NN
               holds buckets NN*65536 .. NN*65536+65535
  offsets.bin  uint32[2^20 + 1] global record index where each bucket starts
  meta.json
"""
import os, sys, json, glob, time, shutil, argparse
import numpy as np
from huggingface_hub import HfApi, snapshot_download
from fp import K, W, BB

REPO = os.environ.get("INDEX_REPO", "mediawizards01/denwix-plag-index")
PARTS, MAX_DF = 16, 50
REC = np.dtype([("fp", "<u8"), ("doc", "<u4")])
PER = (1 << BB) // PARTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lang"); ap.add_argument("--local"); ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    api = HfApi()
    t0 = time.time()
    out = f"final/v2/{a.lang}"
    os.makedirs(out, exist_ok=True)
    offsets = np.zeros((1 << BB) + 1, np.int64)
    total = dropped = 0
    for p in range(PARTS):
        if a.local:
            files = sorted(glob.glob(f"{a.local}/staging/{a.lang}/*/p{p:02d}.bin"))
        else:
            d = snapshot_download(REPO, repo_type="dataset", local_dir="stage",
                                  allow_patterns=[f"staging/{a.lang}/*/p{p:02d}.bin"])
            files = sorted(glob.glob(f"stage/staging/{a.lang}/*/p{p:02d}.bin"))
        rec = np.concatenate([np.fromfile(f, REC) for f in files]) if files else np.empty(0, REC)
        bucket = rec["fp"] & np.uint64((1 << BB) - 1)
        key = (rec["fp"] >> np.uint64(BB)) & np.uint64(0xFFFFFFFF)
        sk = (bucket << np.uint64(32)) | key
        order = np.argsort(sk, kind="stable")
        rec, sk = rec[order], sk[order]
        cnt_sorted = np.unique(sk, return_counts=True)[1]  # documents per shingle (52-bit key)
        keep = np.repeat(cnt_sorted <= MAX_DF, cnt_sorted)
        dropped += int((~keep).sum())
        rec, sk = rec[keep], sk[keep]
        bucket = (sk >> np.uint64(32)).astype(np.int64)
        counts = np.bincount(bucket - p * PER, minlength=PER)
        offsets[p * PER + 1:(p + 1) * PER + 1] = total + np.cumsum(counts)
        o = np.empty((len(rec), 2), "<u4")
        o[:, 0] = (sk & np.uint64(0xFFFFFFFF)).astype("<u4")
        o[:, 1] = rec["doc"]
        fn = f"{out}/fp-{p:02d}.bin"
        o.tofile(fn)
        total += len(rec)
        del rec, o, bucket, keep, sk, order
        print(f"part {p:02d}: {len(files)} shards, total {total} {time.time()-t0:.0f}s", flush=True)
        if not a.no_upload:
            api.upload_file(path_or_fileobj=fn, path_in_repo=f"v2/{a.lang}/fp-{p:02d}.bin", repo_id=REPO,
                            repo_type="dataset", commit_message=f"{a.lang} part {p:02d}")
            os.remove(fn)
            shutil.rmtree("stage", ignore_errors=True)
    # offsets[p*PER] was written by part p-1 (its last end), so the array is continuous
    if total >= 1 << 32:
        raise SystemExit("too many records for uint32 offsets")
    offsets.astype("<u4").tofile(f"{out}/offsets.bin")
    src = a.local or "stage"
    shards = sorted({os.path.basename(os.path.dirname(f)) for f in glob.glob(f"{src}/staging/{a.lang}/*/p00.bin")}) if a.local else \
        sorted({x.split("/")[3] for x in api.list_repo_files(REPO, repo_type="dataset") if x.startswith(f"staging/{a.lang}/")})
    meta = {"version": 2, "lang": a.lang, "source": f"wikipedia-{a.lang} 20231101", "k": K, "w": W, "bb": BB,
            "parts": PARTS, "records": int(total), "dropped_common": dropped, "max_df": MAX_DF,
            "shards": [int(s) for s in shards], "built": time.strftime("%Y-%m-%d"),
            "url": f"https://{a.lang}.wikipedia.org/wiki/{{title}}"}
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    print(meta, flush=True)
    if not a.no_upload:
        for f in ("offsets.bin", "meta.json"):
            api.upload_file(path_or_fileobj=f"{out}/{f}", path_in_repo=f"v2/{a.lang}/{f}", repo_id=REPO,
                            repo_type="dataset", commit_message=f"{a.lang} {f}")
        api.delete_folder(path_in_repo=f"staging/{a.lang}", repo_id=REPO, repo_type="dataset",
                          commit_message=f"{a.lang}: drop staging")


if __name__ == "__main__":
    main()
