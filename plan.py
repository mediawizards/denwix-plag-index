"""Print the GitHub Actions matrix: one job per Wikipedia parquet shard of LANG.

usage: python plan.py LANG [MAX_SHARDS]   (writes matrix=... to $GITHUB_OUTPUT)
"""
import os, sys, json
from huggingface_hub import HfApi

lang = sys.argv[1]
cap = int(sys.argv[2]) if len(sys.argv) > 2 else 0
files = sorted(f for f in HfApi().list_repo_files("wikimedia/wikipedia", repo_type="dataset")
               if f.startswith(f"20231101.{lang}/") and f.endswith(".parquet"))
if cap:
    files = files[:cap]
if not files:
    raise SystemExit(f"no parquet files for {lang}")
inc = [{"shard": i, "file": f} for i, f in enumerate(files)]
line = "matrix=" + json.dumps({"include": inc})
print(line)
if os.environ.get("GITHUB_OUTPUT"):
    open(os.environ["GITHUB_OUTPUT"], "a").write(line + "\n")
