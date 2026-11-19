"""Get LinCE LID data into data/raw/lince/<cfg>/{train,dev}.conll.

Tries, in order:
  1. --from PATH: a zip file or a folder you downloaded yourself (LinCE website, a
     teammate's copy, Kaggle, ...). Any layout works as long as it contains
     <cfg>/train.conll and <cfg>/dev.conll somewhere inside.
  2. The official LinCE server (the same link Hugging Face's lince.py uses).
     This server was timing out on Nov 17, 2026.

Usage:
  python scripts/fetch_lince.py                         # try the server
  python scripts/fetch_lince.py --from ~/Downloads/lince.zip
  python scripts/fetch_lince.py --from /content/drive/MyDrive/lince_download
"""
import argparse
import glob
import io
import os
import shutil
import tempfile
import urllib.request
import zipfile

LINCE_URL = ("https://ritual.uh.edu/lince/libaccess/"
             "eyJ1c2VybmFtZSI6ICJodWdnaW5nZmFjZSBubHAiLCAidXNlcl9pZCI6IDExMSwgImVtYWlsIjogImR1bW15QGVtYWlsLmNvbSJ9")
CONFIGS = ["lid_hineng", "lid_spaeng", "lid_nepeng"]


def find_split(root: str, cfg: str, split: str):
    hits = [p for p in glob.glob(f"{root}/**/{split}.conll", recursive=True) if cfg in p]
    return sorted(hits, key=len)[0] if hits else None


def copy_from_root(root: str, cfg: str, out_root: str) -> bool:
    got = {s: find_split(root, cfg, s) for s in ("train", "dev")}
    if not all(got.values()):
        return False
    os.makedirs(f"{out_root}/{cfg}", exist_ok=True)
    for s, path in got.items():
        shutil.copy(path, f"{out_root}/{cfg}/{s}.conll")
    return True


def unpack(path: str) -> str:
    if os.path.isdir(path):
        return path
    tmp = tempfile.mkdtemp()
    zipfile.ZipFile(path).extractall(tmp)
    for inner in glob.glob(f"{tmp}/**/*.zip", recursive=True):   # zips inside zips
        zipfile.ZipFile(inner).extractall(inner[:-4])
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default=None, help="zip file or folder you downloaded")
    ap.add_argument("--out", default="data/raw/lince")
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--timeout", type=int, default=60)
    a = ap.parse_args()

    root = unpack(os.path.expanduser(a.src)) if a.src else None
    ok = True
    for cfg in a.configs:
        if os.path.exists(f"{a.out}/{cfg}/dev.conll"):
            print(f"{cfg}: already present"); continue
        if root and copy_from_root(root, cfg, a.out):
            print(f"{cfg}: OK from {a.src}"); continue
        try:
            data = urllib.request.urlopen(f"{LINCE_URL}/{cfg}.zip", timeout=a.timeout).read()
            tmp = tempfile.mkdtemp()
            zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
            if copy_from_root(tmp, cfg, a.out):
                print(f"{cfg}: OK from LinCE server"); continue
            print(f"{cfg}: server zip had no train/dev.conll")
        except Exception as e:  # noqa
            print(f"{cfg}: server failed ({type(e).__name__}: {str(e)[:100]})")
        ok = False
    if not ok:
        print("\nSome configs are missing. Download LinCE manually (https://ritual.uh.edu/lince) "
              "or get a copy from a classmate, then rerun with --from <zip or folder>.")


if __name__ == "__main__":
    main()
