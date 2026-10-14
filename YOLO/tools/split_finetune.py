"""finetune_data 세션별 시간순 train/val/test 분할. test 는 <src>_test/ 로 옮긴다(이미지를 열지 않음).
  python tools/split_finetune.py --src "finetune_data(10.08)" [--dry-run]
규칙(세션마다 촬영 시각 순, N장): test = 마지막 round(0.1N), 그 앞 round(0.1N) = val 블록.
  val→test 경계: val 블록 끝 10장 drop. train→val 경계: train 끝 5장 + val 블록 앞 5장 drop. 나머지 train.
drop 은 삭제하지 않고 split.csv 에 표시만 한다.
"""
import argparse, glob, os, shutil
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--frac", type=float, default=0.1)
ap.add_argument("--gap-test", type=int, default=10)
ap.add_argument("--gap-val", type=int, default=5)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
src = a.src.rstrip("/"); dst = src + "_test"
assert not os.path.exists(dst) or a.dry_run, f"{dst} 이미 있음 — 중복 실행 방지"

rows = []
for d in sorted(glob.glob(f"{glob.escape(src)}/*/labels.csv")):
    sess = os.path.basename(os.path.dirname(d))
    df = pd.read_csv(d).sort_values(["timestamp", "filename"]).reset_index(drop=True)
    n = len(df); nt = round(a.frac * n); nv = round(a.frac * n)
    vb, tb = n - nt - nv, n - nt                      # val 블록 시작, test 시작
    split = ["train"] * n
    for i in range(n):
        if i >= tb: split[i] = "test"
        elif i >= tb - a.gap_test: split[i] = "drop"
        elif i >= vb + a.gap_val: split[i] = "val"
        elif i >= vb - a.gap_val: split[i] = "drop"
    df["split"] = split; df["session_dir"] = sess
    rows.append(df)
    c = df.split.value_counts()
    print(f"{sess}: N={n} train {c.get('train',0)} / val {c.get('val',0)} / test {c.get('test',0)} / drop {c.get('drop',0)}"
          f" | test {df.filename[tb]}~{df.filename[n-1]}")

allp = pd.concat(rows)
if a.dry_run:
    raise SystemExit
for sess, df in allp.groupby("session_dir"):
    t = df[df.split == "test"]
    os.makedirs(f"{dst}/{sess}/images")
    for f in t.filename:
        shutil.move(f"{src}/{sess}/images/{f}", f"{dst}/{sess}/images/{f}")
    shutil.copy(f"{src}/{sess}/labels.csv", f"{src}/{sess}/labels.full.csv")   # 원본 보존
    full = pd.read_csv(f"{src}/{sess}/labels.csv")
    full[full.filename.isin(t.filename)].to_csv(f"{dst}/{sess}/labels.csv", index=False)
    full[~full.filename.isin(t.filename)].to_csv(f"{src}/{sess}/labels.csv", index=False)
    shutil.copy(f"{src}/{sess}/sessions.jsonl", f"{dst}/{sess}/sessions.jsonl")
cols = ["session_dir", "filename", "timestamp", "split"]
allp[cols].to_csv(f"{src}/split.csv", index=False)
allp[allp.split == "test"][cols].to_csv(f"{dst}/split.csv", index=False)
print(f"moved test → {dst}/ , manifest {src}/split.csv")
