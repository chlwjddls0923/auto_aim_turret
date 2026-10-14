"""r4 데이터셋: 기존 handover/dataset (train/val/test) + finetune_data(10.08) train/val (finetune_labels). 새 test 는 넣지 않는다.
  python tools/build_dataset_r4.py [--out handover/dataset_r4]
이미지는 심볼릭 링크(원본 불변), 라벨은 복사. 기존 handover/dataset 는 읽기만 한다.
새 파일명: ft1008_15m_<원래이름> / ft1008_20m_<원래이름>. inventory_r4.csv 에 출처·hard negative 여부 기록.
"""
import argparse, glob, os, shutil
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--old", default="handover/dataset")
ap.add_argument("--src", default="finetune_data(10.08)")
ap.add_argument("--labels", default="finetune_labels")
ap.add_argument("--out", default="handover/dataset_r4")
a = ap.parse_args()
assert not os.path.exists(a.out), f"{a.out} 이미 있음"
ROOT = os.path.abspath(".")
PFX = {"both(1.5m_26.10.08)": "ft1008_15m_", "both(2.0m ~ 375, random_26.10.08)": "ft1008_20m_"}
SRCNAME = {"both(1.5m_26.10.08)": "new_1.5m", "both(2.0m ~ 375, random_26.10.08)": "new_2.0m+"}

inv = []
for s in ("train", "val", "test"):
    os.makedirs(f"{a.out}/{s}/images"); os.makedirs(f"{a.out}/{s}/labels")
    for p in sorted(glob.glob(f"{a.old}/{s}/images/*")):
        n = os.path.basename(p); stem = os.path.splitext(n)[0]
        os.symlink(os.path.realpath(p), f"{a.out}/{s}/images/{n}")
        lp = f"{a.old}/{s}/labels/{stem}.txt"
        if os.path.exists(lp): shutil.copy(lp, f"{a.out}/{s}/labels/{stem}.txt")
        inv.append(dict(file=n, split=s, source="old", hard_negative=False))

sp = pd.read_csv(f"{a.src}/split.csv"); sp = sp[sp.split.isin(["train", "val"])]
dl = pd.read_csv(f"{a.labels}/deleted_boxes.csv")
hn = set(zip(dl.session, dl.filename))
for _, r in sp.iterrows():
    n = PFX[r.session_dir] + r.filename; stem = os.path.splitext(r.filename)[0]
    os.symlink(f"{ROOT}/{a.src}/{r.session_dir}/images/{r.filename}", f"{a.out}/{r.split}/images/{n}")
    shutil.copy(f"{a.labels}/{r.session_dir}/labels/{stem}.txt", f"{a.out}/{r.split}/labels/{os.path.splitext(n)[0]}.txt")
    inv.append(dict(file=n, split=r.split, source=SRCNAME[r.session_dir], hard_negative=(r.session_dir, r.filename) in hn))

# 기존 평가용 yaml·목록 복사 (상대 경로 ./val/images/... 그대로 유효)
for f in glob.glob(f"{a.old}/list_*.txt") + glob.glob(f"{a.old}/eval_*.yaml"):
    shutil.copy(f, a.out)
open(f"{a.out}/data.yaml", "w").write("# r4: 기존 + finetune_data(10.08) train/val. path 키 없음 → 이 폴더가 루트\n"
                                      "train: train/images\nval: val/images\nnames:\n  0: tar\n")
iv = pd.DataFrame(inv); iv.to_csv(f"{a.out}/inventory_r4.csv", index=False)
for s in ("train", "val"):
    open(f"{a.out}/list_val_new.txt" if s == "val" else os.devnull, "w").write(
        "".join(f"./val/images/{f}\n" for f in iv[(iv.split == "val") & (iv.source != "old")].file))
t = iv.pivot_table(index="source", columns="split", values="file", aggfunc="count", fill_value=0)
t["hard_negative(train)"] = iv[iv.split == "train"].groupby("source").hard_negative.sum()
t.loc["합계"] = t.sum()
print(t.to_string())
