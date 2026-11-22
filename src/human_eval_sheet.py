"""
human_eval_sheet.py -- build and score a BLIND human-evaluation sheet for several captioning approaches.

Why blind: if the rater knows which model wrote a caption, they tend to favour "their" model. So all captions
of all approaches are mixed in one randomly shuffled list, and the rater only sees
(file_name, reference, prediction). Which approach wrote which row is stored in a separate key file that is
only used after the rating is finished.

The 50 samples are the SAME ones that were used for the translation pipeline
(results/human_eval_translation.csv), so the three approaches can be compared on the same images.

Steps (run from the project root):
    # 1) build the sheet (needs the test predictions written by evaluate.py)
    python src/human_eval_sheet.py make \
        --preds baseline=results/baseline_test_preds.csv transformer=results/transformer_test_preds.csv
        -> results/human_eval_rating_sheet.csv   give this to the rater (Excel / Google Sheets)
        -> results/human_eval_key.csv            item_id -> approach. Do NOT open it before rating is done.

    # 2) the rater fills the two empty columns of the rating sheet with whole numbers from 1 to 5:
    #      fluency_1_5      is the Hindi correct and natural?   1 = broken / not Hindi ... 5 = perfect
    #      correctness_1_5  does it describe what is in the image (file in data/processed/test/images/)?
    #                       1 = completely wrong ... 5 = fully correct.  The reference is only a hint: a
    #                       different but true description is also correct.
    #    (same 1-5 scales as the translation evaluation, so the numbers can be compared)

    # 2b) OPTIONAL, easier than Excel: a browser page that shows the image, the reference and the caption with
    #     1-5 buttons. Open results/human_eval_rater.html, rate, press "Download CSV" and move the downloaded
    #     human_eval_rating_sheet.csv over results/human_eval_rating_sheet.csv.
    python src/human_eval_sheet.py page

    # 3) score it
    python src/human_eval_sheet.py score
        -> prints the mean fluency / correctness of every approach
        -> results/human_eval_baseline_transformer.csv   the un-blinded ratings (commit this one)
"""

import argparse
import json
import os
import random
import sys

import pandas as pd

# Columns the rater has to fill in. Same names as results/human_eval_translation.csv.
RATING_COLS = ["fluency_1_5", "correctness_1_5"]


def read_csv_utf8(path):
    """Read a CSV that may start with a BOM (Excel / Google Sheets add one), keeping empty cells as ''."""
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


def make_sheet(args):
    """Create the shuffled rating sheet and the separate key file."""
    # Refuse to overwrite: re-running 'make' would destroy ratings that were already typed in.
    for path in (args.sheet, args.key):
        if os.path.exists(path) and not args.force:
            sys.exit(f"{path} already exists (it may contain ratings). Use --force only if you want to start over.")

    # The 50 samples (file_name + reference) come from the translation evaluation sheet.
    samples = read_csv_utf8(args.samples)[["file_name", "reference"]]

    # --preds looks like  baseline=results/baseline_test_preds.csv  ->  {"baseline": "results/....csv"}
    systems = {}
    for item in args.preds:
        name, _, path = item.partition("=")
        if not name or not path:
            sys.exit(f"--preds entries must look like name=path, got {item!r}")
        systems[name] = path

    # One row per (approach, sample). The prediction of each approach is looked up by file_name.
    rows = []
    for name, path in systems.items():
        pred = read_csv_utf8(path).set_index("file_name")["prediction"].astype(str)
        missing = [f for f in samples["file_name"] if f not in pred.index]
        if missing:
            sys.exit(f"{path}: no prediction for {len(missing)} of the samples, e.g. {missing[:3]}")
        for f, ref in zip(samples["file_name"], samples["reference"]):
            rows.append({"system": name, "file_name": f, "reference": ref, "prediction": pred[f]})

    # Shuffle with a fixed seed: the order is random for the rater but the same if the sheet is rebuilt.
    random.Random(args.seed).shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["item_id"] = i

    # Rating sheet (what the rater sees): no approach name, two empty rating columns.
    sheet = pd.DataFrame(rows)[["item_id", "file_name", "reference", "prediction"]]
    for col in RATING_COLS:
        sheet[col] = ""
    # Key file: which approach produced which row.
    key = pd.DataFrame(rows)[["item_id", "system"]]

    os.makedirs(os.path.dirname(args.sheet) or ".", exist_ok=True)
    # utf-8-sig so that Excel shows the Hindi text correctly.
    sheet.to_csv(args.sheet, index=False, encoding="utf-8-sig")
    key.to_csv(args.key, index=False, encoding="utf-8-sig")
    print(f"{len(sheet)} items ({len(samples)} samples x {len(systems)} approaches: {', '.join(systems)})")
    print(f"rating sheet -> {args.sheet}")
    print(f"key (do not open before rating) -> {args.key}")


def score_sheet(args):
    """Check the ratings, un-blind them with the key and print the mean scores of every approach."""
    sheet = read_csv_utf8(args.sheet)
    key = read_csv_utf8(args.key)

    # Ratings must be whole numbers 1..5. Blank or other values are listed so the rater can fix them.
    problems = []
    for col in RATING_COLS:
        sheet[col] = pd.to_numeric(sheet[col], errors="coerce")  # blank / text -> NaN
        bad = sheet[~sheet[col].isin([1, 2, 3, 4, 5])]["item_id"].tolist()
        if bad:
            problems.append(f"{col}: {len(bad)} missing or not 1-5, item_id {bad[:10]}{' ...' if len(bad) > 10 else ''}")
    if problems:
        sys.exit("Rating sheet is not complete:\n  " + "\n  ".join(problems))

    # Un-blind: attach the approach name to every rated row.
    data = sheet.merge(key, on="item_id", how="left")
    if data["system"].isna().any():
        sys.exit("some item_id values of the sheet are not in the key file")

    # Mean rating per approach.
    print("| Approach | n | Fluency (1-5) | Correctness (1-5) |")
    print("|---|---|---|---|")
    for name, g in data.groupby("system"):
        print(f"| {name} | {len(g)} | {g['fluency_1_5'].mean():.2f} | {g['correctness_1_5'].mean():.2f} |")

    # Save the un-blinded ratings (this is the file that goes into the repo, like human_eval_translation.csv).
    out = data[["system", "file_name", "reference", "prediction"] + RATING_COLS].sort_values(["system", "file_name"])
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"saved {args.out}")


# Single-file rating page (opened by double-click in a browser, no server needed). The 100 items are embedded
# as JSON inside the page, the images are loaded from disk by their relative path. Ratings are kept in the
# browser (localStorage) after every click, so closing the tab does not lose them, and the "Download CSV" button
# writes the same sheet as the Excel version, filled in, so `score` can read it.
PAGE_HTML = """<!doctype html>
<html lang="hi"><head><meta charset="utf-8"><title>Human evaluation</title>
<style>
 body{font-family:Segoe UI,Arial,sans-serif;max-width:760px;margin:20px auto;padding:0 12px;background:#fafafa;color:#222}
 #top{display:flex;justify-content:space-between;align-items:center}
 #card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:16px;margin-top:10px}
 img{width:336px;height:336px;object-fit:contain;background:#eee;border:1px solid #ccc;display:block;margin:0 auto 12px}
 .txt{font-size:22px;margin:6px 0}.lab{font-size:13px;color:#666;display:block}
 .row{margin:12px 0}.row b{display:inline-block;width:150px}
 .btn{width:44px;height:40px;margin:0 3px;font-size:18px;border:1px solid #888;border-radius:6px;background:#f4f4f4;cursor:pointer}
 .sel{background:#2563eb;color:#fff;border-color:#2563eb}
 .now{outline:3px solid #f59e0b}
 #nav button,#dl{padding:8px 16px;font-size:15px;margin-right:8px;cursor:pointer}
 #hint{font-size:13px;color:#555;margin-top:10px}
</style></head><body>
<div id="top"><h3>Human evaluation (blind)</h3><div id="prog"></div></div>
<div id="card">
 <img id="img" alt="image">
 <div id="fn" class="lab"></div>
 <span class="lab">Reference caption (only a hint)</span><div id="ref" class="txt"></div>
 <span class="lab">Caption to rate</span><div id="pred" class="txt"></div>
 <div class="row" id="rowf"><b>Fluency (1-5)</b></div>
 <div class="row" id="rowc"><b>Correctness (1-5)</b></div>
</div>
<div id="nav" style="margin-top:12px">
 <button id="prev">&larr; Previous</button><button id="next">Next &rarr;</button>
 <button id="dl">Download CSV</button>
</div>
<div id="hint">Fluency: is the Hindi correct and natural? (1 = broken, 5 = perfect). Correctness: does it describe the
image? (1 = completely wrong, 5 = fully correct). Keys: press 1-5 to rate the highlighted row, then the next row, then it
moves on. Left / right arrows = previous / next. Your ratings are saved in this browser automatically.</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);   // {imgdir, items:[...]}
const items = D.items, KEY = 'human_eval_ratings_v1';
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
// rating of every item: start from the browser copy, else from what is already in the sheet
const R = {};
items.forEach(it => { R[it.item_id] = saved[it.item_id] || {f: it.fluency, c: it.correctness}; });
const done = it => R[it.item_id].f && R[it.item_id].c;
let idx = Math.max(0, items.findIndex(it => !done(it)));   // first item that is not rated yet
let target = 'f';                                           // which row the number keys rate now
function persist(){ try { localStorage.setItem(KEY, JSON.stringify(R)); } catch (e) {} }
function mk(rowId, field){
  const row = document.getElementById(rowId);
  row.querySelectorAll('.btn').forEach(b => b.remove());
  for (let v = 1; v <= 5; v++){
    const b = document.createElement('button'); b.className = 'btn'; b.textContent = v;
    b.onclick = () => setRating(field, String(v)); b.dataset.v = v; row.appendChild(b);
  }
}
function render(){
  const it = items[idx], r = R[it.item_id];
  document.getElementById('img').src = D.imgdir + '/' + it.file_name;
  document.getElementById('fn').textContent = it.file_name;
  document.getElementById('ref').textContent = it.reference;
  document.getElementById('pred').textContent = it.prediction;
  [['rowf','f'],['rowc','c']].forEach(([id, f]) => {
    const row = document.getElementById(id);
    row.classList.toggle('now', target === f);
    row.querySelectorAll('.btn').forEach(b => b.classList.toggle('sel', b.dataset.v === r[f]));
  });
  const n = items.filter(done).length;
  document.getElementById('prog').textContent = 'item ' + (idx + 1) + ' / ' + items.length + '   rated: ' + n;
}
function go(d){ idx = Math.min(items.length - 1, Math.max(0, idx + d)); target = 'f'; render(); }
function setRating(field, v){
  R[items[idx].item_id][field] = v; persist();
  if (field === 'f') target = 'c';
  else if (R[items[idx].item_id].f) { if (idx < items.length - 1) { go(1); return; } }
  render();
}
function csvCell(s){ return '"' + String(s).replace(/"/g, '""') + '"'; }
function download(){
  const missing = items.filter(it => !done(it)).length;
  if (missing && !confirm(missing + ' items are not fully rated yet. Download anyway?')) return;
  const head = 'item_id,file_name,reference,prediction,fluency_1_5,correctness_1_5';
  const lines = items.map(it => [it.item_id, it.file_name, it.reference, it.prediction, R[it.item_id].f, R[it.item_id].c]
                                 .map(csvCell).join(','));
  const blob = new Blob(['\\ufeff' + head + '\\r\\n' + lines.join('\\r\\n') + '\\r\\n'], {type: 'text/csv;charset=utf-8'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'human_eval_rating_sheet.csv'; a.click();
}
mk('rowf', 'f'); mk('rowc', 'c');
document.getElementById('prev').onclick = () => go(-1);
document.getElementById('next').onclick = () => go(1);
document.getElementById('dl').onclick = download;
document.addEventListener('keydown', e => {
  if (e.key >= '1' && e.key <= '5') setRating(target, e.key);
  else if (e.key === 'ArrowRight') go(1);
  else if (e.key === 'ArrowLeft') go(-1);
});
render();
</script></body></html>
"""


def make_page(args):
    """Write a single HTML page that shows image + reference + caption for every item and collects the ratings."""
    sheet = read_csv_utf8(args.sheet)
    # Image folder relative to the HTML file, so the page works from any place the repo is copied to.
    imgdir = os.path.relpath(args.images, os.path.dirname(os.path.abspath(args.out))).replace("\\", "/")
    items = [{"item_id": int(r["item_id"]), "file_name": str(r["file_name"]), "reference": str(r["reference"]),
              "prediction": str(r["prediction"]), "fluency": str(r["fluency_1_5"]).strip(),
              "correctness": str(r["correctness_1_5"]).strip()} for _, r in sheet.iterrows()]
    # "</" inside the JSON would end the <script> tag early, so it is escaped.
    data = json.dumps({"imgdir": imgdir, "items": items}, ensure_ascii=False).replace("</", "<\\/")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(PAGE_HTML.replace("__DATA__", data))
    print(f"{len(items)} items -> {args.out}  (open it with a double-click; images are read from {imgdir}/)")
    print("When you are done press 'Download CSV', then move the downloaded file over", args.sheet)


def main():
    ap = argparse.ArgumentParser(description="Blind human-evaluation sheet: make it, then score it")
    sub = ap.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make", help="build the blind rating sheet + key")
    mk.add_argument("--samples", default="results/human_eval_translation.csv",
                    help="sheet that lists the 50 samples (file_name, reference)")
    mk.add_argument("--preds", nargs="+", required=True,
                    help="approaches to compare, as name=predictions.csv (columns file_name, prediction)")
    mk.add_argument("--sheet", default="results/human_eval_rating_sheet.csv")
    mk.add_argument("--key", default="results/human_eval_key.csv")
    mk.add_argument("--seed", type=int, default=0)
    mk.add_argument("--force", action="store_true", help="overwrite an existing sheet / key")
    mk.set_defaults(func=make_sheet)

    sc = sub.add_parser("score", help="check the ratings and print the means")
    sc.add_argument("--sheet", default="results/human_eval_rating_sheet.csv")
    sc.add_argument("--key", default="results/human_eval_key.csv")
    sc.add_argument("--out", default="results/human_eval_baseline_transformer.csv")
    sc.set_defaults(func=score_sheet)

    pg = sub.add_parser("page", help="write a browser page (image + captions + 1-5 buttons) to fill in the ratings")
    pg.add_argument("--sheet", default="results/human_eval_rating_sheet.csv")
    pg.add_argument("--images", default="data/processed/test/images", help="folder with the test images")
    pg.add_argument("--out", default="results/human_eval_rater.html")
    pg.set_defaults(func=make_page)

    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles cannot print Hindi by default
    args.func(args)


if __name__ == "__main__":
    main()
