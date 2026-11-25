import pandas as pd
import requests
import time

CSV_PATH = "/Users/renhirofumi/Desktop/Combined Data.csv"
OUT_CSV = "/Users/renhirofumi/Desktop/compare_105_results.csv"
SUMMARY_TXT = "/Users/renhirofumi/Desktop/compare_105_summary.txt"
URL = "http://localhost:11434/api/generate"

MODELS = ["llama3.2:3b", "qwen2.5:7b"]
LABELS = [
    "Normal",
    "Depression",
    "Suicidal",
    "Anxiety",
    "Stress",
    "Bipolar",
    "Personality disorder",
]

# 1. データ読み込みと前処理（欠損・空文字・重複の除去）
df = pd.read_csv(CSV_PATH).dropna(subset=["statement", "status"])
df["statement"] = df["statement"].astype(str).str.strip()
df = df[df["statement"].str.len() > 0].drop_duplicates(subset=["statement"])

# 2. 各クラス15件ずつ（計105件）を層化抽出
parts = []
for label in LABELS:
    sub = df[df["status"] == label]
    parts.append(sub.sample(n=min(15, len(sub)), random_state=42))
sample = pd.concat(parts, ignore_index=True)

def classify(model_name, text):
    prompt = (
        f"Classify this text into ONE of {LABELS}.\n"
        "Reply with ONLY the exact label name and nothing else.\n\n"
        f"Text: {text[:1000]}"
    )
    try:
        r = requests.post(
            URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=90,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"ERROR: {e}"

def is_refusal(pred):
    low = pred.lower()
    keywords = ["can't", "cannot", "sorry", "helpline", "lifeline", "suicide", "reach out", "1-800"]
    return (pred not in LABELS) and any(k in low for k in keywords)

# 3. 2モデルで同じ105件を順番に分類
records = []
total = len(sample)
for i, row in sample.iterrows():
    rec = {
        "id": row["Unnamed: 0"],
        "true_status": row["status"],
        "statement": row["statement"],
    }
    for m in MODELS:
        pred = classify(m, row["statement"])
        rec[f"{m}_raw"] = pred
        rec[f"{m}_match"] = str(row["status"]).lower() == pred.lower()
        rec[f"{m}_refusal"] = is_refusal(pred)
    records.append(rec)
    print(
        f"[{i+1}/{total}] true={row['status']} | "
        f"llama={rec['llama3.2:3b_raw'][:25]} | "
        f"qwen={rec['qwen2.5:7b_raw'][:25]}"
    )

out_df = pd.DataFrame(records)
out_df.to_csv(OUT_CSV, index=False)

# 4. 集計レポートの作成と表示
lines = ["=== 105-Sample Comparison Summary ==="]
for m in MODELS:
    match_rate = out_df[f"{m}_match"].mean() * 100
    refusal_rate = out_df[f"{m}_refusal"].mean() * 100
    valid_label_rate = out_df[f"{m}_raw"].isin(LABELS).mean() * 100
    lines.append(f"\nModel: {m}")
    lines.append(f"  - Exact match rate : {match_rate:.1f}% ({out_df[f'{m}_match'].sum()}/{total})")
    lines.append(f"  - Valid label rate : {valid_label_rate:.1f}% ({out_df[f'{m}_raw'].isin(LABELS).sum()}/{total})")
    lines.append(f"  - Refusal rate     : {refusal_rate:.1f}% ({out_df[f'{m}_refusal'].sum()}/{total})")

    # クラス別一致数
    by_class = out_df.groupby("true_status")[f"{m}_match"].sum()
    lines.append("  - Matches by true class:")
    for lbl in LABELS:
        lines.append(f"      {lbl}: {int(by_class.get(lbl, 0))}/15")

summary_text = "\n".join(lines)
with open(SUMMARY_TXT, "w") as f:
    f.write(summary_text)

print("\n" + summary_text)
print(f"\nSaved detailed CSV to: {OUT_CSV}")
print(f"Saved summary TXT to : {SUMMARY_TXT}")
