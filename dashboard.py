import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

LOG_FILE = "logs/violations.log"

df = pd.read_csv(LOG_FILE, header=None, names=["raw"])
# Parse the log format: "[2026-06-08 12:51:23] WITHOUT_MASK – 97.3%"
# This is a quick parser; adjust as needed.
data = []
for line in df["raw"]:
    parts = line.split("]")
    if len(parts) < 2:
        continue
    ts_str = parts[0][1:]
    rest = parts[1].strip()
    label_conf = rest.split("–")
    if len(label_conf) < 2:
        continue
    label = label_conf[0].strip()
    conf_str = label_conf[1].strip().replace("%", "")
    try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        confidence = float(conf_str)
        data.append([ts, label, confidence])
    except:
        pass

df_clean = pd.DataFrame(data, columns=["timestamp", "label", "confidence"])
if df_clean.empty:
    print("No violations logged yet.")
else:
    print("Total Violations:", len(df_clean))
    print(df_clean["label"].value_counts().to_string())
    df_clean["date"] = df_clean["timestamp"].dt.date
    daily = df_clean.groupby(["date", "label"]).size().unstack(fill_value=0)
    daily.plot(kind="bar", stacked=True)
    plt.title("Violations per Day")
    plt.show()