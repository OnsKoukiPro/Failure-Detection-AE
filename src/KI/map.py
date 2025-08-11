import os
import re
import pandas as pd
from datetime import datetime

# Configuration
ROOT_DIR = "/app/data"
OUTPUT_CSV = "/app/results/KI/metadata.csv"

def parse_filename(filename):
    """Extract metadata from complex filenames"""
    pattern = (
        r"4 Winkel_EF Messung (\d+)°_Mikrofon (\d+)_(\d+)_"
        r"(\d{4}-\d{2}-\d{2}) (\d{2}-\d{2}-\d{2}\.\d{3})_NIO_nicht beurteilt"
    )
    match = re.match(pattern, filename)
    if match:
        return {
            "angle": int(match.group(1)),
            "microphone": match.group(2),
            "anchor_point": f"Anschlagpunkt{match.group(3)}",
            "date": match.group(4),
            "time": match.group(5).replace("-", ":"),
            "label": "unlabeled"  # All samples initially unlabeled
        }
    return None

def main():
    records = []
    for root, _, files in os.walk(ROOT_DIR):
        for file in files:
            if file.endswith(".wav"):
                meta = parse_filename(file)
                if meta:
                    meta.update({
                        "filepath": os.path.join(root, file),
                        "timestamp": datetime.strptime(
                            f"{meta['date']} {meta['time']}",
                            "%Y-%m-%d %H:%M:%S.%f"
                        )
                    })
                    records.append(meta)

    pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)
    print(f"Metadata saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()