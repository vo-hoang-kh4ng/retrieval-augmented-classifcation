import csv
import json
import os
import sys
from pathlib import Path
import requests


def download_file(url: str, dest_path: Path) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest_path.write_bytes(resp.content)
    return dest_path


def convert_csv_to_json(csv_path: Path, json_path: Path) -> Path:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Try to normalize common NAICS headers
            code = r.get("NAICS Code") or r.get("Code") or r.get("naics_code") or ""
            title = r.get("NAICS Title") or r.get("Title") or r.get("title") or ""
            desc = r.get("Description") or r.get("description") or ""
            rows.append({
                "code": str(code).strip(),
                "title": str(title).strip(),
                "description": str(desc).strip(),
            })
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as out:
        json.dump(rows, out, ensure_ascii=False, indent=2)
    return json_path


def main():
    # Example public CSV for NAICS 2022 structure can vary by source.
    # Replace the URL below with an authoritative CSV link for NAICS 2022 structure.
    url = os.environ.get(
        "NAICS_2022_CSV_URL",
        "https://raw.githubusercontent.com/datasets/naics/master/data/naics_2017.csv",
    )
    # Note: above is a placeholder (2017). For production, set NAICS_2022_CSV_URL to a 2022 CSV.

    root = Path(__file__).resolve().parents[1]
    data_raw = root / "data" / "raw"
    csv_path = data_raw / "naics_2022.csv"
    json_path = data_raw / "naics_codes.json"

    print(f"Downloading: {url}")
    try:
        download_file(url, csv_path)
    except Exception as e:
        print("Download failed:", e)
        sys.exit(1)

    try:
        convert_csv_to_json(csv_path, json_path)
        print(f"Wrote {json_path}")
    except Exception as e:
        print("Convert failed:", e)
        sys.exit(2)


if __name__ == "__main__":
    main()


