import json


def load_experiment_records(records_path: str):
    records = []
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
