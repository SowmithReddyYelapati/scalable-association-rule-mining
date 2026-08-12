from pathlib import Path

import pandas as pd


def load_product_mapping(
    file_path: str | Path,
) -> dict[str, str]:
    """Create a StockCode -> product description mapping."""

    file_path = Path(file_path)

    df = pd.read_excel(
        file_path,
        usecols=["StockCode", "Description"],
    )

    df = df.dropna(subset=["StockCode", "Description"])

    mapping = {}

    for _, row in df.iterrows():
        code = str(row["StockCode"])
        description = str(row["Description"]).strip()

        if code not in mapping:
            mapping[code] = description

    return mapping