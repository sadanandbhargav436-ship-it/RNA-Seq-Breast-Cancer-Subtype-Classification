"""Parse local GTF files and build processed expression matrix and label CSVs."""

import gzip
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd


def parse_gtf(filepath: Path) -> pd.Series:
    gene_expr = {}
    open_fn = gzip.open if str(filepath).endswith(".gz") else open
    with open_fn(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 9 and parts[2] in ["transcript", "exon"]:
                match = re.search(
                    r'gene_id\s+"([^"]+)"', parts[8]
                ) or re.search(r'gene_name\s+"([^"]+)"', parts[8])
                fpkm_m = re.search(r'FPKM\s+"([^"]+)"', parts[8])
                if match and fpkm_m:
                    g_id = match.group(1)
                    val = float(fpkm_m.group(1))
                    gene_expr[g_id] = max(gene_expr.get(g_id, 0.0), val)
    return pd.Series(gene_expr)


def main() -> None:
    gtf_files = list(Path("data").glob("**/*transcripts.gtf.gz"))
    print(f"Found {len(gtf_files)} local GTF sample files.")

    samples = {}
    for f in gtf_files:
        m = re.search(r"GSM\d+", f.name) or re.search(
            r"GSM\d+", str(f.parent)
        )
        if m:
            gsm = m.group(0)
            if gsm not in samples or "with-ref" in f.name.lower():
                samples[gsm] = parse_gtf(f)

    expr_matrix = pd.DataFrame(samples).fillna(0.0).T
    expr_matrix.sort_index(inplace=True)

    labels = {}
    for gsm in expr_matrix.index:
        num = int(gsm.replace("GSM", ""))
        if 1261016 <= num <= 1261021:
            labels[gsm] = "TNBC"
        elif 1261022 <= num <= 1261027:
            labels[gsm] = "Non-TNBC"
        elif 1261028 <= num <= 1261032:
            labels[gsm] = "HER2"
        elif 1261033 <= num <= 1261035:
            labels[gsm] = "Normal"
        else:
            labels[gsm] = "Unknown"

    y = pd.DataFrame(list(labels.items()), columns=["sample_id", "target"])

    os.makedirs("data/processed", exist_ok=True)
    expr_matrix.to_csv("data/processed/GSE52194_expression_matrix.csv")
    y.to_csv("data/processed/GSE52194_labels.csv", index=False)

    print("\nSuccessfully created:")
    print("- data/processed/GSE52194_expression_matrix.csv")
    print("- data/processed/GSE52194_labels.csv")


if __name__ == "__main__":
    main()
