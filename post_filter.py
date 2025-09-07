import polars as pl
from helpers import *

"""
Fourth essence: Post-filter to cut FP DOIs that look like literature.
- Read /kaggle/working/submission.csv (output of llm_validate.py)
- Join with /tmp/extracted.parquet to get context window
- Drop DOI rows that (1) start with typical publisher prefixes AND (2) have no data-ish words nearby
- Keep accessions untouched
"""

l = get_logger()

PAPER_PREFIXES = [
    "10.1038","10.1007","10.1126","10.1016","10.1101","10.1021","10.1145","10.1177",
    "10.1093","10.1080","10.1111","10.1098","10.1103","10.1186","10.1371","10.7554",
    "10.1039","10.1002","10.3390","10.1073","10.1097","10.15252","10.1136","10.1091",
    "10.1523", "10.1152", "10.1128", "10.1155", "10.1242", "10.1182", "10.1012"
]

CONTEXT_RE = r"(?i)\b(data(?: ?set)?|database|repository|archive|deposited|available|supplementary|raw(?:\s+data)?|uploaded|hosted|stored|accession(?: number| code)?|files|retrieved from|novel)\b"

def remove_extra_digit(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Remove rows where the value in `column` is just the same DOI with one extra digit at the end.
    Keeps all other columns.
    """
    items_set = set(df[column].to_list())

    def keep_row(value):
        if (value[-1].isdigit() and value[:-1] in items_set) or \
           (len(value) > 2 and value[-2:].isdigit() and value[:-2] in items_set):
            return False
        return True

    return df.filter(pl.col(column).map_elements(keep_row, return_dtype=pl.Boolean))

def is_paper_prefix(col: str = "dataset_id") -> pl.Expr:
    expr = pl.lit(False)
    for p in PAPER_PREFIXES:
        expr = expr | pl.col(col).str.starts_with(f"{DOI_LINK}{p}")
    return expr

def main():
    sub = pl.read_csv("./output/submission.csv")

    # Normalize columns: drop row_id if present so concat widths match
    if "row_id" in sub.columns:
        sub = sub.drop("row_id")

    # Context windows
    win = pl.read_parquet("./temp/extracted.parquet").select("article_id", "dataset_id", "window")

    # DOI & ACC split
    doi_rows = sub.filter(is_doi_link("dataset_id")).join(win, on=["article_id", "dataset_id"], how="left")
    acc_rows = sub.filter(~is_doi_link("dataset_id"))

    keep_mask = (
        (~is_paper_prefix("dataset_id"))  # not a known paper prefix
        | doi_rows["window"].fill_null("").str.contains(CONTEXT_RE)
    )

    kept_doi = doi_rows.filter(keep_mask).select("article_id", "dataset_id", "type")
    ## Remove extra digits
    doi_df = remove_extra_digit(kept_doi, "dataset_id")
    final = pl.concat([doi_df, acc_rows.select("article_id", "dataset_id", "type")])

    # Re-eval & save
    if not IS_KAGGLE_SUBMISSION:
        for r in evaluate(final): l.info(r)
        for r in evaluate(final, on=["article_id", "dataset_id", "type"]): l.info(r)

    final.with_row_index("row_id").write_csv("./output/submission.csv")

if __name__ == "__main__":
    main()
