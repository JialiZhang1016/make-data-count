
import polars as pl
import re
from collections import Counter

def extract_doi_prefix(doi: str) -> str | None:
    """Extracts '10.xxxx' prefix from a DOI starting with 'https://doi.org/'."""
    if not doi.startswith('https://doi.org/'):
        return None
    # Remove 'https://doi.org/'
    stripped = doi.replace('https://doi.org/', '')
    # Extract '10.xxxx' where xxxx is 4-9 digits until '/'
    if '/' in stripped:
        prefix = stripped.split('/')[0]
        if re.match(r'^10\.\d{4,9}$', prefix):  # Validate: 10. followed by 4-9 digits
            return prefix
    return None

def main(input_path: str, output_path: str) -> None:
    # Read F1_details_xml.csv
    df = pl.read_csv(input_path)

    # Filter for DOI dataset_id
    doi_df = df.filter(pl.col('dataset_id').str.starts_with('https://doi.org/'))

    # Extract prefixes
    doi_df = doi_df.with_columns(
        pl.col('dataset_id').map_elements(extract_doi_prefix, return_dtype=pl.String).alias('doi_prefix')
    ).filter(pl.col('doi_prefix').is_not_null())

    # 1. FP: cat_id == 'FP' and cat_type in ['FP', 'TP']
    fp_df = doi_df.filter(
        (pl.col('cat_id') == 'FP') & (pl.col('cat_type').is_in(['FP', 'TP']))
    )
    fp_prefixes = fp_df['doi_prefix'].to_list()
    fp_freq = Counter(fp_prefixes)

    # 2. TP: cat_id == 'TP' and cat_type in ['FP', 'TP']
    tp_df = doi_df.filter(
        (pl.col('cat_id') == 'TP') & (pl.col('cat_type').is_in(['FP', 'TP']))
    )
    tp_prefixes = tp_df['doi_prefix'].to_list()
    tp_freq = Counter(tp_prefixes)

    # 3. Combine frequencies and filter for FP_Count > 2 and TP_Count == 0
    all_prefixes = set(fp_freq.keys()) | set(tp_freq.keys())
    blacklist = []
    for prefix in all_prefixes:
        fp_count = fp_freq.get(prefix, 0)
        tp_count = tp_freq.get(prefix, 0)
        if fp_count > 2 and tp_count == 0:
            blacklist.append((prefix, fp_count))

    # Sort by FP_Count descending
    blacklist_sorted = sorted(blacklist, key=lambda x: x[1], reverse=True)

    # Print in requested format
    if blacklist_sorted:
        print("\nDOI Prefix Blacklist (FP_Count > 2, TP_Count == 0):")
        print('["' + '","'.join(prefix for prefix, _ in blacklist_sorted) + '"]')
    else:
        print("\nNo DOI prefixes found with FP_Count > 2 and TP_Count == 0")

    # Save to file
    with open(output_path, 'w') as f:
        if blacklist_sorted:
            f.write('["' + '","'.join(prefix for prefix, _ in blacklist_sorted) + '"]\n')
        else:
            f.write('[]\n')
    print(f"\nBlacklist saved to: {output_path}")

if __name__ == '__main__':
    input_path = './temp/F1_details_xml.csv'
    output_path = './temp/doi_prefix_blacklist.txt'
    main(input_path, output_path)
