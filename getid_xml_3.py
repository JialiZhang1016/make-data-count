import re
import polars as pl
from typing import Optional, Tuple

from helpers import *

COMPILED_PATTERNS = {
    # 'ref_header_patterns': [re.compile(r'\b(R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S|BIBLIOGRAPHY|LITERATURE CITED|WORKS CITED|CITED WORKS|ACKNOWLEDGEMENTS)\b[:\s]*', re.IGNORECASE)],    
    'ref_header_patterns': [re.compile(r'\b(R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S|BIBLIOGRAPHY|LITERATURE CITED|WORKS CITED|CITED WORKS|ACKNOWLEDGEMENTS|REFERENCES AND NOTES)\b[:\s]*', re.IGNORECASE)],
    'citation_pattern': re.compile(r'^\s*(\[\d+\]|\(\d+\)|\d+\.|\d+\)|\d+(?=\s|$))\s*'),
    'first_citation_patterns': [
        re.compile(r'^\s*\[1\]\s*'),
        re.compile(r'^\s*\(1\)\s*'),
        re.compile(r'^\s*1\.\s*'),
        re.compile(r'^\s*1\)\s*'),
        re.compile(r'^\s*1(?=\s|$)'),
    ],
}

l = get_logger()

def find_last_reference_header(text: str, header_patterns: list[re.Pattern]) -> Optional[int]:
    last_match_idx = None
    for pattern in header_patterns:
        matches = list(pattern.finditer(text))
        if matches:
            last_match_idx = matches[-1].start()
    return last_match_idx

def find_last_first_citation(text: str) -> Optional[int]:
    lines = text.splitlines()
    last_match_line = None
    for line_num, line in enumerate(lines):
        line = line.strip()
        for pattern in COMPILED_PATTERNS['first_citation_patterns']:
            if pattern.match(line):
                next_lines = lines[line_num:line_num+3]
                if any(COMPILED_PATTERNS['citation_pattern'].match(l.strip()) for l in next_lines[1:]):
                    last_match_line = line_num
                break
    return last_match_line

def find_reference_start(text: str) -> Optional[int]:
    lines = text.splitlines()
    last_first_citation = find_last_first_citation(text)
    if last_first_citation is not None:
        return last_first_citation
    start_search_idx = int(len(lines) * 0.5)
    for i in range(start_search_idx, len(lines)):
        line = lines[i].strip()
        if COMPILED_PATTERNS['citation_pattern'].match(line):
            next_lines = lines[i:i+3]
            if sum(1 for l in next_lines if COMPILED_PATTERNS['citation_pattern'].match(l.strip())) >= 2:
                for j in range(i, max(-1, i-10), -1):
                    if not COMPILED_PATTERNS['citation_pattern'].match(lines[j].strip()):
                        return j + 1
                return max(0, i-10)
    return None

def split_text_and_references(text: str) -> Tuple[str, str]:
    header_idx = find_last_reference_header(text, COMPILED_PATTERNS['ref_header_patterns'])
    if header_idx is not None:
        header_idx2 = find_last_reference_header(text[:header_idx].strip(), COMPILED_PATTERNS['ref_header_patterns'])
        if header_idx2 is not None:
            header_idx3 = find_last_reference_header(text[:header_idx2].strip(), COMPILED_PATTERNS['ref_header_patterns'])
            if header_idx3 is not None:
                return text[:header_idx3].strip(), text[header_idx3:].strip()
            return text[:header_idx2].strip(), text[header_idx2:].strip()
        return text[:header_idx].strip(), text[header_idx:].strip()
    ref_start_line = find_reference_start(text)
    if ref_start_line is not None:
        lines = text.splitlines()
        body = '\n'.join(lines[:ref_start_line])
        refs = '\n'.join(lines[ref_start_line:])
        return body.strip(), refs.strip()
    return text.strip(), ''

def get_splits(df: pl.DataFrame) -> pl.DataFrame:
    bodies, refs = [], []
    for raw_text in df['text']:
        main, ref = split_text_and_references(raw_text)
        bodies.append(main)
        refs.append(ref)
    return df.with_columns(pl.Series('body', bodies), pl.Series('ref', refs))

def tidy_extraction(df) -> pl.DataFrame:
    bad_ids = [f'{DOI_LINK}{e}' for e in ['10.5061/dryad', '10.5281/zenodo', '10.6073/pasta']]
    
    # doi_df = (
    #     df.with_columns(pl.col('body').str.extract_all(r'10\s*\.\s*\d{4,9}\s*/\s*\S+').alias('match'))
    #       .explode('match')
    #       .drop_nulls('match')
    #       .with_columns(
    #           pl.col('match').str.replace_all(r'\s+', '')
    #                          .str.replace(r'[^A-Za-z0-9]+$', '')
    #                          .str.to_lowercase()
    #                          .alias('dataset_id')
    #       )
    #       .group_by('article_id', 'dataset_id')
    #       .agg('match')
    #       .with_columns((DOI_LINK + pl.col('dataset_id')).alias('dataset_id'))
    # )



    doi_df = (
        df.with_columns(pl.col('text').str.extract_all(r'10\s*\.\s*\d{4,9}\s*/\s*\S+').alias('match'))
        .explode('match')
        .drop_nulls('match')
        .with_columns(
            pl.col('match').str.replace_all(r'\s+', '')
                            .str.replace(r'[^A-Za-z0-9]+$', '')
                            .str.to_lowercase()
                            .alias('dataset_id')
        )
        .group_by('article_id', 'dataset_id')
        .agg('match')
        .with_columns((DOI_LINK + pl.col('dataset_id')).alias('dataset_id'))
    )


    # 建议替换的 REGEX_IDS
    REGEX_IDS = (
        r"(?i)\b(?:"
        # Original Patterns
        r"CHEMBL\d+|"
        r"E-(?:GEOD|PROT|MTAB|MEXP)-\d+|EMPIAR-\d+|"
        r"ENS[A-Z]+\d+|" # More general Ensembl pattern
        r"EPI_ISL_\d{5,}|EPI\d{6,7}|"
        r"HPA\d+|CP\d{6,}|IPR\d{6}|PF\d{5}|BX\d{6}|KX\d{6}|K0\d{4}|CAB\d{6}|"
        r"NC_\d{6,}\.\d+|NM_\d{6,}|" # Loosened length constraints
        r"PRJNA\d+|PRJEB\d+|PRJDB\d+|PXD\d+|SAMN\d+|"
        r"GSE\d+|GSM\d+|"
        r"(?:PDB\s?)?[1-9][A-Z0-9]{3}|HMDB\d+|" # Made "PDB" prefix optional
        r"dryad\.[^\s\"<>]+|pasta\/[^\s\"<>]+|"
        r"(?:SR[RPAX]|STH|ERR|DRR|DRX|DRP|ERP|ERX)\d+|"
        r"CVCL_[A-Z0-9]{4}|"
        
        # === New Patterns based on train_labels.csv analysis ===
        r"rs\d+|"                     # dbSNP IDs, e.g., rs33912345
        r"HGNC:\d+|"                  # HGNC IDs, e.g., HGNC:13735
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|" # CATH IDs, e.g., 3.10.180.10
        r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}|" # UniProt IDs
        r"MODEL\d+|"                  # BioModels ID
        r"SRP\d+|"                    # SRA Project
        r"GDS\d+"                     # GEO Datasets


        # === Final additions for edge cases ===
        r"NCT\d+|"          # ClinicalTrials.gov ID
        r"[A-Z]{2}\d{6,}|"   # GenBank accession format (e.g., AF123456)
        r"GPL\d+"           # GEO Platform ID
        r")"
    )
    
    acc_df = (
        df.with_columns(
            pl.col('text').str.extract_all(REGEX_IDS).alias('match')
        )
        .explode('match')
        .drop_nulls('match')
        .with_columns(
            pl.col('match').str.replace_all(r'\s+', '')
                           .str.replace(r'[^A-Za-z0-9]+$', '')
                           .str.replace(r'(?i)^PDB', '')
                           .alias('dataset_id')
        )
        .group_by('article_id', 'dataset_id')
        .agg('match')
        .with_columns(
            pl.when(pl.col('dataset_id').str.starts_with('dryad.'))
              .then(f'{DOI_LINK}10.5061/' + pl.col('dataset_id'))
              .otherwise('dataset_id')
              .alias('dataset_id')
        )
        .with_columns(
            pl.when(pl.col('dataset_id').str.starts_with('pasta/'))
              .then(f'{DOI_LINK}10.6073/' + pl.col('dataset_id'))
              .otherwise('dataset_id')
              .alias('dataset_id')
        )
    )

    df = pl.concat([doi_df, acc_df])
    
    # ======== 智能过滤策略 ========
    # 1. 基本过滤（保留所有可能的匹配）
    df = df.unique(['article_id', 'dataset_id'])
    
    # 2. 添加置信度评分而不是直接过滤
    def calculate_confidence(dataset_id):
        """为每个ID计算置信度分数"""
        score = 0.5  # 基础分数
        
        # 高置信度模式
        high_confidence_patterns = [
            r'10\.\d{4,9}/',  # DOI格式
            r'chebi\.org',    # 已知仓库
            r'ensembl\.org',
            r'GSE\d+',
            r'PRJNA\d+',
            r'CHEMBL\d+',
            r'PXD\d+',
        ]
        
        # 低置信度模式（可能误判）
        low_confidence_patterns = [
            r'^\d+$',  # 纯数字
            r'^\d+\.\d+$',  # 简单小数
            r'^[A-Z]{1,2}\d{1,3}$',  # 短字母数字组合
            r'^Figure\s+\d+',  # 图表引用
            r'^Table\s+\d+',  # 表格引用
        ]
        
        # 应用规则
        for pattern in high_confidence_patterns:
            if re.search(pattern, dataset_id, re.IGNORECASE):
                score += 0.3
        
        for pattern in low_confidence_patterns:
            if re.search(pattern, dataset_id, re.IGNORECASE):
                score -= 0.4
        
        # 确保分数在0-1之间
        return max(0.1, min(1.0, score))
    
    # 添加置信度列
    confidence_scores = [calculate_confidence(id) for id in df['dataset_id'].to_list()]
    df = df.with_columns(pl.Series('confidence', confidence_scores))
    
    # 3. 应用智能过滤（只过滤明显错误的匹配）
    df = (
        df
        # 排除明显错误的匹配（置信度极低）
        .filter(pl.col('confidence') > 0.2)
        # 弱化自身引用过滤
        .filter(
            ~pl.col('dataset_id').str.replace("https?://", "")
            .str.contains(pl.col('article_id').str.replace('_','/'))
        )
        # 更精确的坏ID过滤
        .filter(~pl.col('dataset_id').is_in(bad_ids))
    )
    
    # 保留原有的match去重
    df = df.with_columns(pl.col('match').list.unique())
    # ======== 智能过滤结束 ========
    return df

def get_context_window(text: str, substring: str, window: int = 100) -> str:
    idx = text.find(substring)
    if idx == -1:
        raise ValueError
    start = max(idx - window, 0)
    end = min(idx + len(substring) + window, len(text))
    return text[start:end]

def get_window_df(text_df, ids_df):
    df = ids_df.join(text_df, on='article_id')
    windows = []
    for text, match_ids in df.select('text', 'match').rows():
        windows.append(get_context_window(text, match_ids[0]))
    return df.with_columns(pl.Series('window', windows)).select('article_id', 'dataset_id', 'window')



def preprocess_text(text: str) -> str:
    """
    Tries to join lines that were likely broken during text extraction.
    This is useful for IDs split across newlines.
    """
    lines = text.split('\n')
    processed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Heuristic: If a line ends with a common DOI prefix part or a hyphen,
        # it might be a broken line.
        if (line.strip().endswith('/') or line.strip().endswith('-')) and i + 1 < len(lines):
            # Join with the next line
            processed_lines.append(line.strip() + lines[i+1].strip())
            i += 2 # Skip the next line as it has been merged
        else:
            processed_lines.append(line)
            i += 1
    return '\n'.join(processed_lines)

def main(input_dir: str, parquet_dir: str, output_dir: str) -> None:

    text_df = get_df(input_dir)

    # !!! ADD THIS PREPROCESSING STEP !!!
    original_texts = text_df['text'].to_list()
    preprocessed_texts = [preprocess_text(t) for t in original_texts]
    text_df = text_df.with_columns(pl.Series("text", preprocessed_texts))


    df = get_splits(text_df)
    df = tidy_extraction(df)
    df = get_window_df(text_df, df)
    df.write_parquet(parquet_dir)
    # df.write_parquet('./temp/extracted.parquet_xml')
    df = assume_type(df)
    df.select(['article_id', 'dataset_id', 'type']).with_row_index(name='row_id').write_csv(output_dir)
    if not IS_KAGGLE_SUBMISSION:
        print("*"*10)
        results = evaluate(df)
        for r in results: l.info(r)
        print("*"*10)
        results = evaluate(df, on=['article_id', 'dataset_id', 'type'])
        for r in results: l.info(r)


if __name__=='__main__': 

    input_dir = './temp/parse_xml'
    parquet_dir = './temp/extracted.parquet_xml'
    output_dir = './temp/submission_xml_4.csv'

    main(input_dir, parquet_dir, output_dir)

"""
格式说明： [TP/FP/FN] 其中：
TP (True Positive): 正确识别的数量
FP (False Positive): 错误识别的数量
FN (False Negative): 遗漏的数量
"""


#是在原先代码的基础上修改的
#针对xml转换后的txt:修改正则与过滤规则，可得到600多一点的正确值，但是误识别达到将近两万条
#针对pdf转换后的txt:修改正则与过滤规则，可得到650多一点的正确值，但是误识别达到一万五千条
#注意看一下文件存储的路径是否正确