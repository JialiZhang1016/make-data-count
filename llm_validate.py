import polars as pl
import os

from helpers import *

l = get_logger()

SYS_PROMPT_CLASSIFY_DOI = """
1.1 Rules supporting output A (Primary):
If DOI prefix in the text matches a known data repository:
    Dryad: 10.5061
    Zenodo: 10.5281
    Figshare: 10.6084
    Mendeley Data: 10.24433
    Mendeley Data: 10.17632
    Dataverse: 10.7910/DVN
    OpenNeuro: 10.18112/openneuro
    PANGAEA: 10.1594/PANGAEA
    Neotoma Paleoecology: 10.21233
    ICPSR: 10.3886
    NOAA NCEI: 10.7289
    UK Data Service: 10.5255
    EMPIAR: 10.6019
    Non-DOI dataset accession prefixes:
    NCBI SRA / ENA: SRP, SRA, ERP, ERX
    BioProject: PRJNA, PRJEB, PRJDB
    ProteomeXchange / PRIDE: PXD
    ArrayExpress / EMBL-EBI: E-MTAB, E-  (context needed)
    MetaboLights: MTBLS
    GEO Series: GSE
    GenBank: MN, NC_, CP, MT  (context needed)
    EMDB: EMD-  (context needed)
    EMPIAR: EMPIAR-  (context needed)
    
If the text explicitly uses phrases like "we generated", "we created", "we collected and processed", or "our team developed" to describe the dataset/database, directly attributing its creation to the authors of the paper.
If the text states that the dataset/database was "specifically generated for this study" or "produced as part of the current research".
If the text details the authors' direct involvement in data generation/processing (e.g., describing their own data collection methods, experimental procedures to generate raw data, or unique processing steps tailored for the study).
If the text indicates that the dataset/database "did not exist prior to this research" and was created to address the study's objectives.
If the text refers to the dataset as "our study's dataset", "the database developed in this work", or similar phrases that explicitly link it to the current paper's original efforts.
If the text specifies that the data is "raw data collected by our team" or "processed data derived from our own experiments" without referencing external sources.
If the text mentions that the dataset/database is "made publicly available for the first time through this paper" and is identified as the authors' own creation.
If the text describes the dataset's structure, variables, or parameters as "designed and implemented by our research group" for the study.
If the text includes statements like "we conducted surveys/experiments to gather data for this dataset" or "our fieldwork generated the raw data in this database".
If the text claims the dataset/database is "an original contribution of this study" or "a key output of our research".
If the text notes that the dataset is "expanded or refined from the authors' previously unpublished data" (not derived from external sources).

1.2 Rules supporting output B (Secondary):
If the text explicitly cites a reference when mentioning the dataset/database (e.g., "using the dataset from [Author et al., Year]" or "as reported in [Reference X], the database...").
If the text uses phrases like "we reused the existing dataset", "the database was obtained from prior studies", or "we adopted a published dataset".
If the dataset/database is referred to by a name widely recognized as existing in the field (e.g., "MNIST", "PubMed Central") without the authors claiming creation.
If the text states that the dataset/database was "retrieved from [external source]", "downloaded from [public repository]", or "extracted from [existing records]".
If the text identifies the creator/owner of the dataset/database as a third party (e.g., "the database was developed by [Institution/Author] in 20XX").
If the text only describes applying the dataset/database in the study (e.g., "we analyzed data from [Dataset Name]") without any mention of creating or generating it.
If the text indicates the dataset/database "has been used in previous studies" or "is a well-established resource in the field".
If the text provides a link, DOI, or access path to the dataset/database that points to an external platform (not the paper's supplementary materials or the authors' institutional repository for newly created data).
If the text notes that the dataset/database was "first published in [Reference Y]" or "originally described in [earlier work]".
If the text refers to the dataset as "secondary data" or "publicly available data" without claiming original generation.
If the text mentions the dataset/database was "modified from an existing source" (even with adjustments, the core data is derived from external records).
If the text states that the data was "obtained through collaboration with [external organization]" where the data pre-existed.
If the text describes the dataset as "a benchmark dataset widely used in the field" (implying prior existence).

2. Output
Only output:

A → data repository / dataset

B → literature / non-data resource


Few-shot examples

"Raw images are stored on Figshare (DOI 10.6084/m9.figshare.1234567)." → A

"Sequence reads available under BioProject accession PRJNA765432." → A

"As described in Nature Methods (DOI 10.1038/s41592-020-0793-2)." → B

"See Supplementary Data at Zenodo (10.5281/zenodo.987654)." → A

"Method details published in J. Proteome Res. DOI: 10.1021/acs.jproteome.0c00845." → B

"Data uploaded to Dryad (10.5061/dryad.x1y2z3)." → A

"Referenced paper: DOI 10.1101/2020.01.01.123456 (bioRxiv preprint)." → B

"Metabolomics data in MetaboLights MTBLS1234." → A

"The MRI scans are deposited at OpenNeuro (DOI 10.18112/openneuro.ds000001.v1.0.0)." → A

"Protein structure described in Science (DOI 10.1126/science.abc1234)." → B
""".strip()

def build_df():
    df = pl.read_parquet('./temp/extracted.parquet')
    df.filter(~is_doi_link('dataset_id')).select('article_id', 'dataset_id').write_csv('./temp/accid_sub.csv')
    return df.filter(is_doi_link('dataset_id'))

def build_prompt(tokenizer, df):
    prompts = []
    for doi, text in df.select('dataset_id', 'window').rows():
        messages = [{'role':'system','content': SYS_PROMPT_CLASSIFY_DOI}, {'role':'user', 'content': text}]
        prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
    return df.with_columns(pl.Series('prompt', prompts))

if __name__=='__main__':
    os.environ["VLLM_USE_V1"] = "1"  # CPU backend需要V1
    import vllm
    from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
    model_path = "./models/qwen2.5"
    # 尝试使用transformers库代替vLLM进行CPU推理
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    # 使用transformers库加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
        # 禁用量化相关参数
        attn_implementation="eager"
    )
    # tokenizer已经在上面定义了
    df = build_df()
    df = build_prompt(tokenizer, df)
    prompts = df['prompt'].to_list()
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=["A", "B"])
    outputs = llm.generate(prompts, vllm.SamplingParams(seed=777, temperature=0, skip_special_tokens=True, max_tokens=1, logits_processors=[mclp], logprobs=len(mclp.choices)), use_tqdm=True)
    logprobs = [{lp.decoded_token: lp.logprob for lp in list(lps)} for lps in [output.outputs[0].logprobs[0].values() for output in outputs]]
    choices = [max(d, key=d.get) for d in logprobs]
    types = {'A': True, 'B': False}
    choices = [types[c] for c in choices]
    df = df.with_columns(pl.Series('type', choices))
    df.filter(pl.col('type')).select('article_id', 'dataset_id').write_csv('./temp/doi_sub.csv')
    df = pl.concat([pl.read_csv('./temp/doi_sub.csv'), pl.read_csv('./temp/accid_sub.csv')])
    df = assume_type(df)
    df.select(['article_id', 'dataset_id', 'type']).with_row_index(name='row_id').write_csv('./output/submission.csv')
    if not IS_KAGGLE_SUBMISSION:
        results = evaluate(df)
        for r in results: l.info(r) 
        results = evaluate(df, on=['article_id', 'dataset_id', 'type'])
        for r in results: l.info(r)
