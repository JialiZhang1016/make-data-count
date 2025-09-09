# Make Data Count - LLM Pipeline

**目标**从一篇给定的科学论文（PDF 或 XML 格式）全文中，提取出所有被引用的数据集标识符 (`dataset_id`)，并为每一个引用打上分类标签 (`type`, 分类为 `Primary` 或 `Secondary`)，形成 `(article_id, dataset_id, type)` 的三元组，最大化 **F1-Score**。

You will identify all the data citations (references to research data) from the full text of scientific literature and tag the type of citation (primary or secondary):

* **Primary** - raw or processed data generated as part of the paper, specifically for the study
* **Secondary** - raw or processed data derived or reused from existing records or published data

```
Make Data Count/
├── helpers.py           # 通用工具函数和配置
├── parse.py            # PDF文本提取
├── parse_xml.py        # XML文本提取
├── check_parse.py      # 解析验证
├── check_parse_xml.py  # XML解析验证
├── getid.py            # 数据集ID提取
├── llm_validate.py     # LLM验证
├── post_filter.py      # 后处理过滤
├── data/               # 数据目录
│   ├── train/         # 训练数据
│   │   ├── PDF/       # PDF文件
│   │   └── XML/       # XML文件
│   └── train_labels.csv # 训练标签
├── temp/              # 临时文件目录
│   ├── parse/         # PDF解析结果
│   ├── parse_xml/     # XML解析结果
│   └── extracted.parquet # 提取结果
├── models/            # 模型目录
│   └── qwen2.5/       # LLM模型文件
├── output/            # 输出目录
└── logs/              # 日志目录
```

## Dataset Description

### Data Overview

In this competition, participants will extract all research data referenced in a scientific paper (by their identifier) and classify it based on its context as a primary or secondary citation.

### Paper and Dataset Identifiers

Each object (paper and dataset) has a unique, persistent identifier to represent it. In this competition there will be two types:

- DOIs are used for all papers and some datasets. They take the following form: https://doi.org/[prefix]/[suffix]. Examples:
  https://doi.org/10.1371/journal.pone.0303785
  https://doi.org/10.5061/dryad.r6nq870
- Accession IDs are used for some datasets. They vary in form by individual data repository where the data live. Examples:
  "GSE12345" (Gene Expression Omnibus dataset)
  “PDB 1Y2T” (Protein Data Bank dataset)
  "E-MEXP-568" (ArrayExpress dataset)

### Files

train/{PDF,XML} - the training articles, in PDF and XML format
IMPORTANT: Not all PDF articles have a corresponding XML file (approx. 75% do)
test/{PDF,XML} - the test articles, in PDF and XML format
The rerun test dataset has approximately 2,600 articles.
train_labels.csv - labels for the training articles
article_id - research paper DOI, which will be located in the full text of the paper
dataset_id - the dataset identifier and citation type in the paper.
type - citation type
Primary - raw or processed data generated as part of this paper, specifically for this study
Secondary - raw or processed data derived or reused from existing records or published data
sample_submission.csv - a sample submission file in the correct format
The full text of the scientific papers were downloaded in PDF & XML from at: Europe PMC open access subset.

### Data Citation Mining Examples

To illustrate how research data are mentioned in the scientific literature, here are some examples:
Note: in the text, the dataset identifier may appear with or without the 'https://doi.org' stem.

Paper: https://doi.org/10.1098/rspb.2016.1151
Data: https://doi.org/10.5061/dryad.6m3n9
In-text span: "The data we used in this publication can be accessed from Dryad at doi:10.5061/dryad.6m3n9."
Citation type: Primary

Paper: https://doi.org/10.1098/rspb.2018.1563
Data: https://doi.org/10.5061/dryad.c394c12
In-text span: "Phenotypic data and gene sequences are available from the Dryad Digital Repository: http://dx.doi.org/10.5061/dryad.c394c12"
Citation type: Primary

Paper: https://doi.org/10.1534/genetics.119.302868
Data: https://doi.org/10.25386/genetics.11365982
In-text span: "The authors state that all data necessary for confirming the conclusions presented in the article are represented fully within the article. Supplemental material available at figshare: https://doi.org/10.25386/genetics.11365982."
Citation type: Primary

Paper: https://doi.org/10.1038/sdata.2014.33
Data: GSE37569, GSE45042, GSE28166
In-text span: "Primary data for Agilent and Affymetrix microarray experiments are available at the NCBI Gene Expression Omnibus (GEO, http://www.ncbi.nlm.nih.gov/geo/) under the accession numbers GSE37569, GSE45042 , GSE28166"
Citation type: Primary

Paper: https://doi.org/10.12688/wellcomeopenres.15142.1
Data: pdb 5yfp
In-text span: “Figure 1. Evolution and structure of the exocyst. A) Cartoon representing the major supergroups, which are referred to in the text. The inferred position of the last eukaryotic common ancestor (LECA) is indicated and the supergroups are colour coordinated with all other figures. B) Structure of trypanosome Exo99, modelled using Phyre2 (intensive mode). The model for the WD40/b-propeller (blue) is likely highly accurate. The respective orientations of the a-helical regions may form a solenoid or similar, but due to a lack of confidence in the disordered linker regions this is highly speculative. C and D) Structure of the Saccharomyces cerevisiae exocyst holomeric octameric complex. In C the cryoEM map (at level 0.100) is shown and in D, the fit for all eight subunits (pdb 5yfp). Colours for subunits are shown as a key, and the orientation of the cryoEM and fit are the same for C and D. All structural images were modelled by the authors from PDB using UCSF Chimera.”
Citation type: Secondary

Paper: https://doi.org/10.3389/fimmu.2021.690817
Data: E-MTAB-10217, PRJE43395
In-text span: “The datasets presented in this study can be found in online repositories. The names of the repository/repositories and accession number(s) can be found below: https://www.ebi.ac.uk/arrayexpress/, E-MTAB-10217 and https://www.ebi.ac.uk/ena, PRJE43395.”
Citation type: Secondary

## 脚本功能说明

### helpers.py

- 提供共享的工具函数和配置
- 包含日志设置、文件路径管理
- 定义评估指标和数据处理函数
- 提供通用的数据转换和验证功能

### parse.py

- 使用pymupdf从PDF文件中提取文本
- 支持批量处理多个PDF文件
- 自动跳过已处理的文件
- 将提取的文本保存为TXT格式

### parse_xml.py

- 将XML格式的学术论文转换为TXT格式
- 自动检测XML风格（TEI、HTML、通用XML）
- 智能保留文档结构和段落分隔
- 支持批量处理XML文件

### check_parse.py

- 验证PDF文本提取的质量
- 检查是否遗漏关键数据集引用

### check_parse_xml.py

- 验证XML文本提取的质量
- 检查XML转换过程的准确性
- 评估结构保留效果

### getid.py

- 从文本中提取数据集ID
- 支持多种ID格式（DOI、accession numbers等）
- 使用正则表达式进行模式匹配
- 智能分离正文和参考文献部分
- 提供上下文窗口提取
- 支持DOI和Accession Number两类数据集标识符
- 包含数据过滤和去重功能

### llm_validate.py

- 使用LLM验证提取的数据集引用
- 区分Primary和Secondary引用
- 提供详细的分类规则
- 使用few-shot示例提高准确性

### post_filter.py

- 对LLM验证结果进行后处理
- 过滤掉误识别的文献引用
- 移除重复引用
- 生成最终的提交文件

## 目录结构准备

本地运行前需要创建以下目录结构：

```
project_root/
├── data/                    # 替代 /kaggle/input
│   ├── PDF/                 # PDF文件目录
│   └── train_labels.csv     # 训练标签
├── models/                  # 模型目录
│   └── qwen2.5/            # LLM模型文件
├── temp/                    # 临时文件目录
│   └── train_parse/        # 解析后的文本文件
├── output/                  # 输出目录
└── logs/                   # 日志目录
```

### 环境准备

1. **Python环境**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install polars pymupdf vllm logits-processor-zoo
```

2. **模型准备**

- 下载Qwen2.5模型
- 将模型文件放置在 `./models/qwen2.5` 目录下

### 执行流程和关键结果展示

**创建必要的目录**
mkdir -p data/PDF temp/train_parse output logs

**1. 提取文本**
`python parse.py` # pdf + xml to txt

**2. 验证解析质量**
`python check_parse.py`      # misses: 33 dataset_ids

**3. 提取数据集ID**
`python getid.py`  # 从解析后的文本中提取数据集标识符以及context with window=100

**4. LLM验证**

python llm_validate.py

**5. 后处理过滤**

python post_filter.py

## 改进方向

1. [X] 把目前的内容整合到kaggle的ipynb上面
2. [X] kaggle 成功提交
3. [ ] saperate data and paper
4. [ ] saperate primary and second
