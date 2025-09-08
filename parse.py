import argparse
from pathlib import Path
import pymupdf
import xml.etree.ElementTree as ET
import os
import glob
import re
from helpers import get_logger, PDF_DIR

l = get_logger()

def pdf_to_txt(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(PDF_DIR.glob("*.pdf")) + list(PDF_DIR.glob("*.PDF"))
    existing_txt_files = {f.stem for f in output_dir.glob("*.txt")}
    pdf_count = len(pdf_files)
    for pdf_file in pdf_files:
        txt_file = output_dir / f"{pdf_file.stem}.txt"
        if pdf_file.stem in existing_txt_files:
            continue
        try:
            text = ""
            with pymupdf.open(pdf_file) as doc:
                for page in doc:
                    text += page.get_text()
            txt_file.write_text(text, encoding='utf-8')
        except Exception:
            pass
    return pdf_count

def detect_xml_style(root):
    if '}' in root.tag and 'http://www.tei-c.org/ns/1.0' in root.tag:
        return 'tei'
    html_tags = {'html', 'body', 'div', 'p', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
    for elem in root.iter():
        if elem.tag in html_tags:
            return 'html'
    return 'generic'

def get_block_elements(style):
    if style == 'tei':
        return {
            'p', 'head', 'title', 'abstract', 'div', 'item', 'list', 'table',
            'row', 'cell', 'note', 'quote', 'lg', 'l', 'sp', 'speaker'
        }
    elif style == 'html':
        return {
            'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li',
            'table', 'tr', 'td', 'th', 'blockquote', 'pre', 'section', 'article',
            'header', 'footer', 'nav', 'aside'
        }
    else:
        return {
            'paragraph', 'section', 'title', 'abstract', 'item', 'list', 'entry',
            'cell', 'note', 'block'
        }

def extract_text_with_structure(element, style, block_elements, short_content_threshold=50):
    text_parts = []
    if element.text and element.text.strip():
        cleaned_text = re.sub(r'\s+', ' ', element.text.strip())
        text_parts.append(cleaned_text)
    for child in element:
        child_text = extract_text_with_structure(child, style, block_elements, short_content_threshold)
        if child_text:
            text_parts.append(child_text)
    if element.tail and element.tail.strip():
        cleaned_tail = re.sub(r'\s+', ' ', element.tail.strip())
        text_parts.append(cleaned_tail)
    if '}' in element.tag:
        tag_name = element.tag.split('}', 1)[1]
    else:
        tag_name = element.tag
    result_text = ' '.join(text_parts)
    if tag_name in block_elements:
        if len(result_text) < short_content_threshold:
            return result_text + ' '
        else:
            return result_text + '\n\n'
    else:
        return result_text + ' '

def convert_xml_to_txt(xml_file_path, txt_file_path):
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        style = detect_xml_style(root)
        l.info(f"Detected XML style: {style}")
        block_elements = get_block_elements(style)
        short_content_threshold = 50
        structured_text = extract_text_with_structure(root, style, block_elements, short_content_threshold)
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(structured_text)
        return True
    except ET.ParseError as e:
        l.error(f"Error: Could not parse file {xml_file_path}. It may not be valid XML. Error: {e}")
        return False
    except Exception as e:
        l.error(f"Unknown error processing file {xml_file_path}: {e}")
        return False

def batch_convert_xml_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    xml_files = glob.glob(os.path.join(input_folder, "*.xml"))
    xml_count = len(xml_files)
    overwrite_count = 0
    for xml_file in xml_files:
        original_filename = os.path.splitext(os.path.basename(xml_file))[0]
        txt_file_path = os.path.join(output_folder, original_filename + ".txt")
        if os.path.exists(txt_file_path):
            overwrite_count += 1
        if convert_xml_to_txt(xml_file, txt_file_path):
            l.info(f"Converted: {original_filename}.xml -> {original_filename}.txt")
    return xml_count, overwrite_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf-dir', type=Path, default=PDF_DIR, help='Directory containing PDF files')
    parser.add_argument('--xml-dir', type=str, default='data/train/XML', help='Directory containing XML files')
    parser.add_argument('--output-dir', type=Path, default=Path('temp/parse'), help='Directory to save text files')
    args = parser.parse_args()

    # Process PDFs
    pdf_count = pdf_to_txt(args.output_dir)
    l.info(f"Found and processed {pdf_count} PDF files.")

    # Process XMLs
    xml_count, overwrite_count = batch_convert_xml_folder(args.xml_dir, args.output_dir)
    l.info(f"Found and processed {xml_count} XML files.")
    l.info(f"Overwrote {overwrite_count} text files from XML conversions.")

    # Print summary to terminal
    print(f"Processed {pdf_count} PDF files.")
    print(f"Processed {xml_count} XML files.")
    print(f"Overwrote {overwrite_count} text files from XML conversions.")

if __name__ == "__main__":
    main()