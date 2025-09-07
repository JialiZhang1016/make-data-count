import xml.etree.ElementTree as ET
import os
import glob
import re

def detect_xml_style(root):
    """
    检测XML的风格类型
    """
    # 检查是否有TEI命名空间
    if '}' in root.tag and 'http://www.tei-c.org/ns/1.0' in root.tag:
        return 'tei'
    
    # 检查常见HTML标签
    html_tags = {'html', 'body', 'div', 'p', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
    for elem in root.iter():
        if elem.tag in html_tags:
            return 'html'
    
    # 默认返回通用XML处理
    return 'generic'

def get_block_elements(style):
    """
    根据XML风格返回相应的块级元素列表
    """
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
    else:  # generic XML
        return {
            'paragraph', 'section', 'title', 'abstract', 'item', 'list', 'entry',
            'cell', 'note', 'block'
        }

def extract_text_with_structure(element, style, block_elements, short_content_threshold=50):
    """
    递归提取元素文本，保持结构
    """
    text_parts = []
    
    # 处理当前元素的文本
    if element.text and element.text.strip():
        cleaned_text = re.sub(r'\s+', ' ', element.text.strip())
        text_parts.append(cleaned_text)
    
    # 递归处理子元素
    for child in element:
        child_text = extract_text_with_structure(child, style, block_elements, short_content_threshold)
        if child_text:
            text_parts.append(child_text)
    
    # 处理当前元素的尾部文本
    if element.tail and element.tail.strip():
        cleaned_tail = re.sub(r'\s+', ' ', element.tail.strip())
        text_parts.append(cleaned_tail)
    
    # 确定标签名（处理命名空间）
    if '}' in element.tag:
        tag_name = element.tag.split('}', 1)[1]
    else:
        tag_name = element.tag
    
    # 根据元素类型决定是否添加换行
    result_text = ' '.join(text_parts)
    
    if tag_name in block_elements:
        # 对于块级元素，根据内容长度决定是否换行
        if len(result_text) < short_content_threshold:
            return result_text + ' '
        else:
            return result_text + '\n\n'
    else:
        # 对于内联元素，只添加空格
        return result_text + ' '

def convert_xml_to_txt(xml_file_path, txt_file_path):
    """
    将XML文件转换为TXT文件，自动检测风格并保留文本结构
    """
    try:
        # 解析XML文件
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # 检测XML风格
        style = detect_xml_style(root)
        print(f"检测到XML风格: {style}")
        
        # 获取相应的块级元素列表
        block_elements = get_block_elements(style)
        
        # 短内容长度阈值（字符数）
        short_content_threshold = 50
        
        # 提取结构化文本
        structured_text = extract_text_with_structure(root, style, block_elements, short_content_threshold)
        
        # 写入TXT文件
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(structured_text)
                    
    except ET.ParseError as e:
        print(f"错误：无法解析文件 {xml_file_path}。它可能不是有效的XML。错误信息: {e}")
    except Exception as e:
        print(f"处理文件 {xml_file_path} 时发生未知错误: {e}")

def batch_convert_folder(input_folder, output_folder):
    """
    批量转换一个文件夹中的所有XML文件
    """
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # 查找所有.xml文件
    xml_files = glob.glob(os.path.join(input_folder, "*.xml"))
    
    print(f"在文件夹 '{input_folder}' 中找到 {len(xml_files)} 个XML文件。开始转换...")
    
    for xml_file in xml_files:
        # 获取原始XML文件名（不含扩展名）
        original_filename = os.path.splitext(os.path.basename(xml_file))[0]
        
        # 使用原始文件名创建TXT文件路径
        txt_file_path = os.path.join(output_folder, original_filename + ".txt")
        
        # 转换每个文件
        convert_xml_to_txt(xml_file, txt_file_path)
        print(f"已转换: {original_filename}.xml -> {original_filename}.txt")
        
    print("批量转换完成！")

# === 使用方法 ===
if __name__ == "__main__":
    # 请修改为你存放XML文件的文件夹路径
    input_directory = "data/train/XML"  # 替换为你的XML文件路径
    # 请修改为你希望输出TXT文件的文件夹路径
    output_directory = "temp/parse_xml"  # 输出TXT文件路径
    
    # 运行批量转换
    batch_convert_folder(input_directory, output_directory)