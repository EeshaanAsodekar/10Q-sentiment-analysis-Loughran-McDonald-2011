import os
import glob
import sec_parser as sp

# Directory containing the 10-Q HTML files
input_dir = 'downloaded_10Qs'

# Directory for saving extracted text files
output_dir = 'sec_parser_10Q_txt'
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

# Extract text from the HTML tree level by level and save to a file
def extract_text_level_by_level(tree, output_file_path):
    """
    Traverse the tree, extract text from each node, and save it to a file.

    Args:
        tree (SemanticTree): Parsed tree from 10-Q filing.
        output_file_path (str): Path to save the extracted content.
    """
    with open(output_file_path, 'w', encoding='utf-8') as file:
        # Process each node in the tree
        for node in tree.nodes:
            if isinstance(node.semantic_element, sp.TitleElement):
                # Extract title text
                title_text = node.semantic_element.text
                file.write(f"Title: {title_text}\n")

            # Process child elements if text is available
            for child in node.children:
                if hasattr(child.semantic_element, 'text'):
                    element_text = child.semantic_element.text
                    file.write(f"{element_text}\n")
        
        file.write("\n")  # Add spacing between sections

    print(f"Extracted text saved to {output_file_path}")

# Get all HTML files from the input directory
html_files = glob.glob(os.path.join(input_dir, '*.html'))

# Process each HTML file
for html_file in html_files:
    # Read HTML file content
    with open(html_file, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Parse the HTML content into semantic elements
    elements = sp.Edgar10QParser().parse(html_content)
    tree = sp.TreeBuilder().build(elements)

    # Generate output file path based on input file name
    base_name = os.path.basename(html_file)
    file_name, _ = os.path.splitext(base_name)
    output_text_file = os.path.join(output_dir, f"{file_name}_extracted.txt")

    # Extract text and save it to the output file
    extract_text_level_by_level(tree, output_text_file)
