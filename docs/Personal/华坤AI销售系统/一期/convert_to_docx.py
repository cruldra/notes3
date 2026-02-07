"""
Convert 技术解决方案.md to DOCX with Mermaid diagrams rendered as images.
Steps:
1. Parse MD file, extract all ```mermaid code blocks
2. Render each mermaid block to PNG using mmdc
3. Replace mermaid code blocks with image references
4. Convert the processed MD to DOCX using pandoc
"""
import re
import subprocess
import tempfile
import os
import sys
import shutil

INPUT_FILE = os.path.join(os.path.dirname(__file__), "技术解决方案.md")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "技术解决方案_v1.0.docx")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")

def extract_and_replace_mermaid(md_content: str) -> str:
    """Extract mermaid blocks, render to PNG, replace with image refs."""
    pattern = re.compile(r'```mermaid\s*\n(.*?)```', re.DOTALL)
    matches = list(pattern.finditer(md_content))
    
    if not matches:
        print("No mermaid blocks found.")
        return md_content
    
    print(f"Found {len(matches)} mermaid blocks to render.")
    
    result = md_content
    # Process in reverse to preserve positions
    for i, match in enumerate(reversed(matches)):
        idx = len(matches) - 1 - i
        mermaid_code = match.group(1).strip()
        img_name = f"diagram_{idx + 1}.png"
        img_path = os.path.join(IMAGES_DIR, img_name)
        
        print(f"  Rendering diagram {idx + 1}...")
        
        # Write mermaid code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False, encoding='utf-8') as f:
            f.write(mermaid_code)
            temp_mmd = f.name
        
        try:
            # Render with mmdc (use shell=True on Windows for PATH resolution)
            cmd = f'mmdc -i "{temp_mmd}" -o "{img_path}" -b white -s 2'
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, shell=True)
            
            if proc.returncode != 0:
                print(f"    WARNING: mmdc failed for diagram {idx + 1}: {proc.stderr[:200]}")
                # Keep original mermaid block on failure
                continue
            
            if not os.path.exists(img_path):
                print(f"    WARNING: Image not created for diagram {idx + 1}")
                continue
                
            print(f"    OK -> {img_name}")
            
            # Replace mermaid block with image reference
            replacement = f'![Diagram {idx + 1}](images/{img_name})'
            result = result[:match.start()] + replacement + result[match.end():]
            
        finally:
            os.unlink(temp_mmd)
    
    return result

def convert_to_docx(processed_md: str):
    """Convert processed MD (with image refs) to DOCX using pandoc."""
    # Write processed MD to temp file
    temp_md = os.path.join(os.path.dirname(INPUT_FILE), "_temp_processed.md")
    with open(temp_md, 'w', encoding='utf-8') as f:
        f.write(processed_md)
    
    try:
        cmd = [
            'pandoc', temp_md,
            '-o', OUTPUT_FILE,
            '--from=markdown',
            '--to=docx',
            '--resource-path=' + os.path.dirname(INPUT_FILE),
        ]
        print(f"\nRunning pandoc...")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if proc.returncode != 0:
            print(f"ERROR: pandoc failed: {proc.stderr[:500]}")
            sys.exit(1)
        
        print(f"SUCCESS: Created {OUTPUT_FILE}")
        
        # Show file size
        size = os.path.getsize(OUTPUT_FILE)
        print(f"File size: {size / 1024:.1f} KB")
        
    finally:
        os.unlink(temp_md)

def main():
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Step 1: Extract and render mermaid diagrams
    processed_md = extract_and_replace_mermaid(md_content)
    
    # Step 2: Convert to DOCX
    convert_to_docx(processed_md)

if __name__ == '__main__':
    main()
