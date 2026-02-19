#!/usr/bin/env bash
#
# combine_umap_htmls.sh
# 
# Combines multiple Plotly HTML visualizations into a single tabbed interface.
# Each visualization is embedded in an iframe to preserve interactivity.
#
# Usage:
#   ./combine_umap_htmls.sh input_dir output_file.html
#
# Example:
#   ./combine_umap_htmls.sh sessions combined_umap.html

set -euo pipefail

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_dir> [output_file.html]"
    echo ""
    echo "Combines multiple HTML files in <input_dir> into a single tabbed interface."
    echo "Default output: combined_visualizations.html"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_FILE="${2:-combined_visualizations.html}"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory '$INPUT_DIR' not found"
    exit 1
fi

# Find all HTML files (excluding hidden/system files starting with ._)
# Compatible with Bash 3.2+ (macOS default)
HTML_FILES=()
while IFS= read -r file; do
    HTML_FILES+=("$file")
done < <(find "$INPUT_DIR" -maxdepth 1 -name "*.html" ! -name "._*" ! -name "combined_*.html" | sort)

if [ ${#HTML_FILES[@]} -eq 0 ]; then
    echo "Error: No HTML files found in '$INPUT_DIR'"
    exit 1
fi

echo "Found ${#HTML_FILES[@]} HTML files"
echo "Creating combined visualization: $OUTPUT_FILE"

# Generate the combined HTML
cat > "$OUTPUT_FILE" <<'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Combined UMAP/PCA Visualizations</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            overflow: hidden;
        }
        
        .header {
            background: #2c3e50;
            color: white;
            padding: 15px 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 20px;
            font-weight: 600;
        }
        
        .header p {
            font-size: 13px;
            opacity: 0.8;
            margin-top: 5px;
        }
        
        .tabs-container {
            background: white;
            border-bottom: 1px solid #ddd;
            overflow-x: auto;
            white-space: nowrap;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .tabs {
            display: inline-flex;
            padding: 0 10px;
        }
        
        .tab {
            padding: 12px 20px;
            cursor: pointer;
            border: none;
            background: none;
            color: #666;
            font-size: 13px;
            transition: all 0.2s;
            border-bottom: 3px solid transparent;
            white-space: nowrap;
        }
        
        .tab:hover {
            background: #f8f9fa;
            color: #333;
        }
        
        .tab.active {
            color: #3498db;
            border-bottom-color: #3498db;
            font-weight: 500;
        }
        
        .content {
            position: fixed;
            top: 118px;
            left: 0;
            right: 0;
            bottom: 0;
            background: white;
        }
        
        .viz-frame {
            display: none;
            width: 100%;
            height: 100%;
            border: none;
        }
        
        .viz-frame.active {
            display: block;
        }
        
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #999;
            font-size: 14px;
        }
        
        @media (max-width: 768px) {
            .tab {
                padding: 10px 15px;
                font-size: 12px;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Combined UMAP/PCA Visualizations</h1>
        <p id="file-count"></p>
    </div>
    
    <div class="tabs-container">
        <div class="tabs" id="tabs"></div>
    </div>
    
    <div class="content" id="content">
        <div class="loading">Loading visualizations...</div>
    </div>
    
    <script>
        const files = [
EOF

# Add file paths to JavaScript array
for i in "${!HTML_FILES[@]}"; do
    filename=$(basename "${HTML_FILES[$i]}")
    # Create a display name by removing extension and cleaning up
    display_name="${filename%.html}"
    display_name="${display_name//_umap_pca/}"
    display_name="${display_name//_/ }"
    
    if [ $i -eq $((${#HTML_FILES[@]} - 1)) ]; then
        echo "            {path: '$filename', name: '$display_name'}" >> "$OUTPUT_FILE"
    else
        echo "            {path: '$filename', name: '$display_name'}," >> "$OUTPUT_FILE"
    fi
done

# Continue with the rest of the HTML
cat >> "$OUTPUT_FILE" <<'EOF'
        ];
        
        // Update file count
        document.getElementById('file-count').textContent = `${files.length} visualizations`;
        
        const tabsContainer = document.getElementById('tabs');
        const contentContainer = document.getElementById('content');
        
        // Create tabs and iframes
        files.forEach((file, index) => {
            // Create tab
            const tab = document.createElement('button');
            tab.className = 'tab' + (index === 0 ? ' active' : '');
            tab.textContent = file.name;
            tab.onclick = () => switchTab(index);
            tabsContainer.appendChild(tab);
            
            // Create iframe
            const iframe = document.createElement('iframe');
            iframe.className = 'viz-frame' + (index === 0 ? ' active' : '');
            iframe.id = `frame-${index}`;
            
            // Only load first frame immediately, others on-demand
            if (index === 0) {
                iframe.src = file.path;
            }
            
            contentContainer.appendChild(iframe);
        });
        
        // Remove loading message
        const loading = contentContainer.querySelector('.loading');
        if (loading) loading.remove();
        
        function switchTab(index) {
            // Update tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach((tab, i) => {
                tab.classList.toggle('active', i === index);
            });
            
            // Update frames
            const frames = document.querySelectorAll('.viz-frame');
            frames.forEach((frame, i) => {
                frame.classList.toggle('active', i === index);
                
                // Lazy load iframe content
                if (i === index && !frame.src) {
                    frame.src = files[i].path;
                }
            });
        }
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            const currentIndex = Array.from(document.querySelectorAll('.tab'))
                .findIndex(tab => tab.classList.contains('active'));
            
            if (e.key === 'ArrowLeft' && currentIndex > 0) {
                switchTab(currentIndex - 1);
            } else if (e.key === 'ArrowRight' && currentIndex < files.length - 1) {
                switchTab(currentIndex + 1);
            }
        });
    </script>
</body>
</html>
EOF

echo "✓ Created: $OUTPUT_FILE"
echo ""
echo "Open the file in your browser to view all visualizations in tabs."
echo "Use arrow keys to navigate between tabs."
