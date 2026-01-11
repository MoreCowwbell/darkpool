import json

# Read the notebook
with open('circos_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Settings to change for baseline (no fanning)
changes = {
    'CHORD_ARC_FRACTION = 0.5': 'CHORD_ARC_FRACTION = 1.0                # BASELINE: full arc',
    'BAND_GAP_FRAC = 0.02': 'BAND_GAP_FRAC = 0.0                    # BASELINE: no gap',
    'DIR_GAP_FRAC = 0.02': 'DIR_GAP_FRAC = 0.0                     # BASELINE: no gap',
    'RIBBON_GAP_RAD = 0.002': 'RIBBON_GAP_RAD = 0.0                   # BASELINE: no gap',
    'RIBBON_WIDTH_SCALE_BY_FLOW = True': 'RIBBON_WIDTH_SCALE_BY_FLOW = False      # BASELINE: uniform width',
    'EDGE_RIBBON_SPLITS = 25': 'EDGE_RIBBON_SPLITS = 1                  # BASELINE: single ribbon',
}

# Find and modify cell 1 (config cell)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = cell['source']
        source_text = ''.join(source)

        if 'CHORD_ARC_FRACTION' in source_text and 'BAND_GAP_FRAC' in source_text:
            print(f"Found config cell at index {i}")

            new_source = []
            for line in source:
                modified = False
                for old, new in changes.items():
                    if old in line:
                        # Keep the newline
                        new_source.append(new + '\n')
                        print(f"Changed: {old.split('=')[0].strip()}")
                        modified = True
                        break
                if not modified:
                    new_source.append(line)

            cell['source'] = new_source
            break

# Write the modified notebook
with open('circos_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("\nBaseline settings applied. Re-run notebook to test alignment.")
