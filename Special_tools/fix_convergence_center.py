import json

# Read the notebook
with open('circos_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find allocate_intervals and fix the convergence center calculation
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = cell['source']
        source_text = ''.join(source)

        if 'def allocate_intervals' in source_text and 'arc_center_out' in source_text:
            print(f"Found allocate_intervals cell at index {i}")

            new_source = []
            for line in source:
                # Replace the convergence override for outbound start_pos
                # Old: start_pos = arc_center_out (center of OUT range)
                # New: start_pos = out_range[1] (the boundary where OUT meets IN = true ticker center)
                if "                if RIBBON_CONVERGE_TO_POINT:\n" == line:
                    new_source.append(line)
                elif "                    start_pos = arc_center_out  # Force all ribbons to exact center\n" == line:
                    new_source.append("                    start_pos = out_range[1]  # Force to boundary (ticker center)\n")
                    print("Fixed outbound convergence to out_range[1]")
                elif "                    start_pos = arc_center_in  # Force all ribbons to exact center\n" == line:
                    new_source.append("                    start_pos = in_range[0]  # Force to boundary (ticker center)\n")
                    print("Fixed inbound convergence to in_range[0]")
                else:
                    new_source.append(line)

            cell['source'] = new_source
            break

# Write the modified notebook
with open('circos_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("\nConvergence center fixed. Re-run notebook to test alignment.")
