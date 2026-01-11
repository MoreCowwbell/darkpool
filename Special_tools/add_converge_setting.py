import json

# Read the notebook
with open('circos_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find config cell and add RIBBON_CONVERGE_TO_POINT setting
config_added = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = cell['source']
        source_text = ''.join(source)

        if 'RIBBON_CENTERED = True' in source_text and 'RIBBON_CONVERGE_TO_POINT' not in source_text:
            print(f"Found config cell at index {i}")

            new_source = []
            for line in source:
                new_source.append(line)
                if 'RIBBON_CENTERED = True' in line:
                    new_source.append("RIBBON_CONVERGE_TO_POINT = True         # BASELINE: all ribbons converge to same point (no spreading)\n")
                    print("Added RIBBON_CONVERGE_TO_POINT setting")
                    config_added = True

            cell['source'] = new_source
            break

# Find allocate_intervals function and modify for convergence
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = cell['source']
        source_text = ''.join(source)

        if 'def allocate_intervals' in source_text and "out_info['cursor'] = a0" in source_text:
            print(f"Found allocate_intervals cell at index {i}")

            new_source = []
            for line in source:
                # Override start_pos for outbound when converging (force to arc center)
                if "                start_pos = arc_center_out - total_width / 2\n" == line:
                    new_source.append(line)
                    new_source.append("                if RIBBON_CONVERGE_TO_POINT:\n")
                    new_source.append("                    start_pos = arc_center_out  # Force all ribbons to exact center\n")
                    print("Added convergence override for outbound start_pos")
                # Override start_pos for inbound when converging
                elif "                start_pos = arc_center_in - total_width / 2\n" == line:
                    new_source.append(line)
                    new_source.append("                if RIBBON_CONVERGE_TO_POINT:\n")
                    new_source.append("                    start_pos = arc_center_in  # Force all ribbons to exact center\n")
                    print("Added convergence override for inbound start_pos")
                # Modify cursor advancement for outbound
                elif "        out_info['cursor'] = a0 + out_info['slot_width'] + out_info['gap']\n" == line:
                    new_source.append("        if not RIBBON_CONVERGE_TO_POINT:  # Only advance cursor if not converging\n")
                    new_source.append("            out_info['cursor'] = a0 + out_info['slot_width'] + out_info['gap']\n")
                    print("Modified outbound cursor advancement")
                # Modify cursor advancement for inbound
                elif "        in_info['cursor'] = b0 + in_info['slot_width'] + in_info['gap']\n" == line:
                    new_source.append("        if not RIBBON_CONVERGE_TO_POINT:  # Only advance cursor if not converging\n")
                    new_source.append("            in_info['cursor'] = b0 + in_info['slot_width'] + in_info['gap']\n")
                    print("Modified inbound cursor advancement")
                else:
                    new_source.append(line)

            cell['source'] = new_source
            break

# Write the modified notebook
with open('circos_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("\nConvergence setting added. Re-run notebook to test alignment.")
