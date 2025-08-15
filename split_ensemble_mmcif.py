#!/usr/bin/env python3

import sys
import os
from collections import defaultdict

def split_mmcif_models(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    header_lines = []
    atom_site_started = False
    atom_site_header = []
    model_data = defaultdict(list)
    footer_lines = []
    
    in_atom_site_loop = False
    atom_site_columns = []
    model_col_idx = None
    
    for line in lines:
        line = line.rstrip('\n')
        
        if line.startswith('loop_'):
            if atom_site_started:
                footer_lines.append(line)
            else:
                header_lines.append(line)
            continue
            
        if line.startswith('_atom_site.'):
            atom_site_started = True
            in_atom_site_loop = True
            atom_site_header.append(line)
            atom_site_columns.append(line)
            if line == '_atom_site.pdbx_PDB_model_num':
                model_col_idx = len(atom_site_columns) - 1
            continue
            
        if atom_site_started and not in_atom_site_loop:
            footer_lines.append(line)
            continue
            
        if not atom_site_started:
            header_lines.append(line)
            continue
            
        if in_atom_site_loop and line and not line.startswith('_') and not line.startswith('#'):
            fields = line.split()
            if len(fields) >= len(atom_site_columns):
                if model_col_idx is not None:
                    model_num = fields[model_col_idx]
                else:
                    model_num = '1'
                model_data[model_num].append(line)
        elif in_atom_site_loop and (line.startswith('#') or line == ''):
            in_atom_site_loop = False
            if line:
                footer_lines.append(line)
    
    base_name = os.path.splitext(input_file)[0]
    
    for model_num, data_lines in model_data.items():
        output_file = f"{base_name}_model_{model_num}.cif"
        
        with open(output_file, 'w') as f:
            for line in header_lines:
                f.write(line + '\n')
            
            if atom_site_header:
                # f.write('loop_\n')
                for header_line in atom_site_header:
                    f.write(header_line + '\n')
                
                for data_line in data_lines:
                    f.write(data_line + '\n')
            
            for line in footer_lines:
                f.write(line + '\n')
        
        print(f"Created {output_file} with {len(data_lines)} atoms")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split_ensemble_mmcif.py <input_file.cif>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)
    
    split_mmcif_models(input_file)