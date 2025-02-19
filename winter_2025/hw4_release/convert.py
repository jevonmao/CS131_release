import json

hw_name = "hw4"
with open(f"{hw_name}.ipynb", 'r') as f:
    notebook_data = json.load(f)

# Step 1: Extract all code cells
code_cells = [
    "".join(cell['source']) for cell in notebook_data['cells']
    if cell['cell_type'] == 'code'
]

# Step 2: Extract code cells containing functions
function_cells = []

for idx, code_cell in enumerate(code_cells):
    lines = code_cell.splitlines()

    for line in lines:
        if line.strip():
            if line.strip().startswith("def"):
                function_cells.append(code_cell)
            break

# Step 3: Write import statements and functions to hw0.py
# (Add any additional imports you need to the string below)
with open(f"{hw_name}.py", 'w') as f:
  imports = """from __future__ import print_function
import random
import numpy as np
import time
from PIL import Image
from skimage import color, io
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'"""

  f.write(imports)
  f.write("\n\n")
  for idx, function_cell in enumerate(function_cells):
      f.write(function_cell + "\n\n")