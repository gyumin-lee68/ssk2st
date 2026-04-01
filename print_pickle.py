import pickle
import numpy as np
import os
import glob

# 1. Setting 
# This is an example and you may have to change it to be same as your experiment.
n_fixed = 100        
m_max = 2000         # maximum size of m
num_points = 10      # number of the points of experiments

# Sample Size 
# m varies from 0 to 2000
MM1 = np.linspace(0, m_max, num_points, dtype=int)

# 2. Finding and opening the files
# Define the file as file_path.

try:
    with open(file_path, 'rb') as f:
        results_dict = pickle.load(f)
except Exception as e:
    print(f"Error loading pickle file: {e}")
    exit()

methods = list(results_dict.keys())

# 3. Printing the table
print("\n" + "="*85)
print(f"Power Results Summary (Fixed n={n_fixed})")
print(f"File: {os.path.basename(file_path)}")
print("="*85)

header = f"{'Sample (n, m)':<22}" + "".join([f"{m:>13}" for m in methods])
print(header)
print("-" * len(header))

for j in range(len(MM1)):
    n_val = n_fixed
    m_val = MM1[j]
    row_label = f"n={n_val}, m={m_val}"
    
    row_str = f"{row_label:<22}"
    
    for method in methods:
        data = results_dict[method]
        

        if isinstance(data, np.ndarray):
            if data.ndim > 1:
                
                val = data.mean(axis=0)[j]
            else:
                
                val = data[j]
        else:
            val = 0.0 
            
        # row_str += f"{val:>13.3f}"  # for power comparison
        row_str += f"{1000*val:>13.3f}"  # for running time comparison
    
    print(row_str)

print("="*85 + "\n")
