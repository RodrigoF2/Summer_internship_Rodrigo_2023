import uproot
import os.path
from subprocess import call
import matplotlib.pyplot as plt
import awkward.operations as ak
import awkward_pandas as akpd
import argparse
import torch
import torch.nn as nn
from sklearn import metrics


# Print module versions
print("Uproot version:", uproot.__version__)

# Check if signal file exists
path_sgn = '/user/r/rodrigommf/RodrigoSummer2023/BPSignal.root'
if not os.path.isfile(path_sgn):
    print("File not found:", path_sgn)
else:
    # Open the ROOT file
    data_sgn = uproot.open(path_sgn)

# Check if bkg file exists
path_bkg = '/user/r/rodrigommf/RodrigoSummer2023/BPBackground.root'
if not os.path.isfile(path_bkg):
    print("File not found:", path_bkg)
else:
    # Open the ROOT file
    data_bkg = uproot.open(path_bkg)

# Access the "ntKp" tree within the Bfinder directory
tree_sgn = data_sgn["ntKp"]
tree_bkg = data_bkg["ntKp"]

# Select working variables
variables = ["Balpha", "Btrk1Pt", "Bchi2cl", 'BsvpvDistance', 'BsvpvDisErr', "Btrk1Dxy", 'Btrk1DxyError', 'Btrk1Dz', 'Btrk1DzError', 'Btrk1Dz1', 'Btrk1DzError1', 'Btrk1Dxy1', 'Btrk1DxyError1'] #"dls3D" missisng, track Dxy and Dz not normalized

# Extract data into Awkward Arrays for signal and background
#data_sgn_arrays = {var: tree_sgn.arrays(var) for var in variables}
data_bkg_arrays = {var: tree_bkg.arrays(var) for var in variables}

# Convert Awkward Arrays to PyTorch Tensors for signal
#sgn_tensors = {var: torch.from_numpy(data_sgn_arrays[var]) for var in variables}

# Convert Awkward Arrays to PyTorch Tensors for background
bkg_tensors = {var: torch.from_numpy(data_bkg_arrays[var]) for var in variables}

# Calculate correlation matrix for signal
#sgn_corr_matrix = torch.stack([sgn_tensors[var] for var in variables], dim=1).corrcoef()

# Calculate correlation matrix for background
bkg_corr_matrix = torch.stack([bkg_tensors[var] for var in variables], dim=1).corrcoef()



print("Correlation Matrix for Background:")
print(bkg_corr_matrix)

#plt.imshow(correlation_matrix, cmap='viridis', origin='upper', vmin=-1, vmax=1)
#plt.colorbar()
#plt.xticks(range(len(variables)*2), variables*2, rotation=45)
#plt.yticks(range(len(variables)*2), variables*2)
#plt.title('Correlation Matrix')
#plt.savefig("/user/r/rodrigommf/RodrigoSummer2023/plots/correlation_sgn_matrix.pdf")
#plt.show()
