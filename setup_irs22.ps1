# Miniconda must be installed before running this setup script.
#
# WINDOWS
#========
# Run this script from an Anaconda Powershell prompt, using the base
# Anaconda environment.
# Command: 
# .\setup_env.ps1
#
# MAC AND LINUX
# =============
# Run this command from Terminal.
# Command:
# source setup_env.ps1

conda deactivate
conda create --name trees -y python=3.9

conda activate trees
conda install -y -c conda-forge nodejs
conda install -y -c conda-forge jupyterlab
conda install -y pandas
conda install -y scipy
conda install -y matplotlib
conda install -y seaborn
conda install -y pytest
conda install -y pylint
