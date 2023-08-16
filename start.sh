# import the version number 
VERSION=$(cat ./version.txt)
CONDA_ENV_NAME="synthea_$VERSION"

# activate the environment and start the program
conda init
conda activate "$CONDA_ENV_NAME"
python ./start.py