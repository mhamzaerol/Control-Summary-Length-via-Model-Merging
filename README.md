# cs224n-project
## Setup
1. Clone the repository
2. Create a conda environment with the following command:
```bash
conda create --name cs224n-project python=3.11
```
3. Activate the conda environment with the following command:
```bash
conda activate cs224n-project
```
4. Install the required packages with the following command:
```bash
python -m pip install -r requirements.txt 
```
5. Create a `.env` file in the root directory of the project with the following contents:
```bash
HF_ACCESS_TOKEN=<your_hugging_face_access_token> # For PEFT/Merging experiments (authorizing access to LLaMA)
OPENAI_API_KEY=<your_openai_api_key> # For data preprocessing (re-summarization)
```
6. You may need to log in to Hugging Face with the following command:
```bash
huggingface-cli login
```
## Data Preprocessing
### Running Data Preprocessing
To create initial datasets, run the following command:
```bash
python src/create_datasets.py ./split/
```

### Download Preprocessed Datasets
Alternatively, you can download the preprocessed datasets with the following command:
```bash
gdown "https://drive.google.com/uc?export=download&id=1ogHjdTNJbxoUVrQIWRBavTsqImSlCXQg" -O "datasets.zip"
unzip datasets.zip
rm datasets.zip
```
## Running Experiments
Run the experiments from the main directory (do not `cd` into the experiments directory) with the following command:
```bash
python -m experiments.<experiment_name>
```
### PEFT Experiment
To run a PEFT experiment defined by a configuration file, run the following command:
```bash
python -m experiments.peft_experiment --config_path <path_to_config_file>
```
Refer to the `experiments/configs` directory for the existing configuration files.
Alternatively, you can download the existing runs with the following command:
```bash
gdown "https://drive.google.com/uc?export=download&id=1jk--QGWTyyCJLDfb1lMxGhlhNfAlyUSm" -O "runs.zip"
unzip runs.zip -d experiments/
rm runs.zip
```
### Model Merging Experiment
To run a model merging experiment run the following command:
```bash
python -m experiments.model_merging
```
Refer to the `experiments/model_merging.py` file for the possible arguments to be passed.
Alternatively, you can download the existing generations with the following command:
```bash
gdown "https://drive.google.com/uc?export=download&id=1JVQikjKe599x1sgohMr8lvshHbbpj0kX" -O "files.zip"
unzip files.zip -d experiments/
rm files.zip
```
## Analysis
Refer to the notebook files under the `analysis` directory for the reference codes used for the analysis of the experiments.