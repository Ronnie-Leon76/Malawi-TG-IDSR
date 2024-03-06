# Malawi Public Health Systems LLM Challenge

## Introduction
This repository contains the code and data for the Malawi Public Health Systems LLM Challenge. The challenge is to train an LLM to answer questions about Malawi's public health system. 


## Data
The data is in the `data` directory. The data is in the form of a XLSX workbook.


## Setup

### Requirements
- Python 3.6+
- Dependencies listed in `requirements.txt`

### Technologies Used
- Qdrant Vector Store
- PubmedBERT Embeddings
- Unsloth
- Hugging Face
- FastAPI

### Installation
1. Clone the repository
2. Install the dependencies using `pip install -r requirements.txt`
3. If on Windows, open git bash shell and run the following commands within this directory:
    - `sudo chmod u+x setup.sh`
    - `./setup.sh`
On Linux, you can just run `sudo chmod u+x setup.sh` and `./setup.sh` in the terminal.
4. Activate into your virtual environment using the following commands in Windows:
    - `source venv/Scripts/activate`
For Linux users, you can just run `source venv/bin/activate` in the terminal.
5. Run the script preprocess.py to preprocess the data using the following command:
    - `python preprocess.py`
6. Run the script ingest.py to convert the XLS workbooks into embeddings storing the vector embeddings in Qdrant Vector store:
    - `python ingest.py`
7. Navigate to the Download_GGUF_Model and run the Download_GGUF notebook to download the 8-bit GGUF model from the Hugging Face model hub.
8. There are several directories with different jupyter notebooks:
    - `Exploratory Data Analysis` contains the notebooks used to explore the data
    - `Fine-tuning` contains the notebooks used to fine-tune the model
    - `CPU_Inferencing` contains the notebooks used to run inferencing on the CPU
    - `GPU_inferencing_Yielding_Highest_Score` contains the notebooks used to run inferencing on the GPU yielding the highest score.
    - `static` contains the static files for the web app
    - `templates` contains the HTML templates for the web app
    - `app.py` is the main file for the web app

9. Run the CPU_Inferencing notebook to run inferencing on the CPU


