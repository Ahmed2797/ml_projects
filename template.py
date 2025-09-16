from pathlib import Path
import os
import logging

logging.basicConfig(level=logging.INFO)

project_name = 'mlproject'
list_of_file = [
    f'src/{project_name}/__init__.py',
    f'src/{project_name}/components/__init__.py',
    f'src/{project_name}/components/data_ingestion.py',
    f'src/{project_name}/components/data_transformation.py',
    f'src/{project_name}/components/model_trainer.py',
    f'src/{project_name}/components/model_monitering.py',
    f'src/{project_name}/pipeline/train_pipeline.py',
    f'src/{project_name}/pipeline/predict_pipeline.py',
    f'src/{project_name}/exception.py',
    f'src/{project_name}/logger.py',
    f'src/{project_name}/utils.py',
    'app.py',
    'main.py',
    'requirements.txt'
]

for filepath in list_of_file:
    filepath = Path(filepath)
    file_dir, file_name = os.path.split(filepath)

    
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f'Creating directory: {file_dir} for the file {file_name}')

    
    if (not os.path.exists(filepath) or os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
        logging.info(f'Creating empty file: {file_name}')
    else:
        logging.info(f'{file_name} is already exists')


'''
sre/
└── mlproject/
    ├── __init__.py
    ├── components/
    │   ├── __init__.py
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   ├── model_trainer.py
    │   └── model_monitering.py
    ├── pipeline/
    │   ├── train_pipeline.py
    │   └── predict_pipeline.py
    ├── exception.py
    ├── logger.py
    └── utils.py
app.py
main.py
requirements.txt

'''

