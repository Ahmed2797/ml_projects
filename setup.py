from setuptools import setup,find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()

        #requirements = [reg.replace('/n','') for reg in requirements] 
        requirements = [reg.strip() for reg in requirements if reg.strip()]


        #requirements = [req for req in requirements if HYPEN_E_DOT in req] #-- only '-e .'

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        #requirements = [req for req in requirements if HYPEN_E_DOT not in req]
            
        return requirements

setup(
    name='ml_projects',
    version='0.0.1',
    author='Ahmed',
    author_email='tanvirahmed754575@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)