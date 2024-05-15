from setuptools import setup, find_packages

# 读取 requirements.txt 中的依赖
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='Bert_NER',  
    version='0.0.1', 
    description='This project is a reasearch to test whether a CRF and GRU will help bert in NER',  
    author='Wenjun Zhang',  
    author_email='1378555845gg@gmail.com',
    packages=find_packages(),  
    install_requires=required,  
    
)