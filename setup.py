from io import open
from setuptools import setup, find_packages

with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()

def readme():
    with open('README.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README

classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

setup(
    name='ollama',
    packages=find_packages(),
    version='1.0.0',
    install_requires=requirements,
    license='MIT',
    description='Connect to Ollama API',
    long_description=readme(),
    author='Howw0',
    url='https://github.com/Howw0/ollamaClient',
    download_url='https://github.com/Howw0/ollamaClient.git',
    keywords='ollama',
    classifiers=classifiers,
    )
