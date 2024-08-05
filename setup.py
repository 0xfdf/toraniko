from setuptools import setup, find_packages

requirements = []
with open('requirements.txt', 'r') as fh:
    for line in fh:
        requirements.append(line.strip())

setup(
    name="toraniko",
    version="1.1.0",
    packages=find_packages(),
    install_requires = requirements
)
