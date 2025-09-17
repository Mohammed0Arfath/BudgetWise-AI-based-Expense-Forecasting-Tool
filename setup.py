from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(
    name='BudgetWise',
    version='0.0.1',
    description='A personal expense tracker application',
    author='Mohammed Arfath R',
    author_email='mohammedarfath02003@gmail.com',
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires='>=3.10',
)