from setuptools import setup, find_packages

setup(
    name='eegvix',
    version='1.0.1',
    description='Generating images from EEG signals using Generative Models.',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'tqdm'
    ],
)