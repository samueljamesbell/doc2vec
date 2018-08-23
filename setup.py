import os
from setuptools import find_packages, setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "doc2vec",
    version = "0.0.1",
    author = "Samuel Bell",
    author_email = "samueljamesbell@gmail.com",
    description = "Python implementations of doc2vec algorithms.",
    license = "MIT",
    packages=find_packages(),
    long_description=read('README.md'),
    entry_points = {
        'console_scripts': [
            'doc2vec=doc2vec.doc2vec:main',
            'embeddings-to-tsv=doc2vec.script.embeddings_to_tsv:main',
        ],
    }
)
