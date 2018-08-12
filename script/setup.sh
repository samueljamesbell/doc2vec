#! /bin/bash

pyenv install 3.6.6

cd ..
pyenv virtualenv 3.6.6 doc2vec
cd doc2vec

pip install --upgrade pip

pip install -r requirements.txt

echo "Done"
