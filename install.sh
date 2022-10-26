conda create -n elmer python=3.7
conda activate elmer

pip install -r requirements.txt 

pip install -U git+https://github.com/pltrdy/pyrouge
git clone https://github.com/pltrdy/files2rouge.git 
cd files2rouge
python setup_rouge.py
python setup.py install
