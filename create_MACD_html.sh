#Creating html files from jupyter_notebooks:

NOTEBOOK="MACD_ANALYSIS/CRYPTO/Crypto_Model.ipynb"
HTML="MACD_ANALYSIS/CRYPTO/Crypto_Model.html"

jupyter nbconvert --execute --to html ${NOTEBOOK}
open ${HTML}
