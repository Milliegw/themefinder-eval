# themefinder-eval

python -m venv .venv               # create (once per project)
source .venv/bin/activate          # switch it on (every new terminal)
pip install pandas                 # install into this project only
pip freeze > requirements.txt      # record what's installed
deactivate                         # switch it off