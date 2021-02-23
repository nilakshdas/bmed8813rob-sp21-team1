## Install pyenv and python version
1. https://github.com/pyenv-win/pyenv-win#pyenv-for-windows
2. `$env:Path += ";"+$HOME+"\.pyenv\pyenv-win\bin"`
3. `pyenv install 3.6.8`

## Install latest pip
1. https://www.liquidweb.com/kb/install-pip-windows/ -> download get-pip.py to C:\Users\<your_name>
2. `$env:Path += ";"+$HOME+"\.pyenv\pyenv-win\versions\3.6.8"`
3. `python ~\get-pip.py`

## Install dependencies
1. `$env:Path += ";"+$HOME+"\.pyenv\pyenv-win\versions\3.6.8\Scripts"`
2. `pip install numpngw git+https://github.com/Healthcare-Robotics/assistive-gym.git`

## Run scripts
1. `cd bmed8813rob-sp21-team1`
2. `$env:PYTHONPATH += ";"+$HOME+"\bmed8813rob-sp21-team1"`
3. `python bin/teleop.py`
