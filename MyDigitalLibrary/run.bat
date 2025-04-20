@echo off
REM Replace 'your_env_name' with the name of your conda environment
REM Replace 'app.py' with your Streamlit app file

CALL conda activate ai-dev-env
streamlit run app.py
PAUSE
