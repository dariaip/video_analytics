# Idea
Pursuit car detection can be crutial task for business. The pipeline allows to detect the case automatically.

# Pipeline
- upload video
- cut video into images
- detect objects on each image
- reidentificate objects
- highlight that the object presents on the screen during too long time

# How to run the pipeline
python -m venv venv *(I use python 3.10)*

source venv/bin/activate *(or venv\Scripts\activate in case of using Windows)*

python -m pip install --upgrade pip

python -m pip install --upgrade setuptools

python -m pip install -r requirements.txt

streamlit run app.py