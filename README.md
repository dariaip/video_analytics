# Idea
Pursuit car detection can be a crucial task for a business. The pipeline allows to detect the case automatically.

# Pipeline
- upload video
- cut video into images
- detect objects on each image
- reidentificate objects
- highlight that the object presents on the screen for too long time

# Demo (speeded-up artificially)

https://github.com/dariaip/video_analytics/assets/41380940/462ed0d9-02a9-4ba5-a2aa-ccbbcb6660d6


# How to run the pipeline
```python -m venv venv``` *(I use python 3.10)*

```source venv/bin/activate``` *(or ```venv\Scripts\activate``` in case of using Windows)*

```
python -m pip install --upgrade pip

python -m pip install --upgrade setuptools

python -m pip install -r requirements.txt

streamlit run app.py
```

# Further improvements
The main problem of the solution is its speed. There are two bottlenecks that can speed up the pipeline:
1) Models' optimization (for example, pruning)
2) Algorithm for object comparison between two frames. It compares each object with all objects from the previous frame, but in the case of frequent frames cars don't move significantly, and we can assume that only spatially close objects can be representations of the same object on the different frames. 
