# DeepfakeDetective
## Basic web application skeleton for identifying Deepfakes 
Built using flask and
1. MTCNN for face detection (https://github.com/davidsandberg/facenet)
2. InceptionResNetV1 + DBSCAN for actor face clustering (https://github.com/davidsandberg/facenet)
3. EfficientNet for deepfakedetection per face (https://github.com/lukemelas/EfficientNet-PyTorch)




## Setting up
Clone repository and set up environment by:

<pre><code>$ pip install -r requirements.txt
</code></pre>

## Usage
Once app is up, connect to localhost:5000.
From there, paste any YouTube link into the search bar to estimate the deepfakeness of each actor in the clip.
Currently, a relatively weak model is deployed, this will be changed in future updates.
