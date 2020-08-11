# DeepfakeDetective
## Basic web application skeleton for identifying Deepfakes 
Completely dockerized using docker-compose and flask/nginx/uwsgi.


## Setting up
Clone repository and set up environment by:

<pre><code>$ docker-compose up --build
</code></pre>

## Usage
Once docker-compse is up, connect to localhost:5000.
From there, paste any YouTube link into the search bar to estimate the deepfakeness of each actor in the clip.
Currently, a relatively weak model is deployed, this will be changed in future updates.
