import numpy as np
from itertools import compress
import torchvision.transforms as transforms
from torch.nn import functional as F
from PIL import Image
from pytube import YouTube
import torch
import glob
import os
import shutil
import uuid
import time
import matplotlib.pyplot as plt

# face detection and embeddings modules
from facenet_pytorch import MTCNN, InceptionResnetV1

# own stuff
from FaceDetectionModule import FaceDetection

# for face clustering
from sklearn.cluster import DBSCAN

# model
from EfficientNet import EfficientNet

# html template for predction
html_string_start = open("html_skeleton/prediction_html_prefix.txt", "r").read()

html_string_end = """</body> </html>"""


def predict_single_file(model, faces, agg, device, verbose=False, detective_mode=False):

    """
    needs to be revisited
    """

    # formatting faces
    if detective_mode:
        faces = torch.stack(faces).to(device)
    else:
        faces = torch.as_tensor(faces, dtype=torch.float32).to(device)

    # no gradient computations
    with torch.no_grad():
        # get per face prediction probabilities
        fake_prob = model.forward(faces)
        fake_prob = F.sigmoid(fake_prob)

        # aggregate prediction over all faces associated with file
        if agg == 'mean':
            fake_prob_file = torch.mean(torch.flatten(fake_prob))

        elif agg == 'max':
            fake_prob_file = torch.max(torch.flatten(fake_prob))

        else:
            assert False, "Invalid aggregation over probabilities specified"

        if verbose:
            print(f"Fake prob: {fake_prob_file}")

    return fake_prob_file.item(), fake_prob


def plot_sample_images(ims, title=None, save=False, path=None, name="unnamed", in_percent=False):

    assert not save or path, "To save the plots, please specify path to save"

    title = torch.flatten(title).tolist()
    if in_percent:
        title = [str(np.round(prob*100, 2)) + "%" for prob in title]
    else:
        title = [np.round(prob, 2) for prob in title]

    ims = [im.permute(1, 2, 0).numpy() for im in ims]
    ims = [(im - np.min(im)) / np.ptp(im) for im in ims]

    fig, axs = plt.subplots(nrows=1, ncols=len(ims), figsize=(len(ims)*2, 3)) # dynamically adapt to number of subplots
    for idx, ax in enumerate(axs):
        ax.imshow(ims[idx])
        ax.grid(None)
        ax.set_title(title[idx])
        ax.axis('off')

    plt.savefig(path + f"/{name}.jpg", bbox_inches='tight')

def clean_and_normalize_faces(faces):

    assert faces, 'No faces detected'
    # clean Nones from face list
    faces = [f for f in faces if f is not None]

    # because we keep all faces, some tensors might hold multiple faces > separate those
    faces_extended = []
    for face in faces:
        if face.shape[0] == 1:
            faces_extended.append(face.squeeze(0))
        else:
            for i in range(face.shape[0]):
                faces_extended.append(face[i])
    faces = faces_extended

    print(f"Detected {len(faces)} faces")

    # normalize
    faces = [face.type(torch.int32) / 255.0 for face in faces]
    faces = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(face) for face in
             faces]

    return faces


def cluster_faces(faces, resnet, verbose=True, plot_clusters=False):

    # calculate face embeddings for each face
    faces_embeddings = {}

    # first we create unique mapping between embedding and face
    for face in faces:
        embedding = resnet(face.unsqueeze(0))
        faces_embeddings[embedding] = face

    # reformat for clustering
    encodings = torch.cat(tuple(faces_embeddings.keys()))
    # determine the total number of unique faces found in the clip
    clt = DBSCAN(metric="euclidean", eps=0.75, min_samples=2, n_jobs=-1)
    clt.fit(encodings.detach().numpy())

    # check number of unique faces
    labelIDs = np.unique(clt.labels_)
    numUniqueFaces = len(np.where(labelIDs > -1)[0])

    if verbose:
        print(f" # unique faces detected: {numUniqueFaces}")

    # check that any unique face has been detected
    if not numUniqueFaces:
        return None

    # separate into clusters
    clusters = []
    for cluster_id in range(numUniqueFaces):
        cluster = list(compress(faces, clt.labels_ == cluster_id))

        # only take a max of 10 faces per cluster
        clusters.append(cluster[:10])

    del clt

    return clusters


def init_detective_model(path, device):

    model = EfficientNet.from_name('efficientnet-b0',
                                   override_params={
                                       'num_classes': 1,
                                       'dropout_rate': 0.5,            # does not matter, only eval
                                       'drop_connect_rate': 0.2        # does not matter, only eval
                                   }
                                   )
    model.load_state_dict(torch.load(path + f'checkpoint-{19}.pth.tar')['model'])
    model = model.to(device)
    model.eval()

    return model


def get_path_from_url(url):

     return "static/" + url.replace("/", "")\
                           .replace(":", "")\
                           .replace(".", "")\
                           .replace("?", "_")\
                           .replace("!", "_")


def store_clustered_faces(clusters, path, process_id):

    for cluster_id, cluster in enumerate(clusters):

        actor = cluster[1].permute(1, 2, 0).numpy()
        actor = (actor - np.min(actor)) / np.ptp(actor)

        im = Image.fromarray((actor*255).astype(np.uint8))
        im.save(path + f"/{process_id}_actor{cluster_id}.jpg")


def get_picture_html(path_actor, tag_actor, path_cluster):
    image_html = """<div> <p> <strong> {tag_name_actor} </strong> </p> <picture> <img class="actors" src= "../{path_name_actor}"  height="224" width="224"> </picture>  <picture> <img class="predictions" src= "../{path_name_cluster}"  height="300" width="1200"> </picture> <p>"""
    return image_html.format(tag_name_actor=tag_actor, path_name_actor=path_actor, path_name_cluster=path_cluster)


def generate_html(results_per_cluster, path, process_id):

    picture_html = ""

    if results_per_cluster is not None:
        for cluster_id, (result_actor, _) in enumerate(results_per_cluster):

            # get path to result images
            path_to_cluster_actor = path + f"/{process_id}_actor{cluster_id}.jpg"
            path_to_cluster = path + f"/{process_id}_pred_cluster{cluster_id}.jpg"

            # reformat results to percentages
            result_actor = "   " + str(np.round(result_actor * 100, 2)) + "%" # quick & dirty fix for text position, change later

            picture_html += get_picture_html(path_actor=path_to_cluster_actor,
                                             tag_actor=result_actor,
                                             path_cluster=path_to_cluster)

            file_content = html_string_start + picture_html + "</div>" + html_string_end

    else:
        file_content = html_string_start + "<p> Sorry, could not detect any faces </p>" + html_string_end

    with open('templates/prediction.html', 'w') as f:
        f.write(file_content)


def get_clusters(url, device, process_id):

    # get path
    path = get_path_from_url(url)

    # make a new dir for path
    try:
        os.makedirs(path)
    except FileExistsError:
        """
        store error log?
        """
        # remove old path
        shutil.rmtree(path)
        # new path
        os.makedirs(path)

    # load face detector
    face_detector = MTCNN(image_size=224, margin=10, keep_all=True, device=device, post_process=False).eval()
    face_detection = FaceDetection(face_detector, device, n_frames=30)

    # load model for face embeddings
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    # download youtube clip to mp4 in current dir
    yt = YouTube(url)
    yt.streams.get_highest_resolution().download(output_path=path)

    # get current clip
    clips = glob.glob(os.path.join(path, '*.mp4'))
    assert len(clips) == 1, "Invalid number of clips selected for detection"
    current_clip = clips[0]

    # detect faces
    print("Detecting faces")
    faces = face_detection(current_clip, temporal=False)

    # free memory
    del face_detection
    del face_detector
    torch.cuda.empty_cache()

    # formatting
    faces = clean_and_normalize_faces(faces)

    # cluster faces according to face embeddings
    print("Clustering faces")
    start = time.time()
    clusters = cluster_faces(faces, resnet, verbose=True)
    end = time.time()
    print(f"Clustering took {int(end-start)} sec")

    # check whether a least one cluster was found
    if not clusters:
        return None, path

    # store one face from each cluster to path for html generation later
    print("Storing faces")
    store_clustered_faces(clusters, path, process_id)

    # free memory
    del resnet
    torch.cuda.empty_cache()

    return clusters, path


def predict_clusters(clusters, path_id, model, device):

    # unpack info
    path, process_id = path_id

    if clusters:
        # predict each cluster of faces
        print("Predicting clusters")
        results_per_cluster = []
        for cluster_id, cluster in enumerate(clusters):

            # predict
            fake_prob_file, fake_prob_per_face = predict_single_file(model, cluster, agg='mean',
                                                                     device=device, detective_mode=True)
            # store fake probs per file and per face
            results_per_cluster.append((fake_prob_file, fake_prob_per_face))

            # plot face predictions and save to path
            plot_sample_images(cluster, title=fake_prob_per_face, save=True, path=path,
                               name=f"{process_id}_pred_cluster{cluster_id}", in_percent=True)

            del fake_prob_file, fake_prob_per_face
            torch.cuda.empty_cache()

    else:
        results_per_cluster = None

    # render html
    print("Generating html")
    generate_html(results_per_cluster, path, process_id)


def predict_from_url(url, model, device):

    # first generate uuid for this process (prevents browser caching issues)
    process_id = uuid.uuid1()

    # get the clusters for the url
    clusters, path = get_clusters(url, device, process_id)

    # store path and process id in tuple for less clutter
    path_id = (path, process_id)

    # predict the clusters and render html
    predict_clusters(clusters, path_id, model, device)

    # remove old path, disabled for debugging
    #shutil.rmtree(path)

    return path
