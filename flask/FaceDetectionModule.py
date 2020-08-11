import cv2
from PIL import Image
import numpy as np
import gc


class FaceDetection:
    """
    Facial detection pipeline class. Used to detect faces in the frames of a video file.
    """

    def __init__(self, detector, device,  n_frames=16, batch_size=16, resize=None):
        """
        Constructor for FacialDetection class

        Args:
            detector (): face detector to use
            device: cuda device
            n_frames (): the number of frames that should be extracted from the video
                         frames will be evenly spaced
                          default is None, which results in all frames being extracted
            batch_size (): batch size to use with face detector, default is 16
            resize (): fraction by which frames are resized from original to face detection
                       <1: downsampling
                       >1: upsampling
                       default is None
        """

        self.detector = detector
        self.device = device
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize

    def __call__(self, file, temporal=False):
        """
        This methods extracts frames and faces from a mp4

        Args:
            file (): path + filename of the video

        Returns: list of face images
        """

        # read video from file
        v_cap = cv2.VideoCapture(file)

        # get frame count of video
        v_frames = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # create indices for the specified amount n_frames (random window)

        if not self.n_frames:
            # if number of frames not specified, create index for all frames in video
            frame_indices = np.arange(0, v_frames)
        else:
            if v_frames < self.n_frames:
                print(f"File has less than {self.n_frames} frames. Skipping...")
                return None
            # if number of frames is specified, create n_frames equidistant indices
            if not temporal:
                frame_indices = np.arange(0, v_frames, v_frames/self.n_frames).astype(int)
            # for temporal model, create n_frames/5 equidistant frame windows of length 5
            else:
                start_indices = np.arange(0, v_frames-5, v_frames / (self.n_frames/5.)).astype(int)
                window_list = [[idx, idx+1, idx+2, idx+3, idx+4] for idx in list(start_indices)]
                frame_indices = list(np.array(window_list).flat)

        # init lists to fill with frames and faces
        faces = []
        # batch list for frames
        frame_batch = []

        # Loop through frames
        for frame_index in range(v_frames):

            # grab_frame
            _ = v_cap.grab()

            # if frame is in frame_indices
            if frame_index in frame_indices:

                # Load frame
                success, frame = v_cap.retrieve()

                # if retrieve fails, pass
                if not success:
                    continue

                # colors to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # extract from object
                frame = Image.fromarray(frame)

                # append to frame list
                frame_batch.append(frame)

                del frame
                gc.collect()

                # When batch is full or no more indices
                if len(frame_batch) % self.batch_size == 0 or frame_index == frame_indices[-1]:
                    # detect faces in frame list, append batch to face list
                    # note, if no face is present, None is appended
                    faces.extend(self.detector(frame_batch))
                    # reset frame list
                    frame_batch = []

        # release device resource
        v_cap.release()

        return faces

