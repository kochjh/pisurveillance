# import the necessary packages
from camera.keyclipwriter import KeyClipWriter
from camera.singlemotiondetector import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
from os import listdir
from os.path import isfile, join
import threading
import argparse
import datetime
import imutils
import time
import cv2

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
camera_frame = None
camera_lock = threading.Lock()

video_frame = None
video_lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
# vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


@app.route("/video_list")
def video_list():
    videos = [format_timestamp(f[:-4]) for f in listdir("output/") if isfile(join("output/", f))]
    return render_template("video_list.html", videos=reversed(videos))


@app.route("/video/<name>")
def video(name):
    t = threading.Thread(target=video_player, args=(str(name),), daemon=True)
    t.start()
    return render_template("video.html", video_name=name)


def video_player(name):
    global video_frame, video_lock
    cap = cv2.VideoCapture("output/" + undo_format_timestamp(name) + '.avi')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            with video_lock:
                video_frame = frame.copy()
        else:
            cap.release()
            break
    cap.release()


def detect_motion(frame_count, buffer_size, output, codec, fps):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, camera_frame, camera_lock
    # initialize the motion detector and the total number of frames
    # read thus far
    md = SingleMotionDetector(accumWeight=0.1)
    kcw = KeyClipWriter(bufSize=buffer_size)
    total = 0
    consec_frames = 0
    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        # if the total number of frames has reached a sufficient
        # number to construct a reasonable background model, then
        # continue to process the frame
        if total > frame_count:
            # detect motion in the image
            motion = md.detect(gray)
            # check to see if motion was found in the frame
            if motion is not None:
                # unpack the tuple and draw the box surrounding the
                # "motion area" on the output frame
                # (thresh, (minX, minY, maxX, maxY)) = motion
                # cv2.rectangle(frame, (minX, minY), (maxX, maxY),
                #             (0, 0, 255), 2)
                consec_frames = 0
                # if we are not already recording, start recording
                if not kcw.recording:
                    print('[INFO] start recording')
                    timestamp = datetime.datetime.now()
                    p = "{}/{}.avi".format(output, timestamp.strftime("%Y%m%d-%H%M%S"))
                    kcw.start(p, cv2.VideoWriter_fourcc(*codec), fps)
            else:
                consec_frames += 1

        # update the background model and increment the total number
        # of frames read thus far
        md.update(gray)
        # update the key frame clip buffer
        kcw.update(frame)
        if total <= frame_count:
            total += 1

        if kcw.recording and consec_frames == buffer_size:
            print('[INFO] stop recording')
            kcw.finish()

        # acquire the lock, set the output frame, and release the
        # lock
        with camera_lock:
            camera_frame = frame.copy()


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate_camera(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/play_video")
def play_video():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate_video(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def generate_camera():
    # grab global references to the output frame and lock variables
    global camera_frame, camera_lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with camera_lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if camera_frame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", camera_frame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


def generate_video():
    # grab global references to the output frame and lock variables
    global video_frame, video_lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with video_lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if video_frame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", video_frame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


def format_timestamp(timestamp):
    # input: yyyymmdd-hhmmss
    date = '.'.join([timestamp[6:8], timestamp[4:6], timestamp[0:4]])
    time = ':'.join([timestamp[9:11], timestamp[11:13], timestamp[13:15]])
    # output: dd.mm.yyyy - hh:mm:ss
    return date + ' - ' + time


def undo_format_timestamp(timestamp):
    # input: dd.mm.yyyy - hh:mm:ss
    date = ''.join([timestamp[6:10], timestamp[3:5], timestamp[0:2]])
    time = ''.join([timestamp[13:15], timestamp[16:18], timestamp[19:21]])
    # output: yyyymmdd-hhmmss
    return date + '-' + time


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-p", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-fc", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    ap.add_argument("-o", "--output", required=True,
                    help="path to output directory")
    ap.add_argument("-pc", "--picamera", type=int, default=-1,
                    help="whether or not the Raspberry Pi camera should be used")
    ap.add_argument("-f", "--fps", type=int, default=20,
                    help="FPS of output video")
    ap.add_argument("-c", "--codec", type=str, default="MJPG",
                    help="codec of output video")
    ap.add_argument("-b", "--buffer-size", type=int, default=32,
                    help="buffer size of video clip writer")
    args = vars(ap.parse_args())
    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion, args=(
        args["frame_count"], args['buffer_size'], args['output'], args['codec'], args['fps']))
    t.daemon = True
    t.start()
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()
