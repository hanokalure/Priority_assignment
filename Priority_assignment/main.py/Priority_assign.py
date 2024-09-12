from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib.mailer import Mailer
from mylib import config, thread
import time, csv
import numpy as np
import argparse, imutils
import dlib, cv2, datetime
from itertools import zip_longest

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=False,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-i", "--input", type=str,
                    help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
                    help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
                    help="# of skip frames between detections")
    args = vars(ap.parse_args())

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    print("[INFO] Starting the video..")
    vs = cv2.VideoCapture(args["input"])

    writer = None
    W = None
    H = None

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    totalFrames = 0
    totalDown = 0  # Count for "Enter"
    totalUp = 0    # Count for "Exit"
    x = []
    empty = []
    empty1 = []

    fps = FPS().start()

    # Dictionary to keep track of priority display times
    priority_display_times = {}
    priority_number = 1

    while True:
        ret, frame = vs.read()

        if not ret:
            break

        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

        status = "Waiting"
        rects = []

        if totalFrames % args["skip_frames"] == 0:
            status = "Detecting"
            trackers = []

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > args["confidence"]:
                    idx = int(detections[0, 0, i, 1])

                    if CLASSES[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    trackers.append(tracker)
        else:
            for tracker in trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                rects.append((startX, startY, endX, endY))

        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
        cv2.putText(frame, "-Prediction border - Entrance-", (10, H - 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        objects = ct.update(rects)
        current_time = time.time()

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    # Moving downward (person enters from above and crosses below the center line)
                    if direction > 0 and centroid[1] > H // 2:
                        totalDown += 1  # Increment "Enter" count
                        empty1.append(totalDown)

                        # Assign the priority number only for people who enter
                        if objectID not in priority_display_times:
                            priority_display_times[objectID] = (current_time, priority_number)
                            priority_number += 1

                        to.counted = True

                    # Moving upward (person exits from below and crosses above the center line)
                    elif direction < 0 and centroid[1] < H // 2:
                        totalUp += 1  # Increment "Exit" count
                        empty.append(totalUp)
                        x.append(len(empty) - len(empty1))

                        if sum(x) >= config.Threshold:
                            cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
                            if config.ALERT:
                                Mailer().send(config.MAIL)

                        to.counted = True

            trackableObjects[objectID] = to

        # Draw the priority number if it's within the display time limit
        for (objectID, (display_time, priority_num)) in list(priority_display_times.items()):
            if current_time - display_time <= 0.5:  # Display for 5 seconds
                if objectID in objects:
                    centroid = objects[objectID]
                    text = "Priority No {}".format(priority_num)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
            else:
                del priority_display_times[objectID]

        info = [
            ("Status", status),
        ]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        totalFrames += 1
        fps.update()

    fps.stop()
    print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

    if writer is not None:
        writer.release()

    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
