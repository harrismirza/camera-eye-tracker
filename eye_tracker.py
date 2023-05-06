import cv2
import dlib
import time
from collections import deque
import pyvirtualcam

from multiprocessing import Process, Queue

import queue

CAMERAS_TO_USE = [0, 1]
MOVING_AVG_SIZE = 2
ANALYSIS_INTERVAL = 1
VIRTUAL_CAM_RESOLUTION = (1280, 720, 30)
DEBUG_VIEW = True
SHOW_EYES_POINTS = True
SHOW_EYES_BB = True


def analysis_task(frame_dict_queue, best_cam_queue, cam_ids):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    frame_dict = {}
    ratio_dict = {cam_id: deque(maxlen=MOVING_AVG_SIZE) for cam_id in cam_ids}

    while True:
        try:
            frame_dict = frame_dict_queue.get(False)
        except queue.Empty:
            if frame_dict:
                print("Got Frame, Analysing")
                start = time.time()
                best_cam, eyes = get_best_cam(frame_dict, ratio_dict, detector, predictor)
                analysis_time = time.time() - start
                print(f"Analysis took {analysis_time}s, {analysis_time/(1/VIRTUAL_CAM_RESOLUTION[2]) * 100}% of frame time")
                best_cam_queue.put((best_cam, eyes))
                discard = 0
                frame_dict = {}


def get_bb_from_eye_points(points):
    x_points = [p.x for p in points]
    y_points = [p.y for p in points]
    return min(x_points), min(y_points), max(x_points) - min(x_points), max(y_points) - min(y_points)


def get_eyes_from_frame(frame, detector, predictor):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(frame, 1)
    if (len(rects) != 0):
        rect = rects[0]

        shape = predictor(frame, rect)

        left_eye_points = [shape.part(i) for i in range(36, 42)]
        right_eye_points = [shape.part(i) for i in range(42, 48)]

        return [
            left_eye_points,
            right_eye_points
        ]

    return []


def get_best_cam(frame_dict, ratio_dict, detector, predictor):
    eye_dict = {}
    for cam_id, frame in frame_dict.items():
        eyes = get_eyes_from_frame(frame, detector, predictor)
        eye_dict[cam_id] = eyes
        bb_eyes = [get_bb_from_eye_points(eye) for eye in eyes]

        # print(f"Video {cam_id}")
        # for i, (x, y, w, h) in enumerate(bb_eyes):
        # print(f"Eye {i}: ({w}, {h}) at ({x}, {y}). A: {w*h}")
        if (len(bb_eyes) == 2):
            eye_1_area = bb_eyes[0][2] * bb_eyes[0][3]
            eye_2_area = bb_eyes[1][2] * bb_eyes[1][3]

            eye_ratio = max(eye_1_area, eye_2_area) / min(eye_1_area, eye_2_area)
            ratio_dict[cam_id].append(eye_ratio)
            # print(f"Ratio: {eye_ratio}")
        else:
            ratio_dict[cam_id].append(50)

    # Print best camera
    best_cam = -1
    best_avg_err = 1000000
    for (cam_id, eye_ratios) in ratio_dict.items():
        avg_err = abs(sum(eye_ratios) / len(eye_ratios) - 1)
        if (avg_err < best_avg_err):
            best_cam = cam_id
            best_avg_err = avg_err
    print(f"Best video: {best_cam}, avg_err={best_avg_err}")
    return best_cam, eye_dict


if __name__ == "__main__":
    vids = {cam_id: cv2.VideoCapture(cam_id) for cam_id in CAMERAS_TO_USE}

    last_analysis_time = time.time()
    last_eye_dict = {}
    best_cam_id = CAMERAS_TO_USE[0]

    frame_dict_queue = Queue()
    best_cam_queue = Queue()

    print("Creating Analysis Proc")
    analysis_proc = Process(target=analysis_task, args=(frame_dict_queue, best_cam_queue, list(vids.keys())))

    with pyvirtualcam.Camera(VIRTUAL_CAM_RESOLUTION[0], VIRTUAL_CAM_RESOLUTION[1],
                             VIRTUAL_CAM_RESOLUTION[2]) as virtualcam:
        print("Opened virtual camera, starting analysis proc")

        analysis_proc.start()

        while True:
            vid_eye_ratios = {cam_id: deque(maxlen=MOVING_AVG_SIZE) for (cam_id, vid) in vids.items()}

            try:
                best_cam_id, last_eye_dict = best_cam_queue.get(False)
                print(f"Got Best Cam {best_cam_id}")

            except queue.Empty:
                pass

            vid_frames = {}
            main_ret, main_frame = vids[best_cam_id].read()
            vid_frames[best_cam_id] = main_frame

            for cam_id, vid in vids.items():
                if cam_id != best_cam_id:
                    ret, frame = vid.read()
                    vid_frames[cam_id] = frame

            if time.time() - last_analysis_time > ANALYSIS_INTERVAL:
                frame_dict_queue.put(vid_frames)
                last_analysis_time = time.time()

            if DEBUG_VIEW:
                for cam_id, debug_frame in vid_frames.items():
                    debug_frame = debug_frame.copy()
                    if cam_id in last_eye_dict:
                        last_eyes = last_eye_dict[cam_id]
                        if SHOW_EYES_POINTS:
                            for eye in last_eyes:
                                for point in eye:
                                    # print(point)
                                    cv2.circle(debug_frame, (point.x, point.y), radius=0, color=(0, 255, 0), thickness=3)
                        if SHOW_EYES_BB:
                            for eye in last_eyes:
                                (x, y, w, h) = get_bb_from_eye_points(eye)
                                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=1)
                    cv2.imshow(f"DEBUG: {cam_id}", debug_frame)

            virtual_cam_frame = main_frame.copy()

            virtual_cam_frame = cv2.cvtColor(
                cv2.resize(virtual_cam_frame, (VIRTUAL_CAM_RESOLUTION[0], VIRTUAL_CAM_RESOLUTION[1])),cv2.COLOR_BGR2RGB)

            virtualcam.send(virtual_cam_frame)
            virtualcam.sleep_until_next_frame()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    analysis_proc.kill()
    for cam in vids.values():
        cam.release()
