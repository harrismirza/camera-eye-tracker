import cv2
import dlib

def get_bb_from_eye_points(points):
    x_points = [p.x for p in points]
    y_points = [p.y for p in points]
    return min(x_points), min(y_points), max(x_points) - min(x_points), max(y_points) - min(y_points)

if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

    frame = cv2.imread("Picture1.png", cv2.IMREAD_UNCHANGED)
    analysis_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGBA2GRAY)
    facial_landmarks = frame.copy()
    eye_bb = frame.copy()

    rects = detector(analysis_frame, 1)
    for rect in rects:
        shape = predictor(analysis_frame, rect)
        for i in range(0, 68):
            point = shape.part(i)
            cv2.circle(facial_landmarks, center=(point.x, point.y), radius=0, thickness=3, color=(0, 255, 0, 255))
            cv2.putText(facial_landmarks, text=str(i), org=(point.x + 2, point.y - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, thickness=1, color=(0, 255, 0, 255))

        left_eye_points = [shape.part(i) for i in range(36, 42)]
        right_eye_points = [shape.part(i) for i in range(42, 48)]

        eye_bb_rects = [get_bb_from_eye_points(left_eye_points), get_bb_from_eye_points(right_eye_points)]

        for (x, y, w, h) in eye_bb_rects:
            cv2.rectangle(eye_bb, (x, y), (x+w, y+h), color=(0, 255, 0, 255), thickness=3)

    cv2.imwrite("Picture1_landmark.png", facial_landmarks)
    cv2.imwrite("Picture1_eye_bb.png", eye_bb)