import cv2
import dlib

if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

    camera = cv2.VideoCapture(1)

    while True:
        ret, frame = camera.read()
        rects = detector(frame, 1)
        if len(rects) > 0:
            rect = rects[0]
            shape = predictor(frame, rect)
            for i in range(0, 68):
                point = shape.part(i)
                cv2.circle(frame, center=(point.x, point.y), radius=0, thickness=3, color=(0, 255, 0))
                cv2.putText(frame, text=str(i), org=(point.x + 2, point.y - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.2, thickness=1, color=(0, 255, 0))

        cv2.imshow("Facial Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()