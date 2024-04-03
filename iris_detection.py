""""
   --------------------EYE TRACKING USING  MEDIAPIPE AND OPENCV ----------------------
"""
import mediapipe as mp
import cv2


mp_drawing = mp.solutions.drawing_utils
mp_face_mash = mp.solutions.face_mesh #468 face landmarks
mp_drawing_styles = mp.solutions.drawing_styles
draw_specs = mp_drawing.DrawingSpec(thickness=1,circle_radius=1)

def get_landmark(image):
    face_mash  = mp_face_mash.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5)
    image.flags.writeable = False
    result = face_mash.process(image)
    landmarks = result.multi_face_landmarks[0].landmark
    return result,landmarks

def draw_landmarks(image,result):
    image.flags.writeable = True
    if result.multi_face_landmarks:
        for face_landmark in result in result.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmark,
                landmarks = result.multi_face_landmarks[0].landmark,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
    return image





path_image = 'iriscode.jpg'
img = cv2.imread(path_image)

result,landmarks = get_landmark(image=img)

annotated_img = draw_landmarks(image=annotated_img,result=result)

cv2.imshow("annotated image",annotated_img)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

