import face_recognition
import cv2

video_capture = cv2.VideoCapture(0)
video_capture.set(5, 4) 

path1 = "/home/pi/Desktop/FaceReco/faces_1/"
path2 = "/home/pi/Desktop/FaceReco/faces_2/"
path3 = "/home/pi/Desktop/FaceReco/faces_3/"

known_face_encodings1 = []
known_face_encodings2 = []
known_face_encodings3 = []

for i in range(15):
    image = face_recognition.load_image_file(path1  + "Face." + str(i+1) +".jpg")
    known_face_encodings1.append(face_recognition.face_encodings(image)[0])
print('Pierwsza baza twarzy zostala zaladowana.')
for i in range(15):
    image = face_recognition.load_image_file(path2  + "Face." + str(i+1) +".jpg")
    known_face_encodings2.append(face_recognition.face_encodings(image)[0])
print('Druga baza twarzy zostala zaladowana.')
for i in range(15):
    image = face_recognition.load_image_file(path3  + "Face." + str(i+1) +".jpg")
    known_face_encodings3.append(face_recognition.face_encodings(image)[0])
print('Trzecia baza twarzy zostala zaladowana.')

face_locations = []
face_encodings = []
process_this_frame = True

print('recognition was started')
while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, -1)
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    ##rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        
        
        if(len(face_locations)== 1):
            
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            
            matches1 = face_recognition.compare_faces(known_face_encodings1, face_encodings[0])
            matches2 = face_recognition.compare_faces(known_face_encodings2, face_encodings[0])
            matches3 = face_recognition.compare_faces(known_face_encodings3, face_encodings[0])
            result = {}

            for i in range(15):
                x = 0
                if(matches1[i] == True):
                    x= x +1
                if(matches2[i] == True):
                    x= x +1
                if(matches3[i] == True):
                    x= x +1
                result[i+1] = x 
            print(result)

            max_value = max(result.values())
            for k, v in result.items():
                if (v == max_value and v >= 2):
                    print(k)
            
        elif(len(face_locations) > 1):
            print("too many faces")
            

    process_this_frame = not process_this_frame
    
video_capture.release()
cv2.destroyAllWindows()
