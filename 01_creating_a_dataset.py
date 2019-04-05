import cv2
#import os

cam = cv2.VideoCapture(0)
""""
cam.set(3, 1080) # set video width
cam.set(4, 720) # set video height
""""
cam.set(5, 5) #set video fps

face_detector = cv2.CascadeClassifier('/home/pi/Desktop/FaceReco/haarcascade_frontalface_default.xml')#you must probably edit this
face_id = input('\n write user id and press enter ==>  ')
#face_name = raw_input('\n write user name, surname or pseudo and press enter ==>  ')
count = 0
path = "/home/pi/Desktop/FaceReco/"
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
while(True):
    ret, img = cam.read()
    img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = gray.copy()
    cv2.equalizeHist(gray, hist)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    if len(faces) == 1:    
        count += 1
        cv2.imwrite(path + "/dataset/Face." + str(face_id) + '.' + str(count) + ".jpg", hist[y:y+h,x:x+w])
        if count == 3:
            cv2.imwrite(path + "/faces_1/Face." + str(face_id) + ".jpg", hist[y:y+h,x:x+w]) # +  str(face_name)
        if count == 11:
            cv2.imwrite(path + "/faces_2/Face." + str(face_id) + ".jpg", hist[y:y+h,x:x+w]) # +  str(face_name)
        if count == 19:
            cv2.imwrite(path + "/faces_3/Face." + str(face_id) + ".jpg", hist[y:y+h,x:x+w]) # +  str(face_name)
    else:
        print("\n Only one face please.")
    cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 23: # Take 23 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
