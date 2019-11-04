import cv2
import numpy as np
import time
import pandas as pd
import pickle
def euclidean(a,b):
    return np.sqrt(np.sum(np.power( a - b, 2)))


def calc_ang(d):
    d1 = np.power(d,2)
    d2 = d1[:,0] + d1[:,1] - d1[:,2]
    d3 = 2*d[:,0] * d[:,1]
    d2 = d2/d3
    d2 = np.degrees(np.arccos(d2))
    #print(d2)
    return d2

def fingers_analisys(points,frame):
    t1,t2,t3,t4 = False,False,False,False
    distances = []
    if( points[4] != None and points[8] != None and points[3] != None and points[16] != None and points[20] != None and points[15] != None and points[12] != None and points[16] != None and points[11] != None and points[8] != None and points[12] != None and points[7] != None):
        t1 = True
        d1 = euclidean(np.array(points[4]), np.array(points[8]))
        d2 = euclidean(np.array(points[4]), np.array(points[3]))
        d3 = euclidean(np.array(points[8]), np.array(points[3]))
        cv2.line(frame, points[4], points[8], (255, 0, 255), 1)
        distances.append([d1,d2,d3])


        t2 = True
        d1 = euclidean(np.array(points[8]), np.array(points[12]))
        d2 = euclidean(np.array(points[8]), np.array(points[7]))
        d3 = euclidean(np.array(points[12]), np.array(points[7]))
        cv2.line(frame, points[8], points[12], (255, 0, 255), 1)
        distances.append([d1,d2,d3])



        t3 = True
        d1 = euclidean(np.array(points[12]), np.array(points[16]))
        d2 = euclidean(np.array(points[12]), np.array(points[11]))
        d3 = euclidean(np.array(points[16]), np.array(points[11]))
        cv2.line(frame, points[12], points[16], (255, 0, 255), 1)
        distances.append([d1,d2,d3])

        t4 = True
        d1 = euclidean(np.array(points[16]), np.array(points[20]))
        d2 = euclidean(np.array(points[16]), np.array(points[15]))
        d3 = euclidean(np.array(points[20]), np.array(points[15]))
        cv2.line(frame, points[20], points[16], (255, 0, 255), 1)
        distances.append([d1,d2,d3])

    if(len(distances) > 0):
        d = np.array(distances)
        alfa1 = calc_ang(d)
        d_desl = np.concatenate(([d[:,1]],[d[:,2]],[d[:,0]])).T
        alfa2 = calc_ang(d_desl)
        d_desl = np.concatenate(([d_desl[:,1]],[d_desl[:,2]],[d_desl[:,0]])).T
        alfa3 = calc_ang(d_desl)
        #a1 = (d1**2 + d2**2 -d3**2)/(2*d1*d2)
        #cv2.line(frame, points[4], points[8], (255, 0, 255), 1)
        #print(np.degrees(np.arccos(a1)))
       # print(alfa1.flatten())

                
        result = np.concatenate((d.flatten(),alfa1.flatten(),alfa2.flatten(),alfa3.flatten(),[2]))
     
       # result = np.concatenate((d.flatten(),alfa1.flatten(),alfa2.flatten(),alfa3.flatten()))
      
     #  #print(result)
      #  print(result.shape)
        return result

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22
 
#frame = cv2.imread("hand.jpg")
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

captura = cv2.VideoCapture(0)
threshold = 0.1
count = 0
points_fingers = [4,8,12,16,20]
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
#file_saida = open('data_1.csv','rb')
list_points_ = []
model = pickle.load(open('model_decision.pkl','rb'))
while(1):
    ret, frame = captura.read()
    count += 1
    if(count ==5):
        frameCopy = np.copy(frame)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        aspect_ratio = frameWidth/frameHeight
        inWidth = 320
        inHeight = 210#int(((inWidth/aspect_ratio)*10)//10)
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
     #   print(inHeight, inWidth)
     #   print(frameHeight, frameWidth)
        net.setInput(inpBlob)
    
        output = net.forward()
     
     
        points = []
        init = time.time()
        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (frameWidth, frameHeight))
    
            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap) 
    
            if prob > threshold :
                #cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                #cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    
                # Add the point to the list if the probability is greater than the threshold
                points.append((int(point[0]), int(point[1])))
            else :
                points.append(None)
    
    
     #   print(points)
        # Draw Skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
    
            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
                cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    
    
    
     #   for j in points_fingers:
     #       i = points[j]
      #      if(i != None):
       #        if(i[0] >550):
       #            print("finger {}".format(j), i[0], i[1])
    
        #print(points[4], points[8])     
        result= fingers_analisys(points,frame)
        #print(len(result))
        #result = np.concatenate((result))
        if(result is not  None):

            result = result.reshape((25,))
            print(result.shape)
            #print(np.isnan(result).sum())
            if(np.isnan(result).sum() == 0):
            #    y_ = model.predict(result)
            #    print(y_)
                list_points_.append(result)
        #file_saida.write(result)
                

        #cv2.imshow('Output-Keypoints', frameCopy)
        count = 0
        cv2.imshow('Output-Skeleton', frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break


df = pd.DataFrame(list_points_)
df.to_csv('data1.csv',index=False)
captura.release()
cv2.destroyAllWindows()

