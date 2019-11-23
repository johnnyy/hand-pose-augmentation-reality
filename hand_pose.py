import cv2
import numpy as np
import time
import pandas as pd
import pickle



#
###############Funcoes 
###############
#
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

                
   #     result = np.concatenate((d.flatten(),alfa1.flatten(),alfa2.flatten(),alfa3.flatten(),[2]))
     
        result = np.concatenate((d.flatten(),alfa1.flatten(),alfa2.flatten(),alfa3.flatten()))
      
     #  #print(result)
      #  print(result.shape)
        return result




def draw(obj_list,frame):
    global leny,lenx
   # zeros = np.zeros((leny,lenx))

    for i in range(len(obj_list)):
       # print(obj_list[i][0])
        img_ = obj_list[i][1]
        point = obj_list[i][0]
        mask = obj_list[i][2]

        y_ = point[0] - img_.shape[1]
        x_ = point[1] - img_.shape[0]
        
        if(x_ >= 0 and y_ >= 0):
            a = frame[x_:point[1],y_:point[0]]
            #print(a.shape, img_.shape,mask.shape)
            result = cv2.bitwise_and(img_,mask)
            mask_not = cv2.bitwise_not(mask)
            result_ = cv2.bitwise_and(a,mask_not)
            frame[x_:point[1],y_:point[0]] = cv2.bitwise_or(result,result_) 
#        cv2.rectangle(img_, 
           # cv2.rectangle(frame,(y_, x_),point,(255,255,255), -1)
    
    return frame

def select_element_in_list(obj_list, point_):
    list_return = []
    for i in range(len(obj_list)):
        img_ = obj_list[i][1]
        point = obj_list[i][0]
#        mask = obj_list[i][2]
        

        y_ = point[0] - img_.shape[1]
        x_ = point[1] - img_.shape[0]
        
        if(point_[0] >= y_ and point_[0]<= point[0] and point_[1] >= x_ and point_[1]<= point[1] ):
            list_return.append(obj_list.pop(i))
            break
    return list_return
    

############### Variáveis
###############
#############

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22
 
#frame = cv2.imread("hand.jpg")
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

captura = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi',fourcc, 2.0, (640,480))
#captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
#captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
#captura.set(cv2.CAP_PROP_FPS, 1)
threshold = 0.1
count = 0
points_fingers = [4,8,12,16,20]
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
#file_saida = open('data_1.csv','rb')
list_points_ = []
model = pickle.load(open('model_decision.pkl','rb'))




img1 = cv2.imread('peca1.png',cv2.IMREAD_UNCHANGED)
img1 = cv2.resize(img1,(70,60))
#print(img1.shape)
img1_mask = cv2.cvtColor(img1[:,:,3], cv2.COLOR_GRAY2RGB)
img1 = img1[:,:,:3]

img2 = cv2.imread('peca2.png',cv2.IMREAD_UNCHANGED)
img2 = cv2.resize(img2,(70,60))
#print(img1.shape)
img2_mask = cv2.cvtColor(img2[:,:,3], cv2.COLOR_GRAY2RGB)
img2 = img2[:,:,:3]

img3 = cv2.imread('peca3.png',cv2.IMREAD_UNCHANGED)
img3 = cv2.resize(img3,(70,60))
#print(img1.shape)
img3_mask = cv2.cvtColor(img3[:,:,3], cv2.COLOR_GRAY2RGB)
img3 = img3[:,:,:3]

img4 = cv2.imread('peca4.png',cv2.IMREAD_UNCHANGED)
img4 = cv2.resize(img4,(70,60))
#print(img1.shape)
img4_mask = cv2.cvtColor(img4[:,:,3], cv2.COLOR_GRAY2RGB)
img4 = img4[:,:,:3]



list_obj = []
list_obj_temp  = []
command = 0
flag_command = False

#command 0 = nada a fazer
#command 1 = posicionar obj 1
#command 2 = posicionar obj 2
#command 3 = posicionar obj 3
#command 4 = posicionar obj 4
#command 5 = mover obj selecionado
leny,lenx = 0,0

count_desable_command = 0
select_point = 0
img_draw = None
mask_img_draw = None

while(1):
    ret, frame = captura.read()
    leny,lenx,_ = frame.shape
    img_fi = np.zeros((leny,lenx,3)) 

    
    #frame = cv2.flip(frame,1)
    count += 1
    if(count ==5):
        frameCopy = np.copy(frame)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        aspect_ratio = frameWidth/frameHeight
        inWidth = 320
        inHeight = 210  #int(((inWidth/aspect_ratio)*10)//10)
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
     #   print(inHeight, inWidth)
     #   print(frameHeight, frameWidth)
        net.setInput(inpBlob)
        init_ = time.time()
        output = net.forward()
        end_ = time.time()
        #print(end_ - init_)
     
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

       
        result= fingers_analisys(points,frame)
        if(result is not  None):

            result = result.reshape((1,24))
            print(result.shape)

            if(np.isnan(result).sum() == 0):
                [y_] = model.predict(result)
                y_ = int(y_)
                print(y_)
                
                if(flag_command == False and command == 0 and y_ != 0 and y_ != 6):
                    command = y_
                    flag_command = True
                elif(flag_command ==True and command != 0 and y_ != 0 and command != y_):
                    command = y_
                    
           
            
            
                    
                    
                if(flag_command == True and y_ == 0 and count_desable_command < 2):
                    count_desable_command += 1
                

                elif(flag_command == True and y_ == 0 and count_desable_command > 1 and (command == 5  or command == 4)):
                    command = 0
                    flag_command = False
                    list_obj_temp = []
                    select_point = False
                    list_obj.append([points[4],img_draw, mask_img_draw])
                    
                elif(flag_command == True and y_ == 0 and count_desable_command > 1):
                    command = 0
                    flag_command = False
                    select_point = False

                    list_obj_temp = []

                    list_obj.append([points[8],img_draw, mask_img_draw])
                    
                    
     #           elif(flag_command == True and y_ == 0 and count_desable_command > 1 and command != 4):
     #               command = 0
     ##               flag_command = False
      #              list_obj_temp = []

#                    list_obj.append([points[8],img_draw, mask_img_draw])
                
      
      
                if(flag_command == True and command == 1):
                    print(points[8])
                    list_obj_temp = []
                    img_draw = img1
                    mask_img_draw = img1_mask
                    list_obj_temp.append([points[8],img_draw,mask_img_draw])
                   # frame = draw([points[8],img1],frame)
#                    print("shapeeeee", img1.shape)
 #                   list_obj.append([points[8],img1])
 
                elif(flag_command == True and command == 2):
                    print(points[8])
                    list_obj_temp = []
                    img_draw = img2
                    mask_img_draw = img2_mask
                    list_obj_temp.append([points[8],img_draw,mask_img_draw])
                   # frame = draw([points[8],img1],frame)
#                    print("shapeeeee", img1.shape)
 #                   list_obj.append([points[8],img1])
 
 
                elif(flag_command == True and command == 3):
                    print(points[8])
                    list_obj_temp = []
                    img_draw = img3
                    mask_img_draw = img3_mask
                    list_obj_temp.append([points[8],img_draw,mask_img_draw])
                   # frame = draw([points[8],img1],frame)
#                    print("shapeeeee", img1.shape)
 #                   list_obj.append([points[8],img1])
 
 
                elif(flag_command == True and command == 4):
                    print(points[8])
                    list_obj_temp = []
                    img_draw = img4
                    mask_img_draw = img4_mask
                    list_obj_temp.append([points[4],img_draw,mask_img_draw])
                   # frame = draw([points[8],img1],frame)
#                    print("shapeeeee", img1.shape)
 #                   list_obj.append([points[8],img1])
         #       if(flag_command == False and) 
                 
                elif(flag_command  == True and command == 5):
                    list_obj_temp = []

                    if(select_point == False):
                        
                        list_temp = select_element_in_list(list_obj, points[8])
                        print(len( list_temp ))
                        if(len(list_temp) > 0):
                      #  print(len(list_temp[0]))
                            img_draw = list_temp[0][1]
                            mask_img_draw = list_temp[0][2]
                        
                            list_obj_temp.append([points[4],img_draw,mask_img_draw])

                            select_point = True
                        
                        else:
                            flag_command = False
                            command = 0
                            
                    else:
                        list_obj_temp.append([points[4],img_draw,mask_img_draw])
                        
                elif(flag_command == True and command == 6):
                    list_obj_temp = []
                    flag_command = False
                    command = 0
                
                elif(flag_command == False and y_ == 6):
                    print("delete")
                    list_temp = select_element_in_list(list_obj, points[8])

                    
                print(select_point)
                        
                    #list_obj_temp.append([
                        
  
                    
                    
                
                #if(len(list_obj_temp) > 0 ):
                frame = draw(list_obj_temp,frame)


                    
                

                #list_points_.append(result)

        count = 0
        print("len",len(list_obj))
        frame = draw(list_obj,frame)
        out.write(frame)
        cv2.imshow('Output', frame)
        #cv2.imshow('Output-Skeleton', img_fi)


    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

captura.release()
out.release()

cv2.destroyAllWindows()

