import cv2 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image




def gammaCorrection(img_slice, gamma):
    inv_gamma = 1 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8") 
    img_gamma = cv2.LUT(img_slice, table)
    return img_gamma

def getMinMaxV(img_slice):
    V = cv2.cvtColor(gammaCorrection(img_slice, 2), cv2.COLOR_BGR2HSV)[:, :, 2]
    # _, K = cv2.threshold(V, 120, 255, cv2.THRESH_BINARY)
    # plt.matshow(K, cmap=plt.cm.Blues)
    # plt.show()
    removeMaskV = V[V!=0]
    return np.min(removeMaskV), np.max(removeMaskV)

def removePoint(img_camMask, img_slice, contourArea, originalx, originaly):
    boundaryThresh = 10
    flagArea = True
    flagBoundary = True
    flagCarMask = True
    flagHeight = True
    img_camMask = cv2.bitwise_not(img_camMask)
    # Remove small area contour
    img_sliceArea = img_slice.shape[0] * img_slice.shape[1]
    contourRatio = contourArea/img_sliceArea
    if(contourRatio < 0.01):
        flagArea = False
 
    # Revome point close to Boundary of original image
    if((originalx < boundaryThresh) or ((img_camMask.shape[1]-originalx) < boundaryThresh) or (originaly < boundaryThresh) or ((img_camMask.shape[0]-originaly) < boundaryThresh)):
        flagBoundary = False
    
    # Revome point close to car mask
    if(flagBoundary):
        if(np.count_nonzero(img_camMask[originaly-2 : originaly+3 , originalx-2 : originalx+3]) != 25 ):
            flagCarMask = False
            
    # if(originaly < 500):
    #     flagHeight = False
    
    return flagArea & flagBoundary & flagCarMask & flagHeight

def exec_threshold(file_path):
    # file_path = "ITRI_dataset/seq1/dataset/1681710720_993855093"
    # test2
    # /home/tsaiiast/cv/final_project/ITRI_dataset/seq1/dataset/1681710720_960889218
    # /home/tsaiiast/cv/final_project/ITRI_dataset/seq1/dataset/1681710720_966775996
    # /home/tsaiiast/cv/final_project/ITRI_dataset/seq1/dataset/1681710720_970174987
    # /home/tsaiiast/cv/final_project/ITRI_dataset/seq1/dataset/1681710720_993855093
    # test1
    # /home/tsaiiast/cv/final_project/ITRI_dataset/seq1/dataset/1681710718_75943775
    # /home/tsaiiast/cv/final_project/ITRI_dataset/seq1/dataset/1681710718_81997343
    # /home/tsaiiast/cv/final_project/ITRI_dataset/seq1/dataset/1681710718_73162313
    # /home/tsaiiast/cv/final_project/ITRI_dataset/seq1/dataset/1681710718_85277866
    prob_threshold = 0.3
    shadowGamma = 0.8
    brightGamma = 2.0

    # Read raw image
    img_path = file_path + "/raw_image.jpg"
    img_raw = cv2.imread(img_path)
    # print(img_raw.shape)

    #Read self-camera mask
    cam_path = pd.read_csv(file_path + "/camera.csv" , header=None).to_numpy()    #cam_path shape : (1,1)
    img_camMask = cv2.imread("ITRI_dataset/camera_info" + str(cam_path[0][0]) + "_mask.png")
    img_camMask = cv2.cvtColor(img_camMask, cv2.COLOR_BGR2GRAY)
    img_masked  = cv2.bitwise_and(img_raw, img_raw, mask = cv2.bitwise_not(img_camMask))


    # get marker detect by probability and clip the box x,y value 
    markerDetect = pd.read_csv(file_path + "/detect_road_marker.csv" , header=None).to_numpy()
    prob_mask = (markerDetect[:,5] > prob_threshold)
    markerDetect = markerDetect[prob_mask, :]
    rows = markerDetect.shape[0]
    markerDetect[:, 0:4:2] = np.clip(markerDetect[:, 0:4:2], 0, img_raw.shape[1])
    markerDetect[:, 1:4:2] = np.clip(markerDetect[:, 1:4:2], 0, img_raw.shape[0])
    markerBox = markerDetect[:, 0:4].astype(np.uint16)
    # print(f"rowNum:   {rows}")
    corners = []


    for row in range(rows):
        # get Otsu's binary threshold with mask
        img_slice = img_masked[markerBox[row][1]:markerBox[row][3] , markerBox[row][0]:markerBox[row][2],:]
        minV, maxV = getMinMaxV(img_slice)

        # print(f"row{row} minV:   {minV}")
        # print(f"row{row} maxV:   {maxV}")
        
        if (minV < 70) and (maxV > 200):
            gamma = shadowGamma
        else:
            gamma = brightGamma
            
        img_slice = gammaCorrection(img_slice, gamma)
        img_graySlice = cv2.cvtColor(img_slice, cv2.COLOR_BGR2GRAY)

        # img_graySlice = cv2.dilate(img_graySlice, np.ones((9,9), np.uint8))
        # img_graySlice = cv2.medianBlur(img_graySlice, 21)

        img_blurred = cv2.GaussianBlur(img_graySlice, (5, 5), 0)
        img_getThresh = img_blurred[img_blurred != 0]
        otsuThresh, _ = cv2.threshold(img_getThresh, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        # print(f"row{row} threshold:   {otsuThresh} \n")
        _, img_thresh = cv2.threshold(img_blurred, otsuThresh, 255, cv2.THRESH_BINARY)
        
        
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
        cv2.drawContours(img_graySlice, contours, -1, (0,255,0), thickness=1)
        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, False)
            # drawing points
            for point in approx:
                x, y = point[0]
                flag = removePoint(img_camMask, img_slice, cv2.contourArea(contour), markerBox[row][0] + x, markerBox[row][1] + y)
                if(flag):
                    cv2.circle(img_raw, (markerBox[row][0] + x, markerBox[row][1] + y), 3, (0, 255, 0), -1)
                    corners.append([markerBox[row][0] + x, markerBox[row][1] + y])
                # cv2.circle(img_slice, (x, y), 3, (0, 255, 0), -1)

    ######################################
    #corners :     list of point cloud
    #######################################
    corners = np.array(corners)
    np.save(file_path+'/corners.npy', corners)
    # plt.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
    # plt.savefig(file_path+'/corners.jpg')
    # plt.show()