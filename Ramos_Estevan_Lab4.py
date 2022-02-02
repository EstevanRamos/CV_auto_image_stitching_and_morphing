import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import transform as tf
import cv2
import os
from scipy import interpolate as spy_int
import matplotlib._color_data as mcd

def read_images(image_dir):
    image_files = os.listdir(image_dir)
    image_list = []
    for im in image_files:
        img = mpimg.imread(image_dir + im)
        image_list.append(img)
    return image_list

def select_matches_ransac(pts0, pts1):
    H, mask = cv2.findHomography(pts0.reshape(-1, 1, 2), pts1.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
    choice = np.where(mask.reshape(-1) == 1)[0]
    return pts0[choice], pts1[choice]

def show_points(ax,pts):
    colors = list(mcd.XKCD_COLORS.keys())
    for p in range(pts.shape[0]):
        ax.plot(pts[p,0],pts[p,1],marker='*',linestyle="None",color=colors[p])

def get_control_points_face(src_im):
    # Write special-purpose function, or borrow code from the function we used for stitching
    src_pts_arr = np.zeros((0,2))
    # Add points on boundary as control points
    corners = np.array([[0,0],[0,1],[1,1],[1,0],[0,0]])*(np.array(src_im.shape[:2])-1)
    src_pts_arr = np.vstack((src_pts_arr,interpolate_numpy(corners,25)[:-1]))
    return src_pts_arr

def merge_images(image_list):
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)        
    while len(image_list)>1:
        keys = np.empty(len(image_list), dtype=object)
        descriptors = np.empty(len(image_list), dtype=object)
        num_matches = np.array([[0] * len(image_list)] * len(image_list))
 
        for i in range(len(image_list)):
            keys[i], descriptors[i] = orb.detectAndCompute(image_list[i], mask=None)
        
        #Calculates which images are the closest match
        for i in range(len(image_list)):
            src_points = np.array([p.pt for p in keys[i]])
            for j in range(len(image_list)):
                if i == j:
                    continue
                #Create Matcher object
                matches = matcher.match(descriptors[i], descriptors[j])   
                # Extract data from orb objects and matcher
                dist = np.array([m.distance for m in matches])
                ind1 = np.array([m.queryIdx for m in matches])
                ind2 = np.array([m.trainIdx for m in matches])
                ds = np.argsort(dist)
                
                dest_points  = np.array([p.pt for p in keys[j]])
                good_matches, _ = select_matches_ransac(src_points[ind1[ds]], dest_points[ind2[ds]])
                num_matches[i,j] = good_matches.shape[0]
        #the images with the best mathces are found with the index of the max
        similar = num_matches.argmax()
        im1 = similar//num_matches.shape[0]
        im2 = similar%num_matches.shape[0]
        #get keypoints of the iamages
        keypoints1 = np.array([p.pt for p in keys[im1]])
        keypoints2 = np.array([p.pt for p in keys[im2]])
        #Create Matcher object
        matches = matcher.match(descriptors[im1], descriptors[im2])   
        # Extract data from orb objects and matcher
        dist = np.array([m.distance for m in matches])
        ind1 = np.array([m.queryIdx for m in matches])
        ind2 = np.array([m.trainIdx for m in matches])
        #sort by distance and use ransac
        ds = np.argsort(dist)
        keypoints1,keypoints2 = select_matches_ransac(keypoints1[ind1[ds]], keypoints2[ind2[ds]])
        
        #padding ammount for images
        pad = 100
        #get the two similar images
        img1 = image_list[im1]
        img2 = image_list[im2]
        
        #padd the images to makwe room for transformation
        img1 = np.pad(img1,((pad,pad), (pad,pad),(0, 0)),mode= 'constant')
        img2 = np.pad(img2 ,((pad,pad),(pad,pad),(0,0)),mode= 'constant')
        
        #if the images are different sizes we make them the same size and adjust the points
        if img1.shape[0] > img2.shape[0]:
            img2 = np.pad(img2 ,((pad,pad),(pad,pad),(0,0)),mode= 'constant')
            keypoints2 = keypoints2+pad
            
        if img1.shape[0] < img2.shape[0]:
            img1 = np.pad(img1 ,((pad,pad),(pad,pad),(0,0)),mode= 'constant')
            keypoints1 = keypoints1+pad
        
        #calculate the middle points using keypoints
        mid_x = np.array((keypoints1[:,0]+keypoints2[:,0])/2)
        mid_y = np.array((keypoints1[:,1]+keypoints2[:,1])/2)
        mid = np.column_stack((mid_x,mid_y))
        mid = mid + pad
        #build homographies
        H0 = tf.ProjectiveTransform()
        H0.estimate(keypoints1+pad,mid)
        
        H1 = tf.ProjectiveTransform()
        H1.estimate(keypoints2+pad,mid)
        #warp the images
        warped0 = tf.warp(img1, H0.inverse, output_shape=(img1.shape[0],img1.shape[1]))
        warped1 = tf.warp(img2, H1.inverse, output_shape=(img2.shape[0],img2.shape[1]))
        #combine them
        w = np.maximum(warped0,warped1)
        #show images
        fig, ax = plt.subplots(ncols=3,figsize = (12,4))
        ax[0].imshow(warped1)
        ax[1].imshow(warped0)
        ax[2].imshow(w)
        #turn image back into uint8
        w = w*255
        w = w.astype('uint8')
        #pop the images
        if im1>im2:
            image_list.pop(im1)
            image_list.pop(im2)
        else:
            image_list.pop(im2)
            image_list.pop(im1)
        #append image back to it
        image_list.append(w)
                

def morph(im0,im1,p0,p1,n):
    movie = []
    for i in range(n):
        a = i/(n-1)
        t = p0*(1-a) + (p1*(a))
        H0 = tf.ProjectiveTransform()
        H0.estimate(p0,t)
        w0 = tf.warp(im0,H0.inverse,output_shape=(im0.shape[0],im0.shape[1]))
        
        H1 = tf.ProjectiveTransform()
        H1.estimate(p1,t)
        w1 = tf.warp(im1,H1.inverse,output_shape=(im1.shape[0],im1.shape[1]))
        movie.append(w0*(1-a) + w1*a)
    return movie
                
            
            
            
        
if __name__ == "__main__":
    path = '.\\images\\'
    plt.close('all')
    image_list = []
    m1 = plt.imread(path + 'WF01.jpg')
    m2 = plt.imread(path + 'WF02.jpg')
    m3 = plt.imread(path + 'WF03.jpg')
    image_list.append(m1)
    image_list.append(m2)
    image_list.append(m3)
    merge_images(image_list)
    
#    morph1 = plt.imread( path + "lab4#1.jpg")
#    morph2 = plt.imread(  path + "lab4#2.jpg")
    #pt0 = get_control_points_face(morph1)
    #pt1 = get_control_points_face(morph2)
    
    
    save_video =  True
    use_old_points_0, use_old_points_1  = True, True  # Change this to refine control points or generate new ones
    morph_steps = 5  # Debug using a small number of frames. Once it works, increase to obtain a smoother and longer video
    image_0 = 'mona lisa.jpg'
    image_1 = 'gorilla.jpg'

    points_0 = path +'monalisa_pts_72.npy'
    points_1 = path + 'gorilla_pts_72.npy'

    img0 = plt.imread(path+image_0)
    img1 = plt.imread(path+image_1)
    img0 = img0/np.amax(img0)
    img1 = img1/np.amax(img1)

    if not use_old_points_0:
        pts0 = get_control_points_face(img0)
        np.save(points_0,pts0) # Rename file after running program to indicate source image
    else:
        pts0 = np.load(points_0)

    if not use_old_points_1:
        pts1 = get_control_points_face(img1)
        np.save(points_1,pts1) # Rename file after running program to indicate source image
    else:
        pts1 = np.load(points_1)
        
    movie = morph(img0,img1,pts0,pts1,5)
    fig, ax = plt.subplots(ncols = 5)
    ax[0].imshow(movie[0])
    ax[1].imshow(movie[1])
    ax[2].imshow(movie[2])
    ax[3].imshow(movie[3])
    ax[4].imshow(movie[4])
    
    width = 1280
    height = 720
    FPS = 30
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter('output.mp4', fourcc, float(FPS), (width, height))
    for i in range(len(movie)):
        frame = cv2.cvtColor(movie[i], cv2.COLOR_BGRA2BGR)
        video.write(frame)
    video.release()
    
    fig, ax = plt.subplots(ncols = 2)
    ax[0].imshow(img0)
    ax[1].imshow(img1)
    ax[0].axis('off')
    ax[1].axis('off')
    plt.tight_layout()
    show_points(ax[0],pts0)
    show_points(ax[1],pts1)

