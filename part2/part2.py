#Import required libraries
import numpy as np
import cv2
import os

# Function which detects keypoints and calculates descriptors using SIFT(Scale-Invariant Feature Transform)
def extract_features(image):
    sift=cv2.SIFT_create()
    keypoints,descriptors=sift.detectAndCompute(image,None)
    keypoints=np.float32([keypoint.pt for keypoint in keypoints])#convert keypoints to numpy float32 array
    return keypoints,descriptors

#Function which matches keypoints between two images using BFMatcher and Lowe's ratio test
def find_matches(ptsA,ptsB,descA,descB,ratio_thresh,reproj_thresh):
    matcher=cv2.BFMatcher()
    potential_matches=matcher.knnMatch(descA,descB,2)
    valid_matches=[]
    for match in potential_matches:
        if(len(match)==2)and(match[0].distance<(ratio_thresh*match[1].distance)):
            valid_matches.append((match[0].trainIdx,match[0].queryIdx))        
    if(len(valid_matches)>4):#There should be atleast 5 matches to compute homography matrix
        valid_ptsA=np.float32([ptsA[i]for(_,i)in valid_matches])
        valid_ptsB=np.float32([ptsB[i]for(i,_)in valid_matches])
        H,status=cv2.findHomography(valid_ptsA,valid_ptsB,cv2.RANSAC,reproj_thresh)
        return valid_matches,H,status
    return None #If not enough valid matches found

# Function which visualizes the matches between two images
def visualize_matches(image1,image2,keypoints1,keypoints2,matches,match_status):
    h1,w1=image1.shape[:2]
    h2,w2=image2.shape[:2]
    output_img=np.zeros((max(h1,h2),w1+w2,3),dtype=np.uint8)#Create an empty canvas for the image
    output_img[0:h1,0:w1]=image1
    output_img[0:h2,w1:]=image2
    lines=0
    for((train_idx,query_idx),status)in zip(matches,match_status):
        if status == 1:#Draw a line if the match is good
            pt1=(int(keypoints1[query_idx][0]),int(keypoints1[query_idx][1]))
            pt2=(int(keypoints2[train_idx][0])+w1,int(keypoints2[train_idx][1]))
            if lines <= 250: # Limiting lines to reduce visual clutter
                cv2.line(output_img,pt1,pt2,(0,0,255),1)
                lines += 1
    return output_img

# Function to resize an image by adjusting height according to target_width while having the same aspect ratio
def resize_image(image,target_width):
    height,width=image.shape[:2]
    aspect_ratio=target_width/float(width)
    target_height=(int)(height*aspect_ratio)
    return cv2.resize(image,(target_width,target_height),interpolation=cv2.INTER_AREA)

# Function which crops image to the ROI(Region Of Interest) after stitching
def crop_image(img):
    grayscale_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresholded_image=cv2.threshold(grayscale_image,0,255,cv2.THRESH_BINARY)
    contours,_=cv2.findContours(thresholded_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x,y,w,h=cv2.boundingRect(contours[0])
        return img[y:y+h-1,x:x+w-1]
    return img

# Function which merges/stitches two images
def stitch_images(images,ratio=0.75,reproj_thresh=4.0,display_overlay=False):
    image2,image1=images
    keypoints1,descriptors1=extract_features(image1)
    keypoints2,descriptors2=extract_features(image2)
    x=find_matches(keypoints1,keypoints2,descriptors1,descriptors2,ratio,reproj_thresh)
    if x is None:
        print("Not enough matches found.")
        return None
    matches,homography,status=x
    panorama_image=cv2.warpPerspective(image1,homography,(image1.shape[1] + image2.shape[1],image1.shape[0]))
    panorama_image[0:image2.shape[0],0:image2.shape[1]]=image2
    panorama_image=crop_image(panorama_image)
    if display_overlay:
        visualization=visualize_matches(image1,image2,keypoints1,keypoints2,matches,status)
        return (panorama_image,visualization)
    return panorama_image

# Function which does all steps required for creating a panorama image from its parts
def process_images(input_folder,output_folder):
    image_files=sorted([os.path.join(input_folder,file)for file in os.listdir(input_folder)],key=lambda f:int(os.path.splitext(os.path.basename(f))[0]))
    base_img=cv2.imread(image_files[0])
    base_img=resize_image(base_img,600)
    for idx in range(1,len(image_files)):
        overlay_img=cv2.imread(image_files[idx])
        overlay_img=resize_image(overlay_img,600)
        stitched_result=stitch_images([base_img,overlay_img],display_overlay=True)
        if stitched_result is None:
            continue
        base_img,match_visualization=stitched_result
        folder_name = os.path.basename(input_folder)
        cv2.imwrite(os.path.join(output_folder, f"{folder_name}_matches_{idx}.jpg"), match_visualization)
    cv2.imwrite(os.path.join(output_folder, f"{folder_name}_panorama.jpg"), base_img)
    print(f"Keypoint matchings and panorama image for images in {input_folder} created succesfully and saved in {output_folder}.")

if not os.path.exists("output"):# Create output directory if it doesn't exist
      os.makedirs("output")
# Process all subdirectories in input
for subdir in sorted(os.listdir("input")):
    input_path=os.path.join("input",subdir)
    if os.path.isdir(input_path):
        output_dir=os.path.join("output",subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    process_images(input_path,output_dir)

