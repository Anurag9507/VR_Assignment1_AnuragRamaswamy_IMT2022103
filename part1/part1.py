#Import required libraries
import cv2
import numpy as np
import os

# Function which preprocesses the image
def preprocess(image_path):
    image=cv2.imread(image_path)
    grayscale_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#Convert to grayscale
    #Resizing
    scale=700 / max(image.shape[:2])
    resized_image=cv2.resize(image,(0,0),fx=scale,fy=scale)
    resized_grayscale_image=cv2.resize(grayscale_image,(0,0),fx=scale,fy=scale)
    blurred_image=cv2.GaussianBlur(resized_grayscale_image,(5,5),0)#Gaussian Blur
    binary_image=cv2.adaptiveThreshold(blurred_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)#Binary thresholding
    return resized_image,binary_image,scale

# Function which detects circular edges(coins) in the image
def detect_coin_like_shapes(binary_image,scale):
    contours,_=cv2.findContours(binary_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    coin_like_shapes=[]
    min_coin_size=500*(scale*scale)
    for contour in contours:
        perimeter=cv2.arcLength(contour,True)
        area=cv2.contourArea(contour)
        if(perimeter>0):
            circularity=4*np.pi*(area/(perimeter*perimeter))#to ensure only circular coins detected
            if 0.7<circularity<1.2 and area>min_coin_size:#coin size conditions
                coin_like_shapes.append(contour)
    return coin_like_shapes

# Function which segments out only the coins from the image and turns the background black
def segment_coins_from_background(image,binary_image,coin_like_shapes):
    mask=np.zeros_like(binary_image)#empty mask
    cv2.drawContours(mask,coin_like_shapes,-1,255,thickness=cv2.FILLED)
    segmented_coins= cv2.bitwise_and(image,image,mask=mask)
    canvas=np.zeros_like(image)#blank black canvas 
    canvas[mask == 255]=segmented_coins[mask == 255]
    return canvas

# Function to process all images, apply segmentation and draw contours
def process_coins(input_folder,output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for image_file in sorted(os.listdir(input_folder)):
        input_file=os.path.join(input_folder,image_file)
        output_file=os.path.join(output_folder,f"{os.path.splitext(image_file)[0]}_segmented.jpg")
        resized_image,binary_image,scale=preprocess(input_file)
        coin_like_shapes=detect_coin_like_shapes(binary_image,scale)
        segmented_image=segment_coins_from_background(resized_image,binary_image,coin_like_shapes)
        cv2.drawContours(segmented_image,coin_like_shapes,-1,(0,0,255),2)
        cv2.imwrite(output_file,segmented_image)
        print(f"Number of coins detected in {image_file} = {len(coin_like_shapes)}")

# Process all coins in input
process_coins("input","output")