
from fcmeans import FCM
from matplotlib import pyplot as plt
import cv2
from time import time

# Paths of the images
imgpath1 = plt.imread("img1.jpg")/255 
imgpath2 = plt.imread("img2.jpg")/255 
imgpath3 = plt.imread("img3.jpg")/255 
imgpath4 = plt.imread("img4.jpg")/255 

    
    
# Showing all of the images using matplotlib
f,axarr=plt.subplots(2,2,figsize=(50,50))

axarr[0][0].imshow(imgpath1)
axarr[0][0].set_title("img1",size=50)
axarr[0][0].axis('off')

axarr[0][1].imshow(imgpath2)
axarr[0][1].set_title("img2" , size=50)
axarr[0][1].axis('off')

axarr[1][0].imshow(imgpath3)
axarr[1][0].set_title("img3" , size=50)
axarr[1][0].axis('off')

axarr[1][1].imshow(imgpath4)
axarr[1][1].set_title("img4" , size=50)
axarr[1][1].axis('off')


      

      
# Making img1 the default of the images
img=imgpath1
choosenimg=input("Enter the number of the image you want from 1 to 4: ")


if choosenimg=="1":
        img = imgpath1
    
elif choosenimg=="2":
        img = imgpath2
        
elif choosenimg=="3":
        img = imgpath3
    
elif choosenimg=="4":
        img = imgpath4


else:
        print("You have chosen a number outside the interval [1,4] so the first image is chosen automatically")
        
        

# Taking an object from the method readimage()
pic = img

# To show the image shape
print("Original image shape: ",pic.shape)

# Taking object from time class
new_time = time()

# Num of clusters 
i = input("Enter the number of clusters you want: ")

# Making the num of clusters be the number that the user has entered
fcm = FCM(n_clusters=int(i))

# Resizing the pic
pic_re = pic.reshape(pic.shape[0]*pic.shape[1],pic.shape[2])
print("Resized image ",pic_re.shape)

# Fitting the pic and appling number of clustring on it
fcm.fit(pic_re)

# Finding random centers
fcm_centers = fcm.centers

# Predicting the labels according to the number of clusters
fcm_labels = fcm.predict(pic_re)

# appling centers on labels
pic2show = fcm_centers[fcm_labels]

# Reshaping again
cluster_pic = pic2show.reshape(pic.shape[0],pic.shape[1],pic.shape[2])
print("Cluster image shape: ",cluster_pic.shape)


# Visualization by matplotlib

f,axarr=plt.subplots(1,2,figsize=(50,50))
axarr[0].imshow(pic)
axarr[0].set_title("original image",size=50)
axarr[0].axis('off')
axarr[1].imshow(cluster_pic)
axarr[1].set_title(str(i)+" Clusters" , size=50)
axarr[1].axis('off')

#time taken to make clustering process
print("fuzzy time for clustring is ", time() - new_time, 'seconds')
