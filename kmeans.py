# Resources: 
# https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
# Added video as image input and updating histogram, along with central bounding box
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cap = cv2.VideoCapture(0)


while(1):
    _, img = cap.read()
    img2 = img
    w,h = 640,480

    cv2.rectangle(img, (int(w*0.24), int(h*0.24)), (int(w*0.76), int(h*0.76)), (255,0,0),2)

    cv2.imshow('frame', img)

    img = img2[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)
    plt.axis("off")
    plt.imshow(bar)

    fig.canvas.draw()
    fig.canvas.flush_events()
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()