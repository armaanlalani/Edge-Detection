import numpy as np
from PIL import Image

def CCL(im):
    queue = []
    height = im.shape[0]
    width = im.shape[1]
    label = 1 # current label
    labels = np.zeros((height, width)) # numpy array to store the label of each pixel
    labelled = set() # set used to store the pixels that have already been labelled
    for i in range(height):
        for j in range(width):
            if im[i,j] == 255 and (i,j) not in labelled: # if pixel is an edge and not labelled already
                labels[i, j] = label # add label to the pixel
                labelled.add((i,j)) # add pixel to the set of pixels already labelled
                queue.append((i,j)) # add pixel to the queue
                while len(queue) > 0: # loop until the queue is empty
                    p0, p1 = queue.pop(0) # pop element of the queue
                    for x in range(-1, 2):
                        for y in range(-1, 2):
                            ni, nj = p0 + x, p1 + y # get the indices of the 8 neighbouring pixels
                            if 0 <= ni < height and 0 <= nj < width and (ni,nj) not in labelled and im[ni, nj] == 255:
                                labels[ni, nj] = label # add the current label to the neighbour pixel if it has not been labelled and is also an edge
                                labelled.add((ni,nj)) # add to the labelled set
                                queue.append([ni, nj]) # add the pixel to the queue
                label = label + 1 # increment the current label
    return label

if __name__ == "__main__":
    image = Image.open('output6.png')
    image = image.convert(mode='L')

    data = np.asarray(image)

    print(CCL(data))