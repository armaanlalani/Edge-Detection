import numpy as np
from PIL import Image

def gaussian(x, stddev):
    return 1 / np.sqrt(2 * np.pi) * stddev * np.exp(-(x ** 2) / (2 * stddev ** 2)) # gaussian function

def compute_kernel(sigma):
    dim = 2 * int(2 * sigma + 0.5) + 1 # determines the size of the kernel using sigma --> ensures the size of the kernel is odd
    k_gaussian = np.linspace(-(dim//2), dim//2, dim) # creates the array that will be used to create the outer product
    for i in range(len(k_gaussian)):
        k_gaussian[i] = gaussian(k_gaussian[i], sigma) # 1D gaussian function
    k_gaussian = np.outer(k_gaussian.T, k_gaussian.T) # creates a 2D gaussian function by taking the outer product of the two 1D gaussian functions
    k_gaussian = k_gaussian / k_gaussian.max() # normalizes the kernel to ensure the maximum value is 1
    return k_gaussian

def convolve(im, kern):
    im_h = im.shape[0]
    im_w = im.shape[1]
    kern_h = kern.shape[0]
    kern_w = kern.shape[1]
    output = np.zeros((im_h, im_w))

    kern_size = kern_h * kern_w

    add = [int((kern_h-1)/2), int((kern_w-1)/2)] # size of the additional padded height and weight when filter is placed at edges of the image
    new_im = np.zeros((im_h + 2 * add[0], im_w + 2 * add[1])) # dimensions of the padded image
    new_im[add[0] : im_h + add[0], add[1] : im_w + add[1]] = im # sets the non-padded pixels of the padded image to the pixels of the image being convolved

    for i in range(im_h):
        for j in range(im_w):
            result = kern * new_im[i : i + kern_h, j : j + kern_w] # elementwise multiplication of kernel and appropriate pixels
            output[i, j] = np.sum(result) # adds the elements of the elementwise multiplication
    output = output / kern_size # reduction of pixel values based on kernel size

    return output

def gradient(gx, gy):
    return np.sqrt(gx**2 + gy**2) # gradient magnitude of image

def threshold(im):
    im_h = im.shape[0]
    im_w = im.shape[1]
    th_old = np.sum(im) / (im_h * im_w) # initial threshold value
    while True:
        low = [] # array to hold the pixels below the threshold
        upper = [] # array to hold the pixels above the threshold
        for i in range(im_h):
            for j in range(im_w):
                if im[i, j] < th_old:
                    low.append(im[i,j]) # add respective pixel to low array
                else:
                    upper.append(im[i,j]) # add respective pixel to upper array
        low_avg = sum(low)/len(low) # compute the low average
        upper_avg = sum(upper)/len(upper) # compute the upper average
        th_new = (low_avg + upper_avg) / 2  # compute the new threshold value
        if abs(th_new-th_old) < 0.0001: # check to see if the absolute difference is below epsilon
            break
        th_old = th_new # set the old threshold to the newly computed value
        for i in range(im_h):
            for j in range(im_w):
                if im[i,j] > th_old:
                    im[i,j] = 255 # changes pixel values based on threshold
                else:
                    im[i,j] = 0 # changes pixel values based on threshold
    return im


if __name__ == "__main__":
    image = Image.open('Q4_image_1.jpg') # loads the image
    image = image.convert(mode='L') # converts image to grayscale

    data = np.asarray(image) # converts image to a numpy array
    
    sigma = 1
    gaussian = compute_kernel(sigma) # computes the gaussian kernel based on the inputted sigma
    image_g = convolve(data, gaussian) # convolves the gaussian filter with the image

    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # sobel x
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) # sobel y

    grad_x = convolve(image_g, gx) # computes the gradient in the x direction
    grad_y = convolve(image_g, gy) # computes the gradient in the y direction

    image_grad = gradient(grad_x, grad_y) # determines the madnitude of the gradient based on both directions

    image_th = threshold(image_grad) # determines the threshold of the image

    image2 = Image.fromarray(image_th)
    image2.show()
    # image2.convert('L').save('pic1.jpg')