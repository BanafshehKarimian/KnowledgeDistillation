import os
import cv2
import glob
import gdcm
import torch
import random
import pydicom
import torchvision
import numpy as np
import polars as pl
import torch.nn as nn
import statistics as stats
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from lets_plot import *
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet50
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from lets_plot.mapping import as_discrete
from torch.utils.data import Dataset, DataLoader

def _to_gray(image, method_to_gray: str = "default") -> np.array:
        """
        Convert mammography sceening to gray

        image: The image to convert
        method_to_gray: Which method to use to convert to gray (None correponds to the open cv method
        TODO explore other ways to make it gray, e.g PCA cf. Impact of Image Enhancement Module for Analysis
        of Mammogram Images for Diagnostics of Breast Cancer)

        returns: The image in grayscale
        """
        if method_to_gray == "default":
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

def get_contours(image, thresh_low: int = 5, thresh_high: int = 255):
        """
        Get the list of contours of an image with opencv

        image: The screening mammography
        thresh_low: Lower bound for the threshold of the image
        thresh_high : Upper bound for the threshold of the image

        returns: The list of contours of the image
        """
        # Perform thresholding to create a binary image
        _, binary = cv2.threshold(image, thresh_low, thresh_high, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

def draw_contours(contours, image, biggest = True) -> tuple:
        """
        Draw contour on images

        contours: The list of the contour of the image
        image: The screening mammography
        biggest: If True, draws the biggest contour, else draw other contours

        returns: image with associate contours, binary mask of the contour
        """

        biggest_contour = max(contours, key=cv2.contourArea)
        # Create a mask image with the same size as the original image
        mask = np.zeros_like(image)
        # Convert the mask image to grayscale
        if len(image.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask

        if biggest:
            # Draw the biggest contour on the mask image
            # cv2.drawContours(mask_gray, [biggest_contour], -1, (255, 255, 255), -1)
            cv2.drawContours(mask_gray, [biggest_contour], -1, 255, -1)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(image, [biggest_contour], -1, (255, 0, 0), 3)

            # Set the pixels inside the removed contours to red
        # image[mask_gray == 255] = 255
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            for contour in contours:
                if cv2.contourArea(contour) != cv2.contourArea(biggest_contour):
                    cv2.drawContours(mask_gray, [contour], -1, (255, 255, 255), -1)
                    cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)
        return image, mask_gray
def gamma_correct(image: np.array, gamma: float = 2.2) -> np.array:
        """
        Gamma correct image

        image: The screening mammography
        gamma: The factor to use to gamma correct image

        returns: Gamma corrected image
        """
        lut = np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        return cv2.LUT(image, lut)

def get_edges(image: np.array, method: str = "prewitt") -> np.array:
        """
        Detect edges from an image given a method

        image: Screening mammography
        method: The method for edge detection (here prewitt method is used)

        returns: Edges of the image
        """
        edges = []
        image = image / 255
        if method == "prewitt":
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

            img_prewittx = cv2.filter2D(image, -1, kernelx)
            img_prewitty = cv2.filter2D(image, -1, kernely)

            # Calculate the gradient magnitude
            edges = np.sqrt(np.square(img_prewittx) + np.square(img_prewitty))

            # Normalize the gradient magnitude image
            edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        return edges

def remove_useless_edges(
        edges: np.array,
        kernel_size_closing: tuple = (5, 5),
        thresh_mask_edges: float = 0.95,
        kernel_erosion_shape: tuple = (1, 2),
    ) -> np.array:
        """
        Remove useless edges by adaptive thresholding and small erosion

        edges: Detected edges from image
        kernel_size_closing: Kernel used for closing method (cf. opencv)
        thresh_mask_edges: Adaptative relative thresholding for mask
        kernel_erosion_shape: Kernel of the erosion

        returns: Filtered edges
        """
        # Define the kernel for the closing operation
        kernel = np.ones(kernel_size_closing, np.uint8)

        # Apply the closing operation to the image
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        intensities = closed.flatten()

        # Create a new array of non-zero intensities
        intensities = intensities[intensities > 2]

        # Sort the array of pixel intensities
        intensities.sort()

        # Find the index of the thresh_mask_edges quantile
        index = int(len(intensities) * thresh_mask_edges)

        if index == 0:
            return None
        else:
            # Retrieve the 50th quantile value from the sorted array
            quantile = intensities[index]

        _, edges_thresh = cv2.threshold(closed, quantile, 255, cv2.THRESH_BINARY)

        # Define the kernel for the erosion operation
        kernel = np.ones(kernel_erosion_shape, np.uint8)

        # Apply the erosion operation to the image
        edges_thresh = cv2.erode(edges_thresh, kernel, iterations=1)
        return edges_thresh

def get_convex_hull(edges: np.array) -> np.array:
        """
        Get convex hull from edges of image

        edges: Detected edges from image

        returns: Minimum convex hull (cf. opencv)
        """
        # If the image is fully black
        if edges is None:
            return None
        # Find the non-zero pixels in the image
        points = np.argwhere(edges > 0)

        points = np.array([[elem[1], elem[0]] for elem in points])

        # Calculate the convex hull of the points
        hull = cv2.convexHull(points)
        return hull

def _remove_pectoral_muscle(
        image: np.array,
        method: str = "prewitt",
        kernel_size_closing: tuple = (5, 5),
        thresh_mask_edges: float = 0.95,
        kernel_erosion_shape: tuple = (1, 2),
    ) -> np.array:
        """
        Method to remove pectoral muscle (implementation adapted from
        Removal of pectoral muscle based on topographic map and shape-shifting silhouette)

        image: Screening mammography
        method: The method for edge detection (here prewitt method is used)
        kernel_size_closing: Kernel used for closing method (cf. opencv)
        thresh_mask_edges: Adaptative relative thresholding for mask
        kernel_erosion_shape: Kernel of the erosion

        returns: Image without pectoral muscle
        """
        # Get edges
        edges = get_edges(image=image, method=method)
        edges = remove_useless_edges(
            edges=edges,
            kernel_size_closing=kernel_size_closing,
            thresh_mask_edges=thresh_mask_edges,
            kernel_erosion_shape=kernel_erosion_shape,
        )
        hull = get_convex_hull(edges=edges)
        if hull is None:
            return image
        mask = np.zeros_like(image)

        # Fill the convex hull with 1's in the mask
        cv2.fillConvexPoly(mask, hull, 1)

        # Apply the binary mask to the image
        image = cv2.bitwise_and(image, image, mask=mask)

        return image

def tranform_function(image):
    image = _to_gray(image)
    shape = image.shape
    contours = get_contours(image)
    biggest_contour = max(contours, key=cv2.contourArea)
    mask = draw_contours(contours, image, biggest=True)[1]
    #image  = cv2.bitwise_and(image, mask)#?
    _, y, _, h = cv2.boundingRect(biggest_contour)
    image = image[y + 2 : y + h, :]
    # Resize image to correct shape
    image = cv2.resize(image, shape)
    method = "global"
    if method == "global":
        image = cv2.equalizeHist(image)
    elif method == "clahe":
        clahe = cv2.createCLAHE(
            clipLimit=2, tileGridSize= (8, 8)
        )
        image = clahe.apply(image)
    denoising_method = "NlMD"
    denoise_h = 3
    denoise_block_size = 7
    denoise_search_window = 21
    image = cv2.fastNlMeansDenoising(image, None, denoise_h, denoise_block_size, denoise_search_window)
    if False:
         image = gamma_correct(image, 2.2)
    #sharpening?
    #image_reverted?
    '''image = _remove_pectoral_muscle(
                    image,
                    method= "prewitt",
                    kernel_size_closing= (5, 5),
                    thresh_mask_edges= 0.95,
                    kernel_erosion_shape= (1, 2),
                )'''
    return image

class MammographyDataset(Dataset):
    def __init__(self, meta_df, img_dir, transform=None):
        
        self.df = meta_df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        # Get label from meta data
        label = self.df.get_column('cancer').to_pandas()
        label = label[idx]
        
        # Get image file paths from metadata
        img_fname = self.df.get_column('fname').to_pandas()
        img_fname = img_fname[idx]
        
        # Get image, transform
        img_path = f'{self.img_dir}/{img_fname}.png'
        img = Image.open(img_path)
        
        if self.transform:
            #img = tranform_function(np.array(img))
            img = self.transform(img)
            
        # Get metadata features
        feature_names = [
            'age', 
            'laterality_L', 'laterality_R', 
            'view_AT', 'view_CC', 'view_MLO',
            'implant_0', 'implant_1'
                        ]
        
        meta_features = self.df.select(feature_names).to_pandas()
        meta_features = meta_features.iloc[idx].to_numpy()
        
        return img, meta_features, label