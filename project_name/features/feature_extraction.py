from typing import Dict, Tuple

import cv2
import numpy as np
import scipy.stats as stat
import skimage.feature as feat

class FeatureExctration:
    """Class for the feature extraction of the baseline model"""

    def __init__(self) -> None:
        pass

    #Spectral features

    def split_channels(self, image : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Takes an image and splits it into its three component channels
        Args: 
            file_path(str): the path to the file_path to be processed
        Return: 
            tuple of vectors one for each channel
        """
        return cv2.split(image)

    def mean_channel_value(self, image : np.ndarray) -> float:
        """
        Returns the mean value of the pixel for a given channel
        Args:
            image (mv): vector of pixel values for a color channel
        Return:
            the mean value of the pixels in the input color channel
        """
        return float(np.mean(image))
        
    def channel_std_deviation(self, image : np.ndarray) -> float:
        """
        Returns the the standard deviation of the pixel valeus for a given channel
        Args:
            image (mv): vector of pixel values for a color channel
        Return:
            the standard deviation of the values in the input color channel 
        """
        return float(np.std(image))

    def channel_skew(self, image : np.ndarray) -> float:
        """
        Returns the skew of the pixel values for a given channel
        Args:
            image (mv): vector of pixel values for a color channel
        Return:
            the skew of the pixel values in the input color channel
        """
        return stat.skew(image)

    def channel_mean_difference(self, mean1 : float, mean2 : float) -> float:
        """
        Returns the difference between channel mean pixel values 
        Args:
            image (float): vector of pixel values for a color channel
        Return:
            the difference of the input values
        """
        return mean1 - mean2

    def run(self, file_path : str) -> Dict[str, float]:
        """ 
        Returns the extracted features for the input file in the form of a Dictionary
        Args: 
            file_path (str) : path the the file to be analyzed 
        Return: 
            feats (Dict[str, float]) : dict of the features extracted from the image
        """
        image = cv2.imread(file_path)
        b_channel, g_channel, r_channel = self.split_channels(image)

        #Textural features - only involve blue channel
        glcm = feat.graycomatrix(b_channel, distances = [1], angles = [0, np.pi/4, np.pi/2, 3*np.pi/4])
        contrast = np.mean(feat.graycoprops(glcm,'contrast'))
        entropy = np.mean(feat.graycoprops(glcm,'entropy'))
        energy = np.mean(feat.graycoprops(glcm,'energy'))
        homogeneity = np.mean(feat.graycoprops(glcm,'homogeneity'))        

        #Spectral features
        r_mean = self.mean_channel_value(r_channel)
        b_mean = self.mean_channel_value(b_channel)

        g_mean = self.mean_channel_value(g_channel) #### NOT A FEATURE >:o
        
        b_stdev = self.channel_std_deviation(b_channel)
        b_skewness = self.channel_skew(b_channel)

        rb_diff = self.channel_mean_difference(r_mean, b_mean)
        rg_diff = self.channel_mean_difference(r_mean, g_mean)
        gb_diff = self.channel_mean_difference(g_mean, b_mean)

        feats = {
            "B homogeneity": homogeneity,
            "B contrast" : contrast,
            "B entropy" : entropy,
            "B energy" : energy,
            "B standard deviation" : b_stdev,
            "B skewness": b_skewness,
            "B mean" : b_mean,
            "G - B mean diff" : gb_diff,
            "R - B mean diff" : rb_diff,
            "R - G mean diff" : rg_diff,
            "R mean" : r_mean            
        }
        
        return feats