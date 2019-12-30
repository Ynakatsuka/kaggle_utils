from collections import defaultdict
from functools import partial
import multiprocessing
import operator
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from skimage import feature
from sklearn.metrics.pairwise import euclidean_distances

from .base import BaseFeatureTransformer


class Saliency():
    """Generate saliency map from RGB images with the spectral residual method
        This class implements an algorithm that is based on the spectral
        residual approach (Hou & Zhang, 2007).
    """
    def __init__(self, img, use_numpy_fft=True, gauss_kernel=(5, 5)):
        """Constructor
            This method initializes the saliency algorithm.
            :param img: an RGB input image
            :param use_numpy_fft: flag whether to use NumPy's FFT (True) or
                                  OpenCV's FFT (False)
            :param gauss_kernel: Kernel size for Gaussian blur
        """
        self.use_numpy_fft = use_numpy_fft
        self.gauss_kernel = gauss_kernel
        self.frame_orig = img

        # downsample image for processing
        self.small_shape = (64, 64)
        self.frame_small = cv2.resize(img, self.small_shape[1::-1])

        # whether we need to do the math (True) or it has already
        # been done (False)
        self.need_saliency_map = True

    def get_saliency_map(self):
        """Returns a saliency map
            This method generates a saliency map for the image that was
            passed to the class constructor.
            :returns: grayscale saliency map
        """
        if self.need_saliency_map:
            # haven't calculated saliency map for this image yet
            num_channels = 1
            if len(self.frame_orig.shape) == 2:
                # single channel
                sal = self._get_channel_sal_magn(self.frame_small)
            else:
                # multiple channels: consider each channel independently
                sal = np.zeros_like(self.frame_small).astype(np.float32)
                for c in range(self.frame_small.shape[2]):
                    small = self.frame_small[:, :, c]
                    sal[:, :, c] = self._get_channel_sal_magn(small)

                # overall saliency: channel mean
                sal = np.mean(sal, 2)

            # postprocess: blur, square, and normalize
            if self.gauss_kernel is not None:
                sal = cv2.GaussianBlur(sal, self.gauss_kernel, sigmaX=8,
                                       sigmaY=0)
            sal = sal**2
            sal = np.float32(sal)/np.max(sal)

            # scale up
            sal = cv2.resize(sal, self.frame_orig.shape[1::-1])

            # store a copy so we do the work only once per frame
            self.saliencyMap = sal
            self.need_saliency_map = False

        return self.saliencyMap

    def _get_channel_sal_magn(self, channel):
        """Returns the log-magnitude of the Fourier spectrum
            This method calculates the log-magnitude of the Fourier spectrum
            of a single-channel image. This image could be a regular grayscale
            image, or a single color channel of an RGB image.
            :param channel: single-channel input image
            :returns: log-magnitude of Fourier spectrum
        """
        # do FFT and get log-spectrum
        if self.use_numpy_fft:
            img_dft = np.fft.fft2(channel)
            magnitude, angle = cv2.cartToPolar(np.real(img_dft),
                                               np.imag(img_dft))
        else:
            img_dft = cv2.dft(np.float32(channel),
                              flags=cv2.DFT_COMPLEX_OUTPUT)
            magnitude, angle = cv2.cartToPolar(img_dft[:, :, 0],
                                               img_dft[:, :, 1])

        # get log amplitude
        log_ampl = np.log10(magnitude.clip(min=1e-9))

        # blur log amplitude with avg filter
        log_ampl_blur = cv2.blur(log_ampl, (3, 3))

        # residual
        residual = np.exp(log_ampl - log_ampl_blur)

        # back to cartesian frequency domain
        if self.use_numpy_fft:
            real_part, imag_part = cv2.polarToCart(residual, angle)
            img_combined = np.fft.ifft2(real_part + 1j*imag_part)
            magnitude, _ = cv2.cartToPolar(np.real(img_combined),
                                           np.imag(img_combined))
        else:
            img_dft[:, :, 0], img_dft[:, :, 1] = cv2.polarToCart(residual,
                                                                 angle)
            img_combined = cv2.idft(img_dft)
            magnitude, _ = cv2.cartToPolar(img_combined[:, :, 0],
                                           img_combined[:, :, 1])

        return magnitude

    def calc_magnitude_spectrum(self):
        """Plots the magnitude spectrum
            This method calculates the magnitude spectrum of the image passed
            to the class constructor.
            :returns: magnitude spectrum
        """
        # convert the frame to grayscale if necessary
        if len(self.frame_orig.shape) > 2:
            frame = cv2.cvtColor(self.frame_orig, cv2.COLOR_BGR2GRAY)
        else:
            frame = self.frame_orig

        # expand the image to an optimal size for FFT
        rows, cols = self.frame_orig.shape[:2]
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        frame = cv2.copyMakeBorder(frame, 0, ncols-cols, 0, nrows-rows,
                                   cv2.BORDER_CONSTANT, value=0)

        # do FFT and get log-spectrum
        img_dft = np.fft.fft2(frame)
        spectrum = np.log10(np.abs(np.fft.fftshift(img_dft)))

        # return for plotting
        return 255*spectrum/np.max(spectrum)

    def plot_power_spectrum(self):
        """Plots the power spectrum
            This method plots the power spectrum of the image passed to
            the class constructor.
            :returns: power spectrum
        """
        # convert the frame to grayscale if necessary
        if len(self.frame_orig.shape) > 2:
            frame = cv2.cvtColor(self.frame_orig, cv2.COLOR_BGR2GRAY)
        else:
            frame = self.frame_orig

        # expand the image to an optimal size for FFT
        rows, cols = self.frame_orig.shape[:2]
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        frame = cv2.copyMakeBorder(frame, 0, ncols - cols, 0, nrows - rows,
                                   cv2.BORDER_CONSTANT, value=0)

        # do FFT and get log-spectrum
        if self.use_numpy_fft:
            img_dft = np.fft.fft2(frame)
            spectrum = np.log10(np.real(np.abs(img_dft))**2)
        else:
            img_dft = cv2.dft(np.float32(frame), flags=cv2.DFT_COMPLEX_OUTPUT)
            spectrum = np.log10(img_dft[:, :, 0]**2+img_dft[:, :, 1]**2)

        # radial average
        L = max(frame.shape)
        freqs = np.fft.fftfreq(L)[:L/2]
        dists = np.sqrt(np.fft.fftfreq(frame.shape[0])[:, np.newaxis]**2 +
                        np.fft.fftfreq(frame.shape[1])**2)
        dcount = np.histogram(dists.ravel(), bins=freqs)[0]
        histo, bins = np.histogram(dists.ravel(), bins=freqs,
                                   weights=spectrum.ravel())

        centers = (bins[:-1] + bins[1:]) / 2
        plt.plot(centers, histo/dcount)
        plt.xlabel('frequency')
        plt.ylabel('log-spectrum')
        plt.show()

    def get_proto_objects_map(self, use_otsu=True):
        """Returns the proto-objects map of an RGB image
            This method generates a proto-objects map of an RGB image.
            Proto-objects are saliency hot spots, generated by thresholding
            the saliency map.
            :param use_otsu: flag whether to use Otsu thresholding (True) or
                             a hardcoded threshold value (False)
            :returns: proto-objects map
        """
        saliency = self.get_saliency_map()

        if use_otsu:
            _, img_objects = cv2.threshold(np.uint8(saliency*255), 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            thresh = np.mean(saliency)*255*3
            _, img_objects = cv2.threshold(np.uint8(saliency*255), thresh, 255,
                                           cv2.THRESH_BINARY)
        return img_objects


def get_dullness(img):
    img = Image.fromarray(img)
    # obtain the color palette of the image 
    palette = defaultdict(int)
    for pixel in img.getdata():
        palette[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palette.items(), key=operator.itemgetter(1), reverse=True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = light_shade / shade_count
    dark_percent = dark_shade / shade_count
    return {'light_percent': light_percent, 'dark_percent': dark_percent}


def get_average_pixel_width(img):  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges_sigma1 = feature.canny(gray, sigma=3)
    apw = np.sum(edges_sigma1)/ img.shape[0] / img.shape[1]
    return {'average_pixel_width': apw}


def get_dominant_color(img):
    pixels = img.reshape(-1, 3).astype('float32')

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(img.shape)

    dominant_color = palette[np.argmax(np.unique(labels))]
    dominant_color = (dominant_color / 255).squeeze()
    
    return {
        'dominant_color_r': dominant_color[0], 
        'dominant_color_g': dominant_color[1], 
        'dominant_color_b': dominant_color[2]
    }


def get_blurrness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurness_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return {'blurness_score': blurness_score}


def get_shape(img):
    return {'width': img.shape[0], 'height': img.shape[1]}


def get_brightness_and_saturation_and_contrast(img):
    def get_stats(img):
        x, y = img.shape[0], img.shape[1]
        img = img.reshape(-1, 3)
        return np.concatenate([
            img.mean(axis=0), 
            img.std(axis=0), 
            img.min(axis=0), 
            img.max(axis=0)
        ])
    yuv = get_stats(cv2.cvtColor(img, cv2.COLOR_BGR2YUV)) 
    hls = get_stats(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
    result = {}
    result.update({'yuv_stats_' + str(i): stats for i, stats in enumerate(yuv)})
    result.update({'hls_stats_' + str(i): stats for i, stats in enumerate(hls)})
    return result


def get_colorfullness(img):
    (B, G, R) = cv2.split(img)
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    (rb_mean, rb_std) = (np.mean(rg), np.std(rg))
    (yb_mean, yb_std) = (np.mean(yb), np.std(yb))

    std_root = np.sqrt((rb_std ** 2) + (yb_std ** 2))
    mean_root = np.sqrt((rb_mean ** 2) + (yb_mean ** 2))
    colorfullness = std_root + (0.3 * mean_root)

    return {'colorfullness': colorfullness}


def get_interest_points(img):
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img, None)
    return {'interest_points': len(kp)}


def get_saliency_features(img):
    saliency = Saliency(img).get_saliency_map()
    binary_saliency = np.where(saliency>3*saliency.mean(), 1, 0).astype('uint8')
    prop_background = 1 - binary_saliency.mean()
    
    n_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary_saliency)
    sizes = stats[:, -1]
    countours = stats[:, :-1]
    max_component_size = max(sizes)/img.shape[0]/img.shape[1]
    bbox = countours[np.argmax(sizes)]
    max_component_avg_saliency = saliency[bbox[1]:bbox[3], bbox[0]:bbox[2]].mean()
    s = centroids/[img.shape[0], img.shape[1]]
    dist = euclidean_distances(s)
    mean_dist = dist[~np.eye(dist.shape[0], dtype=bool)].mean()
    max_component_centorid = s[np.argmax(sizes)]
    min_dist_from_third_points = min(
        np.linalg.norm(max_component_centorid - [1/3, 1/3]),
        np.linalg.norm(max_component_centorid - [1/3, 2/3]),
        np.linalg.norm(max_component_centorid - [2/3, 1/3]),
        np.linalg.norm(max_component_centorid - [2/3, 2/3]),
    )
    dist_from_center = np.linalg.norm(s - [0.5, 0.5], axis=1)
    mean_dist_from_center = dist_from_center.mean()
    sum_dist_from_center = dist_from_center.sum()
    
    result = {
        'prop_background': prop_background, 
        'n_components': n_components, 
        'max_component_size': max_component_size, 
        'max_component_avg_saliency': max_component_avg_saliency, 
        'mean_dist': mean_dist, 
        'min_dist_from_third_points': min_dist_from_third_points, 
        'mean_dist_from_center': mean_dist_from_center, 
        'sum_dist_from_center': sum_dist_from_center
    }
    
    return result


def get_face_features(img, cascade_path):
    cascade = cv2.CascadeClassifier(cascade_path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    facerect = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=1, minSize=(20, 20))
    face_area = 0
    face_area_prop = 0
    if len(facerect) > 0:
        for rect in facerect:
            x, y, w, h = rect
            face_area += w * h
        face_area_prop = face_area / img.shape[0] / img.shape[1]

    return {'num_faces': len(facerect), 'face_area': face_area, 'face_area_prop': face_area_prop}


class ImageFeaturesTransformer(BaseFeatureTransformer):
    def __init__(self, path_list, cascade_path=None, workers=1, name=''):
        self.path_list = path_list
        self.workers = workers
        self.name = name
        if cascade_path is None:
            module_path = os.path.dirname(__file__)
            cascade_path = os.path.join(module_path, 'external_data', 'haarcascade_frontalface_alt2.xml')
        self.functions = [
            get_dullness,
            get_average_pixel_width,
            get_blurrness_score,
            get_brightness_and_saturation_and_contrast,
            get_colorfullness,
            get_dominant_color,
            get_interest_points,
            get_saliency_features,
            get_shape,
            partial(get_face_features, cascade_path=cascade_path)
        ]

    def _get_features(self, path):
        img = cv2.imread(path)
        result = {k: v for f in self.functions for k, v in f(img).items()}
        return pd.Series(result)
    
    def _transform(self, paths):
        return pd.Series(paths).apply(self._get_features)
    
    def _parallel_transform(self):
        with multiprocessing.Pool(processes=self.workers) as p:
            splits = np.array_split(self.path_list, self.workers)
            features = p.map(self._transform, splits)
        features = pd.concat(features).reset_index(drop=True)
        features.columns = [self.name + c for c in features.columns]
        return features
        
    def transform(self, dataframe):
        self.features = [self._parallel_transform()]
        dataframe = pd.concat([dataframe]+self.features, axis=1)
        return dataframe
