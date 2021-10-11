import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
'''Class for applying the trained neural network on an example X-ray scattering image.

Input:  test_img:   The image has to be in form of an numpy array. 
        model:      The trained model should be already loaded (see main.py).
'''
class TestModelPatches(object):
    def __init__(self, model, test_img):
        self.model = model
        self.test_img = test_img


    def test_img(model, test_img):
        img = self.test_img
        sizes = img.shape
        points = (np.random.uniform(0, 0.5, 1) * sizes).astype(np.int)
        # take random patch of the image
        imgs_patch = img[points[0]:points[0] + sizes[0] // 2, points[1]:points[1] + sizes[1] // 2]
        #calculate hisotgram of the patch
        hist = normalize(np.log(np.histogram(imgs_patch, bins=256)[0] + 1))
        #predict the upper and lower value for linear clipping via trained model
        pred_1, pred_2 = self.model(torch.Tensor(hist).cuda()).detach().cpu().numpy()
        pred_1 = min(pred_1, pred_2)
        pred_2 = max(pred_1, pred_2)
        m1, m2 = imgs_patch.min(), imgs_patch.max()
        pred_1_rescaled = (pred_1 * (m2 - m1)) + m1
        pred_2_rescaled = (pred_2 * (m2 - m1)) + m1

        #Clip the image with predicted values
        pred_img = np.clip(img, pred_1_rescaled, pred_2_rescaled)
      
        size = 15
        plt.xlabel('Detector pixels', fontsize=size)
        plt.ylabel('Detector pixels',  fontsize=size)  
        plt.imshow(img, extent=[0,img.shape[1],0,img.shape[0]])
        plt.show()
        plt.imshow(pred_img,extent=[0,img.shape[1],0,img.shape[0]])
        plt.xlabel('Detector pixels',  fontsize=size)
        plt.ylabel('Detector pixels',  fontsize=size)
        plt.show()
    
