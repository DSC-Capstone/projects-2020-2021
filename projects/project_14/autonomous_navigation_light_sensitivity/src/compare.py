# import the necessary packages
import os
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2


def view_results(files, outdir):
    os.makedirs(outdir, exist_ok = True)
    imageA, imageB, imageC = cv2.imread(files[0]), cv2.imread(files[1]), cv2.imread(files[2])
    a = baseline_results(imageA, imageB, outdir)
    b = tuned_results(imageA, imageC, outdir)
    plot_ssim(a, b, outdir)
    plot_mse(a, b, outdir)
    

def calculate_mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: Two images have the same dimension
    error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    error /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar" the two images are
    return error
 

def baseline_results(imageA, imageB, outdir):
    # compute the mean squared error and structural similarity
    # index for the images
    s = ssim(imageA, imageB, multichannel=True)
    m = calculate_mse(imageA, imageB)
    # setup the figure
    fig = plt.figure()
    plt.suptitle("SSIM: %.4f, MSE: %.0f" % (s, m))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")
    # save figure
    plt.savefig(os.path.join(outdir, 'baseline.jpg'))
    plt.close()
    baseline_dict = {}
    baseline_dict['ssim'] = s
    baseline_dict['mse'] = m
    
    return baseline_dict
    
def tuned_results(imageA, imageC, outdir):
    # compute the mean squared error and structural similarity
    # index for the images
    s = ssim(imageA, imageC, multichannel=True)
    m = calculate_mse(imageA, imageC)
    # setup the figure
    fig = plt.figure()
    plt.suptitle("SSIM: %.4f, MSE: %.0f" % (s, m))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageC, cmap = plt.cm.gray)
    plt.axis("off")
    # save figure
    plt.savefig(os.path.join(outdir, 'tuned.jpg'))
    plt.close()
    tuned_dict = {}
    tuned_dict['ssim'] = s
    tuned_dict['mse'] = m
    
    return tuned_dict
   

def plot_ssim(dict_1, dict_2, outdir):
    #labels = ['moderate sharpness', 'low sharpness', 'lower white balance', 'low contrast', 'low saturation']
    labels = ['lower white balance']
    x = np.arange(len(labels))
    width = 0.5

    round_base_ssim, round_best_ssim = np.round(dict_1['ssim'], 4), np.round(dict_2['ssim'], 4)

    fig, ax = plt.subplots(figsize=(5,6))
    bar_1 = ax.bar(x - width/2, round_base_ssim, width, label='baseline')
    bar_2 = ax.bar(x + width/2, round_best_ssim, width, label='best tuned')
    ax.set_title('SSIM Comparison', fontsize=15)
    ax.set_xticks(x)
    ax.tick_params(labelrotation=45, labelsize=10)
    ax.set_xticklabels(labels)
    ax.set_ylabel('SSIM', fontsize=15)
    ax.legend(loc=4, prop={'size':10})
    
    def text_label(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    text_label(bar_1)
    text_label(bar_2)

    plt.savefig(os.path.join(outdir,'ssim_comparison.jpg'))
    
    return
    
    
def plot_mse(dict_1, dict_2, outdir):
    
    labels = ['lower white balance']
    x = np.arange(len(labels))
    width = 0.3
    
    round_base_mse, round_tuned_mse = np.round(dict_1['mse'], 0), np.round(dict_2['mse'], 0)

    fig, ax = plt.subplots(figsize=(5,6))
    bar_1 = ax.bar(x - width/2, round_base_mse, width, label='baseline')
    bar_2 = ax.bar(x + width/2, round_tuned_mse, width, label='best tuned', color='y')
    ax.set_title('MSE Comparison', fontsize=15)
    ax.set_xticks(x)
    ax.tick_params(labelrotation=45, labelsize=10)
    ax.set_xticklabels(labels)
    ax.set_ylabel('MSE', fontsize=15)
    ax.legend(loc=1, prop={'size':10})

    def text_label(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    text_label(bar_1)
    text_label(bar_2)
    
    plt.savefig(os.path.join(outdir,'mse_comparison.jpg'))
    
    return
