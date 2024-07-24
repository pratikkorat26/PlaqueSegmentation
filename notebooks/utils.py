import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_side_by_side(image1, image2, title1='', title2=''):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    
    if image1.dtype == np.float32:
        # Denormalize image1
        if mean is not None and std is not None:
            image1 = image1 * std + mean
        
    axes[0].imshow(image1)
    axes[0].set_title(title1)
    axes[0].axis('off')
    
        
    axes[1].imshow(image2)
    axes[1].set_title(title2)
    axes[1].axis('off')
    
    plt.show()