import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_mnist():
    print("Loading MNIST dataset for visualization...")
    try:
        from dataset_loader import load_mnist_data
        
        (X_train, y_train), _ = load_mnist_data(num_train=25, num_test=0)
        
        
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            
            plt.imshow(X_train[i].reshape(28, 28), cmap=plt.cm.binary)
            plt.xlabel(f"Label: {y_train[i]}")
        
        output_path = "dataset_samples.jpg"
        plt.savefig(output_path)
        plt.close()
        
        print(f"\n[SUCCESS] Datset samples saved to: {os.path.abspath(output_path)}")
        print("You can now open this image to see the data.")
        
    except Exception as e:
        print(f"[ERROR] Failed to visualize dataset: {e}")

if __name__ == "__main__":
    visualize_mnist()
