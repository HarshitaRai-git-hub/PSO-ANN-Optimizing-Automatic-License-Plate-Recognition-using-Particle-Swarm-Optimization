import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import os
import sys


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def save_weights(weights, filename="pso_ann_weights.npy"):
    """Saves the best weights to a file."""
    np.save(filename, weights)
    print(f"[INFO] Weights saved to {filename}")

def load_weights(filename="pso_ann_weights.npy"):
    """Loads weights from a file if it exists."""
    if os.path.exists(filename):
        print(f"[INFO] Loading weights from {filename}...")
        return np.load(filename)
    return None


class SimpleANN:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

    def set_weights(self, flat_vector):
        """Reconstruct layers from a flat probability/weight vector."""
        self.weights = []
        self.biases = []
        start = 0
        for i in range(len(self.layer_sizes) - 1):
            w_shape = (self.layer_sizes[i], self.layer_sizes[i+1])
            w_size = self.layer_sizes[i] * self.layer_sizes[i+1]
            b_size = self.layer_sizes[i+1]
            
            w = flat_vector[start : start + w_size].reshape(w_shape)
            start += w_size
            b = flat_vector[start : start + b_size]
            start += b_size
            
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        """Standard feed-forward with sigmoid activation."""
        a = X
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            
            a = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return a

    def get_params_count(self):
        count = 0
        for i in range(len(self.layer_sizes) - 1):
            count += self.layer_sizes[i] * self.layer_sizes[i+1] 
            count += self.layer_sizes[i+1] 
        return count


class PSOClassifier:
    def __init__(self, nn, swarm_size=20, c1=1.5, c2=1.5, w=0.7, initial_weights=None):
        self.nn = nn
        self.swarm_size = swarm_size
        self.params_dim = nn.get_params_count()
        self.c1 = c1  
        self.c2 = c2  
        self.w = w    
        
        
        self.particles = np.random.uniform(-1, 1, (swarm_size, self.params_dim))
        
        
        if initial_weights is not None:
            print("[INFO] Seeding swarm with previous best weights...")
            
            self.particles[0] = initial_weights.copy()
            
            for i in range(1, swarm_size // 2):
                self.particles[i] = initial_weights + np.random.normal(0, 0.05, self.params_dim)

        self.velocities = np.random.uniform(-0.1, 0.1, (swarm_size, self.params_dim))
        self.pbest = self.particles.copy()
        self.pbest_fit = np.full(swarm_size, -np.inf)
        self.gbest = None
        self.gbest_fit = -np.inf

    def fitness(self, particle, X, y):
        self.nn.set_weights(particle)
        y_pred = self.nn.forward(X)
        predictions = np.argmax(y_pred, axis=1)
        
        return np.mean(predictions == y)

    def optimize(self, X, y, iterations=30):
        print(f"Starting PSO Optimization (Swarm: {self.swarm_size}, Dim: {self.params_dim})...")
        
        for i in range(iterations):
            for p in range(self.swarm_size):
                fit = self.fitness(self.particles[p], X, y)
                
                
                if fit > self.pbest_fit[p]:
                    self.pbest_fit[p] = fit
                    self.pbest[p] = self.particles[p].copy()
                    
                
                if fit > self.gbest_fit:
                    self.gbest_fit = fit
                    self.gbest = self.particles[p].copy()
            
            
            r1, r2 = np.random.rand(2)
            self.velocities = (self.w * self.velocities + 
                               self.c1 * r1 * (self.pbest - self.particles) + 
                               self.c2 * r2 * (self.gbest - self.particles))
            self.particles += self.velocities
            
            if (i+1) % 5 == 0 or i == 0:
                print(f"  Iteration {i+1}/{iterations} | Best Accuracy: {self.gbest_fit:.2%}")
        
        return self.gbest


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PSO-ANN Progressive Character Recognition Training")
    print("="*60)

    from dataset_loader import load_mnist_data

    
    stages = [1000, 2000, 3000, 4000, 5000]
    weights_file = "pso_ann_weights.npy"
    
    
    layer_sizes = [784, 32, 10]
    nn = SimpleANN(layer_sizes)

    best_weights = load_weights(weights_file)

    for stage_size in stages:
        print(f"\n>>> STAGE: Training with {stage_size} images...")
        
        
        (X_train, Y_train), (X_test, Y_test) = load_mnist_data(num_train=stage_size, num_test=1000)

        
        pso = PSOClassifier(nn, swarm_size=20, initial_weights=best_weights)

        
        
        iters = 20 if stage_size > 1000 else 30
        best_weights = pso.optimize(X_train, Y_train, iterations=iters)
        
        
        save_weights(best_weights, weights_file)
        print(f"Stage {stage_size} Complete. Current Best Accuracy: {pso.gbest_fit:.2%}")

    print(f"\nFinal Training Complete. Total images used: {stages[-1]}")

    
    print("\nEvaluating final model on test set...")
    nn.set_weights(best_weights)
    test_output = nn.forward(X_test)
    Y_pred = np.argmax(test_output, axis=1)
    test_acc = np.mean(Y_pred == Y_test)
    print(f"Final Test Accuracy: {test_acc:.2%}")

    
    print("\nGenerating final confusion matrix...")
    cm = confusion_matrix(Y_test, Y_pred)
    
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'PSO-ANN Confusion Matrix (Trained on {stages[-1]} images)')
    plt.colorbar()
    
    classes = [str(i) for i in range(10)]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    
    output_filename = "mnist_confusion_matrix_final.jpg"
    plt.savefig(output_filename)
    print(f"\n[SUCCESS] Final confusion matrix saved as: {output_filename}")
    
    plt.show() 
