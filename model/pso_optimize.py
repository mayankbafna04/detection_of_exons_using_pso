import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import train_test_split
from .cnn import build_cnn

class PSOCNNOptimizer:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def fitness(self, params):
        scores = []
        for p in params:
            filters = int(p[0])
            kernel_size = int(p[1])
            dense_units = int(p[2])
            dropout = min(max(p[3], 0.1), 0.7)
            lr = 10 ** (-int(p[4]))

            model = build_cnn(filters, kernel_size, dense_units, dropout)
            model.compile(optimizer=Adam(learning_rate=lr),
                          loss=BinaryCrossentropy(),
                          metrics=[AUC()])

            X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            try:
                model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
                _, auc = model.evaluate(X_val, y_val, verbose=0)
            except:
                auc = 0
            scores.append(1 - auc)
        return np.array(scores)

    def run(self):
        bounds = ([16, 3, 32, 0.1, 3], [128, 7, 256, 0.7, 5])
        optimizer = GlobalBestPSO(n_particles=10, dimensions=5, options={'c1':0.5, 'c2':0.3, 'w':0.9}, bounds=bounds)
        best_cost, best_pos = optimizer.optimize(self.fitness, iters=10)
        return best_pos