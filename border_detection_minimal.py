# border_detection_minimal.py

import numpy as np

def classify_border_and_core_points(X, y=None, p=2, close=100, percentile=60):
    """
    Classify points as 'border' or 'core' based on distance percentile.
    
    Parameters:
    X : np.ndarray
        The dataset (n_samples, n_features).
    y : np.ndarray, optional
        The class labels for the dataset (n_samples,). If None, the function treats all data as one class.
    p : int
        The norm to use for distance calculation (default is Euclidean norm, p=2).
    close : int
        The number of closest points to consider for the distance calculation (default=100).
    percentile : float
        The threshold percentile for defining border points (default=60).
    
    Returns:
    result : dict or tuple
        If y is provided, returns a dictionary with class labels as keys and (border_points, core_points) as values.
        If y is None, returns a tuple (border_points, core_points).
    """
    
    def calculate_distances_close(x, p, close):
        n_samples = x.shape[0]
        distances = np.zeros(n_samples)

        for i in range(n_samples):
            dist_list = []
            for j in range(n_samples):
                if i != j:
                    dist = np.linalg.norm(x[i] - x[j], ord=p)
                    dist_list.append(dist)

            dist_list.sort()
            top_close_dists = dist_list[:close]
            distances[i] = sum(top_close_dists) / close

        return distances
    
    def process_class(X_class):
        distances = calculate_distances_close(X_class, p, close)
        threshold_distance = np.percentile(distances, percentile)

        border_indices = np.where(distances >= threshold_distance)[0]
        core_indices = np.where(distances < threshold_distance)[0]

        border_indices = border_indices[np.argsort(-distances[border_indices])]
        core_indices = core_indices[np.argsort(distances[core_indices])]

        return X_class[border_indices], X_class[core_indices]

    if y is None:
        return process_class(X)
    
    result = {}
    unique_classes = np.unique(y)
    
    for cls in unique_classes:
        X_class = X[y == cls]
        result[cls] = process_class(X_class)
    
    return result
