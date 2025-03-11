# border_detection_minimal.py

import numpy as np

def calculate_distances_close(x, p, close):
    """
    Calculate the average distance to the closest 'close' points for each point in the dataset.
    
    Parameters:
    x : np.ndarray
        The dataset (n_samples, n_features).
    p : int
        The norm to use for distance calculation (default is Euclidean norm, p=2).
    close : int
        The number of nearest points to consider for distance calculation.
    
    Returns:
    distances : np.ndarray
        The array of average distances to the closest 'close' points.
    """
    n_samples = x.shape[0]
    distances = np.zeros(n_samples)

    for i in range(n_samples):
        dist_list = []
        for j in range(n_samples):
            if i != j:
                dist = np.linalg.norm(x[i] - x[j], ord=p)
                dist_list.append(dist)

        # Sort the distances and select the top 'close' closest points
        dist_list.sort()
        top_close_dists = dist_list[:close]

        # Calculate the average of these top 'close' distances
        distances[i] = sum(top_close_dists) / close

    return distances

def classify_border_and_core_points(X, p=2, close=100, percentile=60):
    """
    Classify the points in the dataset as 'border' or 'core' based on the distance percentile.
    
    Parameters:
    X : np.ndarray
        The dataset (n_samples, n_features).
    p : int
        The norm to use for distance calculation (default is Euclidean norm, p=2).
    close : int
        The number of closest points to consider for the distance calculation (default=100).
    percentile : float
        The threshold percentile for defining border points (default=60).
    
    Returns:
    border_points : np.ndarray
        Points classified as 'border', sorted by descending distance.
    core_points : np.ndarray
        Points classified as 'core', sorted by ascending distance.
    """
    # Calculate distances based on the closest 'close' points
    distances = calculate_distances_close(X, p, close)
    
    # Calculate the distance threshold for border points
    threshold_distance = np.percentile(distances, percentile)
    
    # Separate and sort points as border or core based on the threshold
    border_indices = np.where(distances >= threshold_distance)[0]
    core_indices = np.where(distances < threshold_distance)[0]
    
    # Sort border points by descending distance
    border_indices = border_indices[np.argsort(-distances[border_indices])]
    # Sort core points by ascending distance
    core_indices = core_indices[np.argsort(distances[core_indices])]
    
    border_points = X[border_indices]
    core_points = X[core_indices]
    
    return border_points, core_points

def classify_border_and_core_points_multiclass(X, y, p=2, close=100, percentile=60):
    """
    Classify the points in a multiclass dataset as 'border' or 'core' for each class.
    
    Parameters:
    X : np.ndarray
        The dataset (n_samples, n_features).
    y : np.ndarray
        The class labels for the dataset (n_samples,).
    p : int
        The norm to use for distance calculation (default is Euclidean norm, p=2).
    close : int
        The number of closest points to consider for the distance calculation (default=100).
    percentile : float
        The threshold percentile for defining border points (default=60).
    
    Returns:
    class_border_core_dict : dict
        A dictionary where keys are class labels and values are tuples containing
        (border_points, core_points) for each class.
    """
    class_border_core_dict = {}
    
    # Get unique classes in the dataset
    unique_classes = np.unique(y)
    
    for cls in unique_classes:
        # Extract points belonging to the current class
        X_class = X[y == cls]
        
        # Classify points into border and core for the current class
        border_points, core_points = classify_border_and_core_points(X_class, p=p, close=close, percentile=percentile)
        
        # Store the border and core points in the dictionary
        class_border_core_dict[cls] = (border_points, core_points)
    
    return class_border_core_dict
