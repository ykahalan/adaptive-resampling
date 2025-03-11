This repo provides the implementation of the border and core detection as studied in our paper:

# Installation

Assuming you're in current directory run the following command.
```
pip install -e .
```
Or to install as a python package.
```
pip install adaptive-resampling
```

# Example usage
For detecting core and border points of a single class.
```py
# Import the functions from the installed library
from border_detection_minimal.border_detection_minimal import classify_border_and_core_points

# Generate random data points for the example (1000 points in 2D space)
import numpy as np
X = np.random.rand(1000, 2)

# Classify points into border and core points
border_points, core_points = classify_border_and_core_points(X, p=2, close=100, percentile=60)

print(f"Number of border points: {border_points.shape[0]}")
print(f"Number of core points: {core_points.shape[0]}")
```
For detecting core and border points of each class.
```py
# Import the functions from the installed library
from border_detection_minimal.border_detection_minimal import classify_border_and_core_points

# Generate random data points for the example (1000 points in 2D space, with 3 classes)
import numpy as np
np.random.seed(42)
X = np.random.rand(1000, 2)
y = np.random.randint(0, 3, size=1000)  # 3 classes (0, 1, 2)

# Classify border and core points for each class
class_border_core = classify_border_and_core_points(X, y, p=2, close=100, percentile=60)

for cls, (border, core) in class_border_core.items():
    print(f"Class {cls}:")
    print(f"  Number of border points: {border.shape[0]}")
    print(f"  Number of core points: {core.shape[0]}")
```
For oversampling on the border and undersampling on the core as intended per the paper.
```py
# Import necessary libraries
import numpy as np
from adaptive_resampling import classify_border_and_core_points
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Generate synthetic data (1000 samples, 2 features, 3 classes)
np.random.seed(42)
X = np.random.rand(1000, 2)
y = np.random.randint(0, 3, size=1000)  # Classes: 0, 1, 2

# Classify border and core points for each class
class_border_core = classify_border_and_core_points(X, y, p=2, close=100, percentile=60)

# Initialize lists to hold resampled data
X_resampled = []
y_resampled = []

# Iterate over each class to apply resampling strategies
for cls, (border_points, core_points) in class_border_core.items():
    # Prepare labels for border and core points
    y_border = np.full(border_points.shape[0], cls)
    y_core = np.full(core_points.shape[0], cls)

    # Apply SMOTE to border points
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_border_resampled, y_border_resampled = smote.fit_resample(border_points, y_border)

    # Apply Random Undersampling to core points
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_core_resampled, y_core_resampled = rus.fit_resample(core_points, y_core)

    # Combine resampled border and core points
    X_resampled.append(np.vstack((X_border_resampled, X_core_resampled)))
    y_resampled.append(np.hstack((y_border_resampled, y_core_resampled)))

# Concatenate all resampled data
X_resampled = np.vstack(X_resampled)
y_resampled = np.hstack(y_resampled)

# Display the class distribution before and after resampling
print(f"Original class distribution: {Counter(y)}")
print(f"Resampled class distribution: {Counter(y_resampled)}")
```
