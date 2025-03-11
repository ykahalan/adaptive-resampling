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
