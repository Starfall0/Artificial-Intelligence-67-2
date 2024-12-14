from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Generate two clusters of data
X1, y1 = make_blobs(n_samples=100, n_features=2, centers=1,
                           center_box=(-1.0, -1.0), cluster_std=0.45, random_state=69)

X2, y2 = make_blobs(n_samples=100, n_features=2, centers=1,
                           center_box=(1.0, 1.0), cluster_std=0.45, random_state=69)

# Combine data for normalization
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(100), np.ones(100)))

# Normalize z = (x - u)/ s
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Add bias term manually
bias = np.ones((X.shape[0], 1))  # Create a bias column
X = np.hstack((X, bias))

# Perceptron training function
def train_model(X, y, epochs=10, lr=0.1):
    """
    ฟังก์ชันฝึกโมเดล Perceptron ให้เรียนรู้จากข้อมูล X และ y
    
    Args:
        X (2d array): ข้อมูลที่มีลักษณะเป็น 2 มิติ (ตัวอย่างข้อมูลและคุณสมบัติ)
        y (array): เลเบลของข้อมูล (ค่า 0 หรือ 1)
        epochs (int, optional): จำนวนรอบที่โมเดลจะทำการฝึก (ค่าเริ่มต้นคือ 10)
        lr (float, optional): อัตราการเรียนรู้ที่ใช้ปรับค่าน้ำหนัก (ค่าเริ่มต้นคือ 0.1)

    Returns:
        weights: ค่าน้ำหนัก (weights) ที่เรียนรู้จากการฝึกโมเดล
    """
    weights = np.zeros(X.shape[1]) # weight เริ่มต้น
    for _ in range(epochs):
        for i in range(len(y)):
            pred = np.dot(X[i], weights) >= 0
            if pred != y[i]:
                weights += lr * (y[i] - pred) * X[i]
    return weights

# Train the perceptron
w = train_model(X, y, epochs=50, lr=0.01)

# Decision function
def decision_func(x1, x2, w):
    return w[0] * x1 + w[1] * x2 + w[2] # Original: [1]x1 + [1]x2 - 0.5

# Grid for plotting
x1_vals = np.linspace(-3, 3, 300)
x2_vals = np.linspace(-3, 3, 300)
x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)

# Decision boundary
decision_vals = decision_func(x1_grid, x2_grid, w)
print(f"weight0: {w[0]:.4f}, weight1: {w[1]:.4f}, weight2: {w[2]:.4f}")

# Visualization
fig = plt.figure()
plt.contourf(x1_grid, x2_grid, decision_vals, levels=[-np.inf, 0, np.inf], colors=['red', 'blue'], alpha=0.5)
plt.contour(x1_grid, x2_grid, decision_vals, levels=[0], colors=['black'], linewidths=1)
plt.scatter(X1[:, 0], X1[:, 1], c='purple', label="Class 1")
plt.scatter(X2[:, 0], X2[:, 1], c='yellow', label="Class 2")
plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.title('Decision Plane')
plt.grid(True, axis='both')
plt.legend(loc='lower right')
plt.show()
fig.savefig('Out1 - Data Sample.png')