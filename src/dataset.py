import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Membuat dataset acak dengan tren linear
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Nilai X antara 0 hingga 10
y = 2.5 * X + np.random.randn(100, 1) * 3  # Linear dengan noise

# Konversi ke DataFrame
df = pd.DataFrame({'X': X.flatten(), 'y': y.flatten()})

# 2. Membagi dataset menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Membuat model regresi linear
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Prediksi hasil
y_pred = model.predict(X_test)

# 5. Visualisasi
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['X'], y=df['y'], label="Data Asli", color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Garis Regresi")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression dengan Sklearn")
plt.legend()
plt.show()

# 6. Evaluasi model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Koefisien: {model.coef_[0][0]:.2f}, Intercept: {model.intercept_[0]:.2f}")