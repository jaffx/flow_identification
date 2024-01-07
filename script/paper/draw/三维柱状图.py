import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_data = np.arange(10)
y_data = np.arange(10)
z_data = np.arange(10)

dx = dy = 0.8
dz = z_data
print(dz)

ax.bar3d(x_data, y_data, z_data, dx, dy, dz, color="red", alpha=0.6, shade=True)
ax.bar3d(x_data - 1, y_data, z_data, dx, dy, dz, shade=True)
ax.set_zlim(0, 20)
ax.set_title('3D Bar Chart')
ax.set_xlabel('Category')
ax.set_ylabel('Value')
ax.set_zlabel('Height')
plt.show()
