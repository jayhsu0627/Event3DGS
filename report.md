
## Bayer

In the DAVIS 346C event camera which we use in our experiments, F consists of the following tiled 2×2 pattern (RGGB colour filter):
```[[[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 1]]]```




## Synthetic

### Time frame

For each scene, we render a one-second-long 360◦ rotation of camera around the object at 1000 fps as RGB images, resulting in 1000 views. The maximum timestamp  is $1000$ and the minimum is $0$. Therefore, each incremental timestamp $\delta t=1$ ms.

![](./img/syn_timestamp.png)


The naming of the synthetic RGB images are in the form of `r_#####.png` where `#` ranging from $00000$ to $01000$, so we have $1001$ images in total.

| t=0ms                         | t=249ms                    | t=499ms                | t=999ms               |
| :---------------------------: | :------------------------: | :--------------------: | :-------------------: |
| ![](./img/r_00000.png)        |   ![](./img/r_00249.png)   | ![](./img/r_00499.png) | ![](./img/r_00999.png)|


## Real

### Time frame

We record ten objects with the DAVIS 346C colour event camera on a uniform white background. The maximum timestamp  is $0.99999825$ and the minimum is $0$. The incremental timestamp is $7.50\text{E}-07$ ms.

![](./img/real_timestamp.png)

### Camera Pose Calibration
These positions lie on the circle, which corresponds to the correct camera poses, and they are tilted to the rotational axis with an unknown angle offset $\alpha$. In our recordings, we found that $α=2.85\degree$ for the Goatling and Sewing recordings and $\alpha=0.2388\degree$ for the rest of the sequences.

### Density Clipping
For the real scenes, we know that the object always lies inside the cylinder defined by the turntable plate. Hence, to filter the noise and artefacts in the unobserved areas, we force the density to zero everywhere outside of this cylinder:

$$
\sigma(x,y,z) =0 \text{, if } x^2+y^2>r_{max}^2 \text{ or } z>z_{max} \text{ or }  z<z_{min} 
$$

The cylinder parameters zmin, zmax and rmax are tuned manually to fit the recorded experimental setup. In our case, $z_{min} = −0.35$, $z_{max}=0.15$ and $r_{max}=0.25$.

![](./img/density_clipping.png)

```
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define cylinder parameters
z_min = -0.35
z_max = 0.15
r_max = 0.25

# Generate points for plotting
theta = np.linspace(0, 2*np.pi, 100)
z = np.linspace(z_min, z_max, 100)
Z, Theta = np.meshgrid(z, theta)
X = r_max * np.cos(Theta)
Y = r_max * np.sin(Theta)

# Plot cylinder
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.5)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set plot limits
ax.set_xlim(-r_max, r_max)
ax.set_ylim(-r_max, r_max)
ax.set_zlim(z_min, z_max)

# Show plot
plt.title('Cylinder Visualization')
plt.show()
```
