import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Style settings
sns.set_theme(style="whitegrid")

def lorenz_system(x, y, z, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Computes the derivatives of the Lorenz system.
    Using classical Lorenz-63 parameter values.
    """
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

def generate_lorenz_data(dt=0.01, num_steps=10000, initial_state=(1.0, 1.0, 1.0)):
    """
    Generates the Lorenz attractor data using the Euler integration method.
    
    Parameters:
    - dt: Time step for the Euler method.
    - num_steps: Total number of integration steps.
    - initial_state: Tuple with initial (x, y, z) values.
    
    Returns:
    - times: Array of time points.
    - trajectory: A numpy array of shape (num_steps, 3) containing x, y, z coordinates.
    """
    times = np.arange(0, num_steps * dt, dt)
    trajectory = np.zeros((num_steps, 3))
    trajectory[0] = initial_state
    
    for i in range(1, num_steps):
        x, y, z = trajectory[i-1]
        dx, dy, dz = lorenz_system(x, y, z)
        
        # Euler method numerical integration
        trajectory[i, 0] = x + dx * dt
        trajectory[i, 1] = y + dy * dt
        trajectory[i, 2] = z + dz * dt
        
    return times, trajectory

def plot_lorenz_attractor(trajectory):
    """Plots the generated 3D butterfly Lorenz attractor."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
            lw=0.5, color='tab:blue', alpha=0.8)
    
    ax.set_title("Lorenz Attractor (Lorenz-63)")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    plt.show()

if __name__ == "__main__":
    # Generate data
    # print("Generating Lorenz system data using Euler method...")
    # times, data = generate_lorenz_data(dt=0.01, num_steps=10000)
    
    # Plot to verify the bounded chaotic dynamics (the butterfly shape)
    # plot_lorenz_attractor(data)
    
    print("Data Generation module ready to use!")
