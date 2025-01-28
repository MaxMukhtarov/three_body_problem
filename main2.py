import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Constants
G = 6.67430e-11  # Gravitational constant, m^3 kg^-1 s^-2
sun_mass = 1.989e30  # Mass of the Sun, kg

# Number of bodies and their masses (relative to the Sun's mass)
N = 5  # Number of bodies
masses = np.array([1.0, 0.1, 0.01, 0.001, 0.0001]) * sun_mass  # Example masses

# Initial positions (x, y, z) for each body (e.g., in meters)
positions = np.array([
    [0, 0, 0],  # Body 1 (e.g., Sun)
    [1.496e11, 0, 0],  # Body 2 (e.g., Earth)
    [0, 1.496e11, 0],  # Body 3 (e.g., another planet)
    [-1.496e11, 0, 0],  # Body 4
    [0, -1.496e11, 0]  # Body 5
])

# Initial velocities (vx, vy, vz) for each body (e.g., in m/s)
velocities = np.array([
    [0, 0, 0],  # Body 1 (e.g., Sun)
    [0, 29780, 0],  # Body 2 (e.g., Earth)
    [-29780, 0, 0],  # Body 3
    [0, -29780, 0],  # Body 4
    [29780, 0, 0]  # Body 5
])

# State vector: [x1, y1, z1, x2, y2, z2, ..., xN, yN, zN, vx1, vy1, vz1, ..., vxN, vyN, vzN]
y0 = np.concatenate((positions.flatten(), velocities.flatten()))

# Time span for the simulation (e.g., 1 year in seconds)
t_span = (0, 365 * 24 * 3600)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Differential equations for the N-body problem
def n_body_equations(t, y):
    # Extract positions and velocities
    r = y[:3 * N].reshape(N, 3)  # Positions of all bodies
    v = y[3 * N:].reshape(N, 3)  # Velocities of all bodies

    # Initialize accelerations
    a = np.zeros_like(r)

    # Calculate gravitational forces
    for i in range(N):
        for j in range(N):
            if i != j:
                r_ij = r[j] - r[i]  # Vector from body i to body j
                r_ij_norm = np.linalg.norm(r_ij)  # Distance between bodies i and j
                a[i] += G * masses[j] * r_ij / r_ij_norm**3  # Acceleration of body i due to body j

    # Derivatives: [v1, v2, ..., vN, a1, a2, ..., aN]
    dy_dt = np.concatenate((v.flatten(), a.flatten()))
    return dy_dt

# Solve the differential equations
sol = solve_ivp(n_body_equations, t_span, y0, t_eval=t_eval, method='RK45')

# Extract positions
positions_over_time = sol.y[:3 * N, :].reshape(N, 3, -1)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-2e11, 2e11)
ax.set_ylim(-2e11, 2e11)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title(f'{N}-Body Problem Animation')

# Create lines and points for the bodies
lines = [ax.plot([], [], label=f'Body {i+1}')[0] for i in range(N)]
points = [ax.plot([], [], 'o', markersize=10)[0] for i in range(N)]

# Initialization function for the animation
def init():
    for line, point in zip(lines, points):
        line.set_data([], [])
        point.set_data([], [])
    return lines + points

# Animation function
def animate(i):
    for j, (line, point) in enumerate(zip(lines, points)):
        line.set_data(positions_over_time[j, 0, :i], positions_over_time[j, 1, :i])
        point.set_data([positions_over_time[j, 0, i]], [positions_over_time[j, 1, i]])
    return lines + points

# Create the animation
frames = len(t_eval)
interval = 60_000 / frames  # Total duration of 1 minute (60,000 milliseconds)
ani = FuncAnimation(fig, animate, frames=frames, init_func=init, blit=True, interval=interval)

# Show the animation
plt.legend()
plt.grid()
plt.show()