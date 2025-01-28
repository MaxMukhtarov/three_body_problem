import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Constants
G = 6.67430e-11  # Gravitational constant, m^3 kg^-1 s^-2

# Masses of the bodies (same mass for all)
m1 = m2 = m3 = 1.989e30  # Mass of each body (e.g., similar to the Sun)

# Initial positions (x, y, z) for each body (e.g., in meters)
r1_0 = np.array([0, 0, 0])  # Body 1 at origin
r2_0 = np.array([1.496e11, 0, 0])  # Body 2 at 1 AU from Body 1
r3_0 = np.array([0, 1.496e11, 0])  # Body 3 at 1 AU from Body 1 along the y-axis

# Initial velocities (vx, vy, vz) for each body (e.g., in m/s)
v1_0 = np.array([0, 0, 0])  # Body 1 initially stationary
v2_0 = np.array([0, 30000, 0])  # Body 2 initial velocity along y-axis
v3_0 = np.array([-30000, 0, 0])  # Body 3 initial velocity along x-axis

# State vector: [x1, y1, z1, x2, y2, z2, x3, y3, z3, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3]
y0 = np.concatenate((r1_0, r2_0, r3_0, v1_0, v2_0, v3_0))

# Time span for the simulation (e.g., 1 year in seconds)
t_span = (0, 730 * 24 * 3600)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Differential equations for the three-body problem
def three_body_equations(t, y):
    # Extract positions and velocities
    r1 = y[0:3]
    r2 = y[3:6]
    r3 = y[6:9]
    v1 = y[9:12]
    v2 = y[12:15]
    v3 = y[15:18]

    # Distances between bodies
    r12 = np.linalg.norm(r2 - r1)
    r13 = np.linalg.norm(r3 - r1)
    r23 = np.linalg.norm(r3 - r2)

    # Gravitational forces
    a1 = G * (m2 * (r2 - r1) / r12**3 + m3 * (r3 - r1) / r13**3)
    a2 = G * (m1 * (r1 - r2) / r12**3 + m3 * (r3 - r2) / r23**3)
    a3 = G * (m1 * (r1 - r3) / r13**3 + m2 * (r2 - r3) / r23**3)

    # Derivatives: [v1, v2, v3, a1, a2, a3]
    dy_dt = np.concatenate((v1, v2, v3, a1, a2, a3))
    return dy_dt

# Solve the differential equations
sol = solve_ivp(three_body_equations, t_span, y0, t_eval=t_eval, method='RK45')

# Extract positions
r1 = sol.y[0:2, :]  # Only x and y for 2D visualization
r2 = sol.y[3:5, :]
r3 = sol.y[6:8, :]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 12))  # Larger figure size
# Calculate the maximum extent of the bodies' positions
max_extent = max(np.max(np.abs(r1)), np.max(np.abs(r2)), np.max(np.abs(r3)))

# Set the axis limits dynamically
ax.set_xlim(-max_extent, max_extent)
ax.set_ylim(-max_extent, max_extent)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Three-Body Problem Animation (Equal Masses)')

# Create lines and points for the bodies
line1, = ax.plot([], [], color='yellow', label='Body 1')
line2, = ax.plot([], [], color='blue', label='Body 2')
line3, = ax.plot([], [], color='gray', label='Body 3')
point1, = ax.plot([], [], 'o', color='yellow', markersize=10)
point2, = ax.plot([], [], 'o', color='blue', markersize=10)
point3, = ax.plot([], [], 'o', color='gray', markersize=10)

# Initialization function for the animation
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    point1.set_data([], [])
    point2.set_data([], [])
    point3.set_data([], [])
    return line1, line2, line3, point1, point2, point3

# Animation function
def animate(i):
    # Update the lines and points
    line1.set_data(r1[0, :i], r1[1, :i])
    line2.set_data(r2[0, :i], r2[1, :i])
    line3.set_data(r3[0, :i], r3[1, :i])
    point1.set_data([r1[0, i]], [r1[1, i]])  # Pass as sequences
    point2.set_data([r2[0, i]], [r2[1, i]])  # Pass as sequences
    point3.set_data([r3[0, i]], [r3[1, i]])  # Pass as sequences
    return line1, line2, line3, point1, point2, point3

# Create the animation
frames = len(t_eval)
interval = 120_000 / frames  # Total duration of 1 minute (60,000 milliseconds)
ani = FuncAnimation(fig, animate, frames=frames, init_func=init, blit=True, interval=interval)

# Show the animation
plt.legend()
plt.grid()
plt.show()