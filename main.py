import sys
import numpy as np
from scipy.integrate import solve_ivp
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen, QPolygonF
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsItem, QMainWindow

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)

# Masses of the bodies (in kilograms)
mass_earth = 5.972e24   # Mass of the Earth (kg)
mass_sun1 = 1.989e30    # Mass of Sun 1 (kg)
mass_sun2 = 1.989e30    # Mass of Sun 2 (kg)
mass_sun3 = 1.989e30    # Mass of Sun 3 (kg)

# Initial positions (in meters) and velocities (in meters per second)
x_earth, y_earth = 1.496e11, 0  # 1 AU, along x-axis

# Sun positions (in meters)
x_sun1, y_sun1 = -1e11, 0      # Sun 1 is to the left of the Earth
x_sun2, y_sun2 = 2.5e11, 2.5e10  # Sun 2 is to the right and above
x_sun3, y_sun3 = 2.5e11, -2.5e10 # Sun 3 is below Sun 2

# Initial velocities (in m/s)
vx_earth, vy_earth = 0, 29780  # Approximate Earth's velocity around the Sun

# Function to compute the gravitational force between two bodies
def grav_force(m1, m2, r1, r2):
    dist = np.linalg.norm(r1 - r2)
    force_mag = G * m1 * m2 / dist**2
    force_dir = (r2 - r1) / dist
    return force_mag * force_dir

# Define the system of ODEs (positions and velocities)
def three_body_system(t, state):
    # Extract the current state (positions and velocities)
    x1, y1, vx1, vy1 = state[:4]   # Earth's position and velocity
    x2, y2 = x_sun1, y_sun1        # Position of Sun 1
    x3, y3 = x_sun2, y_sun2        # Position of Sun 2
    x4, y4 = x_sun3, y_sun3        # Position of Sun 3

    r1 = np.array([x1, y1])  # Earth position
    r2 = np.array([x2, y2])  # Sun 1 position
    r3 = np.array([x3, y3])  # Sun 2 position
    r4 = np.array([x4, y4])  # Sun 3 position

    # Gravitational forces between Earth and each Sun
    F12 = grav_force(mass_earth, mass_sun1, r1, r2)  # Force from Sun 1
    F13 = grav_force(mass_earth, mass_sun2, r1, r3)  # Force from Sun 2
    F14 = grav_force(mass_earth, mass_sun3, r1, r4)  # Force from Sun 3

    # Net force on the Earth
    net_force = F12 + F13 + F14

    # Equations of motion: F = ma, so a = F / m
    ax = net_force[0] / mass_earth
    ay = net_force[1] / mass_earth

    # Return derivatives (velocity and acceleration)
    return [vx1, vy1, ax, ay]

# Initial state: Earth's position and velocity
initial_state = [x_earth, y_earth, vx_earth, vy_earth]

# Time span for the simulation (e.g., simulate for 2 years)
t_span = (0, 2 * 365 * 24 * 3600)  # 2 years in seconds
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Evaluation times

# Solve the system of ODEs
solution = solve_ivp(three_body_system, t_span, initial_state, t_eval=t_eval)

# Extract the results
x_earth_sol, y_earth_sol = solution.y[0], solution.y[1]

class OrbitViewer(QGraphicsView):
    def __init__(self):
        super().__init__()

        # Set up the scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Convert Earth's trajectory into a list of QPointF for QPolygonF
        trajectory_points = [QPointF(x, y) for x, y in zip(x_earth_sol / 1e10, y_earth_sol / 1e10)]
        
        # Display the Earth's trajectory
        self.earth_trajectory = self.scene.addPolygon(
            QPolygonF(trajectory_points),
            QPen(QColor(0, 0, 255))  # Blue color for Earth trajectory
        )

        # Display the positions of the suns
        self.sun1 = self.scene.addEllipse(x_sun1 / 1e10 - 2, y_sun1 / 1e10 - 2, 4, 4, QPen(QColor(255, 0, 0)), None)
        self.sun2 = self.scene.addEllipse(x_sun2 / 1e10 - 2, y_sun2 / 1e10 - 2, 4, 4, QPen(QColor(255, 0, 0)), None)
        self.sun3 = self.scene.addEllipse(x_sun3 / 1e10 - 2, y_sun3 / 1e10 - 2, 4, 4, QPen(QColor(255, 0, 0)), None)

        # Set the view's scale
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        # You can add a grid or axis here if desired
        painter.setPen(QPen(Qt.black, 1))
        painter.drawRect(0, 0, 1000, 1000)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Earth Orbiting Three Suns')
        self.setGeometry(100, 100, 800, 600)

        self.viewer = OrbitViewer()
        self.setCentralWidget(self.viewer)

# Application setup
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
