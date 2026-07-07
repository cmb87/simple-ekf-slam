import symforce
symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce.codegen import Codegen, PythonConfig, CppConfig
from symforce.values import Values
from pathlib import Path
import sys

dt = sf.Symbol("dt")

# -------------------------------------------
# Camera

x = sf.Symbol("x")
y = sf.Symbol("y")
z = sf.Symbol("z")

omega_r = sf.Symbol("omega_r")
omega_p = sf.Symbol("omega_p")
omega_y = sf.Symbol("omega_y")

vx = sf.Symbol("vx")
vy = sf.Symbol("vy")
vz = sf.Symbol("vz")

q0 = sf.Symbol("q0")
q1 = sf.Symbol("q1")
q2 = sf.Symbol("q2")
q3 = sf.Symbol("q3")

Xwc = sf.V3(x,y,z)
Vwc = sf.V3(vx,vy,vz)
Qwc = sf.V4(q1,q2,q3,q0)

Omega_w = sf.V3(omega_r, omega_p, omega_y)
S = sf.Matrix([*Xwc, *Qwc, *Vwc, *Omega_w])


# -------------------------------------------
# Noise symbols

ax_n = sf.Symbol("ax_n")
ay_n = sf.Symbol("ay_n")
az_n = sf.Symbol("az_n")

gx_n = sf.Symbol("gx_n")
gy_n = sf.Symbol("gy_n")
gz_n = sf.Symbol("gz_n")

An  = sf.V3(ax_n, ay_n, az_n)  # Acceleration noise
Gn  = sf.V3(gx_n, gy_n, gz_n)  # Gyroscope noise


# -------------------------------------------
# Landmarks

xi_l = sf.Symbol("xi_l") # First point observation x
yi_l = sf.Symbol("yi_l") # First point observation y
zi_l = sf.Symbol("zi_l") # First point observation z
theta_l = sf.Symbol("theta_l") # Azimuth angle [0,pi]
phi_l = sf.Symbol("phi_l")     # Elevation angle [0,2pi]
rho_l = sf.Symbol("rho_l")     # inverse depth 1/d

Y_l = sf.V5(xi_l, yi_l, zi_l, theta_l, phi_l) # Landmark state vector

# -------------------------------------------
# Optical Ray Auxillery m(theta,phi)

M_l = sf.V3(sf.sin(theta_l)*sf.cos(phi_l), sf.sin(theta_l)*sf.sin(phi_l), sf.cos(theta_l))


# -------------------------------------------
# Measurement model for the landmark (S + Y_l) -> (u,v)

f = sf.Symbol("f") # Focal length of the camera
dx = sf.Symbol("dx") # PixelSize in X direction
dy = sf.Symbol("dy") # PixelSize in Y direction
u0 = sf.Symbol("u0") # Camera center in X direction
v0 = sf.Symbol("v0") # Camera center in Y direction

u = sf.Symbol("u") # u Pixel projection of the landmark state
v = sf.Symbol("v") # v Pixel projection of the landmark state

# Ray projection of the landmark state
H3d_l = sf.V3(
    xi_l + (1/rho_l) * M_l[0] - x,
    yi_l + (1/rho_l) * M_l[1] - y,
    zi_l + (1/rho_l) * M_l[2] - z
)

# Rotate ray into camera frame W->C from global frame
H3d_l_c = sf.Rot3.from_storage([q1, q2, q3, q0]).inverse()  * H3d_l


# Pixel projection of the landmark state
H2d_l = sf.V2(
    H3d_l_c[0] / H3d_l_c[2] * f / dx + u0,
    H3d_l_c[1] / H3d_l_c[2] * f / dy + v0
)


Y_l_vector = sf.Matrix([xi_l, yi_l, zi_l, theta_l, phi_l, rho_l])

Hr = H2d_l.jacobian(S)
Hl = H2d_l.jacobian(Y_l_vector)

cg = Codegen(
    inputs=Values(S=S, f=f, dx=dx, dy=dy, u0=u0, v0=v0, xi_l=xi_l, yi_l=yi_l, zi_l=zi_l, theta_l=theta_l, phi_l=phi_l, rho_l=rho_l),
    outputs=Values(Hr=Hr, Hl=Hl), config=PythonConfig(), 
    name="landmark_measurement_jacobian"
)

cg.generate_function(output_dir=Path("gen_ekf"))


cg = Codegen(
    inputs=Values(S=S, f=f, dx=dx, dy=dy, u0=u0, v0=v0, xi_l=xi_l, yi_l=yi_l, zi_l=zi_l, theta_l=theta_l, phi_l=phi_l, rho_l=rho_l),
    outputs=Values(h=H2d_l), config=PythonConfig(), 
    name="landmark_measurement"
)

cg.generate_function(output_dir=Path("gen_ekf"))

# -------------------------------------------
# Landmark initialization from the first observation (u,v) and the camera state S
u_m = sf.Symbol("u_m") # measured pixel of the landmark
v_m = sf.Symbol("v_m") # measured pixel of the landmark
rho_m = sf.Symbol("rho_m") 

# As ray
unprojected_ray = sf.V3(
    (u_m - u0) / f,
    (v_m - v0) / f,
    1
)

# Rotated ray into the global frame C=>W
H_m_w = sf.Rot3.from_storage([q1, q2, q3, q0]) * unprojected_ray

theta_m = sf.atan2(sf.sqrt(H_m_w[0]**2 + H_m_w[1]**2), H_m_w[2]) # theta
phi_m   = sf.atan2(H_m_w[1], H_m_w[0])  # phi

# Landmark initialization from the first observation (u,v) and the camera state S
Y_l_init = sf.V6(
    x,
    y,
    z,
    theta_m,
    phi_m,
    rho_m
)


J = Y_l_init.jacobian(sf.Matrix([*S, u_m, v_m, rho_m]))

Gr = Y_l_init.jacobian(S)
Gy = Y_l_init.jacobian(sf.Matrix([u_m, v_m]))
Gs = Y_l_init.jacobian(rho_m)


cg = Codegen(
    inputs=Values(S=S, u=u_m, v=v_m, rho=rho_m, u0=u0, v0=v0, f=f, epsilon=sf.Symbol("epsilon")),
    outputs=Values(Gr=Gr,Gy=Gy,Gs=Gs), config=PythonConfig(), 
    name="landmark_initialization_jacobian"
)


cg.generate_function(output_dir=Path("gen_ekf"))

cg = Codegen(
    inputs=Values(S=S, u=u_m, v=v_m, rho=rho_m, u0=u0, v0=v0, f=f, epsilon=sf.Symbol("epsilon")),
    outputs=Values(Y_l_init=Y_l_init), config=PythonConfig(), 
    name="landmark_initialization"
)

cg.generate_function(output_dir=Path("gen_ekf"))


print(f"Jacobian of the landmark initialization w.r.t. the camera state and the first observation {J.shape}")

# -------------------------------------------
# State update equation for the camera

Qwc_rot = sf.Rot3.from_storage([q1, q2, q3, q0])

rotation_vector = (Omega_w + Gn*dt) * dt
Quat_step = sf.Rot3.from_tangent(rotation_vector)

Qwc_next_rot = Qwc_rot * Quat_step
Qwc_next = sf.V4(Qwc_next_rot.to_storage())


Snext = sf.Matrix.block_matrix([
    [Xwc + (Vwc + An*dt ) * dt],
    [Qwc_next ], # Quat = quaternion defined by the rotation vector (Omega_w  + Gn*dt) * dt
    [Vwc + An*dt],
    [Omega_w + Gn*dt]
])


F = Snext.jacobian(S)

# -------------------------------------------------------------
# Robot motion model Jacobian w.r.t. the camera state and the noise symbols

cg = Codegen(
    inputs=Values(S=S, gx_n=gx_n, gy_n=gy_n, gz_n=gz_n,dt=dt,epsilon=sf.Symbol("epsilon")),
    outputs=Values(F=F), config=PythonConfig(), 
    name="robot_state_update_jacobian"
)

cg.generate_function(output_dir=Path("gen_ekf"))

# -------------------------------------------------------------
# Robot motion model  w.r.t. the camera state and the noise symbols

cg = Codegen(
    inputs=Values(S=S, ax_n=ax_n, ay_n=ay_n, az_n=az_n, gx_n=gx_n, gy_n=gy_n, gz_n=gz_n,dt=dt,epsilon=sf.Symbol("epsilon")),
    outputs=Values(Snext=Snext), config=PythonConfig(), 
    name="robot_state_update"
)

cg.generate_function(output_dir=Path("gen_ekf"))



print(f"Jacobian of the landmark initialization w.r.t. the camera state and the first observation {J.shape}")



# -------------------------------------------

# Whole state vector is assembled from:
# [S, Y_l1, Y_l2, ... , Y_ln] where S is the camera state and Y_li is the i-th landmark state








print(Snext)
print(H2d_l)
print(Y_l_init)
#print(H.shape)