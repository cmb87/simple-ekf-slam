import symforce.symbolic as sf
from symforce.codegen import Codegen, PythonConfig, CppConfig
from symforce.values import Values
from pathlib import Path


def generate_ekf():
    # ------------------------------------------
    # Robot State X = [x, y, theta]
    x = sf.Symbol("x")
    y = sf.Symbol("y")
    theta = sf.Symbol("theta")
    X = sf.V3(x, y, theta)

    # ------------------------------------------
    # Control input U = [v, omega]
    v = sf.Symbol("v")
    omega = sf.Symbol("omega")
    U = sf.V2(v, omega)

    # ------------------------------------------
    # Time step
    dt = sf.Symbol("dt")

    # ------------------------------------------
    # Landmark state L = [xi, yi, alpha, rho] (Inverse Depth)
    # xi, yi: Robot position when landmark was first observed
    # alpha: Bearing angle of landmark from robot position (xi, yi)
    # rho: Inverse depth (1 / distance) from robot position (xi, yi)
    xi = sf.Symbol("xi")
    yi = sf.Symbol("yi")
    alpha = sf.Symbol("alpha")
    rho = sf.Symbol("rho")
    L = sf.V4(xi, yi, alpha, rho)

    # ------------------------------------------
    # Process model:
    # A simple differential drive motion model
    X_next = sf.V3(
        x + sf.cos(theta) * v * dt,
        y + sf.sin(theta) * v * dt,
        theta + omega * dt
    )

    # Process model Jacobians:
    F = X_next.jacobian(X)
    W = X_next.jacobian(U)

    # ------------------------------------------
    # Measurement model (Bearing only, inverse depth)
    # Global position of landmark is:
    # x_L = xi + (1 / rho) * cos(alpha)
    # y_L = yi + (1 / rho) * sin(alpha)
    # To avoid division by zero when rho -> 0, we work with scaling by rho:
    # Relative vector scaled by rho:
    v_x = rho * (xi - x) + sf.cos(alpha)
    v_y = rho * (yi - y) + sf.sin(alpha)

    # Predicted bearing:
    z_pred = sf.V1(sf.atan2(v_y, v_x) - theta)

    # Measurement Jacobians:
    H_X = z_pred.jacobian(X)
    H_L = z_pred.jacobian(L)

    # ------------------------------------------
    # Generate Process Model function
    inputs_process = Values(X=X, U=U, dt=dt)
    outputs_process = Values(X_next=X_next, F=F, W=W)

    codegen_process = Codegen(
        inputs=inputs_process,
        outputs=outputs_process,
        config=PythonConfig(),
        name="process_model"
    )
    
    output_dir = Path("gen_ekf")
    codegen_process.generate_function(output_dir=output_dir)

    # Generate Measurement Model function
    inputs_meas = Values(X=X, L=L)
    outputs_meas = Values(z_pred=z_pred, H_X=H_X, H_L=H_L)

    codegen_meas = Codegen(
        inputs=inputs_meas,
        outputs=outputs_meas,
        config=PythonConfig(),
        name="measurement_model"
    )
    codegen_meas.generate_function(output_dir=output_dir)


    
     # ... (define your symbols X, U, L, X_next, z_pred, and Jacobians as before)
    
     # Generate C++ Process Model function

    codegen_process = Codegen(
        inputs=inputs_process,
        outputs=outputs_process,
        config=CppConfig(),  # <-- Use CppConfig instead of PythonConfig
        name="process_model"
    )
    codegen_process.generate_function(output_dir=Path("gen_ekf_cpp"))




    print("Successfully generated SymForce process and measurement models.")

if __name__ == "__main__":
    generate_ekf()
