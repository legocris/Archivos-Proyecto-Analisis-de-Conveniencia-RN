import os
import numpy as np
from sklearn.model_selection import train_test_split

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def save_to_bin(name, X, y):
    """
    Saves datasets for CUDA processing.

    Format:
    - Content: Transposed (Features x Samples) -> For CUDA Coalesced Access.
    - Filename: Features x Samples.bin
    """

    # 1. Ensure y is (Samples, 1)
    y = y.reshape(-1, 1)

    # 2. Split Data (80% Train, 20% Test)
    # Shapes here are still (Samples, Features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Convert to float32 (Standard for CUDA/GPU processing)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # 4. Define Base Paths
    base_path = f"{name}/Bin"
    paths = {
        "train_x": f"{base_path}/Train/X",
        "train_y": f"{base_path}/Train/Y",
        "test_x":  f"{base_path}/Test/X",
        "test_y":  f"{base_path}/Test/Y",
        "Reg":     f"{name}/Reg",  # Folder for logs
    }

    # 5. Create Directories
    for p in paths.values():
        ensure_dir(p)

    # 6. Save Files
    # Formula: Filename = N_Features x M_Samples
    #          Content  = Transposed Matrix (.T)

    # --- Train X ---
    fn = f"{paths['train_x']}/{X_train.shape[1]}x{X_train.shape[0]}.bin"
    X_train.T.tofile(fn)

    # --- Train Y ---
    fn = f"{paths['train_y']}/{y_train.shape[1]}x{y_train.shape[0]}.bin"
    y_train.T.tofile(fn)

    # --- Test X ---
    fn = f"{paths['test_x']}/{X_test.shape[1]}x{X_test.shape[0]}.bin"
    X_test.T.tofile(fn)

    # --- Test Y ---
    fn = f"{paths['test_y']}/{y_test.shape[1]}x{y_test.shape[0]}.bin"
    y_test.T.tofile(fn)

    print(f"[{name}] Exported.")
    print(f"   Train File: {X_train.shape[1]}x{X_train.shape[0]}.bin (Features x Samples)")

# --- Problemas tipo Checkerboard: muchas regiones disconexas / composicionales ---
# Hipotesis: la red profunda (varias capas estrechas) supera a la angosta
# (1 capa ancha) a igual numero total de neuronas, como en Tablero de Damas.

def make_parity4d(n_samples=2000, seed=42):
    """
    Paridad 4D: generalizacion directa del XOR / Checkerboard a 4 dimensiones.
    y = (x1>0) XOR (x2>0) XOR (x3>0) XOR (x4>0).
    Teorema de Hastad: shallow necesita tamaño exponencial para paridad.
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, size=(n_samples, 4))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0) ^ (X[:, 2] > 0) ^ (X[:, 3] > 0)).astype(int)
    return X, y

def make_checkerboard6x6(n_samples=2000, seed=42):
    """
    Checkerboard 6x6 en [-3,3]x[-3,3], celda de 1x1.
    36 regiones alternadas. Version mas densa que la 4x4 de Gemini.
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-3, 3, size=(n_samples, 2))
    ix = np.floor(X[:, 0] + 3).astype(int)
    iy = np.floor(X[:, 1] + 3).astype(int)
    y = ((ix % 2) ^ (iy % 2)).astype(int)
    return X, y

def make_checker_rotated(n_samples=2000, seed=42):
    """
    Checkerboard 4x4 rotado 45 grados.
    Misma topologia que el checker clasico pero ejes oblicuos:
    obliga a componer rotacion + patron XOR, lo que favorece profundidad.
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-4, 4, size=(n_samples, 2))
    theta = np.pi / 4.0
    c, s = np.cos(theta), np.sin(theta)
    Xr = X[:, 0] * c - X[:, 1] * s
    Yr = X[:, 0] * s + X[:, 1] * c
    ix = np.floor(Xr).astype(int)
    iy = np.floor(Yr).astype(int)
    y = (((ix % 2) + 2) % 2 ^ (((iy % 2) + 2) % 2)).astype(int)
    return X, y

def make_bullseye5(n_samples=2000, seed=42):
    """
    Diana de 5 anillos alternantes 0,1,0,1,0.
    Radios: [0.7, 1.4, 2.1, 2.8], exterior hasta 3.5.
    Muchas fronteras cerradas concentricas disconexas.
    """
    rng = np.random.RandomState(seed)
    n = n_samples
    r = np.sqrt(rng.rand(n)) * 3.5
    theta = rng.uniform(0, 2 * np.pi, n)
    x = r * np.cos(theta) + rng.normal(0, 0.05, n)
    y = r * np.sin(theta) + rng.normal(0, 0.05, n)
    X = np.column_stack((x, y))
    rr = np.sqrt(x ** 2 + y ** 2)
    ring = np.digitize(rr, bins=[0.7, 1.4, 2.1, 2.8])  # 0..4
    y = (ring % 2).astype(int)  # 0,1,0,1,0
    return X, y

def make_sine_highfreq(n_samples=2000, seed=42):
    """
    Checker sinusoidal de alta frecuencia:
    y = (sin(4*x)>0) XOR (sin(4*y)>0).
    Frontera ondulada periodica con muchas alternancias.
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-4, 4, size=(n_samples, 2))
    sx = np.sin(4.0 * X[:, 0]) > 0
    sy = np.sin(4.0 * X[:, 1]) > 0
    y = np.logical_xor(sx, sy).astype(int)
    return X, y

def make_hierarchical_xor(n_samples=2000, seed=42):
    """
    XOR jerarquico (funcion composicional pura, ejemplo Mhaskar/Poggio):
    a = (x1*x2 > 0), b = (x3*x4 > 0), y = a XOR b.
    Estructura de arbol: h3(h1(x1,x2), h2(x3,x4)).
    La red profunda calza con el arbol; la shallow no puede reutilizar.
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, size=(n_samples, 4))
    a = (X[:, 0] * X[:, 1] > 0)
    b = (X[:, 2] * X[:, 3] > 0)
    y = np.logical_xor(a, b).astype(int)
    return X, y

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("Generating DeepWins datasets for CUDA (Features x Samples)...\n")

    X, y = make_parity4d(n_samples=2000)
    save_to_bin("Problem1_Parity4D", X, y)

    X, y = make_checkerboard6x6(n_samples=2000)
    save_to_bin("Problem2_Checkerboard6x6", X, y)

    X, y = make_checker_rotated(n_samples=2000)
    save_to_bin("Problem3_CheckerRotated", X, y)

    X, y = make_bullseye5(n_samples=2000)
    save_to_bin("Problem4_Bullseye5", X, y)

    X, y = make_sine_highfreq(n_samples=2000)
    save_to_bin("Problem5_SineHighFreq", X, y)

    X, y = make_hierarchical_xor(n_samples=2000)
    save_to_bin("Problem6_HierarchicalXOR", X, y)

    print("\nAll datasets generated.")
