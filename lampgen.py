import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
from scipy import signal

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'cylinder_radius': 50.0,
    'cylinder_height': 210,
    'displacement_scale': 12.0,
    'num_theta_points': 800,
    'num_z_points': 200,
    
    'wave_type_1': 'triangle',  # 'sine', 'square', 'sawtooth', 'triangle'    
    'wave1_angle': -np.pi / 8 ,
    'wave1_repetitions': 16,   # Number of horizontal repetitions for wave 1

    'wave_type_2': 'sine', 
    'wave2_angle': np.pi / 4,
    'wave2_repetitions': 4,   # Number of horizontal repetitions for wave 2

    'output_filename': 'cylinder_waves.stl'
}

# ==========================================
# WAVE FUNCTIONS
# ==========================================

def sine_wave(phase):
    return np.sin(2 * np.pi * phase)

def square_wave(phase):
    return signal.square(2 * np.pi * phase)

def sawtooth_wave(phase):
    return signal.sawtooth(2 * np.pi * phase)

def triangle_wave(phase):
    return signal.sawtooth(2 * np.pi * phase, width=0.5)

WAVE_FUNCTIONS = {
    'sine': sine_wave,
    'square': square_wave,
    'sawtooth': sawtooth_wave,
    'triangle': triangle_wave,
}


def create_cylindrical_grid(cylinder_height, num_z_points, num_theta_points):
    z_vals = np.linspace(0, cylinder_height, num_z_points)
    theta_vals = np.linspace(0, 2 * np.pi, num_theta_points)
    Z_cyl, Theta_cyl = np.meshgrid(z_vals, theta_vals, indexing='ij')
    return Z_cyl, Theta_cyl

def compute_wave_interference(Z_cyl, Theta_cyl, cylinder_height, wave_params, wave_type1='sine', wave_type2='sine'):
    wave_func1 = WAVE_FUNCTIONS.get(wave_type1, sine_wave)
    wave_func2 = WAVE_FUNCTIONS.get(wave_type2, sine_wave)
    
    X_normalized = Theta_cyl / (2 * np.pi)
    Y_normalized = Z_cyl / cylinder_height
    
    angle1_rad = wave_params['wave1_angle']
    k1 = wave_params['wave1_repetitions']
    angle2_rad = wave_params['wave2_angle']
    k2 = wave_params['wave2_repetitions']
    
    freq1 = k1 / np.cos(angle1_rad)
    freq2 = k2 / np.cos(angle2_rad)
    
    kx1, ky1 = np.cos(angle1_rad), np.sin(angle1_rad)
    kx2, ky2 = np.cos(angle2_rad), np.sin(angle2_rad)
    
    phase1 = freq1 * (X_normalized * kx1 + Y_normalized * ky1)
    phase2 = freq2 * (X_normalized * kx2 + Y_normalized * ky2)
    
    wave1 = wave_func1(phase1)
    wave2 = wave_func2(phase2)
    
    combined_waves = wave1 + wave2
    
    z_min = combined_waves.min()
    z_max = combined_waves.max()
    
    if z_max - z_min == 0:
        height_data = np.zeros_like(combined_waves)
    else:
        height_data = (combined_waves - z_min) / (z_max - z_min)
    
    return height_data

def convert_to_cartesian(Z_cyl, Theta_cyl, height_data, base_radius, displacement_scale):
    R_cyl = base_radius + (height_data * displacement_scale)
    X_3d = R_cyl * np.cos(Theta_cyl)
    Y_3d = R_cyl * np.sin(Theta_cyl)
    Z_3d = Z_cyl
    return X_3d, Y_3d, Z_3d

def export_to_stl(X_3d, Y_3d, Z_3d, filename):
    print(f"Generating STL mesh for {filename}...")
    
    num_rows, num_cols = X_3d.shape
    num_faces = 2 * (num_rows - 1) * (num_cols - 1)
    cylinder_mesh = mesh.Mesh(np.zeros(num_faces, dtype=mesh.Mesh.dtype))
    
    face_index = 0
    for i in range(num_rows - 1):
        for j in range(num_cols - 1):
            p1 = [X_3d[i, j], Y_3d[i, j], Z_3d[i, j]]
            p2 = [X_3d[i, j+1], Y_3d[i, j+1], Z_3d[i, j+1]]
            p3 = [X_3d[i+1, j+1], Y_3d[i+1, j+1], Z_3d[i+1, j+1]]
            p4 = [X_3d[i+1, j], Y_3d[i+1, j], Z_3d[i+1, j]]
            
            cylinder_mesh.vectors[face_index] = np.array([p1, p2, p3])
            face_index += 1
            cylinder_mesh.vectors[face_index] = np.array([p1, p3, p4])
            face_index += 1
    
    cylinder_mesh.save(filename)


def generate_cylinder_with_waves(config):
    
    Z_cyl, Theta_cyl = create_cylindrical_grid(
        config['cylinder_height'],
        config['num_z_points'],
        config['num_theta_points']
    )
    
    wave_params = {
        'wave1_angle': config['wave1_angle'],
        'wave1_repetitions': config['wave1_repetitions'],
        'wave2_angle': config['wave2_angle'],
        'wave2_repetitions': config['wave2_repetitions'],
    }
    height_data = compute_wave_interference(
        Z_cyl, Theta_cyl, config['cylinder_height'],
        wave_params, config['wave_type_1'], config['wave_type_2']
    )
    
    X_3d, Y_3d, Z_3d = convert_to_cartesian(
        Z_cyl, Theta_cyl, height_data,
        config['cylinder_radius'],
        config['displacement_scale']
    )
        
    export_to_stl(X_3d, Y_3d, Z_3d, config['output_filename'])
    return X_3d, Y_3d, Z_3d

if __name__ == "__main__":
    generate_cylinder_with_waves(CONFIG)
    
