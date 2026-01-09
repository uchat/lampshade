# lampshade
A python script that generate lamp shades based on wave wave interference patterns.  Currently support 4 wave forms (sine, triangle, saw tooth, square).  It will generate the STL file that you can directly import into your slicer and print it in vase mode.  I found that the line width between 0.6 and 0.8 gave good results.

You can try different configurations and see how the shade looks like.  The parameters are self-explanatory except num_theta_points and num_z_points, which defines the resolution of the STL.

```
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
```

While the script has MIT license, any STL file you generate with it is yours.
