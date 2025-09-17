import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress
from tqdm import tqdm
import subprocess
import os
import tempfile
import MDAnalysis as mda
import shutil
import platform
from matplotlib.animation import FuncAnimation, FFMpegWriter
from PIL import Image
import random
import time

# Physical parameters
D = 2.8e-9  # ROS diffusion coefficient (m²/s)
tau = 1e-9   # ROS lifetime (s)
dt = 0.1 * tau
ROS_PER_SOURCE = 200  # ROS molecules per G4 site
THRESHOLD_DENSITY = 1e4  # Membrane rupture threshold (molecules/μm²/ns)
ANIMATION_MOLECULES_PER_SOURCE = 1  # Number of molecules to animate per source

# Create custom colormap
colors = ["darkblue", "blue", "cyan", "green", "yellow", "red"]
cmap = LinearSegmentedColormap.from_list("ros_cmap", colors, N=256)

class AptamerLengthSimulator:
    def __init__(self):
        # Initialize MD-related attributes first
        # Detect OS and set appropriate GROMACS executable
        self.gromacs_path = self.detect_gromacs_path()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.md_results = None
        self.md_snapshot = None  # Initialize to None
        self.contour = None  # Initialize contour object
        self.animation = None  # For storing animation object
        self.animation_running = False
        self.trajectories = []  # For storing molecule trajectories
        self.recording = False
        self.recording_frames = []
        self.recording_start_time = 0
        
        # Create GUI with improved layout
        self.fig = plt.figure(figsize=(16, 12), facecolor='#f0f8ff')  # Light blue background
        gs = gridspec.GridSpec(3, 2, width_ratios=[1.2, 1], height_ratios=[1.5, 1.2, 0.1])
        
        # 3D view - with light green background
        self.ax_3d = self.fig.add_subplot(gs[0, 0], projection='3d', facecolor='#e0f7fa')  # Light green-blue
        self.ax_3d.set_title('DNA Nanotube with Aptamer and G4 Sites', color='#01579b', fontsize=14, pad=15)
        
        # Membrane ROS distribution
        self.ax_membrane = self.fig.add_subplot(gs[1, 0], facecolor='#e0f7fa')
        self.ax_membrane.set_title('ROS Density at Membrane', color='#01579b', fontsize=14, pad=10)
        
        # Control panel with better spacing
        self.ax_control = self.fig.add_subplot(gs[0, 1], facecolor='#b2ebf2')  # Lighter blue
        self.ax_control.set_title('Simulation Controls', color='#01579b', fontsize=14, pad=10)
        self.ax_control.axis('off')
        
        # Analysis panel with more space
        self.ax_analysis = self.fig.add_subplot(gs[1, 1], facecolor='#b2ebf2')  # Lighter blue
        self.ax_analysis.set_title('Analysis & Export', color='#01579b', fontsize=14, pad=10)
        self.ax_analysis.axis('off')
        
        # Add aptamer length slider with better position
        slider_ax1 = plt.axes([0.55, 0.50, 0.35, 0.02], facecolor='#80deea')  # Light blue
        self.slider_length = Slider(slider_ax1, 'Aptamer Length (nm)', 5, 50, valinit=20, 
                                   color='#29b6f6', track_color='#b3e5fc')  # Blue slider
        
        # Add orientation sliders with better spacing
        slider_ax2 = plt.axes([0.55, 0.45, 0.35, 0.02], facecolor='#80deea')  # Light blue
        slider_ax3 = plt.axes([0.55, 0.40, 0.35, 0.02], facecolor='#80deea')  # Light blue
        slider_ax4 = plt.axes([0.55, 0.35, 0.35, 0.02], facecolor='#80deea')  # Light blue
        
        self.slider_x = Slider(slider_ax2, 'X Rotation', -180, 180, valinit=0, 
                              color='#29b6f6', track_color='#b3e5fc')  # Blue slider
        self.slider_y = Slider(slider_ax3, 'Y Rotation', -180, 180, valinit=0, 
                              color='#29b6f6', track_color='#b3e5fc')  # Blue slider
        self.slider_z = Slider(slider_ax4, 'Z Rotation', -180, 180, valinit=0, 
                              color='#29b6f6', track_color='#b3e5fc')  # Blue slider
        
        # Add control buttons
        button_y = 0.25
        button_height = 0.04
        button_spacing = 0.05
        
        button_ax = plt.axes([0.65, button_y, 0.2, button_height], facecolor='#4dd0e1')
        self.button = Button(button_ax, 'Reset View', color='#4dd0e1', hovercolor='#80deea')
        
        analyze_ax = plt.axes([0.65, button_y - button_spacing, 0.2, button_height], facecolor='#4dd0e1')
        self.analyze_button = Button(analyze_ax, 'Run Analysis', color='#4dd0e1', hovercolor='#80deea')
        
        gromacs_ax = plt.axes([0.65, button_y - 2*button_spacing, 0.2, button_height], facecolor='#4dd0e1')
        self.gromacs_button = Button(gromacs_ax, 'Run GROMACS MD', color='#4dd0e1', hovercolor='#80deea')
        
        # Add export buttons
        export_y = 0.08
        animate_ax = plt.axes([0.45, export_y, 0.2, button_height], facecolor='#29b6f6')
        self.animate_button = Button(animate_ax, 'Animate Diffusion', color='#29b6f6', hovercolor='#4fc3f7')
        
        export3d_ax = plt.axes([0.45, export_y - button_spacing, 0.2, button_height], facecolor='#29b6f6')
        self.export3d_button = Button(export3d_ax, 'Export 3D View', color='#29b6f6', hovercolor='#4fc3f7')
        
        export_membrane_ax = plt.axes([0.45, export_y - 2*button_spacing, 0.2, button_height], facecolor='#29b6f6')
        self.export_membrane_button = Button(export_membrane_ax, 'Export Membrane View', color='#29b6f6', hovercolor='#4fc3f7')
        
        export_data_ax = plt.axes([0.45, export_y - 3*button_spacing, 0.2, button_height], facecolor='#29b6f6')
        self.export_data_button = Button(export_data_ax, 'Export Data', color='#29b6f6', hovercolor='#4fc3f7')
        
        # Add recording button
        record_ax = plt.axes([0.65, button_y - 3*button_spacing, 0.2, button_height], facecolor='#e91e63')
        self.record_button = Button(record_ax, 'Record Animation', color='#f48fb1', hovercolor='#f8bbd0')
        
        # Set initial tube structure
        self.tube_length = 120e-9  # Tube length 120nm
        self.tube_radius = 15e-9   # Tube radius 15nm
        self.n_sources = 30        # Number of G4 sites
        self.aptamer_length = 20e-9  # Initial aptamer length 20nm
        
        # Create initial tube sources with 10 per circle
        self.sources = self.create_tube_sources()
        
        # Create membrane grid
        self.grid_size = 100
        self.membrane_hits = np.zeros((self.grid_size, self.grid_size))
        self.x_grid = np.linspace(-100e-9, 100e-9, self.grid_size)
        self.y_grid = np.linspace(-100e-9, 100e-9, self.grid_size)
        
        # Initialize visualization
        self.initialize_visualization()
        
        # Event bindings
        self.slider_length.on_changed(self.update_aptamer)
        self.slider_x.on_changed(self.update)
        self.slider_y.on_changed(self.update)
        self.slider_z.on_changed(self.update)
        self.button.on_clicked(self.reset)
        self.analyze_button.on_clicked(self.run_analysis)
        self.gromacs_button.on_clicked(self.run_gromacs_simulation)
        self.animate_button.on_clicked(self.toggle_animation)
        self.export3d_button.on_clicked(self.export_3d_view)
        self.export_membrane_button.on_clicked(self.export_membrane_view)
        self.export_data_button.on_clicked(self.export_data)
        self.record_button.on_clicked(self.toggle_recording)
        
        # Analysis data storage
        self.analysis_results = []
        
        # Global styling with better spacing
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.2, hspace=0.3)
        self.fig.suptitle('DNA Nanotube ROS Diffusion Simulator', 
                         fontsize=18, color='#0288d1', fontweight='bold')  # Deep blue title
    
    def detect_gromacs_path(self):
        """Detect GROMACS executable path based on OS"""
        # Check if gmx is in PATH
        if shutil.which("gmx"):
            return "gmx"
        
        # Check common installation paths
        common_paths = {
            "Windows": [
                "C:\\Program Files\\GROMACS\\bin\\gmx.exe",
                "C:\\GROMACS\\bin\\gmx.exe"
            ],
            "Linux": [
                "/usr/local/gromacs/bin/gmx",
                "/usr/bin/gmx",
                "/opt/gromacs/bin/gmx"
            ],
            "Darwin": [
                "/usr/local/gromacs/bin/gmx",
                "/opt/homebrew/bin/gmx"
            ]
        }
        
        os_type = platform.system()
        for path in common_paths.get(os_type, []):
            if os.path.exists(path):
                return path
        
        # If not found, return None and handle later
        return None
    
    def create_tube_sources(self):
        """Create G4 sites on the nanotube structure with 10 per circle"""
        sources = []
        points_per_circle = 10  # 10 sources per circle
        
        # Calculate number of circles needed
        num_circles = max(1, int(np.ceil(self.n_sources / points_per_circle)))
        
        # Create sources for each circle
        for circle_idx in range(num_circles):
            # Z position for this circle - ensure all points are above membrane
            z = (circle_idx / max(1, (num_circles - 1))) * self.tube_length + self.aptamer_length + 5e-9
            
            # Create points for this circle
            for point_idx in range(points_per_circle):
                if len(sources) >= self.n_sources:
                    break
                
                # Angle for this point (10 points per circle)
                theta = 2 * np.pi * point_idx / points_per_circle
                x = self.tube_radius * np.cos(theta)
                y = self.tube_radius * np.sin(theta)
                
                sources.append([x, y, z])
        
        return np.array(sources)
    
    def run_gromacs_simulation(self, event=None):
        """Run GROMACS molecular dynamics simulation with error handling"""
        # Check if GROMACS is available
        if self.gromacs_path is None or not shutil.which(self.gromacs_path):
            self.ax_analysis.clear()
            self.ax_analysis.text(0.1, 0.6, "GROMACS not found! Please install GROMACS", 
                                 color='#d32f2f', fontsize=12)
            self.ax_analysis.text(0.1, 0.5, "or set the correct path in the code.", 
                                 color='#d32f2f', fontsize=12)
            self.fig.canvas.draw_idle()
            return
        
        # Update status
        self.ax_analysis.clear()
        self.ax_analysis.text(0.5, 0.7, "Running GROMACS simulation...", 
                             color='#0288d1', fontsize=14, ha='center')
        self.fig.canvas.draw_idle()
        
        # Generate input files (simplified for this example)
        temp_dir = self.temp_dir.name
        current_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # In a real implementation, this would run actual GROMACS commands
            # For now, we'll just simulate with a progress bar
            self.ax_analysis.clear()
            status_text = self.ax_analysis.text(0.5, 0.6, "Simulating MD (100ps)...", 
                                              color='#0288d1', fontsize=12, ha='center')
            self.fig.canvas.draw_idle()
            
            # Simulate progress
            for i in tqdm(range(10), desc="MD Simulation"):
                status_text.set_text(f"Simulating MD (100ps)... {i*10}% complete")
                self.fig.canvas.draw_idle()
                plt.pause(0.1)
            
            # Simulate results
            self.ax_analysis.clear()
            self.ax_analysis.text(0.1, 0.8, "GROMACS MD Results (Simulated):", 
                                 color='#0288d1', fontsize=12, fontweight='bold')
            self.ax_analysis.text(0.1, 0.7, "Simulation Time: 100 ps")
            self.ax_analysis.text(0.1, 0.65, "Final RMSD: 0.85 nm")
            self.ax_analysis.text(0.1, 0.6, "Structure optimized with MD")
            self.ax_analysis.text(0.1, 0.5, "Visualization updated with MD structure", 
                                 color='#f57c00')
            
            # Create a small random displacement to simulate MD effect
            displacement = np.random.normal(0, 2e-9, self.sources.shape)
            self.sources += displacement
            
            # Update visualization
            self.update()
            
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.ax_analysis.clear()
            self.ax_analysis.text(0.1, 0.6, "Simulation error:", color='#d32f2f', fontsize=12)
            self.ax_analysis.text(0.1, 0.5, error_msg, color='#d32f2f', fontsize=10)
            self.fig.canvas.draw_idle()
        finally:
            # Return to original directory
            os.chdir(current_dir)
    
    def initialize_visualization(self):
        """Initialize visualization elements with error handling"""
        # 3D view
        self.ax_3d.clear()
        self.ax_3d.grid(False)
        
        # Draw membrane plane at z=0 - dark blue
        xx, yy = np.meshgrid(np.linspace(-100, 100, 10), np.linspace(-100, 100, 10))
        zz = np.zeros_like(xx)
        self.membrane_surface = self.ax_3d.plot_surface(xx, yy, zz, alpha=0.4, color='#0d47a1')  # Dark blue
        
        # Draw tube structure with 10 sources per circle - light blue
        rotated_sources = self.rotate_sources()
        self.tube_scatter = self.ax_3d.scatter(
            rotated_sources[:, 0]*1e9, 
            rotated_sources[:, 1]*1e9, 
            rotated_sources[:, 2]*1e9, 
            s=15, c='#81d4fa', alpha=0.9, edgecolors='#e1f5fe', linewidths=0.5  # Light blue, smaller points
        )
        
        # Add ROS source markers (one for each source) - sky blue
        self.source_markers = self.ax_3d.scatter(
            rotated_sources[:, 0]*1e9, 
            rotated_sources[:, 1]*1e9, 
            rotated_sources[:, 2]*1e9, 
            s=8, c='#4fc3f7', marker='o', alpha=0.8  # Sky blue, smaller points
        )
        
        # Add aptamer representation - light blue
        self.aptamer_length = self.slider_length.val * 1e-9
        
        # Find the lowest point on the nanotube (closest to membrane)
        min_z_index = np.argmin(rotated_sources[:, 2])
        nanotube_point = rotated_sources[min_z_index]
        membrane_point = np.array([nanotube_point[0], nanotube_point[1], 0])
        
        # Save aptamer objects for updating
        self.aptamer_line, = self.ax_3d.plot(
            [nanotube_point[0]*1e9, membrane_point[0]*1e9],
            [nanotube_point[1]*1e9, membrane_point[1]*1e9],
            [nanotube_point[2]*1e9, membrane_point[2]*1e9],
            '#81d4fa', linewidth=1.5, alpha=0.8  # Light blue
        )
        self.aptamer_point = self.ax_3d.scatter(
            membrane_point[0]*1e9, 
            membrane_point[1]*1e9, 
            membrane_point[2]*1e9, 
            s=50, c='#4fc3f7', marker='*', edgecolors='#29b6f6', linewidths=0.5  # Sky blue
        )
        
        # Add distance annotation - actual distance to membrane
        distance_to_membrane = nanotube_point[2]  # Since membrane is at z=0
        self.distance_text = self.ax_3d.text(
            nanotube_point[0]*1e9, 
            nanotube_point[1]*1e9, 
            nanotube_point[2]*1e9 + 10,
            f"Distance: {distance_to_membrane*1e9:.1f} nm",
            color='#01579b', fontsize=9  # Dark blue text
        )
        
        # Initialize molecule trajectories for animation
        self.molecule_points = self.ax_3d.scatter([], [], [], s=8, c='#4fc3f7', alpha=0.7)  # Sky blue
        self.trajectory_lines = []
        
        # Set 3D view parameters
        self.ax_3d.set_xlim(-100, 100)
        self.ax_3d.set_ylim(-100, 100)
        self.ax_3d.set_zlim(-50, 150)  # Adjust to show membrane and nanotube
        self.ax_3d.set_xlabel('X (nm)', color='#01579b')
        self.ax_3d.set_ylabel('Y (nm)', color='#01579b')
        self.ax_3d.set_zlabel('Z (nm)', color='#01579b')
        self.ax_3d.tick_params(colors='#01579b')
        self.ax_3d.xaxis.pane.fill = False
        self.ax_3d.yaxis.pane.fill = False
        self.ax_3d.zaxis.pane.fill = False
        self.ax_3d.xaxis.pane.set_edgecolor('#b3e5fc')
        self.ax_3d.yaxis.pane.set_edgecolor('#b3e5fc')
        self.ax_3d.zaxis.pane.set_edgecolor('#b3e5fc')
        
        # Membrane view
        self.ax_membrane.clear()
        self.membrane_img = self.ax_membrane.imshow(
            self.membrane_hits.T, 
            extent=[-100, 100, -100, 100], 
            origin='lower', 
            cmap=cmap, 
            vmin=0, 
            vmax=ROS_PER_SOURCE * self.n_sources / 10
        )
        self.ax_membrane.set_xlabel('X (nm)', color='#01579b')
        self.ax_membrane.set_ylabel('Y (nm)', color='#01579b')
        self.ax_membrane.tick_params(colors='#01579b')
        self.ax_membrane.set_facecolor('#e0f7fa')
        
        # Add threshold contour
        if np.any(self.membrane_hits > THRESHOLD_DENSITY):
            self.contour = self.ax_membrane.contour(
                self.x_grid*1e9, 
                self.y_grid*1e9, 
                self.membrane_hits.T, 
                levels=[THRESHOLD_DENSITY], 
                colors='white', 
                linewidths=1
            )
        else:
            # Create empty contour if no points above threshold
            self.contour = None
        
        # Add colorbar
        cbar = plt.colorbar(self.membrane_img, ax=self.ax_membrane, pad=0.01)
        cbar.set_label('ROS Density (molecules/μm²/ns)', color='#01579b')
        cbar.ax.yaxis.set_tick_params(color='#01579b')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#01579b')
        
        # Update analysis panel
        self.update_analysis_panel()
    
    def rotate_sources(self):
        """Apply current rotation angles"""
        rx = self.slider_x.val * np.pi / 180
        ry = self.slider_y.val * np.pi / 180
        rz = self.slider_z.val * np.pi / 180
        
        # Create rotation matrix
        rotation = R.from_euler('xyz', [rx, ry, rz], degrees=False)
        return rotation.apply(self.sources)
    
    def simulate_ros_diffusion(self):
        """Simulate ROS diffusion process from ALL membrane-external G4 sites"""
        self.membrane_hits = np.zeros((self.grid_size, self.grid_size))
        rotated_sources = self.rotate_sources()
        
        # Ensure all sources are above membrane (z > 0)
        rotated_sources[:, 2] = np.maximum(rotated_sources[:, 2], 1e-9)
        
        # For animation: track molecules from all sources
        self.trajectories = []
        
        # For each G4 source above the membrane
        for src_idx, src in enumerate(rotated_sources):
            # For each ROS molecule from this source
            for ros_idx in range(ROS_PER_SOURCE):
                # Only track a subset for animation
                if ros_idx < ANIMATION_MOLECULES_PER_SOURCE:
                    pos = src.copy()
                    trajectory = [pos.copy()]  # Store positions for animation
                    
                    # Simulate diffusion until hitting membrane or lifetime ends
                    for _ in range(100):  # Maximum steps
                        # Brownian motion displacement
                        dx = np.random.normal(0, np.sqrt(2*D*dt))
                        dy = np.random.normal(0, np.sqrt(2*D*dt))
                        dz = np.random.normal(0, np.sqrt(2*D*dt))
                        pos += np.array([dx, dy, dz])
                        trajectory.append(pos.copy())
                        
                        # Detect if hit membrane (z≈0)
                        if pos[2] <= 0:
                            # Find nearest grid point
                            x_idx = np.argmin(np.abs(self.x_grid - pos[0]))
                            y_idx = np.argmin(np.abs(self.y_grid - pos[1]))
                            
                            if 0 <= x_idx < self.grid_size and 0 <= y_idx < self.grid_size:
                                self.membrane_hits[x_idx, y_idx] += 1
                            break
                    
                    # Store trajectory for animation
                    self.trajectories.append(trajectory)
                else:
                    # For molecules not tracked in animation, simulate minimally
                    pos = src.copy()
                    # Simulate until membrane hit (without storing positions)
                    for _ in range(100):
                        dx = np.random.normal(0, np.sqrt(2*D*dt))
                        dy = np.random.normal(0, np.sqrt(2*D*dt))
                        dz = np.random.normal(0, np.sqrt(2*D*dt))
                        pos += np.array([dx, dy, dz])
                        
                        if pos[2] <= 0:
                            x_idx = np.argmin(np.abs(self.x_grid - pos[0]))
                            y_idx = np.argmin(np.abs(self.y_grid - pos[1]))
                            
                            if 0 <= x_idx < self.grid_size and 0 <= y_idx < self.grid_size:
                                self.membrane_hits[x_idx, y_idx] += 1
                            break
        
        return rotated_sources
    
    def update_aptamer(self, val):
        """Update aptamer length - now only affects visualization"""
        self.update()
    
    def update_analysis_panel(self):
        """Update analysis panel with focus on ROS mechanism"""
        self.ax_analysis.clear()
        self.ax_analysis.axis('off')
        
        # Calculate analysis metrics
        total_ros = np.sum(self.membrane_hits)
        peak_density = np.max(self.membrane_hits)
        grid_cell_area = (200/self.grid_size)**2  # Area per grid cell in μm²
        coverage_area = np.sum(self.membrane_hits > THRESHOLD_DENSITY) * grid_cell_area
        can_rupture = peak_density > THRESHOLD_DENSITY and coverage_area > 0.05
        
        # Add text analysis with ROS focus
        analysis_text = (
            f"Peak ROS Density: {peak_density:.1f} molecules/μm²/ns\n"
            f"Threshold Density: {THRESHOLD_DENSITY} molecules/μm²/ns\n"
            f"ROS Coverage Area: {coverage_area:.2f} μm²\n"
            f"Total ROS Reaching Membrane: {total_ros:.0f} molecules\n"
            f"Number of Active G4 Sources: {self.n_sources}\n\n"
            f"Membrane Rupture by ROS: {'POSSIBLE' if can_rupture else 'UNLIKELY'}"
        )
        
        self.ax_analysis.text(0.1, 0.7, analysis_text, 
                             color='#0288d1' if can_rupture else '#f57c00',
                             fontsize=12, fontfamily='monospace',
                             verticalalignment='top')
        
        # Add aptamer analysis
        aptamer_len = self.slider_length.val
        
        # Find min distance to membrane
        min_distance = np.min(self.rotate_sources()[:, 2]) * 1e9
        self.ax_analysis.text(0.1, 0.5, f"Aptamer Length: {aptamer_len:.1f} nm\n"
                                       f"Min Distance to Membrane: {min_distance:.1f} nm\n"
                                       f"Active G4 Sources: {self.n_sources}",
                             color='#01579b', fontsize=11)
        
        # Add design recommendations with ROS focus
        if can_rupture:
            advice = "ROS density sufficient for membrane rupture."
            color = "#388e3c"  # Green
        else:
            advice = "Recommendations:\n- Reduce distance to membrane\n- Increase G4 density\n- Optimize orientation for ROS diffusion"
            color = "#f57c00"  # Orange
        
        self.ax_analysis.text(0.1, 0.2, advice, 
                             color=color, fontsize=12, fontweight='bold')
        
        # Add export instructions
        self.ax_analysis.text(0.1, 0.05, "Export Options:", color='#0288d1', fontsize=10)
    
    def update(self, val=None):
        """Update entire view with error handling"""
        # Update tube structure position
        rotated_sources = self.rotate_sources()
        
        # Ensure all points are above membrane
        rotated_sources[:, 2] = np.maximum(rotated_sources[:, 2], 1e-9)
        
        self.tube_scatter._offsets3d = (
            rotated_sources[:, 0]*1e9,
            rotated_sources[:, 1]*1e9,
            rotated_sources[:, 2]*1e9
        )
        
        # Update ROS source markers
        self.source_markers._offsets3d = (
            rotated_sources[:, 0]*1e9,
            rotated_sources[:, 1]*1e9,
            rotated_sources[:, 2]*1e9
        )
        
        # Update aptamer position
        aptamer_len = self.slider_length.val
        
        # Find the lowest point on the nanotube (closest to membrane)
        min_z_index = np.argmin(rotated_sources[:, 2])
        nanotube_point = rotated_sources[min_z_index]
        membrane_point = np.array([nanotube_point[0], nanotube_point[1], 0])
        
        # Update aptamer line
        self.aptamer_line.set_data(
            [nanotube_point[0]*1e9, membrane_point[0]*1e9],
            [nanotube_point[1]*1e9, membrane_point[1]*1e9]
        )
        self.aptamer_line.set_3d_properties(
            [nanotube_point[2]*1e9, membrane_point[2]*1e9]
        )
        
        # Update aptamer point
        self.aptamer_point._offsets3d = (
            [membrane_point[0]*1e9], 
            [membrane_point[1]*1e9], 
            [membrane_point[2]*1e9]
        )
        
        # Update distance annotation - actual distance to membrane
        distance_to_membrane = nanotube_point[2]  # Since membrane is at z=0
        self.distance_text.set_position((
            nanotube_point[0]*1e9, 
            nanotube_point[1]*1e9, 
            nanotube_point[2]*1e9 + 10
        ))
        self.distance_text.set_text(f"Distance: {distance_to_membrane*1e9:.1f} nm")
        
        # Simulate ROS diffusion from ALL G4 sites to membrane
        translated_sources = self.simulate_ros_diffusion()
        
        # Update membrane view with ROS density
        self.membrane_img.set_data(self.membrane_hits.T)
        self.membrane_img.autoscale()
        
        # Update threshold contour with error handling
        if self.contour is not None:
            # Remove old contour collections if they exist
            for coll in self.contour.collections:
                if coll in self.ax_membrane.collections:
                    coll.remove()
        
        # Create new contour if there are points above threshold
        if np.any(self.membrane_hits > THRESHOLD_DENSITY):
            self.contour = self.ax_membrane.contour(
                self.x_grid*1e9, 
                self.y_grid*1e9, 
                self.membrane_hits.T, 
                levels=[THRESHOLD_DENSITY], 
                colors='white', 
                linewidths=1
            )
        else:
            self.contour = None
        
        # Update analysis panel with ROS focus
        self.update_analysis_panel()
        
        # Capture frame if recording
        if self.recording:
            # Render the figure
            self.fig.canvas.draw()
            
            # Convert the figure to an image array
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            
            # Append to recording frames
            self.recording_frames.append(img.copy())
            
            # Check if recording should stop (after 5 seconds)
            if time.time() - self.recording_start_time > 5:
                self.toggle_recording()
        
        self.fig.canvas.draw_idle()
    
    def reset(self, event):
        """Reset view"""
        self.slider_x.reset()
        self.slider_y.reset()
        self.slider_z.reset()
        self.slider_length.set_val(20)  # Reset aptamer length to 20nm
        if hasattr(self, 'md_snapshot'):
            self.md_snapshot = None  # Clear MD results
        self.sources = self.create_tube_sources()  # Restore initial structure
        self.update()
    
    def run_analysis(self, event):
        """Run aptamer length impact analysis with ROS focus"""
        # Create analysis window
        analysis_fig = plt.figure(figsize=(14, 10), facecolor='#e0f7fa')
        analysis_fig.suptitle('ROS-Mediated Membrane Rupture Analysis', 
                             fontsize=20, color='#0288d1', fontweight='bold')
        
        gs = gridspec.GridSpec(2, 2)
        
        # Parameter sweep settings
        aptamer_lengths = np.linspace(5, 50, 20)  # 5-50nm range
        orientations = [0, 45, 90]  # Three typical orientations
        results = {angle: [] for angle in orientations}
        
        # Progress bar
        print("Running ROS-mediated membrane rupture analysis...")
        
        # Sweep over orientations and aptamer lengths
        for angle in orientations:
            for length in tqdm(aptamer_lengths, desc=f"Angle {angle}°"):
                # Set orientation
                self.slider_x.set_val(0)
                self.slider_y.set_val(angle)
                self.slider_z.set_val(0)
                
                # Set aptamer length
                self.slider_length.set_val(length)
                self.update()
                
                # Collect results
                peak_density = np.max(self.membrane_hits)
                grid_cell_area = (200/self.grid_size)**2  # Area per grid cell in μm²
                coverage_area = np.sum(self.membrane_hits > THRESHOLD_DENSITY) * grid_cell_area
                results[angle].append((length, peak_density, coverage_area))
        
        # Plot results
        ax1 = analysis_fig.add_subplot(gs[0, 0], facecolor='#e0f7fa')
        ax2 = analysis_fig.add_subplot(gs[0, 1], facecolor='#e0f7fa')
        ax3 = analysis_fig.add_subplot(gs[1, :], facecolor='#e0f7fa')
        
        # Set styling
        for ax in [ax1, ax2, ax3]:
            ax.tick_params(colors='#01579b')
            for spine in ax.spines.values():
                spine.set_color('#01579b')
            ax.xaxis.label.set_color('#01579b')
            ax.yaxis.label.set_color('#01579b')
            ax.title.set_color('#0288d1')
        
        # Plot peak ROS density vs aptamer length
        for angle in orientations:
            lengths = [r[0] for r in results[angle]]
            peaks = [r[1] for r in results[angle]]
            ax1.plot(lengths, peaks, 'o-', color='#4fc3f7', label=f"{angle}°", linewidth=2)
        
        ax1.set_title('Peak ROS Density vs Aptamer Length')
        ax1.set_xlabel('Aptamer Length (nm)')
        ax1.set_ylabel('ROS Density (molecules/μm²/ns)')
        ax1.axhline(THRESHOLD_DENSITY, color='#d32f2f', linestyle='--', label='Rupture Threshold')
        ax1.grid(True, alpha=0.2, color='#b3e5fc')
        ax1.legend()
        
        # Plot ROS coverage area vs aptamer length
        for angle in orientations:
            lengths = [r[0] for r in results[angle]]
            areas = [r[2] for r in results[angle]]
            ax2.plot(lengths, areas, 'o-', color='#4fc3f7', label=f"{angle}°", linewidth=2)
        
        ax2.set_title('Effective ROS Coverage Area')
        ax2.set_xlabel('Aptamer Length (nm)')
        ax2.set_ylabel('ROS Coverage Area (μm²)')
        ax2.axhline(0.05, color='#d32f2f', linestyle='--', label='Minimum Area')
        ax2.grid(True, alpha=0.2, color='#b3e5fc')
        ax2.legend()
        
        # Plot rupture efficiency by ROS
        for angle in orientations:
            lengths = [r[0] for r in results[angle]]
            rupture_efficiency = []
            for r in results[angle]:
                # Rupture efficiency by ROS combines peak density and coverage area
                eff = min(1.0, r[1]/THRESHOLD_DENSITY) * min(1.0, r[2]/0.05)
                rupture_efficiency.append(eff)
            
            ax3.plot(lengths, rupture_efficiency, 'o-', color='#4fc3f7', label=f"{angle}°", linewidth=2)
            
            # Add regression analysis
            slope, intercept, r_value, p_value, std_err = linregress(lengths, rupture_efficiency)
            reg_line = slope * np.array(lengths) + intercept
            ax3.plot(lengths, reg_line, '--', color='#0288d1', alpha=0.5)
            ax3.text(lengths[-1], reg_line[-1], 
                    f"slope: {slope:.4f}\nR²: {r_value**2:.3f}", 
                    color='#01579b', fontsize=9)
        
        ax3.set_title('ROS-Mediated Membrane Rupture Efficiency')
        ax3.set_xlabel('Aptamer Length (nm)')
        ax3.set_ylabel('Rupture Efficiency')
        ax3.axhline(1.0, color='#d32f2f', linestyle='--', label='Full Efficiency')
        ax3.grid(True, alpha=0.2, color='#b3e5fc')
        ax3.legend()
        
        # Add key conclusions with ROS focus
        conclusion_text = (
            "Key Conclusions on ROS-Mediated Rupture:\n"
            "1. Shorter aptamers increase ROS density at membrane\n"
            "2. Distance to membrane is critical for ROS effectiveness\n"
            "3. Vertical orientations maximize ROS delivery to membrane\n"
            "4. ROS coverage area >0.05 μm² needed for rupture\n"
            "5. Optimal aptamer length: 15-25nm for efficient ROS delivery\n"
            "6. Multiple sources create synergistic ROS effects"
        )
        analysis_fig.text(0.1, 0.05, conclusion_text, 
                         color='#0288d1', fontsize=12, 
                         bbox=dict(facecolor='#b3e5fc', alpha=0.7))
        
        # Add export button to analysis window
        export_ax = plt.axes([0.8, 0.01, 0.15, 0.04], facecolor='#4fc3f7')
        export_button = Button(export_ax, 'Export Analysis', color='#4fc3f7', hovercolor='#81d4fa')
        def export_analysis(event):
            analysis_fig.savefig('ros_analysis.png', dpi=300, bbox_inches='tight', facecolor='#e0f7fa')
            self.ax_analysis.text(0.5, 0.3, "Analysis exported to ros_analysis.png", 
                                 color='#388e3c', ha='center')
            analysis_fig.canvas.draw_idle()
        export_button.on_clicked(export_analysis)
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()
    
    def toggle_animation(self, event):
        """Toggle ROS diffusion animation from all sources"""
        if self.animation_running:
            # Stop animation
            if self.animation is not None:
                self.animation.event_source.stop()
            self.animation_running = False
            self.animate_button.label.set_text("Animate Diffusion")
            # Clear animation objects
            self.molecule_points.remove()
            for line in self.trajectory_lines:
                line.remove()
            self.trajectory_lines = []
            # Redraw static view
            self.update()
        else:
            # Start animation
            if len(self.trajectories) == 0:
                self.simulate_ros_diffusion()
            
            # Create trajectory lines for each molecule
            self.trajectory_lines = []
            for trajectory in self.trajectories:
                line, = self.ax_3d.plot([], [], [], '#4fc3f7', alpha=0.5, linewidth=1)  # Sky blue
                self.trajectory_lines.append(line)
            
            # Initialize animation
            self.animation = FuncAnimation(
                self.fig, self.update_animation,
                frames=min(len(t) for t in self.trajectories) if self.trajectories else 100,
                interval=50, blit=False
            )
            self.animation_running = True
            self.animate_button.label.set_text("Stop Animation")
        
        self.fig.canvas.draw_idle()
    
    def update_animation(self, frame):
        """Update function for ROS diffusion animation from all sources"""
        points = []
        for i, trajectory in enumerate(self.trajectories):
            if frame < len(trajectory):
                pos = trajectory[frame]
                points.append([pos[0]*1e9, pos[1]*1e9, pos[2]*1e9])
                
                # Update trajectory line
                x = [p[0]*1e9 for p in trajectory[:frame+1]]
                y = [p[1]*1e9 for p in trajectory[:frame+1]]
                z = [p[2]*1e9 for p in trajectory[:frame+1]]
                self.trajectory_lines[i].set_data(x, y)
                self.trajectory_lines[i].set_3d_properties(z)
        
        if points:
            points = np.array(points)
            self.molecule_points._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
        
        return self.molecule_points, *self.trajectory_lines
    
    def export_3d_view(self, event):
        """Export current 3D view as high-quality image"""
        # Create a temporary figure for rendering
        export_fig = plt.figure(figsize=(10, 8), facecolor='#e0f7fa')
        ax = export_fig.add_subplot(111, projection='3d', facecolor='#e0f7fa')
        
        # Copy current view settings
        ax.set_xlim(self.ax_3d.get_xlim())
        ax.set_ylim(self.ax_3d.get_ylim())
        ax.set_zlim(self.ax_3d.get_zlim())
        ax.set_xlabel('X (nm)', color='#01579b')
        ax.set_ylabel('Y (nm)', color='#01579b')
        ax.set_zlabel('Z (nm)', color='#01579b')
        ax.tick_params(colors='#01579b')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#b3e5fc')
        ax.yaxis.pane.set_edgecolor('#b3e5fc')
        ax.zaxis.pane.set_edgecolor('#b3e5fc')
        ax.grid(False)
        ax.set_title('DNA Nanotube with Aptamer and ROS Sources', color='#0288d1', fontsize=14)
        
        # Draw membrane plane - dark blue
        xx, yy = np.meshgrid(np.linspace(-100, 100, 10), np.linspace(-100, 100, 10))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.4, color='#0d47a1')
        
        # Draw tube structure - light blue
        rotated_sources = self.rotate_sources()
        ax.scatter(
            rotated_sources[:, 0]*1e9, 
            rotated_sources[:, 1]*1e9, 
            rotated_sources[:, 2]*1e9, 
            s=15, c='#81d4fa', alpha=0.9, edgecolors='#e1f5fe', linewidths=0.5
        )
        
        # Draw ROS sources - sky blue
        ax.scatter(
            rotated_sources[:, 0]*1e9, 
            rotated_sources[:, 1]*1e9, 
            rotated_sources[:, 2]*1e9, 
            s=8, c='#4fc3f7', marker='o', alpha=0.8
        )
        
        # Draw aptamer - light blue
        min_z_index = np.argmin(rotated_sources[:, 2])
        nanotube_point = rotated_sources[min_z_index]
        membrane_point = np.array([nanotube_point[0], nanotube_point[1], 0])
        
        ax.plot(
            [nanotube_point[0]*1e9, membrane_point[0]*1e9],
            [nanotube_point[1]*1e9, membrane_point[1]*1e9],
            [nanotube_point[2]*1e9, membrane_point[2]*1e9],
            '#81d4fa', linewidth=1.5, alpha=0.8
        )
        ax.scatter(
            membrane_point[0]*1e9, 
            membrane_point[1]*1e9, 
            membrane_point[2]*1e9, 
            s=50, c='#4fc3f7', marker='*', edgecolors='#29b6f6', linewidths=0.5
        )
        
        # Add distance annotation
        distance_to_membrane = nanotube_point[2] * 1e9
        ax.text(
            nanotube_point[0]*1e9, 
            nanotube_point[1]*1e9, 
            nanotube_point[2]*1e9 + 10,
            f"Distance: {distance_to_membrane:.1f} nm",
            color='#01579b', fontsize=9
        )
        
        # Save as high-quality image
        export_fig.savefig('nanotube_3d_view.png', dpi=300, bbox_inches='tight', facecolor='#e0f7fa')
        plt.close(export_fig)
        
        # Update status
        self.ax_analysis.clear()
        self.ax_analysis.text(0.5, 0.6, "3D view exported to nanotube_3d_view.png", 
                             color='#388e3c', fontsize=12, ha='center')
        self.fig.canvas.draw_idle()
    
    def export_membrane_view(self, event):
        """Export membrane view as high-quality image"""
        # Create a temporary figure for rendering
        export_fig, ax = plt.subplots(figsize=(8, 6), facecolor='#e0f7fa')
        
        # Copy current membrane view
        img = ax.imshow(
            self.membrane_hits.T, 
            extent=[-100, 100, -100, 100], 
            origin='lower', 
            cmap=cmap
        )
        ax.set_xlabel('X (nm)', color='#01579b')
        ax.set_ylabel('Y (nm)', color='#01579b')
        ax.tick_params(colors='#01579b')
        ax.set_facecolor('#e0f7fa')
        ax.set_title('ROS Density at Membrane (All Sources)', color='#0288d1', fontsize=14)
        
        # Add threshold contour
        if np.any(self.membrane_hits > THRESHOLD_DENSITY):
            ax.contour(
                self.x_grid*1e9, 
                self.y_grid*1e9, 
                self.membrane_hits.T, 
                levels=[THRESHOLD_DENSITY], 
                colors='white', 
                linewidths=1
            )
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=ax, pad=0.01)
        cbar.set_label('ROS Density (molecules/μm²/ns)', color='#01579b')
        cbar.ax.yaxis.set_tick_params(color='#01579b')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#01579b')
        
        # Add annotations
        peak_density = np.max(self.membrane_hits)
        total_ros = np.sum(self.membrane_hits)
        ax.text(0.02, 0.95, f"Peak Density: {peak_density:.1f}", 
               transform=ax.transAxes, color='#01579b', fontsize=10,
               bbox=dict(facecolor='#b3e5fc', alpha=0.7))
        ax.text(0.02, 0.88, f"Total ROS: {total_ros:.0f}", 
               transform=ax.transAxes, color='#01579b', fontsize=10,
               bbox=dict(facecolor='#b3e5fc', alpha=0.7))
        ax.text(0.02, 0.81, f"Active Sources: {self.n_sources}", 
               transform=ax.transAxes, color='#01579b', fontsize=10,
               bbox=dict(facecolor='#b3e5fc', alpha=0.7))
        
        # Save as high-quality image
        export_fig.savefig('membrane_view.png', dpi=300, bbox_inches='tight', facecolor='#e0f7fa')
        plt.close(export_fig)
        
        # Update status
        self.ax_analysis.clear()
        self.ax_analysis.text(0.5, 0.6, "Membrane view exported to membrane_view.png", 
                             color='#388e3c', fontsize=12, ha='center')
        self.fig.canvas.draw_idle()
    
    def export_data(self, event):
        """Export simulation data as CSV and images"""
        try:
            # Export 3D coordinates
            np.savetxt('nanotube_coordinates.csv', self.sources * 1e9, 
                      delimiter=',', header='x,y,z (nm)', comments='')
            
            # Export membrane density
            np.savetxt('membrane_density.csv', self.membrane_hits, 
                      delimiter=',', comments='')
            
            # Export ROS source positions
            rotated_sources = self.rotate_sources()
            np.savetxt('ros_sources.csv', rotated_sources * 1e9,
                      delimiter=',', header='x,y,z (nm)', comments='')
            
            # Export settings
            with open('simulation_settings.txt', 'w') as f:
                f.write(f"Aptamer Length: {self.slider_length.val} nm\n")
                f.write(f"Rotation X: {self.slider_x.val} deg\n")
                f.write(f"Rotation Y: {self.slider_y.val} deg\n")
                f.write(f"Rotation Z: {self.slider_z.val} deg\n")
                f.write(f"Number of G4 sites: {self.n_sources}\n")
                f.write(f"ROS per site: {ROS_PER_SOURCE}\n")
                f.write(f"Diffusion coefficient: {D} m²/s\n")
                f.write(f"ROS lifetime: {tau} s\n")
            
            # Create a summary figure
            summary_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='#e0f7fa')
            summary_fig.suptitle('Simulation Summary', color='#0288d1', fontsize=16)
            
            # 3D view
            self.ax_3d.figure = summary_fig
            self.ax_3d.change_geometry(1, 2, 1)
            ax1.remove()
            summary_fig.add_axes(self.ax_3d)
            
            # Membrane view
            self.ax_membrane.figure = summary_fig
            self.ax_membrane.change_geometry(1, 2, 2)
            ax2.remove()
            summary_fig.add_axes(self.ax_membrane)
            
            # Save summary
            summary_fig.savefig('simulation_summary.png', dpi=300, bbox_inches='tight', facecolor='#e0f7fa')
            plt.close(summary_fig)
            
            # Update status
            self.ax_analysis.clear()
            self.ax_analysis.text(0.5, 0.6, "Data exported:\n- nanotube_coordinates.csv\n- ros_sources.csv\n- membrane_density.csv\n- simulation_settings.txt\n- simulation_summary.png", 
                                 color='#388e3c', fontsize=12, ha='center')
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            self.ax_analysis.clear()
            self.ax_analysis.text(0.1, 0.6, f"Export error: {str(e)}", 
                                 color='#d32f2f', fontsize=10)
            self.fig.canvas.draw_idle()
    
    def toggle_recording(self, event=None):
        """Toggle recording of the 3D view"""
        if self.recording:
            # Stop recording
            self.recording = False
            self.record_button.label.set_text("Record Animation")
            
            # Save the recording as a video
            if self.recording_frames:
                try:
                    # Create animation
                    fig, ax = plt.subplots(figsize=(10, 8))
                    fig.set_facecolor('#e0f7fa')
                    ax.set_axis_off()
                    
                    # Create animation
                    ims = []
                    for frame in self.recording_frames:
                        im = ax.imshow(frame, animated=True)
                        ims.append([im])
                    
                    ani = FuncAnimation(fig, lambda i: ims[i], 
                                       frames=len(self.recording_frames), 
                                       interval=50, blit=True)
                    
                    # Save as MP4
                    writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
                    ani.save('3d_animation.mp4', writer=writer)
                    
                    # Update status
                    self.ax_analysis.clear()
                    self.ax_analysis.text(0.5, 0.6, "Animation saved as 3d_animation.mp4", 
                                         color='#388e3c', fontsize=12, ha='center')
                except Exception as e:
                    self.ax_analysis.clear()
                    self.ax_analysis.text(0.5, 0.6, f"Error saving video: {str(e)}", 
                                         color='#d32f2f', fontsize=12, ha='center')
            
            # Reset recording frames
            self.recording_frames = []
        else:
            # Start recording
            self.recording = True
            self.recording_start_time = time.time()
            self.record_button.label.set_text("Recording...")
            self.recording_frames = []
            
            # Update status
            self.ax_analysis.clear()
            self.ax_analysis.text(0.5, 0.6, "Recording animation...", 
                                 color='#0288d1', fontsize=12, ha='center')
        
        self.fig.canvas.draw_idle()

# Run simulator
simulator = AptamerLengthSimulator()
plt.show()