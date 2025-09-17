import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib as mpl
from scipy.stats import linregress
from tqdm import tqdm
import time
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Set a professional style for the interface
plt.style.use('default')
mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8

# Physical parameters
D_ros = 2.8e-9  # ROS diffusion coefficient (m²/s)
D_drug = 1.0e-9  # Drug diffusion coefficient (m²/s)
tau = 1e-9       # ROS lifetime (s)
dt = 0.1 * tau
ROS_PER_SOURCE = 200    # ROS molecules per G4 site
DRUG_MOLECULES = 500    # Drug molecules
THRESHOLD_DENSITY = 1e4 # Membrane rupture threshold (molecules/μm²/ns)
ANIMATION_MOLECULES = 50 # Molecules to show in animation

# Color maps
colors = ["darkblue", "blue", "cyan", "green", "yellow", "red"]
ros_cmap = LinearSegmentedColormap.from_list("ros_cmap", colors, N=256)
drug_cmap = LinearSegmentedColormap.from_list("drug_cmap", ["purple", "magenta", "pink"], N=256)

def calculate_pore_diameter(ros_density):
    """Calculate pore diameter based on ROS density"""
    base_diameter = 20  # Minimum pore diameter (nm)
    scaling_factor = 15  # Concentration scaling factor
    adjusted_density = max(ros_density, 1000)  
    return base_diameter + scaling_factor * np.log10(adjusted_density / 1000)

class DNANanotubeSimulator:
    def __init__(self):
        # Initialize simulation state
        self.animation_running = False
        self.trajectories = []
        self.drug_trajectories = []
        self.recording = False
        self.recording_frames = []
        self.recording_start_time = 0
        self.translocation_animating = False
        self.pore_diameter = 0
        self.pore_center = None
        self.cell_radius = 500  # Cell curvature radius (nm)
        self.drug_delivery_complete = False
        self.simulation_mode = "ros"
        self.current_tab = "control"  # Track current tab
        self.performance_data = []
        
        # DNA Nanotube parameters
        self.nanotube_length = 120e-9  # 120nm
        self.nanotube_radius = 5e-9    # 5nm
        self.n_rings = 5               # Number of rings along Z-axis
        self.sources_per_ring = 8      # Sources per ring
        self.aptamer_length = 20e-9    # Initial aptamer length 20nm
        self.drug_size = 5             # Drug molecule size (nm)
        self.drug_inside = 0           # Drug molecules inside cell
        
        # Create nanotube structure
        self.sources = self.create_nanotube_sources()
        self.drug_positions = self.create_drug_molecules()
        
        # Create membrane grid
        self.grid_size = 100
        self.membrane_hits = np.zeros((self.grid_size, self.grid_size))
        self.drug_hits = np.zeros((self.grid_size, self.grid_size))
        self.x_grid = np.linspace(-100e-9, 100e-9, self.grid_size)
        self.y_grid = np.linspace(-100e-9, 100e-9, self.grid_size)
        
        # Create the main figure with optimized layout
        self.fig = plt.figure(figsize=(14, 9), facecolor='#f5f5f5')
        self.fig.suptitle('DNA Nanotube Drug Delivery Simulator', 
                         fontsize=16, color='#0288d1', fontweight='bold')
        
        # Use GridSpec for precise layout control
        self.gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1], 
                                   wspace=0.25, hspace=0.3, left=0.05, right=0.95, 
                                   bottom=0.07, top=0.90)
        
        # Create visualization areas
        self.ax_3d = self.fig.add_subplot(self.gs[0, 0], projection='3d', facecolor='#e8f5e9')
        self.ax_membrane = self.fig.add_subplot(self.gs[1, 0], facecolor='#e8f5e9')
        
        # Create control and analysis areas with tabs
        self.create_control_panel()
        self.create_analysis_panel()
        
        # Initialize visualization
        self.initialize_visualization()
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def create_control_panel(self):
        """Create the control panel with tabs"""
        # Create a rectangle for the control panel
        control_rect = [0.68, 0.07, 0.27, 0.86]
        self.control_bg = Rectangle((control_rect[0], control_rect[1]), control_rect[2], control_rect[3], 
                                   transform=self.fig.transFigure, facecolor='#eeeeee', 
                                   edgecolor='#bdbdbd', alpha=0.8)
        self.fig.patches.append(self.control_bg)
        
        # Create tabs
        tab_height = 0.04
        self.tab_axes = {}
        tabs = ['control', 'simulation', 'analysis', 'export']
        
        for i, tab in enumerate(tabs):
            tab_width = 0.27 / len(tabs)
            tab_x = 0.68 + i * tab_width
            tab_ax = plt.axes([tab_x, 0.93, tab_width, tab_height], facecolor='#e0e0e0')
            tab_ax.text(0.5, 0.5, tab.capitalize(), ha='center', va='center', 
                       fontsize=9, transform=tab_ax.transAxes)
            tab_ax.set_navigate(False)
            tab_ax.set_xticks([])
            tab_ax.set_yticks([])
            for spine in tab_ax.spines.values():
                spine.set_color('#9e9e9e')
            self.tab_axes[tab] = tab_ax
        
        # Highlight current tab
        self.update_tab_highlight()
        
        # Create control content
        self.create_control_content()
        
    def create_control_content(self):
        """Create content for the control tab"""
        # Clear any existing controls
        if hasattr(self, 'control_elements'):
            for element in self.control_elements:
                if hasattr(element, 'remove'):
                    element.remove()
        
        self.control_elements = []
        
        # Add title
        title_ax = plt.axes([0.69, 0.88, 0.25, 0.03])
        title_ax.text(0.5, 0.5, 'Simulation Controls', ha='center', va='center', 
                     fontsize=11, fontweight='bold', color='#0288d1')
        title_ax.set_xticks([])
        title_ax.set_yticks([])
        title_ax.set_facecolor('none')
        self.control_elements.append(title_ax)
        
        # Add parameter sliders
        y_pos = 0.82
        y_step = 0.06
        
        # Aptamer length slider
        slider_ax = plt.axes([0.70, y_pos, 0.24, 0.02])
        self.slider_length = Slider(slider_ax, 'Aptamer (nm)', 5, 50, valinit=20, 
                                   color='#29b6f6', track_color='#b3e5fc')
        self.slider_length.on_changed(self.update_aptamer)
        self.control_elements.append(slider_ax)
        y_pos -= y_step
        
        # Rotation sliders
        slider_ax = plt.axes([0.70, y_pos, 0.24, 0.02])
        self.slider_x = Slider(slider_ax, 'X Rotation', -180, 180, valinit=0, 
                              color='#29b6f6', track_color='#b3e5fc')
        self.slider_x.on_changed(self.update)
        self.control_elements.append(slider_ax)
        y_pos -= y_step
        
        slider_ax = plt.axes([0.70, y_pos, 0.24, 0.02])
        self.slider_y = Slider(slider_ax, 'Y Rotation', -180, 180, valinit=0, 
                              color='#29b6f6', track_color='#b3e5fc')
        self.slider_y.on_changed(self.update)
        self.control_elements.append(slider_ax)
        y_pos -= y_step
        
        slider_ax = plt.axes([0.70, y_pos, 0.24, 0.02])
        self.slider_z = Slider(slider_ax, 'Z Rotation', -180, 180, valinit=0, 
                              color='#29b6f6', track_color='#b3e5fc')
        self.slider_z.on_changed(self.update)
        self.control_elements.append(slider_ax)
        y_pos -= y_step
        
        # Drug size slider
        slider_ax = plt.axes([0.70, y_pos, 0.24, 0.02])
        self.slider_drug_size = Slider(slider_ax, 'Drug Size (nm)', 1, 30, valinit=5, 
                                      color='#ab47bc', track_color='#e1bee7')
        self.slider_drug_size.on_changed(self.update_drug_size)
        self.control_elements.append(slider_ax)
        y_pos -= y_step
        
        # Mode selection
        mode_ax = plt.axes([0.70, y_pos-0.02, 0.24, 0.04])
        self.mode_radio = RadioButtons(mode_ax, ('ROS Model', 'Drug Delivery'), active=0)
        self.mode_radio.on_clicked(self.change_mode)
        self.control_elements.append(mode_ax)
        y_pos -= y_step + 0.02
        
        # Action buttons
        btn_width = 0.11
        btn_height = 0.04
        btn_y = y_pos
        
        # Row 1
        reset_ax = plt.axes([0.70, btn_y, btn_width, btn_height])
        self.reset_button = Button(reset_ax, 'Reset', color='#ef5350', hovercolor='#e57373')
        self.reset_button.on_clicked(self.reset)
        self.control_elements.append(reset_ax)
        
        animate_ax = plt.axes([0.82, btn_y, btn_width, btn_height])
        self.animate_button = Button(animate_ax, 'Animate', color='#29b6f6', hovercolor='#4fc3f7')
        self.animate_button.on_clicked(self.toggle_animation)
        self.control_elements.append(animate_ax)
        
        btn_y -= btn_height + 0.02
        
        # Row 2
        translocate_ax = plt.axes([0.70, btn_y, btn_width, btn_height])
        self.translocation_button = Button(translocate_ax, 'Translocate', color='#7b1fa2', hovercolor='#9c27b0')
        self.translocation_button.on_clicked(self.start_translocation)
        self.control_elements.append(translocate_ax)
        
        drug_ax = plt.axes([0.82, btn_y, btn_width, btn_height])
        self.drug_button = Button(drug_ax, 'Drug Sim', color='#7b1fa2', hovercolor='#9c27b0')
        self.drug_button.on_clicked(self.start_drug_diffusion)
        self.control_elements.append(drug_ax)
        
        btn_y -= btn_height + 0.02
        
        # Row 3
        analyze_ax = plt.axes([0.70, btn_y, btn_width, btn_height])
        self.analyze_button = Button(analyze_ax, 'Analyze', color='#388e3c', hovercolor='#66bb6a')
        self.analyze_button.on_clicked(self.run_analysis)
        self.control_elements.append(analyze_ax)
        
        export_ax = plt.axes([0.82, btn_y, btn_width, btn_height])
        self.export_button = Button(export_ax, 'Export', color='#ffa000', hovercolor='#ffb300')
        self.export_button.on_clicked(self.export_data)
        self.control_elements.append(export_ax)
        
    def create_analysis_panel(self):
        """Create the analysis panel"""
        # Create a dedicated analysis area
        self.analysis_ax = plt.axes([0.69, 0.07, 0.25, 0.35])
        self.analysis_ax.set_facecolor('#fafafa')
        self.analysis_ax.set_xticks([])
        self.analysis_ax.set_yticks([])
        for spine in self.analysis_ax.spines.values():
            spine.set_color('#e0e0e0')
        
        # Add title
        self.analysis_ax.text(0.5, 0.95, 'Analysis Results', ha='center', va='top', 
                             fontsize=11, fontweight='bold', color='#0288d1',
                             transform=self.analysis_ax.transAxes)
        
        # Initial placeholder text
        self.analysis_ax.text(0.5, 0.5, 'Run simulation to see analysis results', 
                             ha='center', va='center', color='#9e9e9e', style='italic',
                             transform=self.analysis_ax.transAxes)
        
    def update_tab_highlight(self):
        """Update tab highlighting based on current selection"""
        for tab, ax in self.tab_axes.items():
            if tab == self.current_tab:
                ax.set_facecolor('#ffffff')
                for spine in ax.spines.values():
                    spine.set_color('#0288d1')
                    spine.set_linewidth(1.5)
            else:
                ax.set_facecolor('#e0e0e0')
                for spine in ax.spines.values():
                    spine.set_color('#9e9e9e')
                    spine.set_linewidth(1.0)
        
    def on_click(self, event):
        """Handle click events for tab switching"""
        if event.inaxes in self.tab_axes.values():
            for tab, ax in self.tab_axes.items():
                if ax == event.inaxes:
                    self.current_tab = tab
                    self.update_tab_highlight()
                    
                    # Update content based on selected tab
                    if tab == 'control':
                        self.create_control_content()
                    elif tab == 'analysis':
                        self.update_analysis_panel()
                    
                    self.fig.canvas.draw_idle()
                    break
    
    def create_nanotube_sources(self):
        """Create G4 sites along the DNA nanotube"""
        sources = []
        
        # Create sources along the nanotube length
        for ring_idx in range(self.n_rings):
            # Position along the nanotube (z-axis)
            z = (ring_idx / (self.n_rings - 1)) * self.nanotube_length + self.aptamer_length
            
            # Create points around the circumference
            for source_idx in range(self.sources_per_ring):
                angle = 2 * np.pi * source_idx / self.sources_per_ring
                x = self.nanotube_radius * np.cos(angle)
                y = self.nanotube_radius * np.sin(angle)
                
                sources.append([x, y, z])
        
        return np.array(sources)
    
    def create_drug_molecules(self):
        """Create drug molecules inside the nanotube"""
        positions = []
        
        # Random distribution inside the nanotube
        for _ in range(DRUG_MOLECULES):
            r = np.random.uniform(0, self.nanotube_radius * 0.8)
            theta = np.random.uniform(0, 2 * np.pi)
            z = np.random.uniform(0, self.nanotube_length) + self.aptamer_length
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            positions.append([x, y, z])
        
        return np.array(positions)
    
    def create_nanotube_mesh(self, center=(0, 0, 0)):
        """Create a cylindrical mesh for the DNA nanotube"""
        length = self.nanotube_length
        radius = self.nanotube_radius
        resolution = 20
        
        # Create cylinder mesh
        z = np.linspace(0, length, resolution)
        theta = np.linspace(0, 2 * np.pi, resolution)
        theta_grid, z_grid = np.meshgrid(theta, z)
        
        # Cylinder coordinates
        x_grid = radius * np.cos(theta_grid) + center[0]
        y_grid = radius * np.sin(theta_grid) + center[1]
        z_grid = z_grid + center[2] - length/2
        
        return x_grid, y_grid, z_grid
    
    def initialize_visualization(self):
        """Initialize visualization elements"""
        # 3D view
        self.ax_3d.clear()
        self.ax_3d.grid(False)
        
        # Draw membrane plane
        xx, yy = np.meshgrid(np.linspace(-100, 100, 10), np.linspace(-100, 100, 10))
        zz = np.zeros_like(xx)
        self.membrane_surface = self.ax_3d.plot_surface(xx, yy, zz, alpha=0.4, color='#0d47a1')
        
        # Draw DNA nanotube
        rotated_sources = self.rotate_sources()
        center = np.mean(rotated_sources, axis=0)
        X, Y, Z = self.create_nanotube_mesh(center)
        
        # Plot nanotube as a surface
        self.nanotube_surface = self.ax_3d.plot_surface(
            X*1e9, Y*1e9, Z*1e9, 
            color='#4fc3f7', alpha=0.7, edgecolor='#01579b', linewidth=0.5
        )
        
        # Add G4 source markers
        self.source_markers = self.ax_3d.scatter(
            rotated_sources[:, 0]*1e9, 
            rotated_sources[:, 1]*1e9, 
            rotated_sources[:, 2]*1e9, 
            s=30, c='#ff9800', marker='o', alpha=1.0
        )
        
        # Add aptamer representation
        min_z_index = np.argmin(rotated_sources[:, 2])
        nanotube_point = rotated_sources[min_z_index]
        membrane_point = np.array([nanotube_point[0], nanotube_point[1], 0])
        
        self.aptamer_line, = self.ax_3d.plot(
            [nanotube_point[0]*1e9, membrane_point[0]*1e9],
            [nanotube_point[1]*1e9, membrane_point[1]*1e9],
            [nanotube_point[2]*1e9, membrane_point[2]*1e9],
            '#81d4fa', linewidth=1.5, alpha=0.8
        )
        
        self.aptamer_point = self.ax_3d.scatter(
            membrane_point[0]*1e9, 
            membrane_point[1]*1e9, 
            membrane_point[2]*1e9, 
            s=50, c='#4fc3f7', marker='*', edgecolors='#29b6f6', linewidths=0.5
        )
        
        # Add distance annotation
        distance_to_membrane = nanotube_point[2]
        self.distance_text = self.ax_3d.text(
            nanotube_point[0]*1e9, 
            nanotube_point[1]*1e9, 
            nanotube_point[2]*1e9 + 10,
            f"Distance: {distance_to_membrane*1e9:.1f} nm",
            color='#01579b', fontsize=9
        )
        
        # Add drug molecules
        self.drug_points = self.ax_3d.scatter(
            self.drug_positions[:, 0]*1e9,
            self.drug_positions[:, 1]*1e9,
            self.drug_positions[:, 2]*1e9,
            s=30, c='purple', alpha=0.8
        )
        
        # Set 3D view parameters
        self.ax_3d.set_xlim(-100, 100)
        self.ax_3d.set_ylim(-100, 100)
        self.ax_3d.set_zlim(-50, 150)
        self.ax_3d.set_xlabel('X (nm)', color='#01579b')
        self.ax_3d.set_ylabel('Y (nm)', color='#01579b')
        self.ax_3d.set_zlabel('Z (nm)', color='#01579b')
        self.ax_3d.tick_params(colors='#01579b')
        self.ax_3d.set_title('3D Nanotube Visualization', color='#0288d1', fontsize=11)
        
        # Membrane view
        self.ax_membrane.clear()
        self.membrane_img = self.ax_membrane.imshow(
            self.membrane_hits.T, 
            extent=[-100, 100, -100, 100], 
            origin='lower', 
            cmap=ros_cmap, 
            vmin=0, 
            vmax=ROS_PER_SOURCE * self.n_rings * self.sources_per_ring / 10
        )
        self.ax_membrane.set_xlabel('X (nm)', color='#01579b')
        self.ax_membrane.set_ylabel('Y (nm)', color='#01579b')
        self.ax_membrane.tick_params(colors='#01579b')
        self.ax_membrane.set_facecolor('#e8f5e9')
        self.ax_membrane.set_title('Membrane ROS Density', color='#0288d1', fontsize=11)
        
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
        """Simulate ROS diffusion from G4 sites"""
        self.membrane_hits = np.zeros((self.grid_size, self.grid_size))
        rotated_sources = self.rotate_sources()
        
        # Ensure all points are above membrane
        rotated_sources[:, 2] = np.maximum(rotated_sources[:, 2], 1e-9)
        
        # Initialize trajectories
        self.trajectories = []
        
        # For each G4 source
        for src_idx, src in enumerate(rotated_sources):
            # For each ROS molecule
            for ros_idx in range(ROS_PER_SOURCE):
                # Only track a subset for animation
                if ros_idx < ANIMATION_MOLECULES:
                    pos = src.copy()
                    trajectory = [pos.copy()]
                    
                    # Simulate diffusion
                    for _ in range(100):
                        # Brownian motion
                        dx = np.random.normal(0, np.sqrt(2*D_ros*dt))
                        dy = np.random.normal(0, np.sqrt(2*D_ros*dt))
                        dz = np.random.normal(0, np.sqrt(2*D_ros*dt))
                        pos += np.array([dx, dy, dz])
                        trajectory.append(pos.copy())
                        
                        # Check if hit membrane
                        if pos[2] <= 0:
                            x_idx = np.argmin(np.abs(self.x_grid - pos[0]))
                            y_idx = np.argmin(np.abs(self.y_grid - pos[1]))
                            
                            if 0 <= x_idx < self.grid_size and 0 <= y_idx < self.grid_size:
                                self.membrane_hits[x_idx, y_idx] += 1
                            break
                    
                    # Store trajectory
                    self.trajectories.append(trajectory)
                else:
                    # Minimal simulation for non-tracked molecules
                    pos = src.copy()
                    for _ in range(100):
                        dx = np.random.normal(0, np.sqrt(2*D_ros*dt))
                        dy = np.random.normal(0, np.sqrt(2*D_ros*dt))
                        dz = np.random.normal(0, np.sqrt(2*D_ros*dt))
                        pos += np.array([dx, dy, dz])
                        
                        if pos[2] <= 0:
                            x_idx = np.argmin(np.abs(self.x_grid - pos[0]))
                            y_idx = np.argmin(np.abs(self.y_grid - pos[1]))
                            
                            if 0 <= x_idx < self.grid_size and 0 <= y_idx < self.grid_size:
                                self.membrane_hits[x_idx, y_idx] += 1
                            break
        
        # Calculate pore diameter
        peak_density = np.max(self.membrane_hits)
        self.pore_diameter = calculate_pore_diameter(peak_density)
        
        return rotated_sources
    
    def update_aptamer(self, val):
        """Update aptamer length"""
        self.update()
    
    def update_drug_size(self, val):
        """Update drug size"""
        self.drug_size = val
        self.update_analysis_panel()
    
    def update_analysis_panel(self):
        """Update analysis panel with current results"""
        self.analysis_ax.clear()
        self.analysis_ax.set_facecolor('#fafafa')
        self.analysis_ax.set_xticks([])
        self.analysis_ax.set_yticks([])
        for spine in self.analysis_ax.spines.values():
            spine.set_color('#e0e0e0')
        
        # Calculate analysis metrics
        total_ros = np.sum(self.membrane_hits)
        peak_density = np.max(self.membrane_hits)
        grid_cell_area = (200/self.grid_size)**2
        coverage_area = np.sum(self.membrane_hits > THRESHOLD_DENSITY) * grid_cell_area
        can_rupture = peak_density > THRESHOLD_DENSITY and coverage_area > 0.05
        
        # Calculate pore diameter
        pore_diameter = calculate_pore_diameter(peak_density)
        can_translocate = pore_diameter > self.nanotube_radius * 2e9
        can_drug_pass = pore_diameter > self.drug_size
        
        # Add text analysis
        y_pos = 0.85
        line_height = 0.07
        
        # Title
        self.analysis_ax.text(0.5, 0.95, 'Analysis Results', ha='center', va='top', 
                             fontsize=11, fontweight='bold', color='#0288d1')
        
        # Mode-specific analysis
        if self.simulation_mode == "ros":
            analysis_lines = [
                f"Peak ROS Density: {peak_density:.1f} molecules/μm²/ns",
                f"Threshold: {THRESHOLD_DENSITY} molecules/μm²/ns",
                f"Coverage Area: {coverage_area:.2f} μm²",
                f"Total ROS: {total_ros:.0f} molecules",
                f"Pore Diameter: {pore_diameter:.1f} nm",
                "",
                f"Membrane Rupture: {'POSSIBLE' if can_rupture else 'UNLIKELY'}",
                f"Nanotube Entry: {'POSSIBLE' if can_translocate else 'UNLIKELY'}"
            ]
            
            for i, line in enumerate(analysis_lines):
                color = '#0288d1' if can_rupture else '#f57c00'
                self.analysis_ax.text(0.05, y_pos - i*line_height, line, 
                                     color=color, fontsize=9, fontfamily='monospace')
        else:
            analysis_lines = [
                f"Drug Size: {self.drug_size} nm",
                f"Pore Diameter: {pore_diameter:.1f} nm",
                f"Size Ratio: {self.drug_size/pore_diameter:.2f}",
                "",
                f"Drug Passage: {'POSSIBLE' if can_drug_pass else 'NOT POSSIBLE'}"
            ]
            
            for i, line in enumerate(analysis_lines):
                color = '#7b1fa2' if can_drug_pass else '#d32f2f'
                self.analysis_ax.text(0.05, y_pos - i*line_height, line, 
                                     color=color, fontsize=9, fontfamily='monospace')
        
        # Add aptamer analysis
        aptamer_len = self.slider_length.val
        min_distance = np.min(self.rotate_sources()[:, 2]) * 1e9
        
        aptamer_info = [
            f"Aptamer Length: {aptamer_len:.1f} nm",
            f"Min Distance: {min_distance:.1f} nm",
            f"G4 Sources: {self.n_rings * self.sources_per_ring}"
        ]
        
        for i, line in enumerate(aptamer_info):
            self.analysis_ax.text(0.05, 0.25 - i*line_height, line, 
                                 color='#01579b', fontsize=9)
        
        # Add design recommendations
        if self.simulation_mode == "ros":
            if can_rupture and can_translocate:
                advice = "Design optimal for delivery"
                color = "#388e3c"
            elif can_rupture:
                advice = "Increase pore size for entry"
                color = "#f57c00"
            else:
                advice = "Reduce distance, increase G4 density"
                color = "#f57c00"
        else:
            if can_drug_pass:
                advice = "Drug can pass through pore"
                color = "#7b1fa2"
            else:
                advice = "Reduce drug size or increase pore size"
                color = "#d32f2f"
        
        self.analysis_ax.text(0.05, 0.05, advice, color=color, fontsize=9, fontweight='bold')
    
    def update(self, val=None):
        """Update entire view"""
        # Update tube structure position
        rotated_sources = self.rotate_sources()
        rotated_sources[:, 2] = np.maximum(rotated_sources[:, 2], 1e-9)
        
        # Update source markers
        self.source_markers._offsets3d = (
            rotated_sources[:, 0]*1e9,
            rotated_sources[:, 1]*1e9,
            rotated_sources[:, 2]*1e9
        )
        
        # Update nanotube surface
        center = np.mean(rotated_sources, axis=0)
        X, Y, Z = self.create_nanotube_mesh(center)
        self.nanotube_surface.remove()
        self.nanotube_surface = self.ax_3d.plot_surface(
            X*1e9, Y*1e9, Z*1e9, 
            color='#4fc3f7', alpha=0.7, edgecolor='#01579b', linewidth=0.5
        )
        
        # Update drug molecules
        self.drug_points._offsets3d = (
            self.drug_positions[:, 0]*1e9,
            self.drug_positions[:, 1]*1e9,
            self.drug_positions[:, 2]*1e9
        )
        
        # Update aptamer position
        min_z_index = np.argmin(rotated_sources[:, 2])
        nanotube_point = rotated_sources[min_z_index]
        membrane_point = np.array([nanotube_point[0], nanotube_point[1], 0])
        
        self.aptamer_line.set_data(
            [nanotube_point[0]*1e9, membrane_point[0]*1e9],
            [nanotube_point[1]*1e9, membrane_point[1]*1e9]
        )
        self.aptamer_line.set_3d_properties(
            [nanotube_point[2]*1e9, membrane_point[2]*1e9]
        )
        
        self.aptamer_point._offsets3d = (
            [membrane_point[0]*1e9], 
            [membrane_point[1]*1e9], 
            [membrane_point[2]*1e9]
        )
        
        # Update distance annotation
        distance_to_membrane = nanotube_point[2]
        self.distance_text.set_position((
            nanotube_point[0]*1e9, 
            nanotube_point[1]*1e9, 
            nanotube_point[2]*1e9 + 10
        ))
        self.distance_text.set_text(f"Distance: {distance_to_membrane*1e9:.1f} nm")
        
        # Simulate ROS diffusion
        translated_sources = self.simulate_ros_diffusion()
        
        # Update membrane view
        self.membrane_img.set_data(self.membrane_hits.T)
        self.membrane_img.autoscale()
        
        # Update threshold contour
        if self.contour is not None:
            for coll in self.contour.collections:
                if coll in self.ax_membrane.collections:
                    coll.remove()
        
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
        
        # Update analysis panel
        self.update_analysis_panel()
        
        self.fig.canvas.draw_idle()
    
    def reset(self, event):
        """Reset view"""
        self.slider_x.reset()
        self.slider_y.reset()
        self.slider_z.reset()
        self.slider_length.set_val(20)
        self.slider_drug_size.set_val(5)
        self.sources = self.create_nanotube_sources()
        self.drug_positions = self.create_drug_molecules()
        self.pore_diameter = 0
        self.drug_delivery_complete = False
        self.update()
    
    def change_mode(self, label):
        """Change simulation mode"""
        self.simulation_mode = "ros" if label == "ROS Model" else "drug"
        self.update_analysis_panel()
    
    def toggle_animation(self, event):
        """Toggle ROS diffusion animation"""
        if self.animation_running:
            if self.animation is not None:
                self.animation.event_source.stop()
            self.animation_running = False
            self.animate_button.label.set_text("Animate")
        else:
            if len(self.trajectories) == 0:
                self.simulate_ros_diffusion()
            
            self.trajectory_lines = []
            for trajectory in self.trajectories[:ANIMATION_MOLECULES]:
                line, = self.ax_3d.plot([], [], [], '#4fc3f7', alpha=0.5, linewidth=1)
                self.trajectory_lines.append(line)
            
            self.animation = FuncAnimation(
                self.fig, self.update_ros_animation,
                frames=min(len(t) for t in self.trajectories[:ANIMATION_MOLECULES]) if self.trajectories else 100,
                interval=50, blit=False
            )
            self.animation_running = True
            self.animate_button.label.set_text("Stop")
        
        self.fig.canvas.draw_idle()
    
    def update_ros_animation(self, frame):
        """Update ROS diffusion animation"""
        points = []
        for i, trajectory in enumerate(self.trajectories[:ANIMATION_MOLECULES]):
            if frame < len(trajectory):
                pos = trajectory[frame]
                points.append([pos[0]*1e9, pos[1]*1e9, pos[2]*1e9])
                
                x = [p[0]*1e9 for p in trajectory[:frame+1]]
                y = [p[1]*1e9 for p in trajectory[:frame+1]]
                z = [p[2]*1e9 for p in trajectory[:frame+1]]
                self.trajectory_lines[i].set_data(x, y)
                self.trajectory_lines[i].set_3d_properties(z)
        
        if points:
            points = np.array(points)
            self.source_markers._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
        
        return self.source_markers, *self.trajectory_lines
    
    def start_translocation(self, event):
        """Start membrane translocation"""
        if self.translocation_animating:
            return
            
        self.ax_analysis.clear()
        self.ax_analysis.text(0.5, 0.7, "Starting translocation...", 
                             color='#0288d1', fontsize=12, ha='center')
        self.fig.canvas.draw_idle()
        
        pore_pos = self.visualize_pore_formation()
        
        if self.pore_diameter < self.nanotube_radius * 2e9:
            self.ax_analysis.clear()
            self.ax_analysis.text(0.5, 0.6, "Pore too small for entry!", 
                                 color='#d32f2f', fontsize=12, ha='center')
            self.fig.canvas.draw_idle()
            return
        
        self.translocation_animating = True
        self.translocation_button.label.set_text("Working...")
        self.simulate_translocation(pore_pos)
        self.translocation_animating = False
        self.translocation_button.label.set_text("Translocate")
    
    def start_drug_diffusion(self, event):
        """Start drug diffusion simulation"""
        if self.animation_running:
            return
            
        if self.pore_diameter == 0:
            self.ax_analysis.clear()
            self.ax_analysis.text(0.5, 0.6, "Run ROS simulation first!", 
                                 color='#d32f2f', fontsize=12, ha='center')
            self.fig.canvas.draw_idle()
            return
            
        if self.pore_diameter < self.drug_size:
            self.ax_analysis.clear()
            self.ax_analysis.text(0.5, 0.6, "Drug too large for pore!", 
                                 color='#d32f2f', fontsize=12, ha='center')
            self.fig.canvas.draw_idle()
            return
        
        self.drug_trajectories = []
        for pos in self.drug_positions[:ANIMATION_MOLECULES]:
            self.drug_trajectories.append([pos.copy()])
        
        self.drug_points = self.ax_3d.scatter([], [], [], s=30, c='purple', alpha=0.8)
        
        self.drug_delivery_complete = False
        self.animation = FuncAnimation(
            self.fig, self.update_drug_animation,
            frames=200,
            interval=50, blit=False
        )
        self.animation_running = True
        self.drug_button.label.set_text("Diffusing...")
    
    def update_drug_animation(self, frame):
        """Update drug diffusion animation"""
        if self.drug_delivery_complete:
            return
        
        new_positions = []
        completed_molecules = 0
        
        for i, trajectory in enumerate(self.drug_trajectories):
            if len(trajectory) > 0:
                pos = trajectory[-1].copy()
                
                dx = np.random.normal(0, np.sqrt(2*D_drug*dt))
                dy = np.random.normal(0, np.sqrt(2*D_drug*dt))
                dz = np.random.normal(0, np.sqrt(2*D_drug*dt))
                pos += np.array([dx, dy, dz])
                
                if pos[2] < 0:
                    completed_molecules += 1
                else:
                    trajectory.append(pos)
                    new_positions.append(pos)
        
        if new_positions:
            new_positions = np.array(new_positions)
            self.drug_points._offsets3d = (
                new_positions[:, 0]*1e9,
                new_positions[:, 1]*1e9,
                new_positions[:, 2]*1e9
            )
        
        if completed_molecules >= len(self.drug_trajectories):
            self.drug_delivery_complete = True
            self.animation.event_source.stop()
            self.animation_running = False
            self.drug_button.label.set_text("Drug Sim")
            
            efficiency = completed_molecules / len(self.drug_trajectories) * 100
            
            result_text = (
                f"Delivery Complete!\n"
                f"Efficiency: {efficiency:.1f}%\n"
                f"Drug Size: {self.drug_size} nm\n"
                f"Pore Size: {self.pore_diameter:.1f} nm"
            )
            self.ax_3d.text2D(
                0.3, 0.8, result_text, 
                color='purple', fontsize=12,
                bbox=dict(facecolor='#f3e5f5', alpha=0.8),
                transform=self.ax_3d.transAxes
            )
        
        return self.drug_points
    
    def visualize_pore_formation(self):
        """Visualize pore formation on membrane"""
        xx, yy = np.meshgrid(np.linspace(-100, 100, 50), 
                             np.linspace(-100, 100, 50))
        zz = np.zeros_like(xx)
        
        max_idx = np.unravel_index(np.argmax(self.membrane_hits), self.membrane_hits.shape)
        pore_x = self.x_grid[max_idx[0]] * 1e9
        pore_y = self.y_grid[max_idx[1]] * 1e9
        
        peak_density = np.max(self.membrane_hits)
        self.pore_diameter = calculate_pore_diameter(peak_density)
        
        for i in range(50):
            for j in range(50):
                x = xx[i, j]
                y = yy[i, j]
                
                distance_to_pore = np.sqrt((x - pore_x)**2 + (y - pore_y)**2)
                
                depth_factor = min(1.0, self.membrane_hits[max_idx[0], max_idx[1]] / (2 * THRESHOLD_DENSITY))
                
                pore_depth = -depth_factor * 30 * np.exp(-(distance_to_pore**2) / (self.pore_diameter**2))
                
                zz[i, j] = pore_depth
        
        radius = self.cell_radius
        zz += (xx**2 + yy**2) / (2 * radius)
        
        if hasattr(self, 'membrane_surface') and self.membrane_surface:
            self.membrane_surface.remove()
            
        self.membrane_surface = self.ax_3d.plot_surface(
            xx, yy, zz, alpha=0.6, color='#0d47a1', 
            cmap='viridis', antialiased=True
        )
        
        if hasattr(self, 'pore_center') and self.pore_center:
            self.pore_center.remove()
            
        pore_z = -depth_factor * 30 + (pore_x**2 + pore_y**2) / (2 * radius)
        self.pore_center = self.ax_3d.scatter(
            [pore_x], [pore_y], [pore_z], 
            s=100, c='red', marker='x', alpha=0.8
        )
        
        self.ax_3d.text(
            pore_x, pore_y, pore_z + 15,
            f"Pore: Ø{self.pore_diameter:.1f} nm",
            color='red', fontsize=10
        )
        
        return pore_x, pore_y, pore_z
    
    def simulate_translocation(self, pore_pos):
        """Simulate nanotube translocation"""
        original_sources = self.sources.copy()
        original_drugs = self.drug_positions.copy()
        
        start_z = 50
        
        n_steps = 30
        for step in range(n_steps):
            if not self.translocation_animating:
                break
                
            progress = step / n_steps
            current_z = start_z * (1 - progress) + pore_pos[2] * progress - 20
            
            self.update_nanotube_position(
                pore_pos[0], pore_pos[1], current_z
            )
            
            if current_z < pore_pos[2] + 10:
                self.highlight_translocation(pore_pos)
            
            plt.pause(0.05)
        
        result_text = (
            f"Translocation SUCCESSFUL!\n"
            f"Pore diameter: {self.pore_diameter:.1f} nm\n"
            f"Nanotube diameter: {self.nanotube_radius*2e9:.1f} nm"
        )
        self.ax_3d.text2D(
            0.3, 0.9, result_text, 
            color='green', fontsize=12,
            bbox=dict(facecolor='#e0f7fa', alpha=0.8),
            transform=self.ax_3d.transAxes
        )
        
        self.sources = original_sources
        self.drug_positions = original_drugs
        self.update()
    
    def update_nanotube_position(self, target_x, target_y, z_pos):
        """Update nanotube position"""
        current_center = np.mean(self.sources, axis=0)
        displacement = np.array([
            target_x*1e-9 - current_center[0],
            target_y*1e-9 - current_center[1],
            z_pos*1e-9 - current_center[2]
        ])
        
        self.sources += displacement
        self.drug_positions += displacement
        self.update()
    
    def highlight_translocation(self, pore_pos):
        """Highlight translocation process"""
        theta = np.linspace(0, 2*np.pi, 20)
        z = np.linspace(pore_pos[2] - 10, pore_pos[2] + 10, 5)
        
        x_cyl = []
        y_cyl = []
        z_cyl = []
        
        for zi in z:
            x_cyl.append(pore_pos[0] + (self.pore_diameter/2) * np.cos(theta))
            y_cyl.append(pore_pos[1] + (self.pore_diameter/2) * np.sin(theta))
            z_cyl.append(np.ones_like(theta) * zi)
        
        if not hasattr(self, 'pore_cylinder'):
            self.pore_cylinder = self.ax_3d.plot_surface(
                np.array(x_cyl), np.array(y_cyl), np.array(z_cyl),
                alpha=0.3, color='red'
            )
        else:
            self.pore_cylinder.remove()
            self.pore_cylinder = self.ax_3d.plot_surface(
                np.array(x_cyl), np.array(y_cyl), np.array(z_cyl),
                alpha=0.3, color='red'
            )
        
        if not hasattr(self, 'translocation_indicator'):
            self.translocation_indicator = self.ax_3d.text(
                pore_pos[0], pore_pos[1], pore_pos[2] + 20,
                "NANOTUBE ENTERING CELL!", 
                color='red', fontsize=12, fontweight='bold'
            )
        else:
            self.translocation_indicator.set_position((pore_pos[0], pore_pos[1], pore_pos[2] + 20))
        
        self.fig.canvas.draw_idle()
    
    def run_analysis(self, event):
        """Run comprehensive analysis"""
        # Placeholder for analysis function
        print("Running comprehensive analysis...")
        
    def export_data(self, event):
        """Export simulation data"""
        # Placeholder for export function
        print("Exporting simulation data...")

# Run the simulator
simulator = DNANanotubeSimulator()
plt.show()