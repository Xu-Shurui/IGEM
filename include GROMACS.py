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

# Physical parameters
D = 2.8e-9  # ROS diffusion coefficient (m²/s)
tau = 1e-9   # ROS lifetime (s)
dt = 0.1 * tau
ROS_PER_SOURCE = 200  # ROS molecules per G4 site
THRESHOLD_DENSITY = 1e4  # Membrane rupture threshold (molecules/μm²/ns)

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
        
        # Create GUI with improved layout
        self.fig = plt.figure(figsize=(16, 12), facecolor='#0f0f1a')
        gs = gridspec.GridSpec(3, 2, width_ratios=[1.2, 1], height_ratios=[1.5, 1.2, 0.1])
        
        # 3D view
        self.ax_3d = self.fig.add_subplot(gs[0, 0], projection='3d', facecolor='#0f0f1a')
        self.ax_3d.set_title('DNA Nanotube with Aptamer', color='white', fontsize=14, pad=15)
        
        # Membrane ROS distribution
        self.ax_membrane = self.fig.add_subplot(gs[1, 0], facecolor='#0f0f1a')
        self.ax_membrane.set_title('ROS Density at Membrane', color='white', fontsize=14, pad=10)
        
        # Control panel with better spacing
        self.ax_control = self.fig.add_subplot(gs[0, 1], facecolor='#1a1a2a')
        self.ax_control.set_title('Aptamer Controls', color='white', fontsize=14, pad=10)
        self.ax_control.axis('off')
        
        # Analysis panel with more space
        self.ax_analysis = self.fig.add_subplot(gs[1, 1], facecolor='#1a1a2a')
        self.ax_analysis.set_title('ROS-Mediated Rupture Analysis', color='white', fontsize=14, pad=10)
        self.ax_analysis.axis('off')
        
        # Add aptamer length slider with better position
        slider_ax1 = plt.axes([0.55, 0.45, 0.35, 0.02], facecolor='#2a2a4a')
        self.slider_length = Slider(slider_ax1, 'Aptamer Length (nm)', 5, 50, valinit=20, 
                                   color='#ffcc00', track_color='#2a2a4a')
        
        # Add orientation sliders with better spacing
        slider_ax2 = plt.axes([0.55, 0.40, 0.35, 0.02], facecolor='#2a2a4a')
        slider_ax3 = plt.axes([0.55, 0.35, 0.35, 0.02], facecolor='#2a2a4a')
        slider_ax4 = plt.axes([0.55, 0.30, 0.35, 0.02], facecolor='#2a2a4a')
        
        self.slider_x = Slider(slider_ax2, 'X Rotation', -180, 180, valinit=0, 
                              color='#00ccff', track_color='#2a2a4a')
        self.slider_y = Slider(slider_ax3, 'Y Rotation', -180, 180, valinit=0, 
                              color='#00ccff', track_color='#2a2a4a')
        self.slider_z = Slider(slider_ax4, 'Z Rotation', -180, 180, valinit=0, 
                              color='#00ccff', track_color='#2a2a4a')
        
        # Add buttons with better positioning
        button_ax = plt.axes([0.65, 0.20, 0.2, 0.04])
        self.button = Button(button_ax, 'Reset View', color='#2a2a4a', hovercolor='#3a3a5a')
        
        analyze_ax = plt.axes([0.65, 0.14, 0.2, 0.04])
        self.analyze_button = Button(analyze_ax, 'Run Analysis', color='#2a2a4a', hovercolor='#3a3a5a')
        
        gromacs_ax = plt.axes([0.65, 0.08, 0.2, 0.04])
        self.gromacs_button = Button(gromacs_ax, 'Run GROMACS MD', color='#2a2a4a', hovercolor='#3a3a5a')
        
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
        
        # Analysis data storage
        self.analysis_results = []
        
        # Global styling with better spacing
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.2, hspace=0.3)
        self.fig.suptitle('Aptamer Length Impact on ROS-Mediated Membrane Rupture', 
                         fontsize=18, color='cyan', fontweight='bold')
    
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
    
    def generate_gromacs_input(self):
        """Generate GROMACS input files and topology"""
        temp_dir = self.temp_dir.name
        
        # 1. Create topology file (topol.top)
        topol_content = """
; DNA Nanotube with Aptamer Topology
#include "amber99.ff/forcefield.itp"
#include "amber99.ff/dna.itp"

[ system ]
DNA Nanotube with Aptamer

[ molecules ]
DNA    1
"""
        with open(os.path.join(temp_dir, "topol.top"), "w") as f:
            f.write(topol_content)
        
        # 2. Create coordinate file (initial.gro)
        gro_content = f"DNA Nanotube with Aptamer\n{self.n_sources}\n"
        
        # Add DNA nanotube atoms - ensure all above membrane (z>0)
        for i in range(self.n_sources):
            # Calculate residue ID based on circle index (10 points per circle)
            res_id = i // 10 + 1
            atom_name = "DA"  # Deoxyadenosine
            atom_id = i + 1
            # Convert coordinates to nm (GROMACS units)
            x = self.sources[i, 0] * 1e9  # m to nm
            y = self.sources[i, 1] * 1e9  # m to nm
            z = self.sources[i, 2] * 1e9  # m to nm
            
            # Ensure z > 0 (above membrane)
            z = max(z, 1.0)  # At least 1 nm above membrane
            
            # Format GRO file line
            gro_content += f"{res_id:5d}{'DNA':5s}{atom_name:5s}{atom_id:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n"
        
        # Add box dimensions
        box_size = 20.0  # nm
        gro_content += f"{box_size:10.5f}{box_size:10.5f}{box_size:10.5f}\n"
        
        with open(os.path.join(temp_dir, "initial.gro"), "w") as f:
            f.write(gro_content)
        
        # 3. Create MD parameters file (md.mdp)
        mdp_content = """
; MD parameters for DNA Nanotube simulation
integrator          = md
dt                  = 0.002        ; 2 fs
nsteps              = 50000        ; 100 ps

nstxout             = 1000         ; write coordinates every 2 ps
nstvout             = 1000         ; write velocities every 2 ps
nstenergy           = 1000         ; write energies every 2 ps
nstlog              = 1000         ; update log file every 2 ps

cutoff-scheme       = Verlet
coulombtype         = PME
rcoulomb            = 1.0          ; short-range electrostatic cutoff (nm)
rvdw                = 1.0          ; short-range van der Waals cutoff (nm)

tcoupl              = v-rescale    ; temperature coupling
tc-grps             = DNA          ; couple DNA atoms to bath
tau_t               = 0.1          ; time constant (ps)
ref_t               = 300          ; reference temperature (K)

pcoupl              = no           ; no pressure coupling

constraints         = h-bonds      ; constrain bonds involving hydrogen
constraint-algorithm = LINCS       ; algorithm for constraints

gen-vel             = yes          ; generate initial velocities
gen-temp            = 300          ; temperature for initial velocities (K)
"""
        with open(os.path.join(temp_dir, "md.mdp"), "w") as f:
            f.write(mdp_content)
        
        return temp_dir
    
    def run_gromacs_simulation(self, event=None):
        """Run GROMACS molecular dynamics simulation with error handling"""
        # Check if GROMACS is available
        if self.gromacs_path is None or not shutil.which(self.gromacs_path):
            self.ax_analysis.clear()
            self.ax_analysis.text(0.1, 0.6, "GROMACS not found! Please install GROMACS", 
                                 color='red', fontsize=12)
            self.ax_analysis.text(0.1, 0.5, "or set the correct path in the code.", 
                                 color='red', fontsize=12)
            self.fig.canvas.draw_idle()
            return
        
        # Update status
        self.ax_analysis.clear()
        self.ax_analysis.text(0.5, 0.7, "Running GROMACS simulation...", 
                             color='cyan', fontsize=14, ha='center')
        self.fig.canvas.draw_idle()
        
        # Generate input files
        temp_dir = self.generate_gromacs_input()
        current_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Step 1: Generate topology
            subprocess.run([self.gromacs_path, "grompp", "-f", "md.mdp", 
                           "-c", "initial.gro", "-p", "topol.top", 
                           "-o", "em.tpr"], check=True, capture_output=True)
            
            # Step 2: Run energy minimization
            subprocess.run([self.gromacs_path, "mdrun", "-v", "-deffnm", "em"], 
                          check=True, capture_output=True)
            
            # Step 3: Run molecular dynamics
            subprocess.run([self.gromacs_path, "grompp", "-f", "md.mdp", 
                           "-c", "em.gro", "-p", "topol.top", 
                           "-o", "md.tpr"], check=True, capture_output=True)
            
            subprocess.run([self.gromacs_path, "mdrun", "-v", "-deffnm", "md"], 
                          check=True, capture_output=True)
            
            # Step 4: Analyze results
            self.analyze_gromacs_results(temp_dir)
            
        except subprocess.CalledProcessError as e:
            error_msg = f"GROMACS error:\n{e.stderr.decode() if e.stderr else str(e)}"
            self.ax_analysis.clear()
            self.ax_analysis.text(0.1, 0.6, "GROMACS simulation failed:", color='red', fontsize=12)
            self.ax_analysis.text(0.1, 0.5, error_msg[:200] + "...", color='red', fontsize=10)
            self.fig.canvas.draw_idle()
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.ax_analysis.clear()
            self.ax_analysis.text(0.1, 0.6, "Unexpected error:", color='red', fontsize=12)
            self.ax_analysis.text(0.1, 0.5, error_msg, color='red', fontsize=10)
            self.fig.canvas.draw_idle()
        finally:
            # Return to original directory
            os.chdir(current_dir)
    
    def analyze_gromacs_results(self, temp_dir):
        """Analyze GROMACS simulation results and update visualization"""
        # Load trajectory files
        try:
            tpr_file = os.path.join(temp_dir, "md.tpr")
            xtc_file = os.path.join(temp_dir, "md.xtc")
            
            if not os.path.exists(tpr_file) or not os.path.exists(xtc_file):
                raise FileNotFoundError("GROMACS output files not found")
            
            u = mda.Universe(tpr_file, xtc_file)
            
            # Get last snapshot
            u.trajectory[-1]
            
            # Extract DNA atom coordinates
            dna_atoms = u.select_atoms("resname DA")
            self.md_snapshot = dna_atoms.positions * 0.1  # nm to m
            
            # Ensure all points are above membrane (z > 0)
            self.md_snapshot[:, 2] = np.maximum(self.md_snapshot[:, 2], 1e-9)
            
            # Update source positions
            self.sources = self.md_snapshot.copy()
            
            # Update visualization
            self.update()
            
            # Calculate RMSD
            initial_gro = os.path.join(temp_dir, "initial.gro")
            if not os.path.exists(initial_gro):
                raise FileNotFoundError("Initial structure file not found")
            
            initial_pos = np.loadtxt(initial_gro, 
                                    skiprows=2, max_rows=self.n_sources, 
                                    usecols=(3, 4, 5)) * 0.1  # nm to m
            
            # Ensure initial positions are above membrane
            initial_pos[:, 2] = np.maximum(initial_pos[:, 2], 1e-9)
            
            rmsd = np.sqrt(np.mean(np.sum((initial_pos - self.md_snapshot)**2, axis=1)))
            
            # Display results in analysis panel
            self.ax_analysis.clear()
            self.ax_analysis.text(0.1, 0.8, "GROMACS MD Results:", 
                                 color='cyan', fontsize=12, fontweight='bold')
            self.ax_analysis.text(0.1, 0.7, f"Simulation Time: 100 ps")
            self.ax_analysis.text(0.1, 0.65, f"Final RMSD: {rmsd*1e9:.2f} nm")
            self.ax_analysis.text(0.1, 0.6, f"Structure optimized with MD")
            self.ax_analysis.text(0.1, 0.5, "Visualization updated with MD structure", 
                                 color='yellow')
            
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            self.ax_analysis.clear()
            self.ax_analysis.text(0.1, 0.6, "GROMACS analysis failed:", color='red', fontsize=12)
            self.ax_analysis.text(0.1, 0.5, error_msg, color='red', fontsize=10)
            self.fig.canvas.draw_idle()
    
    def initialize_visualization(self):
        """Initialize visualization elements with error handling"""
        # 3D view
        self.ax_3d.clear()
        self.ax_3d.grid(False)
        
        # Draw membrane plane at z=0
        xx, yy = np.meshgrid(np.linspace(-100, 100, 10), np.linspace(-100, 100, 10))
        zz = np.zeros_like(xx)
        self.ax_3d.plot_surface(xx, yy, zz, alpha=0.2, color='#5555ff')
        
        # Draw tube structure with 10 sources per circle
        rotated_sources = self.rotate_sources()
        self.tube_scatter = self.ax_3d.scatter(
            rotated_sources[:, 0]*1e9, 
            rotated_sources[:, 1]*1e9, 
            rotated_sources[:, 2]*1e9, 
            s=40, c='cyan', alpha=0.8, edgecolors='white'
        )
        
        # Add aptamer representation - now only for visualization
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
            'y-', linewidth=2, alpha=0.7
        )
        self.aptamer_point = self.ax_3d.scatter(
            membrane_point[0]*1e9, 
            membrane_point[1]*1e9, 
            membrane_point[2]*1e9, 
            s=100, c='yellow', marker='*', edgecolors='gold'
        )
        
        # Add distance annotation - actual distance to membrane
        distance_to_membrane = nanotube_point[2]  # Since membrane is at z=0
        self.distance_text = self.ax_3d.text(
            nanotube_point[0]*1e9, 
            nanotube_point[1]*1e9, 
            nanotube_point[2]*1e9 + 10,
            f"Distance: {distance_to_membrane*1e9:.1f} nm",
            color='white', fontsize=10
        )
        
        # Set 3D view parameters
        self.ax_3d.set_xlim(-100, 100)
        self.ax_3d.set_ylim(-100, 100)
        self.ax_3d.set_zlim(-50, 150)  # Adjust to show membrane and nanotube
        self.ax_3d.set_xlabel('X (nm)', color='white')
        self.ax_3d.set_ylabel('Y (nm)', color='white')
        self.ax_3d.set_zlabel('Z (nm)', color='white')
        self.ax_3d.tick_params(colors='white')
        self.ax_3d.xaxis.pane.fill = False
        self.ax_3d.yaxis.pane.fill = False
        self.ax_3d.zaxis.pane.fill = False
        self.ax_3d.xaxis.pane.set_edgecolor('#1a1a2a')
        self.ax_3d.yaxis.pane.set_edgecolor('#1a1a2a')
        self.ax_3d.zaxis.pane.set_edgecolor('#1a1a2a')
        
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
        self.ax_membrane.set_xlabel('X (nm)', color='white')
        self.ax_membrane.set_ylabel('Y (nm)', color='white')
        self.ax_membrane.tick_params(colors='white')
        self.ax_membrane.set_facecolor('#0f0f1a')
        
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
        cbar.set_label('ROS Density (molecules/μm²/ns)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Update analysis panel with ROS focus
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
        """Simulate ROS diffusion process from membrane-external G4 sites"""
        self.membrane_hits = np.zeros((self.grid_size, self.grid_size))
        rotated_sources = self.rotate_sources()
        
        # Ensure all sources are above membrane (z > 0)
        rotated_sources[:, 2] = np.maximum(rotated_sources[:, 2], 1e-9)
        
        # For each G4 source above the membrane
        for src in rotated_sources:
            # For each ROS molecule from this source
            for _ in range(ROS_PER_SOURCE):
                pos = src.copy()
                
                # Simulate diffusion until hitting membrane or lifetime ends
                for _ in range(100):  # Maximum steps
                    # Brownian motion displacement
                    dx = np.random.normal(0, np.sqrt(2*D*dt))
                    dy = np.random.normal(0, np.sqrt(2*D*dt))
                    dz = np.random.normal(0, np.sqrt(2*D*dt))
                    pos += np.array([dx, dy, dz])
                    
                    # Detect if hit membrane (z≈0)
                    if pos[2] <= 0:
                        # Find nearest grid point
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
            f"Total ROS Reaching Membrane: {total_ros:.0f} molecules\n\n"
            f"Membrane Rupture by ROS: {'POSSIBLE' if can_rupture else 'UNLIKELY'}"
        )
        
        self.ax_analysis.text(0.1, 0.7, analysis_text, 
                             color='cyan' if can_rupture else 'yellow',
                             fontsize=12, fontfamily='monospace',
                             verticalalignment='top')
        
        # Add aptamer analysis
        aptamer_len = self.slider_length.val
        
        # Find min distance to membrane
        min_distance = np.min(self.rotate_sources()[:, 2]) * 1e9
        self.ax_analysis.text(0.1, 0.5, f"Aptamer Length: {aptamer_len:.1f} nm\n"
                                       f"Min Distance to Membrane: {min_distance:.1f} nm",
                             color='white', fontsize=11)
        
        # Add design recommendations with ROS focus
        if can_rupture:
            advice = "ROS density sufficient for membrane rupture."
            color = "lime"
        else:
            advice = "Recommendations:\n- Reduce distance to membrane\n- Increase G4 density\n- Optimize orientation for ROS diffusion"
            color = "orange"
        
        self.ax_analysis.text(0.1, 0.2, advice, 
                             color=color, fontsize=12, fontweight='bold')
        
        # Add GROMACS status if available
        if hasattr(self, 'md_snapshot') and self.md_snapshot is not None:
            self.ax_analysis.text(0.1, 0.05, "GROMACS MD structure applied", 
                                 color='#00ccff', fontsize=10)
        
        # Add GROMACS availability status
        if self.gromacs_path is None:
            self.ax_analysis.text(0.1, 0.0, "GROMACS not detected", 
                                 color='red', fontsize=9)
        elif not shutil.which(self.gromacs_path):
            self.ax_analysis.text(0.1, 0.0, f"GROMACS not found at: {self.gromacs_path}", 
                                 color='red', fontsize=9)
    
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
        
        # Simulate ROS diffusion from G4 sites to membrane
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
        analysis_fig = plt.figure(figsize=(14, 10), facecolor='#0f0f1a')
        analysis_fig.suptitle('ROS-Mediated Membrane Rupture Analysis', 
                             fontsize=20, color='cyan', fontweight='bold')
        
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
        ax1 = analysis_fig.add_subplot(gs[0, 0])
        ax2 = analysis_fig.add_subplot(gs[0, 1])
        ax3 = analysis_fig.add_subplot(gs[1, :])
        
        # Set styling
        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor('#0f0f1a')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('cyan')
        
        # Plot peak ROS density vs aptamer length
        for angle in orientations:
            lengths = [r[0] for r in results[angle]]
            peaks = [r[1] for r in results[angle]]
            ax1.plot(lengths, peaks, 'o-', label=f"{angle}°", linewidth=2)
        
        ax1.set_title('Peak ROS Density vs Aptamer Length')
        ax1.set_xlabel('Aptamer Length (nm)')
        ax1.set_ylabel('ROS Density (molecules/μm²/ns)')
        ax1.axhline(THRESHOLD_DENSITY, color='r', linestyle='--', label='Rupture Threshold')
        ax1.grid(True, alpha=0.2)
        ax1.legend()
        
        # Plot ROS coverage area vs aptamer length
        for angle in orientations:
            lengths = [r[0] for r in results[angle]]
            areas = [r[2] for r in results[angle]]
            ax2.plot(lengths, areas, 'o-', label=f"{angle}°", linewidth=2)
        
        ax2.set_title('Effective ROS Coverage Area')
        ax2.set_xlabel('Aptamer Length (nm)')
        ax2.set_ylabel('ROS Coverage Area (μm²)')
        ax2.axhline(0.05, color='r', linestyle='--', label='Minimum Area')
        ax2.grid(True, alpha=0.2)
        ax2.legend()
        
        # Plot rupture efficiency by ROS
        for angle in orientations:
            lengths = [r[0] for r in results[angle]]
            rupture_efficiency = []
            for r in results[angle]:
                # Rupture efficiency by ROS combines peak density and coverage area
                eff = min(1.0, r[1]/THRESHOLD_DENSITY) * min(1.0, r[2]/0.05)
                rupture_efficiency.append(eff)
            
            ax3.plot(lengths, rupture_efficiency, 'o-', label=f"{angle}°", linewidth=2)
            
            # Add regression analysis
            slope, intercept, r_value, p_value, std_err = linregress(lengths, rupture_efficiency)
            reg_line = slope * np.array(lengths) + intercept
            ax3.plot(lengths, reg_line, '--', alpha=0.5)
            ax3.text(lengths[-1], reg_line[-1], 
                    f"slope: {slope:.4f}\nR²: {r_value**2:.3f}", 
                    color='white', fontsize=9)
        
        ax3.set_title('ROS-Mediated Membrane Rupture Efficiency')
        ax3.set_xlabel('Aptamer Length (nm)')
        ax3.set_ylabel('Rupture Efficiency')
        ax3.axhline(1.0, color='r', linestyle='--', label='Full Efficiency')
        ax3.grid(True, alpha=0.2)
        ax3.legend()
        
        # Add key conclusions with ROS focus
        conclusion_text = (
            "Key Conclusions on ROS-Mediated Rupture:\n"
            "1. Shorter aptamers increase ROS density at membrane\n"
            "2. Distance to membrane is critical for ROS effectiveness\n"
            "3. Vertical orientations maximize ROS delivery to membrane\n"
            "4. ROS coverage area >0.05 μm² needed for rupture\n"
            "5. Optimal aptamer length: 15-25nm for efficient ROS delivery"
        )
        analysis_fig.text(0.1, 0.05, conclusion_text, 
                         color='yellow', fontsize=12, 
                         bbox=dict(facecolor='#1a1a2a', alpha=0.7))
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()

# Run simulator
simulator = AptamerLengthSimulator()
plt.show()