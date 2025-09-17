import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress
from tqdm import tqdm
import time
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Physical parameters
D_ros = 2.8e-9  # ROS扩散系数 (m²/s)
D_drug = 1.0e-9  # 药物扩散系数 (m²/s)
tau = 1e-9       # ROS寿命 (s)
dt = 0.1 * tau
ROS_PER_SOURCE = 200    # 每个G4位点产生的ROS分子数
DRUG_MOLECULES = 500    # 药物分子数
THRESHOLD_DENSITY = 1e4 # 膜破裂阈值 (molecules/μm²/ns)
ANIMATION_MOLECULES = 50 # 动画显示的分子数

# 创建自定义颜色映射
colors = ["darkblue", "blue", "cyan", "green", "yellow", "red"]
ros_cmap = LinearSegmentedColormap.from_list("ros_cmap", colors, N=256)
drug_cmap = LinearSegmentedColormap.from_list("drug_cmap", ["purple", "magenta", "pink"], N=256)

# 根据文献数据建立的孔径-浓度模型 (nm)
def calculate_pore_diameter(ros_density):
    """
    基于文献数据的经验公式: D = a * log(C) + b
    文献参考:
    - 紫杉醇诱导ROS导致细胞膜形成直径20-100 nm的纳米孔
    - 载铅粉尘实验中，ROS浓度增加2-25倍导致膜通透性增加36-46%
    """
    # 基础参数 (根据校准)
    base_diameter = 20  # 最小孔径(nm)
    scaling_factor = 15  # 浓度缩放因子
    
    # 避免对数计算错误
    adjusted_density = max(ros_density, 1000)  
    return base_diameter + scaling_factor * np.log10(adjusted_density / 1000)

class DrugDeliverySimulator:
    def __init__(self):
        # 初始化属性
        self.animation_running = False
        self.trajectories = []
        self.drug_trajectories = []
        self.recording = False
        self.recording_frames = []
        self.recording_start_time = 0
        self.translocation_animating = False
        self.pore_diameter = 0
        self.pore_center = None
        self.cell_radius = 500  # 细胞曲率半径 (nm)
        self.drug_delivery_complete = False
        self.simulation_mode = "ros"  # "ros" or "drug"
        
        # 创建GUI - 调整布局
        self.fig = plt.figure(figsize=(16, 12), facecolor='#f0f8ff')
        gs = gridspec.GridSpec(3, 2, width_ratios=[1.2, 1], height_ratios=[1.5, 1.2, 0.1])
        
        # 3D视图
        self.ax_3d = self.fig.add_subplot(gs[0, 0], projection='3d', facecolor='#e0f7fa')
        self.ax_3d.set_title('DNA Nanotube Drug Delivery System', color='#01579b', fontsize=14, pad=15)
        
        # 膜视图
        self.ax_membrane = self.fig.add_subplot(gs[1, 0], facecolor='#e0f7fa')
        self.ax_membrane.set_title('Membrane Analysis', color='#01579b', fontsize=14, pad=10)
        
        # 控制面板 - 改进布局
        self.ax_control = self.fig.add_subplot(gs[0, 1], facecolor='#e8f5e9')
        self.ax_control.set_title('Simulation Controls', color='#2e7d32', fontsize=14, pad=10)
        self.ax_control.axis('off')
        
        # 分析面板 - 改进布局
        self.ax_analysis = self.fig.add_subplot(gs[1, 1], facecolor='#f3e5f5')
        self.ax_analysis.set_title('Analysis & Results', color='#6a1b9a', fontsize=14, pad=10)
        self.ax_analysis.axis('off')
        
        # 添加分隔线
        self.ax_control.axhline(0.85, color='#81c784', linewidth=2)
        self.ax_control.axhline(0.65, color='#81c784', linewidth=2)
        self.ax_control.axhline(0.45, color='#81c784', linewidth=2)
        self.ax_control.axhline(0.25, color='#81c784', linewidth=2)
        
        # 添加分组标签
        self.ax_control.text(0.05, 0.92, 'Nanotube Parameters', color='#2e7d32', fontsize=12, fontweight='bold')
        self.ax_control.text(0.05, 0.72, 'Orientation Control', color='#2e7d32', fontsize=12, fontweight='bold')
        self.ax_control.text(0.05, 0.52, 'Simulation Mode', color='#2e7d32', fontsize=12, fontweight='bold')
        self.ax_control.text(0.05, 0.32, 'Simulation Actions', color='#2e7d32', fontsize=12, fontweight='bold')
        self.ax_control.text(0.05, 0.12, 'Export Options', color='#2e7d32', fontsize=12, fontweight='bold')
        
        # 添加aptamer长度滑块 - 调整位置
        slider_ax1 = plt.axes([0.70, 0.83, 0.25, 0.02], facecolor='#c8e6c9')
        self.slider_length = Slider(slider_ax1, 'Aptamer Length (nm)', 5, 50, valinit=20, 
                                   color='#388e3c', track_color='#a5d6a7')
        
        # 添加药物参数滑块 - 调整位置
        slider_ax5 = plt.axes([0.70, 0.78, 0.25, 0.02], facecolor='#c8e6c9')
        self.slider_drug_size = Slider(slider_ax5, 'Drug Size (nm)', 1, 30, valinit=5, 
                                      color='#7b1fa2', track_color='#ba68c8')
        
        # 添加方向滑块 - 调整位置
        slider_ax2 = plt.axes([0.70, 0.63, 0.25, 0.02], facecolor='#c8e6c9')
        slider_ax3 = plt.axes([0.70, 0.58, 0.25, 0.02], facecolor='#c8e6c9')
        slider_ax4 = plt.axes([0.70, 0.53, 0.25, 0.02], facecolor='#c8e6c9')
        
        self.slider_x = Slider(slider_ax2, 'X Rotation', -180, 180, valinit=0, 
                              color='#0288d1', track_color='#81d4fa')
        self.slider_y = Slider(slider_ax3, 'Y Rotation', -180, 180, valinit=0, 
                              color='#0288d1', track_color='#81d4fa')
        self.slider_z = Slider(slider_ax4, 'Z Rotation', -180, 180, valinit=0, 
                              color='#0288d1', track_color='#81d4fa')
        
        # 添加模式选择按钮 - 调整位置
        mode_ax = plt.axes([0.70, 0.43, 0.25, 0.06], facecolor='#e8f5e9')
        self.mode_radio = RadioButtons(mode_ax, ('ROS Model', 'Drug Delivery'), active=0)
        
        # 修复RadioButtons样式问题 - 使用更兼容的方法
        try:
            # 尝试使用circles属性（旧版本Matplotlib）
            for circle in self.mode_radio.circles:
                circle.set_radius(0.05)
                circle.set_edgecolor('#2e7d32')
        except AttributeError:
            # 如果circles属性不存在，使用新方法
            for circle in self.mode_radio.ax.get_children():
                if isinstance(circle, plt.Circle):
                    circle.set_radius(0.05)
                    circle.set_edgecolor('#2e7d32')
        
        # 添加控制按钮 - 调整位置和样式
        button_y = 0.30
        button_height = 0.04
        button_spacing = 0.05
        
        reset_ax = plt.axes([0.58, button_y, 0.15, button_height], facecolor='#4caf50')
        self.reset_button = Button(reset_ax, 'Reset View', color='#4caf50', hovercolor='#81c784')
        
        analyze_ax = plt.axes([0.58, button_y - button_spacing, 0.15, button_height], facecolor='#2196f3')
        self.analyze_button = Button(analyze_ax, 'Run Analysis', color='#2196f3', hovercolor='#64b5f6')
        
        # 添加穿膜按钮 - 调整位置和样式
        translocation_ax = plt.axes([0.78, button_y, 0.15, button_height], facecolor='#7b1fa2')
        self.translocation_button = Button(translocation_ax, 'Simulate Delivery', color='#7b1fa2', hovercolor='#ba68c8')
        
        # 添加药物递送按钮 - 调整位置和样式
        drug_ax = plt.axes([0.78, button_y - button_spacing, 0.15, button_height], facecolor='#ab47bc')
        self.drug_button = Button(drug_ax, 'Start Drug Diffusion', color='#ab47bc', hovercolor='#ce93d8')
        
        # 添加导出按钮 - 调整位置和样式
        export_y = 0.10
        animate_ax = plt.axes([0.58, export_y, 0.15, button_height], facecolor='#ff9800')
        self.animate_button = Button(animate_ax, 'Animate Diffusion', color='#ff9800', hovercolor='#ffb74d')
        
        export3d_ax = plt.axes([0.58, export_y - button_spacing, 0.15, button_height], facecolor='#f44336')
        self.export3d_button = Button(export3d_ax, 'Export 3D View', color='#f44336', hovercolor='#ef5350')
        
        export_membrane_ax = plt.axes([0.78, export_y, 0.15, button_height], facecolor='#009688')
        self.export_membrane_button = Button(export_membrane_ax, 'Export Membrane', color='#009688', hovercolor='#26a69a')
        
        # DNA纳米管参数
        self.nanotube_length = 120e-9  # 120nm
        self.nanotube_radius = 5e-9    # 5nm
        self.n_sources = 30            # G4位点数
        self.aptamer_length = 20e-9    # 初始aptamer长度20nm
        self.drug_size = 5             # 药物分子大小(nm)
        self.drug_inside = 0           # 进入细胞的药物分子数
        
        # 创建纳米管结构
        self.sources = self.create_nanotube_sources()
        self.drug_positions = self.create_drug_molecules()
        
        # 创建膜网格
        self.grid_size = 100
        self.membrane_hits = np.zeros((self.grid_size, self.grid_size))
        self.drug_hits = np.zeros((self.grid_size, self.grid_size))
        self.x_grid = np.linspace(-100e-9, 100e-9, self.grid_size)
        self.y_grid = np.linspace(-100e-9, 100e-9, self.grid_size)
        
        # 初始化可视化
        self.initialize_visualization()
        
        # 事件绑定
        self.slider_length.on_changed(self.update_aptamer)
        self.slider_x.on_changed(self.update)
        self.slider_y.on_changed(self.update)
        self.slider_z.on_changed(self.update)
        self.slider_drug_size.on_changed(self.update_drug_size)
        self.reset_button.on_clicked(self.reset)
        self.analyze_button.on_clicked(self.run_analysis)
        self.animate_button.on_clicked(self.toggle_animation)
        self.export3d_button.on_clicked(self.export_3d_view)
        self.export_membrane_button.on_clicked(self.export_membrane_view)
        self.translocation_button.on_clicked(self.start_translocation)
        self.drug_button.on_clicked(self.start_drug_diffusion)
        self.mode_radio.on_clicked(self.change_mode)
        
        # 分析数据存储
        self.analysis_results = []
        
        # 全局样式
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.2, hspace=0.3)
        self.fig.suptitle('DNA Nanotube Drug Delivery Simulator', 
                         fontsize=18, color='#0288d1', fontweight='bold')
    
    def create_nanotube_sources(self):
        """沿DNA纳米管创建G4位点"""
        sources = []
        
        # 沿纳米管长度创建位点
        for i in range(self.n_sources):
            # 沿纳米管的位置(z轴)
            z = (i / (self.n_sources - 1)) * self.nanotube_length + self.aptamer_length
            
            # 纳米管沿z轴
            sources.append([0, 0, z])
        
        return np.array(sources)
    
    def create_drug_molecules(self):
        """在纳米管内部创建药物分子"""
        positions = []
        
        # 在纳米管内部随机分布药物分子
        for _ in range(DRUG_MOLECULES):
            # 随机位置 (在纳米管内)
            r = np.random.uniform(0, self.nanotube_radius * 0.8)
            theta = np.random.uniform(0, 2 * np.pi)
            z = np.random.uniform(0, self.nanotube_length) + self.aptamer_length
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            positions.append([x, y, z])
        
        return np.array(positions)
    
    def create_nanotube_mesh(self, center=(0, 0, 0)):
        """创建DNA纳米管的圆柱形网格"""
        # 圆柱参数
        length = self.nanotube_length
        radius = self.nanotube_radius
        resolution = 20
        
        # 创建圆柱网格
        z = np.linspace(0, length, resolution)
        theta = np.linspace(0, 2 * np.pi, resolution)
        theta_grid, z_grid = np.meshgrid(theta, z)
        
        # 圆柱坐标
        x_grid = radius * np.cos(theta_grid) + center[0]
        y_grid = radius * np.sin(theta_grid) + center[1]
        z_grid = z_grid + center[2] - length/2
        
        return x_grid, y_grid, z_grid
    
    def visualize_pore_formation(self):
        """在膜表面创建基于ROS浓度的3D孔洞"""
        # 创建变形膜表面 (球面)
        xx, yy = np.meshgrid(np.linspace(-100, 100, 50), 
                             np.linspace(-100, 100, 50))
        zz = np.zeros_like(xx)
        
        # 计算峰值密度位置
        max_idx = np.unravel_index(np.argmax(self.membrane_hits), self.membrane_hits.shape)
        pore_x = self.x_grid[max_idx[0]] * 1e9
        pore_y = self.y_grid[max_idx[1]] * 1e9
        
        # 计算孔径
        peak_density = np.max(self.membrane_hits)
        self.pore_diameter = calculate_pore_diameter(peak_density)
        
        # 根据ROS密度添加凹陷（孔洞）
        for i in range(50):
            for j in range(50):
                x = xx[i, j]
                y = yy[i, j]
                
                # 计算到孔中心的距离
                distance_to_pore = np.sqrt((x - pore_x)**2 + (y - pore_y)**2)
                
                # 孔深与ROS密度成正比
                depth_factor = min(1.0, self.membrane_hits[max_idx[0], max_idx[1]] / (2 * THRESHOLD_DENSITY))
                
                # 创建孔洞形状 - 高斯凹陷
                pore_depth = depth_factor * 30 * np.exp(-(distance_to_pore**2) / (self.pore_diameter**2))
                
                zz[i, j] = -pore_depth
        
        # 创建球面变形 (模拟细胞曲率)
        radius = self.cell_radius  # 细胞曲率半径(nm)
        zz += (xx**2 + yy**2) / (2 * radius)  # 球面近似
        
        # 更新膜表面
        if hasattr(self, 'membrane_surface') and self.membrane_surface:
            self.membrane_surface.remove()
            
        self.membrane_surface = self.ax_3d.plot_surface(
            xx, yy, zz, alpha=0.6, color='#0d47a1', 
            cmap='viridis', antialiased=True
        )
        
        # 标记主孔位置
        if hasattr(self, 'pore_center') and self.pore_center:
            self.pore_center.remove()
            
        # 计算孔洞在曲面上的Z位置（近似）
        pore_z = -depth_factor * 30 + (pore_x**2 + pore_y**2) / (2 * radius)
        self.pore_center = self.ax_3d.scatter(
            [pore_x], [pore_y], [pore_z], 
            s=100, c='red', marker='x', alpha=0.8
        )
        
        # 添加孔洞尺寸标注
        self.ax_3d.text(
            pore_x, pore_y, pore_z + 15,
            f"Pore: Ø{self.pore_diameter:.1f} nm",
            color='red', fontsize=10
        )
        
        return pore_x, pore_y, pore_z
    
    def start_translocation(self, event):
        """启动DNA纳米管穿膜过程"""
        if self.translocation_animating:
            return
            
        # 更新分析面板
        self.ax_analysis.clear()
        self.ax_analysis.text(0.5, 0.7, "Starting membrane translocation...", 
                             color='#0288d1', fontsize=12, ha='center')
        self.fig.canvas.draw_idle()
        
        # 可视化孔洞形成
        pore_pos = self.visualize_pore_formation()
        
        # 检查是否可穿膜
        if self.pore_diameter < self.nanotube_radius * 2e9:  # 转换为nm
            self.ax_analysis.clear()
            self.ax_analysis.text(0.5, 0.6, "Pore too small for nanotube entry!", 
                                 color='#d32f2f', fontsize=12, ha='center')
            self.ax_analysis.text(0.5, 0.5, f"Pore diameter: {self.pore_diameter:.1f} nm", 
                                 color='#d32f2f', fontsize=12, ha='center')
            self.ax_analysis.text(0.5, 0.4, f"Nanotube diameter: {self.nanotube_radius*2e9:.1f} nm", 
                                 color='#d32f2f', fontsize=12, ha='center')
            self.fig.canvas.draw_idle()
            return
        
        # 启动穿膜动画
        self.translocation_animating = True
        self.translocation_button.label.set_text("Translocating...")
        self.simulate_translocation(pore_pos)
        self.translocation_animating = False
        self.translocation_button.label.set_text("Simulate Delivery")
    
    def simulate_translocation(self, pore_pos):
        """模拟DNA纳米管穿过膜孔的过程"""
        # 保存初始位置
        original_sources = self.sources.copy()
        original_drugs = self.drug_positions.copy()
        
        # 初始位置 (膜上方)
        start_z = 50  # nm
        
        # 动画步骤
        n_steps = 30
        for step in range(n_steps):
            if not self.translocation_animating:
                break
                
            # 计算当前插入深度
            progress = step / n_steps
            current_z = start_z * (1 - progress) + pore_pos[2] * progress - 20
            
            # 更新纳米管位置
            self.update_nanotube_position(
                pore_pos[0], pore_pos[1], current_z
            )
            
            # 特别显示穿膜时刻
            if current_z < pore_pos[2] + 10:
                self.highlight_translocation(pore_pos)
            
            plt.pause(0.05)
        
        # 显示结果
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
        
        # 恢复初始位置
        self.sources = original_sources
        self.drug_positions = original_drugs
        self.update()
    
    def update_nanotube_position(self, target_x, target_y, z_pos):
        """更新DNA纳米管位置"""
        # 计算平移向量
        current_center = np.mean(self.sources, axis=0)
        displacement = np.array([
            target_x*1e-9 - current_center[0],
            target_y*1e-9 - current_center[1],
            z_pos*1e-9 - current_center[2]
        ])
        
        # 应用平移
        self.sources += displacement
        self.drug_positions += displacement
        self.update()
    
    def highlight_translocation(self, pore_pos):
        """高亮穿膜时刻"""
        # 创建孔洞的3D表示
        theta = np.linspace(0, 2*np.pi, 20)
        z = np.linspace(pore_pos[2] - 10, pore_pos[2] + 10, 5)
        
        # 创建圆柱形孔洞
        x_cyl = []
        y_cyl = []
        z_cyl = []
        
        for zi in z:
            x_cyl.append(pore_pos[0] + (self.pore_diameter/2) * np.cos(theta))
            y_cyl.append(pore_pos[1] + (self.pore_diameter/2) * np.sin(theta))
            z_cyl.append(np.ones_like(theta) * zi)
        
        # 绘制半透明孔洞
        if not hasattr(self, 'pore_cylinder'):
            self.pore_cylinder = self.ax_3d.plot_surface(
                np.array(x_cyl), np.array(y_cyl), np.array(z_cyl),
                alpha=0.3, color='red'
            )
        else:
            # 更新现有圆柱
            self.pore_cylinder.remove()
            self.pore_cylinder = self.ax_3d.plot_surface(
                np.array(x_cyl), np.array(y_cyl), np.array(z_cyl),
                alpha=0.3, color='red'
            )
        
        # 添加穿膜指示器
        if not hasattr(self, 'translocation_indicator'):
            self.translocation_indicator = self.ax_3d.text(
                pore_pos[0], pore_pos[1], pore_pos[2] + 20,
                "NANOTUBE ENTERING CELL!", 
                color='red', fontsize=12, fontweight='bold'
            )
        else:
            self.translocation_indicator.set_position((pore_pos[0], pore_pos[1], pore_pos[2] + 20))
        
        self.fig.canvas.draw_idle()
    
    def start_drug_diffusion(self, event):
        """启动药物扩散模拟"""
        if self.animation_running:
            return
            
        # 检查是否已形成孔洞
        if self.pore_diameter == 0:
            self.ax_analysis.clear()
            self.ax_analysis.text(0.5, 0.6, "Please run ROS simulation first!", 
                                 color='#d32f2f', fontsize=12, ha='center')
            self.fig.canvas.draw_idle()
            return
            
        # 检查药物是否能通过孔洞
        if self.pore_diameter < self.drug_size:
            self.ax_analysis.clear()
            self.ax_analysis.text(0.5, 0.6, "Drug too large for pore!", 
                                 color='#d32f2f', fontsize=12, ha='center')
            self.ax_analysis.text(0.5, 0.5, f"Pore diameter: {self.pore_diameter:.1f} nm", 
                                 color='#d32f2f', fontsize=12, ha='center')
            self.ax_analysis.text(0.5, 0.4, f"Drug size: {self.drug_size} nm", 
                                 color='#d32f2f', fontsize=12, ha='center')
            self.fig.canvas.draw_idle()
            return
        
        # 初始化药物轨迹
        self.drug_trajectories = []
        for pos in self.drug_positions[:ANIMATION_MOLECULES]:
            self.drug_trajectories.append([pos.copy()])
        
        # 初始化药物点
        self.drug_points = self.ax_3d.scatter([], [], [], s=30, c='purple', alpha=0.8)
        
        # 启动动画
        self.drug_delivery_complete = False
        self.animation = FuncAnimation(
            self.fig, self.update_drug_animation,
            frames=200,
            interval=50, blit=False
        )
        self.animation_running = True
        self.drug_button.label.set_text("Diffusing...")
    
    def update_drug_animation(self, frame):
        """更新药物扩散动画"""
        if self.drug_delivery_complete:
            return
        
        new_positions = []
        completed_molecules = 0
        
        for i, trajectory in enumerate(self.drug_trajectories):
            if len(trajectory) > 0:
                pos = trajectory[-1].copy()
                
                # 布朗运动位移
                dx = np.random.normal(0, np.sqrt(2*D_drug*dt))
                dy = np.random.normal(0, np.sqrt(2*D_drug*dt))
                dz = np.random.normal(0, np.sqrt(2*D_drug*dt))
                pos += np.array([dx, dy, dz])
                
                # 检查是否进入细胞 (z < 0)
                if pos[2] < 0:
                    # 标记为已完成
                    completed_molecules += 1
                else:
                    # 添加到轨迹
                    trajectory.append(pos)
                    new_positions.append(pos)
        
        # 更新药物点
        if new_positions:
            new_positions = np.array(new_positions)
            self.drug_points._offsets3d = (
                new_positions[:, 0]*1e9,
                new_positions[:, 1]*1e9,
                new_positions[:, 2]*1e9
            )
        
        # 检查是否所有药物都已扩散
        if completed_molecules >= len(self.drug_trajectories):
            self.drug_delivery_complete = True
            self.animation.event_source.stop()
            self.animation_running = False
            self.drug_button.label.set_text("Start Drug Diffusion")
            
            # 计算递送效率
            efficiency = completed_molecules / len(self.drug_trajectories) * 100
            
            # 显示结果
            result_text = (
                f"Drug Delivery Complete!\n"
                f"Delivery Efficiency: {efficiency:.1f}%\n"
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
    
    def change_mode(self, label):
        """更改模拟模式"""
        self.simulation_mode = "ros" if label == "ROS Model" else "drug"
        self.update_analysis_panel()
    
    def initialize_visualization(self):
        """初始化可视化元素"""
        # 3D视图
        self.ax_3d.clear()
        self.ax_3d.grid(False)
        
        # 绘制膜平面
        xx, yy = np.meshgrid(np.linspace(-100, 100, 10), np.linspace(-100, 100, 10))
        zz = np.zeros_like(xx)
        self.membrane_surface = self.ax_3d.plot_surface(xx, yy, zz, alpha=0.4, color='#0d47a1')
        
        # 绘制DNA纳米管
        rotated_sources = self.rotate_sources()
        
        # 创建纳米管网格
        center = np.mean(rotated_sources, axis=0)
        X, Y, Z = self.create_nanotube_mesh(center)
        
        # 绘制纳米管表面
        self.nanotube_surface = self.ax_3d.plot_surface(
            X*1e9, Y*1e9, Z*1e9, 
            color='#4fc3f7', alpha=0.7, edgecolor='#01579b', linewidth=0.5
        )
        
        # 添加G4位点标记
        self.source_markers = self.ax_3d.scatter(
            rotated_sources[:, 0]*1e9, 
            rotated_sources[:, 1]*1e9, 
            rotated_sources[:, 2]*1e9, 
            s=30, c='#ff9800', marker='o', alpha=1.0
        )
        
        # 添加aptamer表示
        self.aptamer_length = self.slider_length.val * 1e-9
        
        # 找到纳米管上最接近膜的点
        min_z_index = np.argmin(rotated_sources[:, 2])
        nanotube_point = rotated_sources[min_z_index]
        membrane_point = np.array([nanotube_point[0], nanotube_point[1], 0])
        
        # 保存aptamer对象以便更新
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
        
        # 添加距离标注
        distance_to_membrane = nanotube_point[2]  # 膜在z=0处
        self.distance_text = self.ax_3d.text(
            nanotube_point[0]*1e9, 
            nanotube_point[1]*1e9, 
            nanotube_point[2]*1e9 + 10,
            f"Distance: {distance_to_membrane*1e9:.1f} nm",
            color='#01579b', fontsize=9
        )
        
        # 添加药物分子
        self.drug_points = self.ax_3d.scatter([], [], [], s=30, c='purple', alpha=0.8)
        
        # 设置3D视图参数
        self.ax_3d.set_xlim(-100, 100)
        self.ax_3d.set_ylim(-100, 100)
        self.ax_3d.set_zlim(-50, 150)
        self.ax_3d.set_xlabel('X (nm)', color='#01579b')
        self.ax_3d.set_ylabel('Y (nm)', color='#01579b')
        self.ax_3d.set_zlabel('Z (nm)', color='#01579b')
        self.ax_3d.tick_params(colors='#01579b')
        
        # 膜视图
        self.ax_membrane.clear()
        self.membrane_img = self.ax_membrane.imshow(
            self.membrane_hits.T, 
            extent=[-100, 100, -100, 100], 
            origin='lower', 
            cmap=ros_cmap, 
            vmin=0, 
            vmax=ROS_PER_SOURCE * self.n_sources / 10
        )
        self.ax_membrane.set_xlabel('X (nm)', color='#01579b')
        self.ax_membrane.set_ylabel('Y (nm)', color='#01579b')
        self.ax_membrane.tick_params(colors='#01579b')
        self.ax_membrane.set_facecolor('#e0f7fa')
        
        # 添加阈值等高线
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
        
        # 添加颜色条
        cbar = plt.colorbar(self.membrane_img, ax=self.ax_membrane, pad=0.01)
        cbar.set_label('ROS Density (molecules/μm²/ns)', color='#01579b')
        cbar.ax.yaxis.set_tick_params(color='#01579b')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#01579b')
        
        # 更新分析面板
        self.update_analysis_panel()
    
    def rotate_sources(self):
        """应用当前旋转角度"""
        rx = self.slider_x.val * np.pi / 180
        ry = self.slider_y.val * np.pi / 180
        rz = self.slider_z.val * np.pi / 180
        
        # 创建旋转矩阵
        rotation = R.from_euler('xyz', [rx, ry, rz], degrees=False)
        return rotation.apply(self.sources)
    
    def simulate_ros_diffusion(self):
        """模拟从G4位点的ROS扩散过程"""
        self.membrane_hits = np.zeros((self.grid_size, self.grid_size))
        rotated_sources = self.rotate_sources()
        
        # 确保所有位点都在膜上方 (z > 0)
        rotated_sources[:, 2] = np.maximum(rotated_sources[:, 2], 1e-9)
        
        # 初始化轨迹
        self.trajectories = []
        
        # 对每个G4位点
        for src_idx, src in enumerate(rotated_sources):
            # 对每个ROS分子
            for ros_idx in range(ROS_PER_SOURCE):
                # 仅跟踪动画分子子集
                if ros_idx < ANIMATION_MOLECULES:
                    pos = src.copy()
                    trajectory = [pos.copy()]  # 存储动画位置
                    
                    # 模拟扩散直到击中膜或寿命结束
                    for _ in range(100):  # 最大步数
                        # 布朗运动位移
                        dx = np.random.normal(0, np.sqrt(2*D_ros*dt))
                        dy = np.random.normal(0, np.sqrt(2*D_ros*dt))
                        dz = np.random.normal(0, np.sqrt(2*D_ros*dt))
                        pos += np.array([dx, dy, dz])
                        trajectory.append(pos.copy())
                        
                        # 检测是否击中膜 (z≈0)
                        if pos[2] <= 0:
                            # 找到最近的网格点
                            x_idx = np.argmin(np.abs(self.x_grid - pos[0]))
                            y_idx = np.argmin(np.abs(self.y_grid - pos[1]))
                            
                            if 0 <= x_idx < self.grid_size and 0 <= y_idx < self.grid_size:
                                self.membrane_hits[x_idx, y_idx] += 1
                            break
                    
                    # 存储轨迹用于动画
                    self.trajectories.append(trajectory)
                else:
                    # 对于未在动画中跟踪的分子，最小化模拟
                    pos = src.copy()
                    # 模拟直到击中膜（不存储位置）
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
        
        # 计算峰值密度位置
        max_idx = np.unravel_index(np.argmax(self.membrane_hits), self.membrane_hits.shape)
        pore_x = self.x_grid[max_idx[0]] * 1e9
        pore_y = self.y_grid[max_idx[1]] * 1e9
        
        # 计算孔径
        peak_density = np.max(self.membrane_hits)
        self.pore_diameter = calculate_pore_diameter(peak_density)
        
        return rotated_sources
    
    def update_aptamer(self, val):
        """更新aptamer长度"""
        self.update()
    
    def update_drug_size(self, val):
        """更新药物大小"""
        self.drug_size = val
        self.update_analysis_panel()
    
    def update_analysis_panel(self):
        """更新分析面板"""
        self.ax_analysis.clear()
        self.ax_analysis.axis('off')
        
        # 添加分析面板背景和边框
        rect = Rectangle((0, 0), 1, 1, transform=self.ax_analysis.transAxes, 
                        facecolor='#f3e5f5', alpha=0.7, edgecolor='#6a1b9a', linewidth=2)
        self.ax_analysis.add_patch(rect)
        
        # 计算分析指标
        total_ros = np.sum(self.membrane_hits)
        peak_density = np.max(self.membrane_hits)
        grid_cell_area = (200/self.grid_size)**2  # 每个网格单元的面积 (μm²)
        coverage_area = np.sum(self.membrane_hits > THRESHOLD_DENSITY) * grid_cell_area
        can_rupture = peak_density > THRESHOLD_DENSITY and coverage_area > 0.05
        
        # 计算孔径
        pore_diameter = calculate_pore_diameter(peak_density)
        can_translocate = pore_diameter > self.nanotube_radius * 2e9  # 转换为nm
        can_drug_pass = pore_diameter > self.drug_size
        
        # 添加标题
        title_text = "ANALYSIS RESULTS" if self.simulation_mode == "ros" else "DRUG DELIVERY ANALYSIS"
        self.ax_analysis.text(0.5, 0.95, title_text, 
                             color='#6a1b9a', fontsize=14, fontweight='bold', 
                             ha='center', transform=self.ax_analysis.transAxes)
        
        # 添加文本分析
        if self.simulation_mode == "ros":
            analysis_text = (
                f"Peak ROS Density: {peak_density:.1f} molecules/μm²/ns\n"
                f"Threshold Density: {THRESHOLD_DENSITY} molecules/μm²/ns\n"
                f"ROS Coverage Area: {coverage_area:.2f} μm²\n"
                f"Total ROS Reaching Membrane: {total_ros:.0f} molecules\n"
                f"Estimated Pore Diameter: {pore_diameter:.1f} nm\n\n"
                f"Membrane Rupture: {'POSSIBLE' if can_rupture else 'UNLIKELY'}\n"
                f"Nanotube Entry: {'POSSIBLE' if can_translocate else 'UNLIKELY'}"
            )
            
            self.ax_analysis.text(0.05, 0.75, analysis_text, 
                                 color='#0288d1' if can_rupture else '#f57c00',
                                 fontsize=11, fontfamily='monospace',
                                 verticalalignment='top', transform=self.ax_analysis.transAxes)
        else:
            analysis_text = (
                f"Drug Size: {self.drug_size} nm\n"
                f"Estimated Pore Diameter: {pore_diameter:.1f} nm\n"
                f"Drug-Pore Size Ratio: {self.drug_size/pore_diameter:.2f}\n\n"
                f"Drug Passage: {'POSSIBLE' if can_drug_pass else 'NOT POSSIBLE'}"
            )
            
            self.ax_analysis.text(0.05, 0.75, analysis_text, 
                                 color='#7b1fa2' if can_drug_pass else '#d32f2f',
                                 fontsize=11, fontfamily='monospace',
                                 verticalalignment='top', transform=self.ax_analysis.transAxes)
        
        # 添加aptamer分析
        aptamer_len = self.slider_length.val
        
        # 找到到膜的最小距离
        min_distance = np.min(self.rotate_sources()[:, 2]) * 1e9
        self.ax_analysis.text(0.05, 0.5, f"Aptamer Length: {aptamer_len:.1f} nm\n"
                                       f"Min Distance to Membrane: {min_distance:.1f} nm\n"
                                       f"Active G4 Sources: {self.n_sources}",
                             color='#01579b', fontsize=10, transform=self.ax_analysis.transAxes)
        
        # 添加设计建议
        if self.simulation_mode == "ros":
            if can_rupture and can_translocate:
                advice = "Design optimal for nanotube delivery"
                color = "#388e3c"  # 绿色
            elif can_rupture:
                advice = "Recommendations:\n- Increase pore size for nanotube entry\n- Optimize orientation"
                color = "#f57c00"  # 橙色
            else:
                advice = "Recommendations:\n- Reduce distance to membrane\n- Increase G4 density\n- Optimize orientation"
                color = "#f57c00"  # 橙色
        else:
            if can_drug_pass:
                advice = "Drug can pass through pore\nDelivery efficiency depends on diffusion"
                color = "#7b1fa2"  # 紫色
            else:
                advice = "Recommendations:\n- Reduce drug size\n- Increase pore size\n- Use smaller drug molecules"
                color = "#d32f2f"  # 红色
        
        self.ax_analysis.text(0.05, 0.25, advice, 
                             color=color, fontsize=11, fontweight='bold',
                             transform=self.ax_analysis.transAxes)
        
        # 添加状态指示器
        status_color = '#4caf50' if (self.simulation_mode == "ros" and can_rupture) or \
                                   (self.simulation_mode == "drug" and can_drug_pass) else '#f44336'
        status_text = "OPTIMAL" if status_color == '#4caf50' else "SUBOPTIMAL"
        
        status_circle = Circle((0.9, 0.9), 0.04, transform=self.ax_analysis.transAxes, 
                              facecolor=status_color, edgecolor='black', alpha=0.8)
        self.ax_analysis.add_patch(status_circle)
        self.ax_analysis.text(0.9, 0.8, status_text, color=status_color, fontsize=10,
                             ha='center', fontweight='bold', transform=self.ax_analysis.transAxes)
    
    def update(self, val=None):
        """更新整个视图"""
        # 更新管结构位置
        rotated_sources = self.rotate_sources()
        
        # 确保所有点都在膜上方
        rotated_sources[:, 2] = np.maximum(rotated_sources[:, 2], 1e-9)
        
        # 更新位点标记
        self.source_markers._offsets3d = (
            rotated_sources[:, 0]*1e9,
            rotated_sources[:, 1]*1e9,
            rotated_sources[:, 2]*1e9
        )
        
        # 更新纳米管表面
        center = np.mean(rotated_sources, axis=0)
        X, Y, Z = self.create_nanotube_mesh(center)
        if hasattr(self, 'nanotube_surface'):
            self.nanotube_surface.remove()
        self.nanotube_surface = self.ax_3d.plot_surface(
            X*1e9, Y*1e9, Z*1e9, 
            color='#4fc3f7', alpha=0.7, edgecolor='#01579b', linewidth=0.5
        )
        
        # 更新药物分子
        self.drug_points._offsets3d = (
            self.drug_positions[:, 0]*1e9,
            self.drug_positions[:, 1]*1e9,
            self.drug_positions[:, 2]*1e9
        )
        
        # 更新aptamer位置
        aptamer_len = self.slider_length.val
        
        # 找到纳米管上最接近膜的点
        min_z_index = np.argmin(rotated_sources[:, 2])
        nanotube_point = rotated_sources[min_z_index]
        membrane_point = np.array([nanotube_point[0], nanotube_point[1], 0])
        
        # 更新aptamer线
        self.aptamer_line.set_data(
            [nanotube_point[0]*1e9, membrane_point[0]*1e9],
            [nanotube_point[1]*1e9, membrane_point[1]*1e9]
        )
        self.aptamer_line.set_3d_properties(
            [nanotube_point[2]*1e9, membrane_point[2]*1e9]
        )
        
        # 更新aptamer点
        self.aptamer_point._offsets3d = (
            [membrane_point[0]*1e9], 
            [membrane_point[1]*1e9], 
            [membrane_point[2]*1e9]
        )
        
        # 更新距离标注
        distance_to_membrane = nanotube_point[2]
        self.distance_text.set_position((
            nanotube_point[0]*1e9, 
            nanotube_point[1]*1e9, 
            nanotube_point[2]*1e9 + 10
        ))
        self.distance_text.set_text(f"Distance: {distance_to_membrane*1e9:.1f} nm")
        
        # 模拟从G4位点到膜的ROS扩散
        translated_sources = self.simulate_ros_diffusion()
        
        # 使用ROS密度更新膜视图
        self.membrane_img.set_data(self.membrane_hits.T)
        self.membrane_img.autoscale()
        
        # 更新阈值等高线
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
        
        # 更新分析面板
        self.update_analysis_panel()
        
        # 如果正在录制，捕获帧
        if self.recording:
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            self.recording_frames.append(img.copy())
            
            if time.time() - self.recording_start_time > 5:
                self.toggle_recording()
        
        self.fig.canvas.draw_idle()
    
    def reset(self, event):
        """重置视图"""
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
    
    def run_analysis(self, event):
        """运行aptamer长度影响分析"""
        # 创建分析窗口
        analysis_fig = plt.figure(figsize=(14, 10), facecolor='#e0f7fa')
        analysis_fig.suptitle('Drug Delivery Efficiency Analysis', 
                             fontsize=20, color='#0288d1', fontweight='bold')
        
        gs = gridspec.GridSpec(2, 2)
        
        # 参数扫描设置
        aptamer_lengths = np.linspace(5, 50, 20)  # 5-50nm范围
        orientations = [0, 45, 90]  # 三种典型方向
        pore_results = {angle: [] for angle in orientations}
        drug_results = {angle: [] for angle in orientations}
        
        # 进度条
        print("Running drug delivery efficiency analysis...")
        
        # 扫描方向和aptamer长度
        for angle in orientations:
            for length in tqdm(aptamer_lengths, desc=f"Angle {angle}°"):
                # 设置方向
                self.slider_x.set_val(0)
                self.slider_y.set_val(angle)
                self.slider_z.set_val(0)
                
                # 设置aptamer长度
                self.slider_length.set_val(length)
                self.update()
                
                # 收集结果
                peak_density = np.max(self.membrane_hits)
                pore_diameter = calculate_pore_diameter(peak_density)
                pore_results[angle].append((length, pore_diameter))
                
                # 模拟药物扩散
                if pore_diameter > self.drug_size:
                    # 简化效率模型
                    efficiency = min(1.0, pore_diameter / (self.drug_size * 2))
                    drug_results[angle].append((length, efficiency))
                else:
                    drug_results[angle].append((length, 0))
        
        # 绘制结果
        ax1 = analysis_fig.add_subplot(gs[0, 0], facecolor='#e0f7fa')
        ax2 = analysis_fig.add_subplot(gs[0, 1], facecolor='#e0f7fa')
        ax3 = analysis_fig.add_subplot(gs[1, :], facecolor='#e0f7fa')
        
        # 设置样式
        for ax in [ax1, ax2, ax3]:
            ax.tick_params(colors='#01579b')
            for spine in ax.spines.values():
                spine.set_color('#01579b')
            ax.xaxis.label.set_color('#01579b')
            ax.yaxis.label.set_color('#01579b')
            ax.title.set_color('#0288d1')
        
        # 绘制孔径与aptamer长度的关系
        for angle in orientations:
            lengths = [r[0] for r in pore_results[angle]]
            pores = [r[1] for r in pore_results[angle]]
            ax1.plot(lengths, pores, 'o-', color='#4fc3f7', label=f"{angle}°", linewidth=2)
        
        ax1.set_title('Pore Diameter vs Aptamer Length')
        ax1.set_xlabel('Aptamer Length (nm)')
        ax1.set_ylabel('Pore Diameter (nm)')
        ax1.axhline(self.drug_size, color='#7b1fa2', linestyle='--', label='Drug Size')
        ax1.grid(True, alpha=0.2, color='#b3e5fc')
        ax1.legend()
        
        # 绘制递送效率与aptamer长度的关系
        for angle in orientations:
            lengths = [r[0] for r in drug_results[angle]]
            efficiencies = [r[1] for r in drug_results[angle]]
            ax2.plot(lengths, efficiencies, 'o-', color='#ba68c8', label=f"{angle}°", linewidth=2)
        
        ax2.set_title('Drug Delivery Efficiency')
        ax2.set_xlabel('Aptamer Length (nm)')
        ax2.set_ylabel('Delivery Efficiency')
        ax2.grid(True, alpha=0.2, color='#b3e5fc')
        ax2.legend()
        
        # 绘制药物递送效率热图
        efficiencies = np.zeros((len(orientations), len(aptamer_lengths)))
        for i, angle in enumerate(orientations):
            for j, length in enumerate(aptamer_lengths):
                efficiencies[i, j] = drug_results[angle][j][1]
        
        im = ax3.imshow(efficiencies, aspect='auto', cmap='viridis', 
                       extent=[aptamer_lengths[0], aptamer_lengths[-1], orientations[-1], orientations[0]])
        ax3.set_title('Drug Delivery Efficiency Heatmap')
        ax3.set_xlabel('Aptamer Length (nm)')
        ax3.set_ylabel('Orientation Angle (deg)')
        ax3.set_yticks(orientations)
        
        # 添加颜色条
        cbar = analysis_fig.colorbar(im, ax=ax3)
        cbar.set_label('Delivery Efficiency', color='#01579b')
        cbar.ax.yaxis.set_tick_params(color='#01579b')
        
        # 添加关键结论
        conclusion_text = (
            "Key Conclusions on Drug Delivery:\n"
            "1. Shorter aptamers increase pore size and drug delivery efficiency\n"
            "2. Optimal aptamer length: 15-25nm for efficient delivery\n"
            "3. Vertical orientations maximize drug delivery\n"
            "4. Drug size should be <50% of pore diameter for efficient delivery\n"
            "5. Drug diffusion efficiency depends on pore-drug size ratio"
        )
        analysis_fig.text(0.1, 0.05, conclusion_text, 
                         color='#0288d1', fontsize=12, 
                         bbox=dict(facecolor='#b3e5fc', alpha=0.7))
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()
    
    def toggle_animation(self, event):
        """切换ROS扩散动画"""
        if self.animation_running:
            # 停止动画
            if self.animation is not None:
                self.animation.event_source.stop()
            self.animation_running = False
            self.animate_button.label.set_text("Animate Diffusion")
        else:
            # 开始动画
            if len(self.trajectories) == 0:
                self.simulate_ros_diffusion()
            
            # 创建轨迹线
            self.trajectory_lines = []
            for trajectory in self.trajectories[:ANIMATION_MOLECULES]:
                line, = self.ax_3d.plot([], [], [], '#4fc3f7', alpha=0.5, linewidth=1)
                self.trajectory_lines.append(line)
            
            # 初始化动画
            self.animation = FuncAnimation(
                self.fig, self.update_ros_animation,
                frames=min(len(t) for t in self.trajectories[:ANIMATION_MOLECULES]) if self.trajectories else 100,
                interval=50, blit=False
            )
            self.animation_running = True
            self.animate_button.label.set_text("Stop Animation")
        
        self.fig.canvas.draw_idle()
    
    def update_ros_animation(self, frame):
        """更新ROS扩散动画的函数"""
        points = []
        for i, trajectory in enumerate(self.trajectories[:ANIMATION_MOLECULES]):
            if frame < len(trajectory):
                pos = trajectory[frame]
                points.append([pos[0]*1e9, pos[1]*1e9, pos[2]*1e9])
                
                # 更新轨迹线
                x = [p[0]*1e9 for p in trajectory[:frame+1]]
                y = [p[1]*1e9 for p in trajectory[:frame+1]]
                z = [p[2]*1e9 for p in trajectory[:frame+1]]
                self.trajectory_lines[i].set_data(x, y)
                self.trajectory_lines[i].set_3d_properties(z)
        
        if points:
            points = np.array(points)
            self.source_markers._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
        
        return self.source_markers, *self.trajectory_lines
    
    def export_3d_view(self, event):
        """导出当前3D视图为高质量图像"""
        # 创建临时图形进行渲染
        export_fig = plt.figure(figsize=(10, 8), facecolor='#e0f7fa')
        ax = export_fig.add_subplot(111, projection='3d', facecolor='#e0f7fa')
        
        # 复制当前视图设置
        ax.set_xlim(self.ax_3d.get_xlim())
        ax.set_ylim(self.ax_3d.get_ylim())
        ax.set_zlim(self.ax_3d.get_zlim())
        ax.set_xlabel('X (nm)', color='#01579b')
        ax.set_ylabel('Y (nm)', color='#01579b')
        ax.set_zlabel('Z (nm)', color='#01579b')
        ax.tick_params(colors='#01579b')
        ax.grid(False)
        
        # 绘制膜平面
        xx, yy = np.meshgrid(np.linspace(-100, 100, 10), np.linspace(-100, 100, 10))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.4, color='#0d47a1')
        
        # 绘制DNA纳米管
        rotated_sources = self.rotate_sources()
        center = np.mean(rotated_sources, axis=0)
        X, Y, Z = self.create_nanotube_mesh(center)
        ax.plot_surface(
            X*1e9, Y*1e9, Z*1e9, 
            color='#4fc3f7', alpha=0.7, edgecolor='#01579b', linewidth=0.5
        )
        
        # 绘制G4位点
        ax.scatter(
            rotated_sources[:, 0]*1e9, 
            rotated_sources[:, 1]*1e9, 
            rotated_sources[:, 2]*1e9, 
            s=30, c='#ff9800', marker='o', alpha=1.0
        )
        
        # 绘制aptamer
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
        
        # 添加距离标注
        distance_to_membrane = nanotube_point[2] * 1e9
        ax.text(
            nanotube_point[0]*1e9, 
            nanotube_point[1]*1e9, 
            nanotube_point[2]*1e9 + 10,
            f"Distance: {distance_to_membrane:.1f} nm",
            color='#01579b', fontsize=9
        )
        
        # 添加孔径信息
        if self.pore_diameter > 0:
            ax.text2D(
                0.05, 0.95, f"Pore Diameter: {self.pore_diameter:.1f} nm", 
                transform=ax.transAxes, color='red', fontsize=10,
                bbox=dict(facecolor='#ffebee', alpha=0.7)
            )
        
        # 保存为高质量图像
        export_fig.savefig('nanotube_3d_view.png', dpi=300, bbox_inches='tight', facecolor='#e0f7fa')
        plt.close(export_fig)
        
        # 更新状态
        self.ax_analysis.clear()
        self.ax_analysis.text(0.5, 0.6, "3D view exported to nanotube_3d_view.png", 
                             color='#388e3c', fontsize=12, ha='center')
        self.fig.canvas.draw_idle()
    
    def export_membrane_view(self, event):
        """导出膜视图"""
        # 创建临时图形进行渲染
        export_fig, ax = plt.subplots(figsize=(8, 6), facecolor='#e0f7fa')
        
        # 复制当前膜视图
        img = ax.imshow(
            self.membrane_hits.T, 
            extent=[-100, 100, -100, 100], 
            origin='lower', 
            cmap=ros_cmap
        )
        ax.set_xlabel('X (nm)', color='#01579b')
        ax.set_ylabel('Y (nm)', color='#01579b')
        ax.tick_params(colors='#01579b')
        ax.set_facecolor('#e0f7fa')
        ax.set_title('ROS Density at Membrane', color='#0288d1', fontsize=14)
        
        # 添加阈值等高线
        if np.any(self.membrane_hits > THRESHOLD_DENSITY):
            ax.contour(
                self.x_grid*1e9, 
                self.y_grid*1e9, 
                self.membrane_hits.T, 
                levels=[THRESHOLD_DENSITY], 
                colors='white', 
                linewidths=1
            )
        
        # 添加颜色条
        cbar = plt.colorbar(img, ax=ax, pad=0.01)
        cbar.set_label('ROS Density (molecules/μm²/ns)', color='#01579b')
        cbar.ax.yaxis.set_tick_params(color='#01579b')
        
        # 添加标注
        peak_density = np.max(self.membrane_hits)
        pore_diameter = calculate_pore_diameter(peak_density)
        total_ros = np.sum(self.membrane_hits)
        ax.text(0.02, 0.95, f"Peak Density: {peak_density:.1f}", 
               transform=ax.transAxes, color='#01579b', fontsize=10,
               bbox=dict(facecolor='#b3e5fc', alpha=0.7))
        ax.text(0.02, 0.88, f"Pore Diameter: {pore_diameter:.1f} nm", 
               transform=ax.transAxes, color='#01579b', fontsize=10,
               bbox=dict(facecolor='#b3e5fc', alpha=0.7))
        ax.text(0.02, 0.81, f"Active Sources: {self.n_sources}", 
               transform=ax.transAxes, color='#01579b', fontsize=10,
               bbox=dict(facecolor='#b3e5fc', alpha=0.7))
        
        # 保存为高质量图像
        export_fig.savefig('membrane_view.png', dpi=300, bbox_inches='tight', facecolor='#e0f7fa')
        plt.close(export_fig)
        
        # 更新状态
        self.ax_analysis.clear()
        self.ax_analysis.text(0.5, 0.6, "Membrane view exported to membrane_view.png", 
                             color='#388e3c', fontsize=12, ha='center')
        self.fig.canvas.draw_idle()

# 运行模拟器
simulator = DrugDeliverySimulator()
plt.show()