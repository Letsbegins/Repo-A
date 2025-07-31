import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import norm

def double_gaussian(x, params):
    """
    双高斯函数模型
    params: [intensity, width, pos1, pos2]
    - intensity: 两个高斯的共同强度
    - width: 两个高斯的共同峰宽(标准差)
    - pos1, pos2: 两个高斯的位置
    """
    intensity, width, pos1, pos2 = params
    gaussian1 = intensity * np.exp(-0.5 * ((x - pos1) / width) ** 2)
    gaussian2 = intensity * np.exp(-0.5 * ((x - pos2) / width) ** 2)
    return gaussian1 + gaussian2

def create_mask(x, center, mask_width=3):
    """
    创建掩码，排除中心 ±mask_width 像素的数据
    """
    mask = np.abs(x - center) > mask_width
    return mask

def residuals(params, x_data, y_data, mask):
    """
    计算残差函数（仅使用未被掩码的数据点）
    """
    model = double_gaussian(x_data[mask], params)
    return y_data[mask] - model

def fit_double_gaussian(x_data, y_data, initial_guess, center_pos, mask_width=3):
    """
    拟合双高斯函数
    
    Parameters:
    - x_data, y_data: 输入数据
    - initial_guess: 初始参数猜测 [intensity, width, pos1, pos2]
    - center_pos: 双峰结构的中心位置
    - mask_width: 掩码宽度（像素）
    
    Returns:
    - result: 优化结果
    - fitted_params: 拟合参数
    - peak_distance: 两峰之间的距离
    """
    # 创建掩码
    mask = create_mask(x_data, center_pos, mask_width)
    
    # 设置参数边界（可选）
    # bounds格式: ([lower_bounds], [upper_bounds])
    bounds = ([0, 0.1, x_data.min(), x_data.min()], 
              [np.inf, 10, x_data.max(), x_data.max()])
    
    # 使用Trust Region Reflective算法进行拟合
    result = least_squares(
        residuals, 
        initial_guess, 
        args=(x_data, y_data, mask),
        bounds=bounds,
        method='trf'  # Trust Region Reflective
    )
    
    fitted_params = result.x
    peak_distance = abs(fitted_params[3] - fitted_params[2])  # |pos2 - pos1|
    
    return result, fitted_params, peak_distance

def generate_demo_data():
    """
    生成演示数据：双高斯 + 噪声 + 中心杂音
    """
    # 数据范围
    x = np.linspace(-20, 20, 200)
    
    # 真实参数
    true_intensity = 5.0
    true_width = 2.0
    true_pos1 = -4.0
    true_pos2 = 6.0
    
    # 生成双高斯信号
    y_clean = double_gaussian(x, [true_intensity, true_width, true_pos1, true_pos2])
    
    # 添加随机噪声
    noise = np.random.normal(0, 0.2, len(x))
    
    # 在中心添加杂音（模拟实际情况）
    center = (true_pos1 + true_pos2) / 2
    center_indices = np.abs(x - center) <= 3
    artifact_noise = np.random.normal(0, 1.0, np.sum(center_indices))
    noise[center_indices] += artifact_noise
    
    y_noisy = y_clean + noise
    
    return x, y_noisy, [true_intensity, true_width, true_pos1, true_pos2]

def plot_results(x_data, y_data, fitted_params, mask, center_pos, mask_width):
    """
    绘制拟合结果
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 生成拟合曲线
    x_fit = np.linspace(x_data.min(), x_data.max(), 500)
    y_fit = double_gaussian(x_fit, fitted_params)
    
    # 分别绘制两个高斯分量
    intensity, width, pos1, pos2 = fitted_params
    gauss1 = intensity * np.exp(-0.5 * ((x_fit - pos1) / width) ** 2)
    gauss2 = intensity * np.exp(-0.5 * ((x_fit - pos2) / width) ** 2)
    
    # 上图：原始数据和拟合结果
    ax1.scatter(x_data[mask], y_data[mask], alpha=0.6, s=20, color='blue', label='Used data')
    ax1.scatter(x_data[~mask], y_data[~mask], alpha=0.6, s=20, color='red', label='Masked data')
    ax1.plot(x_fit, y_fit, 'g-', linewidth=2, label='Double Gaussian fit')
    ax1.plot(x_fit, gauss1, 'orange', linestyle='--', alpha=0.7, label='Gaussian 1')
    ax1.plot(x_fit, gauss2, 'purple', linestyle='--', alpha=0.7, label='Gaussian 2')
    
    # 标记掩码区域
    ax1.axvspan(center_pos - mask_width, center_pos + mask_width, 
                alpha=0.2, color='red', label=f'Masked region (±{mask_width} pixels)')
    
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Intensity')
    ax1.set_title('Double Gaussian Fitting with Center Masking')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 下图：残差
    residual_vals = y_data[mask] - double_gaussian(x_data[mask], fitted_params)
    ax2.scatter(x_data[mask], residual_vals, alpha=0.6, s=20, color='blue')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Fitting Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数：演示双高斯拟合
    """
    print("双高斯拟合演示")
    print("=" * 50)
    
    # 生成演示数据
    x_data, y_data, true_params = generate_demo_data()
    true_distance = abs(true_params[3] - true_params[2])
    
    print(f"真实参数:")
    print(f"  Intensity: {true_params[0]:.3f}")
    print(f"  Width: {true_params[1]:.3f}")
    print(f"  Position 1: {true_params[2]:.3f}")
    print(f"  Position 2: {true_params[3]:.3f}")
    print(f"  真实峰间距: {true_distance:.3f}")
    print()
    
    # 估计中心位置（用于掩码）
    center_estimate = x_data[np.argmax(y_data)]
    
    # 初始参数猜测
    intensity_guess = np.max(y_data) * 0.6  # 考虑到是两个峰的叠加
    width_guess = 2.0
    pos1_guess = center_estimate - 5
    pos2_guess = center_estimate + 5
    initial_guess = [intensity_guess, width_guess, pos1_guess, pos2_guess]
    
    print(f"初始猜测:")
    print(f"  Intensity: {initial_guess[0]:.3f}")
    print(f"  Width: {initial_guess[1]:.3f}")
    print(f"  Position 1: {initial_guess[2]:.3f}")
    print(f"  Position 2: {initial_guess[3]:.3f}")
    print()
    
    # 执行拟合
    mask_width = 3
    result, fitted_params, peak_distance = fit_double_gaussian(
        x_data, y_data, initial_guess, center_estimate, mask_width
    )
    
    # 显示结果
    print("拟合结果:")
    print(f"  优化成功: {result.success}")
    print(f"  迭代次数: {result.nfev}")
    print(f"  残差平方和: {np.sum(result.fun**2):.6f}")
    print()
    
    print(f"拟合参数:")
    print(f"  Intensity: {fitted_params[0]:.3f}")
    print(f"  Width: {fitted_params[1]:.3f}")
    print(f"  Position 1: {fitted_params[2]:.3f}")
    print(f"  Position 2: {fitted_params[3]:.3f}")
    print(f"  拟合峰间距: {peak_distance:.3f}")
    print()
    
    print(f"误差分析:")
    print(f"  峰间距误差: {abs(peak_distance - true_distance):.3f}")
    print(f"  相对误差: {abs(peak_distance - true_distance)/true_distance*100:.2f}%")
    
    # 计算掩码
    mask = create_mask(x_data, center_estimate, mask_width)
    
    # 绘制结果
    plot_results(x_data, y_data, fitted_params, mask, center_estimate, mask_width)
    
    return fitted_params, peak_distance

if __name__ == "__main__":
    fitted_params, distance = main()