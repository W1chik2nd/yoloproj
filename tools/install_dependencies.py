#!/usr/bin/env python3
"""
GUI依赖安装脚本
运行此脚本将自动安装GUI所需的所有依赖包
"""

import subprocess
import sys
import os

def install_package(package_name):
    """安装单个包"""
    try:
        print(f"正在安装 {package_name}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package_name], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {package_name} 安装成功")
            return True
        else:
            print(f"✗ {package_name} 安装失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ {package_name} 安装异常: {e}")
        return False

def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def main():
    print("=== GUI依赖安装脚本 ===")
    print()
    
    # 定义所需的包
    required_packages = [
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("PyQt5", "PyQt5"),
        ("ultralytics", "ultralytics"),
    ]
    
    # 检查当前状态
    print("检查当前依赖状态...")
    missing_packages = []
    
    for package, import_name in required_packages:
        if check_package(package, import_name):
            print(f"✓ {package} 已安装")
        else:
            print(f"✗ {package} 未安装")
            missing_packages.append(package)
    
    if not missing_packages:
        print("\n所有依赖都已安装!")
        return
    
    print(f"\n需要安装 {len(missing_packages)} 个包:")
    for package in missing_packages:
        print(f"  - {package}")
    
    # 询问用户是否继续
    response = input("\n是否继续安装? (y/n): ").lower().strip()
    if response not in ['y', 'yes', '是']:
        print("安装取消")
        return
    
    print("\n开始安装依赖...")
    
    # 安装缺失的包
    success_count = 0
    for package in missing_packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n安装完成: {success_count}/{len(missing_packages)} 个包安装成功")
    
    if success_count == len(missing_packages):
        print("\n✓ 所有依赖安装成功! 现在可以运行GUI了")
        print("运行命令: python src/gui.py")
    else:
        print("\n⚠ 部分依赖安装失败，请检查错误信息")
        print("也可以手动运行: pip install opencv-python numpy PyQt5 ultralytics")

if __name__ == "__main__":
    main() 