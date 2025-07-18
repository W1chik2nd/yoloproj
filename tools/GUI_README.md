# GUI追踪应用使用说明

## 修复内容

已修复原始GUI应用中的以下问题：
- **依赖导入失败**: 添加了完整的依赖检查和错误处理
- **onnxruntime DLL错误**: 提供了简化版追踪器作为备用方案
- **界面稳定性**: 改进了错误处理和用户反馈
- **中文界面**: 完全本地化的用户界面

## 快速开始

### 1. 安装依赖

**方式一：使用安装脚本**
```bash
python src/install_dependencies.py
```

**方式二：手动安装**
```bash
pip install opencv-python numpy PyQt5 ultralytics
```

### 2. 运行GUI应用

```bash
python src/gui.py
```

## 功能特点

### 双模式追踪
- **完整模式**: 使用原始的DeepSORT追踪器（需要onnxruntime）
- **简化模式**: 使用基础追踪器（自动降级，无需onnxruntime）

### 摄像头支持
- 支持Camera 0和Camera 2
- 实时预览和追踪结果对比
- 自动摄像头检测和错误提示

### 用户界面
- 左侧：原始摄像头画面
- 右侧：追踪结果画面  
- 底部：摄像头选择和控制按钮
- 错误弹窗：友好的错误提示

## 故障排除

### 常见问题

**1. 依赖包导入失败**
```
[ERROR] 缺少以下依赖包: opencv-python, numpy, PyQt5
```
**解决方案**: 运行 `python src/install_dependencies.py`

**2. YOLO模型加载失败**
```
[ERROR] 无法初始化YOLO: [Errno 2] No such file or directory: 'weights/yolo11s.pt'
```
**解决方案**: 确保 `weights/yolo11s.pt` 文件存在

**3. 摄像头无法打开**
```
[ERROR] 无法打开摄像头 0
```
**解决方案**: 
- 检查摄像头是否被其他程序占用
- 尝试切换到Camera 2
- 确认摄像头硬件连接正常

**4. onnxruntime导入失败**
```
[WARNING] 导入完整模块失败: DLL load failed
[INFO] 使用简化追踪模式
```
**解决方案**: 这是正常的降级行为，应用会自动使用简化版追踪器

### 虚拟环境问题

如果使用虚拟环境，请确保：
1. 激活虚拟环境: `& 'C:\Users\ASUS\yolovs\Scripts\Activate.ps1'`
2. 在虚拟环境中安装依赖
3. 在虚拟环境中运行GUI

## 技术细节

### 追踪器降级策略
```python
# 优先使用完整追踪器
try:
    from pip2dual import process_detections_and_tracks, yolo, create_tracker
    use_full_tracker = True
except Exception:
    # 降级到简化追踪器
    use_full_tracker = False
    # 使用基础YOLO检测 + 简单ID分配
```

### 错误处理机制
- 依赖检查在启动时进行
- 运行时错误通过信号传递到GUI
- 摄像头故障自动恢复
- 追踪失败时显示原始画面

## 后续改进

可以考虑的优化方向：
1. 添加更多摄像头支持
2. 保存录像功能
3. 追踪参数调节界面
4. 性能监控显示
5. 多目标类别支持 