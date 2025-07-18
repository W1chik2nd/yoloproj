# 文件名: convert_to_onnx.py

print("--- 转换脚本开始运行 ---")
import torch
import torchreid # 您可能需要先安装: pip install torchreid

# ---------- 配置 ---------- #
# 1. 指定您要转换的 .pt 文件路径
PT_MODEL_PATH = "weights/osnet_x1_0_msmt17.pt"

# 2. 指定转换后输出的 .onnx 文件路径
ONNX_MODEL_PATH = "weights/osnet_x1_0_msmt17.onnx"

# 3. 指定模型名称和输入尺寸 (对于OSNet通常是固定的)
MODEL_NAME = 'osnet_x1_0'
INPUT_SHAPE = (1, 3, 256, 128) # (batch_size, channels, height, width)

# -------------------------- #

print(f"正在加载PyTorch模型: {PT_MODEL_PATH}")

# 使用torchreid库来构建模型结构
# 这确保了模型结构与加载的权重是匹配的
model = torchreid.models.build_model(
    name=MODEL_NAME,
    num_classes=1041,  # num_classes可以随便填，因为我们只用特征提取部分
    loss='softmax',
    pretrained=False # 我们将手动加载自己的权重，而不是下载预训练模型
)

# 加载您的.pt权重文件
checkpoint = torch.load(PT_MODEL_PATH, map_location=torch.device('cpu'))

# 根据.pt文件的结构加载权重
# 通常权重保存在'state_dict'键中，如果不是，您可能需要检查.pt文件的结构
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

# 将模型设置为评估模式
model.eval()
print("PyTorch模型加载成功。")

# 创建一个符合模型输入的虚拟张量 (dummy input)
dummy_input = torch.randn(INPUT_SHAPE, requires_grad=True)

print(f"正在将模型导出为ONNX格式，保存至: {ONNX_MODEL_PATH}")

# 使用torch.onnx.export进行转换
torch.onnx.export(
    model,                        # 要转换的模型
    dummy_input,                  # 虚拟输入
    ONNX_MODEL_PATH,              # 输出路径
    export_params=True,           # 导出模型参数
    opset_version=11,             # ONNX算子集版本
    do_constant_folding=True,     # 是否执行常量折叠优化
    input_names=['input'],        # 输入节点的名称
    output_names=['output'],      # 输出节点的名称
    dynamic_axes={'input' : {0 : 'batch_size'},    # 允许batch_size是动态的
                  'output' : {0 : 'batch_size'}}
)

print("转换成功！")