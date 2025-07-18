# test_import.py
# 这是一个专用的诊断脚本，用于找出导入失败的根本原因。
import traceback

print("--- 正在尝试导入 'tracking_logic.py' ---")
print("如果导入失败，下面将会打印出详细的错误信息。")
print("-" * 40)

try:
    import tracking_logic
    print("\n--- 成功！---")
    print("模块 'tracking_logic' 可以被成功导入。")
    print("这意味着问题可能与PyQt的交互有关，而非模块本身。")

except Exception as e:
    print("\n--- 导入失败！---")
    print("已捕获到导致导入失败的根本原因:")
    # 打印完整的、详细的错误堆栈信息
    traceback.print_exc()

print("-" * 40)
# 阻塞窗口，防止闪退
input("诊断结束。按 Enter 键退出...")