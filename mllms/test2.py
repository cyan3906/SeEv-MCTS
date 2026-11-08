code_str = """
nums = [1, 1, 6, 9]
res = ((1 + 1) * 9) + 6
print(f"24点计算结果: {res}")
"""

# 多行代码只能用 exec，无需导入
exec(code_str)
