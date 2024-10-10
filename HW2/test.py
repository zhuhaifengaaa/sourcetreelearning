import re
text = "The quick brown fox jumps over the lazy dog"

# 把文本中的所有空格替换成分号
result = re.sub(r"\s", "", text)

print(result)
# 输出："The;quick;brown;fox;jumps;over;the;lazy;dog"