# 大语言模型生成抽奖程序（问题2：滚动5秒版）

```python
import tkinter as tk
from tkinter import messagebox
import random
import time

# 30人虚拟名单
names = [
    "张伟", "王芳", "李娜", "刘强", "陈静", "杨洋", "赵敏", "黄磊", "周杰", "吴倩",
    "徐磊", "孙俪", "马超", "朱婷", "胡军", "郭峰", "林芳", "何平", "高翔", "罗成",
    "郑爽", "梁波", "宋佳", "唐磊", "许晴", "韩雪", "冯刚", "曹颖", "彭飞", "董洁"
]

root = tk.Tk()
root.title("随机抽奖")
root.geometry("300x200")
root.resizable(False, False)

# 显示名字的标签
label = tk.Label(root, text="点击按钮开始抽奖", font=("微软雅黑", 20, "bold"))
label.pack(expand=True, fill="both")


def start_lottery():
    # 禁用按钮，防止重复点击
    btn.config(state="disabled", bg="#f0f0f0")
    end_time = time.time() + 5  # 滚动5秒

    def roll():
        if time.time() < end_time:
            label.config(text=random.choice(names))
            root.after(50, roll)  # 每50毫秒刷新一次名字
        else:
            winner = random.choice(names)
            label.config(text=winner)
            # 按钮背景改为蓝色并恢复可用
            btn.config(state="normal", bg="blue",
                       activebackground="blue", fg="white")
            # 弹出中奖结果
            messagebox.showinfo("中奖结果", f"🎉 恭喜【{winner}】中奖！")

    roll()


# 抽奖按钮
btn = tk.Button(root, text="开始抽奖", font=("微软雅黑", 14),
                bg="#4CAF50", fg="white", command=start_lottery)
btn.pack(pady=10, ipadx=20, ipady=5)

root.mainloop()
```
