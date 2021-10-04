import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

cat = Image.open("./cat.png")
ghost = Image.open("./ghost.png")

# figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
fig = plt.figure(figsize=(8,10))

# add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# 個別のグラフにタイトルをつける
ax1.set_title("cat")
ax2.set_title("ghost")

# 全体にタイトルをつける
fig.suptitle("graphs")

# 画像を表示
ax1.imshow(cat)
ax2.imshow(ghost)

plt.tight_layout()
plt.show()
# 保存
plt.savefig("hoge.png", facecolor = "lightgray", tight_layout = True)