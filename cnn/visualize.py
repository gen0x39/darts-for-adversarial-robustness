import numpy as np
import matplotlib.pyplot as plt
import os
import torch
    
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 敵対的サンプルの保存
def save_adversarial_img(adv_sample, input, target, epsilon, step):
    cpu = torch.device('cpu')

    # Reshape
    # adv_sample : [batchsize, 3, 32, 32] -> [32, 32, 3]
    tmp = adv_sample[0] # [3, 32, 32]
    tmp = tmp.to(cpu)

    # cast to numpy
    red = tmp[0].detach().numpy().copy()    # [32, 32]
    green = tmp[1].detach().numpy().copy()  # [32, 32]
    blue = tmp[2].detach().numpy().copy()   # [32, 32]
    adv_img = np.stack([red, green, blue], axis = 2)

    # 元画像の保存
    tmp = input[0]
    tmp = tmp.to(cpu)

    red = tmp[0].detach().numpy().copy()    # [32, 32]
    green = tmp[1].detach().numpy().copy()  # [32, 32]
    blue = tmp[2].detach().numpy().copy()   # [32, 32]
    ori_img = np.stack([red, green, blue], axis = 2)

    # figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
    fig = plt.figure(figsize=(8,10))

    # add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # 個別のグラフにタイトルをつける
    title1 = "original (lagel = " + classes[target[0]] + ")"
    ax1.set_title(title1)
    title2 = "adversarial (epsilon = " + str(epsilon) + ")"
    ax2.set_title(title2)

    # 画像を表示
    ax1.imshow(ori_img)
    ax2.imshow(adv_img)

    plt.tight_layout()
    #plt.show()
    # 保存
    fname = "adv-" + str(step) + ".png"
    save_dir = "./adversarial_example"
    plt.savefig(os.path.join(save_dir, fname), dpi = 64, facecolor = "lightgray", tight_layout = True)

    print(np.all(adv_img == ori_img))
    print(ori_img)
    print(adv_img)
