import matplotlib.pyplot  as plt
import numpy as np
import os

def vis_fn(imgs, masks, preds, pad, save, save_dir, nEpoch, n_show=5, fix_show=False):

    fig, axes = plt.subplots(n_show, 3, figsize=(10, n_show*3), sharey=True, sharex=True)
    
    if fix_show:
        range_list = [i for i in range(n_show)]
    else:
        num = imgs.shape[0]
        range_list = np.random.randint(num, size=num)
    
    for i in range(n_show):
        idx = range_list[i]
        img = np.squeeze(imgs[idx])
        mask = np.squeeze(masks[idx])
        pred = np.squeeze(preds[idx])

        axes[i, 0].imshow(img[pad:-pad,pad:-pad,:])
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 2].imshow(pred, cmap='gray')

        if i == 0:
            axes[i, 0].set_title('org')
            axes[i, 1].set_title('mask')
            axes[i, 2].set_title('pred')

    fig.tight_layout()

    if save:
        if not os.path.exists(save_dir + 'images/'):
            os.makedirs(save_dir + 'images/')
        name = save_dir + 'images/' + str(nEpoch) + '.jpg'
        plt.savefig(name, dpi=150)
    else:
        fig.show()
        plt.show()
        plt.pause(5)

    plt.close()
