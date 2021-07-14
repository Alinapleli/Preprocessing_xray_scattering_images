import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import mpld3


class TestModelPatches(object):
    def __init__(self, model, loader: PatchLoader):
        self.model = model
        self.loader = loader
        self.criterion = nn.L1Loss(reduction='none')

    def get_loss_scatter(self):
        self.model.eval()
        upper_losses = []
        lower_losses = []
        losses = []
        for data in self.loader:
            for hist, labels in zip(data[0], data[1]):
                if not hist.numel():
                    continue
                predicted = self.model(hist)
                mse_up = self.criterion(labels[:, 1], predicted[:, 1]).detach().cpu().numpy().flatten()
                mse_low = self.criterion(labels[:, 0], predicted[:, 0]).detach().cpu().numpy().flatten()
                upper_losses.append(mse_up)
                lower_losses.append(mse_low)
                losses.append([np.mean(mse_up_ + mse_low_) for mse_up_, mse_low_ in zip(mse_up, mse_low)])
            upper_losses = np.asarray(upper_losses)
            lower_losses = np.asarray(lower_losses)
            losses = np.asarray(losses)
            cm = plt.get_cmap('jet')
            fig, axs = plt.subplots(ncols=1, figsize=(12, 8))
            scatter = axs.scatter(lower_labels * 100, 100 - upper_labels * 100, c=losses[0])
            axs.grid()
            axs.set_title('Performance Upper and Lower Labels', fontsize=14)
            axs.set_xlabel('Lower cut (%)', fontsize=14)
            axs.set_ylabel('Upper cut (%)', fontsize=14);
            ax2 = fig.add_axes([0.95, 0.1, 0.01, 0.8])
            norm = mpl.colors.Normalize(losses[0].min(), losses[0].max())
            mpl.colorbar.ColorbarBase(ax2, cmap=cm, norm=norm)
            labels = ['mse= {:.3f}'.format(losses[0][i]) for i in range(len(losses))]
            tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
            mpld3.plugins.connect(fig, tooltip)
            mpld3.enable_notebook()

    def get_loss_statistics(self):
        self.model.eval()
        upper_losses = []
        lower_losses = []
        for data in self.loader:
            for hist, labels in zip(data[0], data[1]):
                if not hist.numel():
                    continue
                predicted = self.model(hist)
                upper_losses += self.criterion(labels[1], predicted[1]).detach().cpu().numpy().flatten().tolist()
                lower_losses += self.criterion(labels[0], predicted[0]).detach().cpu().numpy().flatten().tolist()
            plt.hist(upper_losses, bins=50, label='upper');
            plt.hist(lower_losses, bins=50, label='lower');
            plt.xlabel('Mean absolut loss (between predicted and labeled clipping values)', fontsize=12)
            plt.ylabel('Occurrences', fontsize=12)
            plt.legend()
            plt.show()

    def compare_imgs(self):
        hist1, labels1, imgs1 = next(iter(self.loader.test_iteration()))
        hist2, labels2, imgs2 = next(iter(self.loader.test_iteration()))
        hist3, labels3, imgs3 = next(iter(self.loader.test_iteration()))
        hist4, labels4, imgs4 = next(iter(self.loader.test_iteration()))
        hists = [hist1, hist2, hist3, hist4]
        print(len(hist1))
        imgs = [imgs1, imgs2, imgs3, imgs4]
        self._compare_imgs_median(hists, labels1, imgs)

    def _compare_imgs_median(self, hists, labels, imgs):
        self.model.eval()
        pred = []
        for hist in hists:
            pred.append(self.model(hist))
        imgs = [np.flip(img, 1) for img in imgs]
        predicted_img_1, predicted_img_2, predicted_img_3, predicted_img_4 = self._clip_imgs(imgs[0],pred[0]), \
                                                                             self._clip_imgs(imgs[1], pred[1]), \
                                                                             self._clip_imgs(imgs[2], pred[2]), \
                                                                             self._clip_imgs(imgs[3], pred[3])

        pred = [p.detach().cpu().numpy() for p in pred]
        pred_median_low = [statistics.median([pred[0][i][0], pred[1][i][0], pred[2][i][0], pred[3][i][0]]) for i in
                           range(len(pred[0]))]
        pred_median_up = [statistics.median([pred[0][i][1], pred[1][i][1], pred[2][i][1], pred[3][i][1]]) for i in
                          range(len(pred[0]))]
        pred_median_tensor_low, pred_median_tensor_up = torch.tensor(pred_median_low, dtype=torch.float32, device='cuda'), \
                                                        torch.tensor(pred_median_up, dtype=torch.float32, device='cuda')
        predicted_img_median = self._clip_imgs_(imgs[0], pred_median_tensor_low, pred_median_tensor_up)
        target_imgs = self._clip_imgs(imgs[0], labels)
        for i in range(len(imgs[0])):
            print("Img:", i)
            print('Labeled Clipping Values:', labels[i].detach().cpu().numpy())
            print('Predicted Clipping Values:', pred[0][i])
            fig, axs = plt.subplots(ncols=6, figsize=(15, 10))
            axs[0].imshow(target_imgs[i])
            axs[0].set_xlim([0, target_imgs[i].shape[1]])
            axs[0].set_ylim([0, target_imgs[i].shape[0]])
            axs[0].set_xlabel('Pixels of the raw image', fontsize=12)
            axs[0].set_title('Target img')
            axs[0].set_ylabel('Pixels of the raw image', fontsize=12)
            axs[1].imshow(predicted_img_1[i])
            axs[1].set_xlim([0, predicted_img_1[i].shape[1]])
            axs[1].set_ylim([0, predicted_img_1[i].shape[0]])
            axs[1].set_xlabel('Pixels of the raw image', fontsize=12)
            axs[1].set_title('Pred img 1')
            axs[1].set_ylabel('Pixels of the raw image', fontsize=12)
            axs[2].imshow(predicted_img_2[i])
            axs[2].set_xlim([0, predicted_img_2[i].shape[1]])
            axs[2].set_ylim([0, predicted_img_2[i].shape[0]])
            axs[2].set_xlabel('Pixels of the raw image', fontsize=12)
            axs[2].set_title('Pred img 2')
            axs[2].set_ylabel('Pixels of the raw image', fontsize=12)
            axs[3].imshow(predicted_img_3[i])
            axs[3].set_xlim([0, predicted_img_3[i].shape[1]])
            axs[3].set_ylim([0, predicted_img_3[i].shape[0]])
            axs[3].set_xlabel('Pixels of the raw image', fontsize=12)
            axs[3].set_title('Pred img 3')
            axs[3].set_ylabel('Pixels of the raw image', fontsize=12)
            axs[4].imshow(predicted_img_4[i])
            axs[4].set_xlim([0, predicted_img_4[i].shape[1]])
            axs[4].set_ylim([0, predicted_img_4[i].shape[0]])
            axs[4].set_xlabel('Pixels of the raw image', fontsize=12)
            axs[4].set_title('Pred img 4')
            axs[4].set_ylabel('Pixels of the raw image', fontsize=12)
            axs[5].imshow(predicted_img_median[i])
            axs[5].set_xlim([0, predicted_img_median[i].shape[1]])
            axs[5].set_ylim([0, predicted_img_median[i].shape[0]])
            axs[5].set_xlabel('Pixels of the raw image', fontsize=12)
            axs[5].set_title('Pred img median')
            axs[5].set_ylabel('Pixels of the raw image', fontsize=12)
            plt.show()

    def _compare_imgs(self, hists, labels, imgs):
        self.model.eval()
        predicted = self.model(hists)
        target_imgs = self._clip_imgs(imgs, labels)
        predicted_imgs = self._clip_imgs(imgs, predicted)
        for i in range(len(imgs)):
            print('Labeled Clipping Values:', labels[i].detach().cpu().numpy())
            print('Predicted Clipping Values:', predicted[i].detach().cpu().numpy())
            fig, axs = plt.subplots(ncols=2, figsize=(12, 8))
            axs[0].imshow(target_imgs[i])
            axs[1].imshow(predicted_imgs[i])
            plt.show()

    def _clip_imgs(self, imgs, labels):
        label1_ = [label.detach().cpu().numpy()[0] for label in labels]
        label2_ = [label.detach().cpu().numpy()[1] for label in labels]
        m1 = [img.min() for img in imgs]
        m2 = [img.max() for img in imgs]
        return [np.clip(img, label1, label2) for img, label1, label2 in zip(imgs, label1_, label2_)]

    def _clip_imgs_(self, imgs, low, up):
        label1_ = [label.detach().cpu().numpy() for label in low]
        label2_ = [label.detach().cpu().numpy() for label in up]
        m1 = [img.min() for img in imgs]
        m2 = [img.max() for img in imgs]
        return [np.clip(img, label1, label2) for img, label1, label2 in zip(imgs, label1_, label2_)]
