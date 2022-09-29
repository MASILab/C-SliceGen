from email.mime import image
from fileinput import filename
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
import logging
import sys
from collections import defaultdict
from pathlib import Path
#from utils import mkdir
import numpy as np

def mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path

class Visualizer(object):
    def __init__(self,
                 output_dir,
                 batch_size,
                 loss_xlim=None,
                 loss_ylim=None):
        self.output_dir = Path(output_dir)
        self.train_outdir = mkdir(self.output_dir / 'train')
        self.val_outdir = mkdir(self.output_dir / 'val')
        self.test_outdir = mkdir(self.output_dir / 'test_blsa_pdf')
        # self.recons_dir = mkdir(self.output_dir / 'recons')
        # self.gen_dir = mkdir(self.output_dir / 'gen')
        self.loss_xlim = loss_xlim
        self.loss_ylim = loss_ylim
        sns.set()
        self.rows, self.cols = 8, 16
        self.at_start = True

        self.history = defaultdict(list)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(self.output_dir / 'log.txt'),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.print_header = True
        self.batch_size = batch_size
        
    def plot_subgrid(self, x_t, x_c, x_gen, target_label, seg_output, filename=None,seg=False):
        # print(type(images))
        if seg:
            rows, cols = 5,self.batch_size
            datas = [x_t, x_c, x_gen, target_label, seg_output]
        else:
            if x_t is not None:
                rows, cols = x_gen.shape[1]*2 + 1, self.batch_size
            else:
                rows, cols = x_gen.shape[1] + 1, self.batch_size
            datas = [x_t, x_c, x_gen]
        scale = .75
        fig, ax = plt.subplots(figsize=(cols * scale, rows * scale))
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])

        inner_grid = gridspec.GridSpec(rows, cols, fig, wspace=.05, hspace=.05,
                                       left=0, right=1, top=1, bottom=0)

        
        cmap = 'gray'

        # for s, images in enumerate(datas):
        #     if images is None:
        #         images = np.zeros(x_t.shape)
        #     else:
        #         images = images.detach().cpu().numpy()
        #     for i in range(len(images)):
        #         ax = plt.subplot(inner_grid[s, i])
        #         ax.set_axis_off()
        #         ax.set_xticks([])
        #         ax.set_yticks([])
        #         ax.set_aspect('equal')
        #         if len(images.shape) == 4:
        #             ax.imshow(images[i, 0], interpolation='none', aspect='equal',
        #                     cmap=cmap, vmin=0, vmax=1)
        #         else:
        #             ax.imshow(images[i], interpolation='none', aspect='equal',
        #                     cmap=cmap)

        for b in range(x_gen.shape[0]):
            r = 0
            for s, images in enumerate(datas):
                if images is None:
                    continue
                else:
                    images = images.detach().cpu().numpy()
                
                if len(images.shape) == 4:
              
                    for i in range(images.shape[1]):
                        ax = plt.subplot(inner_grid[r, b])
                        ax.set_axis_off()
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_aspect('equal')
                        
                        ax.imshow(images[b, i], interpolation='none', aspect='equal',
                                cmap=cmap, vmin=0, vmax=1)
        

                        r += 1
                else:
                    ax = plt.subplot(inner_grid[r, b])
                    ax.set_axis_off()
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect('equal')
                
                    ax.imshow(images[b, 0], interpolation='none', aspect='equal',
                            cmap=cmap)
                    r += 1

        # plt.show()

            # if bbox is not None:
            #     bbox = bbox.reshape(-1, bbox.shape[-1])
            #     d0, d1, d0_len, d1_len = bbox[i]
            #     ax.add_patch(Rectangle(
            #         (d1 - .5, d0 - .5), d1_len, d0_len, lw=1,
            #         edgecolor='red', fill=False))
            # fig.add_subplot(ax)

        if filename is not None:
            plt.savefig(str(filename))
        plt.close(fig)

    def plot(self, epoch, x_t, x_c, x_gen, target_label, seg_output, idx, phase, seg):
        # if self.at_start:
        #     self.plot_subgrid(x_t, x_c, x_recon, x_gen, 
        #                       self.recons_dir / f'groundtruth.png')
        #     # self.plot_subgrid(x_c, self.recons_dir / f'original.png')
        #     self.at_start = False
        if phase == 'train':
            self.plot_subgrid(x_t, x_c,  x_gen, target_label, seg_output,
                            self.train_outdir / f'{epoch:04d}_{idx}.pdf',seg)
        elif phase == 'val':
            self.plot_subgrid(x_t, x_c, x_gen, target_label, seg_output, 
                            self.val_outdir / f'{epoch:04d}_{idx}.pdf',seg)
        else:
            self.plot_subgrid(x_t, x_c, x_gen, target_label, seg_output, 
                            self.test_outdir / '{}.pdf'.format(epoch[0]),seg)
        # self.plot_subgrid(x_t, x_c, x_recon, x_gen, self.gen_dir / f'{epoch:04d}.png')
    
    def plot_test(self, epoch, x, mask, bbox, x_recon, x_gen, save_root):
        if self.at_start:
            self.plot_subgrid(x * mask + .5 * (1 - mask), bbox,
                              self.recons_dir / f'groundtruth.png')
            self.plot_subgrid(x, bbox, self.recons_dir / f'original.png')
            self.at_start = False
        self.plot_subgrid(x * mask + x_recon * (1 - mask), bbox,
                          save_root + '/' + f'reco_{epoch:04d}.png')
        self.plot_subgrid(x_gen, None, save_root + '/' + f'gen_{epoch:04d}.png')

    def plot_loss(self, epoch, losses):
        for name, val in losses.items():
            self.history[name].append(val)

        fig, ax_trace = plt.subplots(1,3,figsize=(6, 4))
        ax_trace[0].set_ylabel('loss')
        ax_trace[0].set_xlabel('epochs')
        ax_trace[1].set_ylabel('rmse')
        ax_trace[1].set_xlabel('epochs')
        ax_trace[2].set_ylabel('loss')
        ax_trace[2].set_xlabel('epochs')
        if self.loss_xlim is not None:
            ax_trace[0].set_xlim(self.loss_xlim)
            ax_trace[1].set_xlim(self.loss_xlim)
            ax_trace[2].set_xlim(self.loss_xlim)
        if self.loss_ylim is not None:
            ax_trace[0].set_ylim(self.loss_ylim)
            ax_trace[1].set_ylim(self.loss_ylim)
            ax_trace[2].set_ylim(self.loss_ylim)
        for label, loss in self.history.items():
            # print(label)
            if 'lr' not in label and label!='total' and 'mse' not in label and 'l1' not in label and 'ssim' not in label and 'kl' not in label:
                # loss = loss.cpu().numpy()
                ax_trace[0].plot(loss, '-', label=label)
            elif 'kl' in label or 'ssim' in label:
                ax_trace[2].plot(loss, '-', label=label)
            elif 'rmse' in label or 'l1' in label:
                ax_trace[1].plot(loss, '-', label=label)
        if len(self.history) > 1:
            ax_trace[0].legend(loc='upper right', prop={'size': 6})
            ax_trace[1].legend(loc='upper right', prop={'size': 6})
            ax_trace[2].legend(loc='upper right', prop={'size': 6})
        plt.tight_layout()
        plt.savefig(str(self.output_dir / 'loss.png'), dpi=300)
        plt.close(fig)

        if self.print_header:
            logging.info(' ' * 7 + '  '.join(
                f'{key:>12}' for key in sorted(losses)))
            self.print_header = False
        logging.info(f'[{epoch:4}] ' + '  '.join(
            f'{val:12.4f}' for _, val in sorted(losses.items())))