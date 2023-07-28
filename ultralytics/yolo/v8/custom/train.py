# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torchvision

from ultralytics.nn.tasks import CustomModel, attempt_load_one_weight
from ultralytics.yolo import v8
from ultralytics.yolo.data import CustomDataset, build_dataloader
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.yolo.utils.plotting import plot_images, plot_results
from ultralytics.yolo.utils.torch_utils import is_parallel, strip_optimizer, torch_distributed_zero_first, de_parallel


class CustomTrainer(BaseTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a CustomTrainer object with optional configuration overrides and callbacks."""
        if overrides is None:
            overrides = {}
        overrides['task'] = 'custom'
        super().__init__(cfg, overrides, _callbacks)
        self.compute_loss = Loss(num_cls=self.data['nc'])

    def set_model_attributes(self):
        """Set the YOLO model's class names from the loaded dataset."""
        self.model.names = self.data['names']

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Returns a modified PyTorch model configured for training YOLO."""
        model = CustomModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        pretrained = self.args.pretrained
        for m in model.modules():
            if not pretrained and hasattr(m, 'reset_parameters'):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training

        # Update defaults
        if self.args.imgsz == 640:
            self.args.imgsz = 224

        return model

    def setup_model(self):
        """
        load/create/download model for any task
        """
        # Custom models require special handling

        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model = str(self.model)
        # Load a YOLO model locally, from torchvision, or from Ultralytics assets
        if model.endswith('.pt'):
            self.model, _ = attempt_load_one_weight(model, device='cpu')
            for p in self.model.parameters():
                p.requires_grad = True  # for training
        elif model.endswith('.yaml'):
            self.model = self.get_model(cfg=model)
        elif model in torchvision.models.__dict__:
            pretrained = True
            self.model = torchvision.models.__dict__[model](weights='IMAGENET1K_V1' if pretrained else None)
        else:
            FileNotFoundError(f'ERROR: model={model} not found locally or online. Please check model name.')
        CustomModel.reshape_outputs(self.model, self.data['nc'])

        return  # dont return ckpt. Custom doesn't support resume

    def build_dataset(self, img_path, mode='train', batch=None):
        """Build YOLO Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """TODO: manage splits differently."""
        # Calculate stride - check if model is initialized

        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            print("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images and classes."""
        batch['img'] = batch['img'].to(self.device)
        batch['cls'] = batch['cls'].to(self.device)
        batch['pose'] = batch['pose'].to(self.device)

        return batch

    def progress_string(self):
        """Returns a formatted string showing training progress."""
        return ('\n' + '%11s' * (4 + len(self.loss_names))) % \
            ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def get_validator(self):
        """Returns an instance of CustomValidator for validation."""
        self.loss_names = ['loss']
        return v8.custom.CustomValidator(self.test_loader, self.save_dir)

    def criterion(self, preds, batch):
        """Compute loss for YOLO prediction and ground-truth."""
        loss = self.compute_loss(preds, batch)
        return loss

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def resume_training(self, ckpt):
        """Resumes training from a given checkpoint."""
        pass

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, classify=True)  # save results.png

    def final_eval(self):
        """Evaluate trained model and save validation results."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                # TODO: validate best.pt after training completes
                # if f is self.best:
                #     LOGGER.info(f'\nValidating {f}...')
                #     self.validator.args.save_json = True
                #     self.metrics = self.validator(model=f)
                #     self.metrics.pop('fitness', None)
                #     self.run_callbacks('on_fit_epoch_end')
        LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(images=batch['img'],
                    batch_idx=torch.arange(len(batch['img'])),
                    cls=batch['cls'].squeeze(-1),
                    fname=self.save_dir / f'train_batch{ni}.jpg')

# batch*n
def normalize_vector( v):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v
    
# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out

#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3]#batch*3
    y_raw = poses[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

#euler batch*4
#output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)  
def compute_rotation_matrix_from_euler(euler):
    batch=euler.shape[0]
        
    c1=torch.cos(euler[:,0]).view(batch,1)#batch*1 
    s1=torch.sin(euler[:,0]).view(batch,1)#batch*1 
    c2=torch.cos(euler[:,2]).view(batch,1)#batch*1 
    s2=torch.sin(euler[:,2]).view(batch,1)#batch*1 
    c3=torch.cos(euler[:,1]).view(batch,1)#batch*1 
    s3=torch.sin(euler[:,1]).view(batch,1)#batch*1 
        
    row1=torch.cat((c2*c3,          -s2,    c2*s3         ), 1).view(-1,1,3) #batch*1*3
    row2=torch.cat((c1*s2*c3+s1*s3, c1*c2,  c1*s2*s3-s1*c3), 1).view(-1,1,3) #batch*1*3
    row3=torch.cat((s1*s2*c3-c1*s3, s1*c2,  s1*s2*s3+c1*c3), 1).view(-1,1,3) #batch*1*3
        
    matrix = torch.cat((row1, row2, row3), 1) #batch*3*3
     
        
    return matrix

#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
class GeodesicLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, ortho6d, m2):
        m1 = compute_rotation_matrix_from_ortho6d(ortho6d)
        # m2 = compute_rotation_matrix_from_euler(euler)
        m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
        
        cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2        
        theta = torch.acos(torch.clamp(cos, -1+self.eps, 1-self.eps))
        return torch.mean(theta)
    
# Criterion class for computing training losses
class Loss:
    def __init__(self, num_cls):  # model must be de-paralleled

        device = 'cuda:0'  # get model device

        self.rot_hyp = 0.75
        self.cls_hyp = 1

        self.nc = num_cls  # number of classes
        self.device = device
        self.pose = 6

        self.rotloss = GeodesicLoss().to(device)
        self.rot_cls = [1,4,5,6,7,8,9,10]


    def __call__(self, preds, batch):
        """Calculate the sum of the loss for cls and pose multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        # feats = preds[1] if isinstance(preds, tuple) else preds
        if isinstance(preds, tuple):
            pred_poses = preds[0]
            pred_scores = preds[1]
        else:
            pred_poses, pred_scores = preds.split(
                (self.pose, self.nc), 1)

        batch_size = pred_scores.shape[0]

        # rot loss
        for i in range(batch_size):
            if int(batch['cls'][i]) in self.rot_cls:
                loss[0] += self.rotloss(pred_poses[i:i+1,...], batch['pose'][i:i+1,...])

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = torch.nn.functional.cross_entropy(pred_scores, batch['cls'], reduction='sum')

        loss[0] *= self.rot_hyp / batch_size  # rot gain
        loss[1] *= self.cls_hyp  # cls gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize YOLO model given training data and device."""
    model = cfg.model or 'yolov8n.pt'
    data = cfg.data or 'coco128.yaml'  # or yolo.CustomDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = CustomTrainer(overrides=args)
        trainer.train()


if __name__ == '__main__':
    train()
