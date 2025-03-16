import os
import torch
import random
import numpy as np
from tqdm import tqdm
import clip_ldc as clip
import torch.nn.functional as F
import torchvision.transforms as T
from alisuretool.Tools import Tools
from torch.utils.data import DataLoader
from datasets.utils import DatasetWrapper
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


MODEL_CACHE_DIR = './model/clip'
DATA_ROOT = './your/data/path'
LOG_ROOT = './result/log'


class MyTransform(object):

    @staticmethod
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    @staticmethod
    def transform_train(size, scale=(0.8, 1.0)):
        funcs = [
            T.RandomResizedCrop(size=size, scale=scale, interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5), MyTransform._convert_image_to_rgb, ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]
        return Compose(funcs)

    @staticmethod
    def transform_test(size):
        funcs = [
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size), MyTransform._convert_image_to_rgb, ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]
        return Compose(funcs)

    pass


class Config10Dataset(object):

    def __init__(self, dataset_name, seed=2024, shots=16, backbone="RN50", lr=0.001, batch_size=64, train_epoch=50,
                 loss_lambda=[1.0, 1.0, 1.0, 1.0, 1.0], fuse_type=2):
        self.setup_seed(seed)

        self.seed = seed
        self.shots = shots
        self.lr = lr
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.backbone = backbone  # RN50 RN101 ViT-B/32 ViT-B/16

        self.loss_lambda = loss_lambda
        self.fuse_type = fuse_type

        _dataset_info = self.dataset_info()
        self.dataset_name = dataset_name
        assert self.dataset_name in _dataset_info.keys()
        self.data_path = os.path.join(DATA_ROOT, _dataset_info[self.dataset_name][2])
        self.dataset = _dataset_info[self.dataset_name][0](self.data_path, self.shots)
        self.num_classes = _dataset_info[self.dataset_name][1]

        self.cache_dir = MODEL_CACHE_DIR
        pass

    def get_detail(self):
        detail_str = (f"dataset_name={self.dataset_name}, shots={self.shots}, lr={self.lr}, seed={self.seed}, "
                      f"train_epoch={self.train_epoch}, batch_size={self.batch_size}, backbone={self.backbone}, "
                      f"num_classes={self.num_classes}, loss_lambda={self.loss_lambda}, fuse_type={self.fuse_type}")
        return detail_str

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        pass

    @staticmethod
    def get_gpu_id():
        """
        torch.cuda.set_device(get_gpu_id())
        """
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_id, free = 0, 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            now_free = (info.free // 1048576) / 1024  # info.total, info.free, info.used
            if now_free > free:
                free = now_free
                gpu_id = i
            pass
        pynvml.nvmlShutdown()
        return gpu_id

    @staticmethod
    def dataset_info():
        from datasets.oxford_pets import OxfordPets
        from datasets.eurosat import EuroSAT
        from datasets.ucf101 import UCF101
        from datasets.sun397 import SUN397
        from datasets.caltech101 import Caltech101
        from datasets.dtd import DescribableTextures
        from datasets.fgvc import FGVCAircraft
        from datasets.food101 import Food101
        from datasets.oxford_flowers import OxfordFlowers
        from datasets.stanford_cars import StanfordCars

        return {"caltech101": [Caltech101, 100, "caltech-101"], "dtd": [DescribableTextures, 47, "dtd"],
                "fgvc": [FGVCAircraft, 100, "fgvc_aircraft"], "eurosat": [EuroSAT, 10, "eurosat"],
                "food101": [Food101, 101, "food-101"], "oxford_flowers": [OxfordFlowers, 102, "oxford_flowers"],
                "oxford_pets": [OxfordPets, 37, "oxford_pets"], "stanford_cars": [StanfordCars, 196, "stanford_cars"],
                "sun397": [SUN397, 397, "sun397"], "ucf101": [UCF101, 101, "ucf101"]}

    pass


class ConfigImageDomainShift(object):

    def __init__(self, seed=2024, shots=16, backbone="RN50", lr=0.001, batch_size=64, train_epoch=50,
                 loss_lambda=[1.0, 1.0, 1.0, 1.0, 1.0], fuse_type=2, has_ood=True):
        Config10Dataset.setup_seed(seed)

        self.seed = seed
        self.shots = shots
        self.lr = lr
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.backbone = backbone  # RN50 RN101 ViT-B/32 ViT-B/16

        self.loss_lambda = loss_lambda
        self.fuse_type = fuse_type
        self.has_ood = has_ood

        self.num_classes = 1000
        self.dataset_name = "imagenet"
        self.data_path_imagenet = os.path.join(DATA_ROOT, 'imagenet/images')
        self.data_path_imagenet_v2 = os.path.join(DATA_ROOT, 'imagenetv2/imagenetv2-matched-frequency-format-val')
        self.data_path_imagenet_sketch = os.path.join(DATA_ROOT, 'imagenet-sketch/images')

        from datasets.imagenet import MyImageNet
        from datasets.imagenetv2 import ImageNetV2
        from datasets.imagenet_sketch import ImageNetSketch
        self.dataset = MyImageNet(self.data_path_imagenet, self.shots, 'train', MyTransform.transform_train(224))
        self.test_set = MyImageNet(root=self.data_path_imagenet, num_shots=self.shots,
                                   split='test', transform=MyTransform.transform_test(224))
        self.test_set_v2 = ImageNetV2(root=self.data_path_imagenet_v2, transform=MyTransform.transform_test(224))
        self.test_set_sketch = ImageNetSketch(root=self.data_path_imagenet_sketch, transform=MyTransform.transform_test(224))

        self.cache_dir = MODEL_CACHE_DIR
        pass

    def get_detail(self):
        detail_str = (f"dataset_name={self.dataset_name}, shots={self.shots}, lr={self.lr}, seed={self.seed}, "
                      f"train_epoch={self.train_epoch}, batch_size={self.batch_size}, backbone={self.backbone}, "
                      f"num_classes={self.num_classes}, loss_lambda={self.loss_lambda}, fuse_type={self.fuse_type}")
        return detail_str

    pass


class MyScheduler(object):

    def __init__(self, optimizer, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0) -> None:
        self.optimizer = optimizer
        self.optimizer.param_groups[0]['lr'] = 0

        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(0, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        self.schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        self.schedule = np.concatenate((warmup_schedule, self.schedule))
        self.id = 0
        assert len(self.schedule) == epochs * niter_per_ep

    def step(self):
        self.optimizer.param_groups[0]['lr'] = self.schedule[self.id]
        self.id += 1
        pass

    pass


class Eval(object):

    def __init__(self, batch_size, clip_model, val_loader, text_feats):
        self.clip_model = clip_model
        self.text_feats = text_feats
        self.val_loader = val_loader
        self.batch_size = batch_size
        pass

    def eval(self, best_beta=None):
        self.clip_model.eval()
        all_labels, all_logits = [], []
        with torch.no_grad():
            with tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc='Evaluate') as tqdm_eval:
                for _, (images, labels) in tqdm_eval:
                    clip_logits, mlp_logits, ada_logits, tot_logits = self.clip_model.my_forward(images.cuda(),
                                                                                                 self.text_feats)
                    all_logits.append([clip_logits, mlp_logits, ada_logits, tot_logits])
                    all_labels.append(labels)
                    pass
                pass
            pass
        all_labels = torch.cat(all_labels, dim=0)

        result_acc = {}
        acc = self.cal_acc(torch.cat([one[0] for one in all_logits], dim=0), all_labels) * 100.
        result_acc["clip_logits"] = acc
        Tools.print(f"test all_clip_logits acc={acc:.2f}%")
        acc = self.cal_acc(torch.cat([one[1] for one in all_logits], dim=0), all_labels) * 100.
        result_acc["mlp_logits"] = acc
        Tools.print(f"test all_mlp_logits acc={acc:.2f}%")
        acc = self.cal_acc(torch.cat([one[2] for one in all_logits], dim=0), all_labels) * 100.
        result_acc["ada_logits"] = acc
        Tools.print(f"test all_ada_logits acc={acc:.2f}%")
        acc = self.cal_acc(torch.cat([one[3] for one in all_logits], dim=0), all_labels) * 100.
        result_acc["tot_logits"] = acc
        Tools.print(f"test all_tot_logits acc={acc:.2f}%")

        if best_beta is None:
            best_beta, last_acc, best_acc = self.search_hp(torch.cat([one[1] for one in all_logits], dim=0),
                                                           torch.cat([one[2] for one in all_logits], dim=0), all_labels)
            result_acc["acc"] = best_acc
            Tools.print(f"val best beta = {best_beta:.4f} => last_acc={last_acc:.2f}% [best_acc={best_acc}]")
            return best_beta, result_acc
        else:
            logits = self.fuse_logits(torch.cat([one[1] for one in all_logits], dim=0),
                                      torch.cat([one[2] for one in all_logits], dim=0), beta=best_beta)
            acc = self.cal_acc(logits, all_labels) * 100.
            result_acc["acc"] = acc
            Tools.print(f"test acc={acc:.2f}%")
            return best_beta, result_acc
        # return best_beta, acc

    @staticmethod
    def fuse_logits(mlp_logits, clip_logits, beta=1.0):
        return beta * mlp_logits + (1 - beta) * clip_logits

    @staticmethod
    def cal_acc(logits, labels):
        pred = torch.argmax(logits, -1)
        acc_num = (pred == labels.cuda()).sum().item()
        return 1.0 * acc_num / len(labels)

    def search_hp(self, mlp_logits, clip_logits, all_labels, start=0, end=1, step=50):
        beta_list = [i * (end - start) / step + start for i in range(step + 1)]
        accs, best_beta, best_acc = [], start, 0.
        for beta in beta_list:
            logits = self.fuse_logits(mlp_logits, clip_logits, beta=beta)
            acc = self.cal_acc(logits, all_labels) * 100.
            accs.append((beta, acc))
            if acc > best_acc:
                best_acc = acc
                best_beta = beta
        return best_beta, accs[-1][-1], best_acc

    pass


class AvgACC:
    def __init__(self) -> None:
        self.acc_num = 0
        self.total = 0
        pass

    def step(self, logits, labels):
        pred = torch.argmax(logits, -1)
        acc_num = (pred == labels.cuda()).sum().item()
        total = len(labels)
        self.acc_num += acc_num
        self.total += total
        pass

    def cal(self):
        return 0.00 if self.total == 0 else 1.0 * self.acc_num / self.total

    pass


class Runner(object):

    def __init__(self, config):
        self.config = config

        Tools.print(f"Preparing {self.config.backbone} model.")
        self.clip_model, self.preprocess = clip.load(self.config.backbone, download_root=self.config.cache_dir,
                                                     num_classes=self.config.num_classes, config=self.config)
        self.clip_model.eval()

        Tools.print("Getting cached textual weights W ...")
        self.text_feats = self.clip_classifier(
            os.path.join(self.config.cache_dir, f"{self.config.dataset_name}_{self.config.backbone}_textfeats.pt"),
            self.config.dataset.classnames, self.config.dataset.template, self.clip_model)

        # Preparation for training
        for param in self.clip_model.parameters():
            param.requires_grad = False
            pass
        for name, param in self.clip_model.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
            pass

        Tools.print(f"Preparing {self.config.dataset_name} dataset.")
        if self.config.dataset_name != "imagenet":
            self.train_loader = DataLoader(
                DatasetWrapper(self.config.dataset.train_x, input_size=224, transform=MyTransform.transform_train(224), is_train=True),
                batch_size=self.config.batch_size, num_workers=8, shuffle=True, drop_last=False, pin_memory=(torch.cuda.is_available()))
            self.val_loader = DataLoader(
                DatasetWrapper(self.config.dataset.val, input_size=224, transform=self.preprocess, is_train=False),
                batch_size=64, num_workers=8, shuffle=False, drop_last=False, pin_memory=(torch.cuda.is_available()))
            self.test_loader = DataLoader(
                DatasetWrapper(self.config.dataset.test, input_size=224, transform=self.preprocess, is_train=False),
                batch_size=64, num_workers=8, shuffle=False, drop_last=False, pin_memory=(torch.cuda.is_available()))
            self.test_loader_list = [self.test_loader]
        else:
            self.train_loader = DataLoader(self.config.dataset, self.config.batch_size, num_workers=8, shuffle=True)
            self.val_loader = None
            self.test_loader = DataLoader(dataset=self.config.test_set, batch_size=self.config.batch_size, num_workers=8, shuffle=False)
            self.test_loader_v2 = DataLoader(dataset=self.config.test_set_v2, batch_size=self.config.batch_size, num_workers=8, shuffle=False)
            self.test_loader_sketch = DataLoader(dataset=self.config.test_set_sketch, batch_size=self.config.batch_size, num_workers=8, shuffle=False)
            self.test_loader_list = [self.test_loader, self.test_loader_v2, self.test_loader_sketch] if self.config.has_ood else [self.test_loader]
            pass

        self.optimizer = torch.optim.AdamW(self.clip_model.parameters(), lr=self.config.lr / 10, weight_decay=1e-4, eps=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.config.train_epoch * len(self.train_loader))

        self.eval = Eval(self.config.batch_size, self.clip_model, self.test_loader, self.text_feats)
        pass

    def train_epoch(self, epoch):
        self.clip_model.adapter.train()
        self.clip_model.visual.adapter.train()

        train_acc, train_loss = AvgACC(), 0.0
        loss_list = [0, 0, 0, 0, 0]
        with tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"epoch {epoch}") as tqdm_train:
            for _, (images, labels) in tqdm_train:
                images, labels = images.cuda(), labels.cuda()
                clip_logits, mlp_logits, ada_logits, total_logits = self.clip_model.my_forward(images, self.text_feats)
                loss, losses = self.get_loss(labels, clip_logits, mlp_logits, ada_logits, total_logits,
                                             lambda_value=self.config.loss_lambda)
                train_loss += loss.item()
                train_acc.step(mlp_logits, labels)

                for i, l in enumerate(losses):
                    loss_list[i] += l.item()
                tqdm_train.set_postfix(cur_loss=loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

            train_acc_result = train_acc.cal()
            train_loss = train_loss / len(self.train_loader)
            pass

        Tools.print(f"train acc={train_acc_result}, "
                    f"[l1_loss, ce_loss] => {[one / len(self.train_loader) for one in loss_list]}")
        return train_loss

    def train(self):
        for epoch in range(self.config.train_epoch):
            loss = self.train_epoch(epoch)
            Tools.print(f"Epoch: {epoch}, loss: {loss:.4f}, "
                        f"lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:.8f}")
            pass
        return self.test()

    def test(self):
        self.eval.clip_model = self.clip_model
        val_best_beta = None
        if self.val_loader:
            self.eval.val_loader = self.val_loader
            val_best_beta, val_result_acc = self.eval.eval()
            pass
        test_acc_list = []
        for test_loader in self.test_loader_list:
            self.eval.val_loader = test_loader
            val_best_beta, test_result_acc = self.eval.eval(best_beta=val_best_beta)
            test_acc_list.append(test_result_acc)
            pass
        return test_acc_list

    @staticmethod
    def clip_classifier(feat_path, classnames, template, clip_model):
        if os.path.exists(feat_path):
            Tools.print(f"Loading texture features from {feat_path}")
            text_feats = torch.load(feat_path, map_location='cpu')
            return text_feats.cuda()

        with torch.no_grad():
            clip_weights = []
            for classname in classnames:
                classname = classname.replace('_', ' ')
                if isinstance(template, list):
                    texts = [t.format(classname) for t in template]
                elif isinstance(template, dict):
                    texts = template[classname]

                texts = clip.tokenize(texts).cuda()
                # prompt ensemble for ImageNet
                class_embeddings = clip_model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                clip_weights.append(class_embedding)
                pass

            clip_weights = torch.stack(clip_weights, dim=1).cuda()
            torch.save(clip_weights, Tools.new_dir(feat_path))

        return clip_weights

    @staticmethod
    def get_loss(labels, clip_logits, mlp_logits, ada_logits, total_logits, lambda_value=[1.0, 1.0, 1.0, 1.0, 1.0]):
        ce_loss = F.cross_entropy(mlp_logits, labels) * lambda_value[0]
        ce_loss2 = F.cross_entropy(ada_logits, labels) * lambda_value[1]
        ce_loss3 = F.cross_entropy(total_logits, labels) * lambda_value[2]

        l1_loss1 = F.l1_loss(mlp_logits, clip_logits) * lambda_value[3]
        l1_loss2 = F.l1_loss(ada_logits, clip_logits) * lambda_value[4]

        loss = l1_loss1 + l1_loss2 + ce_loss + ce_loss2 + ce_loss3
        return loss, [l1_loss1, l1_loss2, ce_loss, ce_loss2, ce_loss3]

    pass


class AllExperiments(object):

    def __init__(self):
        self.seed = 2024
        self.datasets = "imagenet/fgvc/caltech101/stanford_cars/dtd/eurosat/oxford_flowers/food101/oxford_pets/sun397/ucf101"
        pass

    def main_experiment_1_zero_shot(self):
        log_txt_path = Tools.new_dir(os.path.join(LOG_ROOT, "1_main_experiment_1_zero_shot.txt"))
        backbone_list = ["RN50", "ViT-B/16"]
        for backbone in backbone_list:
            self.experiment_one(backbone=backbone, train_epoch=0, has_ood=False, log_txt_path=log_txt_path)
            pass
        pass

    def main_experiment_2_few_shot(self):
        log_txt_path = Tools.new_dir(os.path.join(LOG_ROOT, "1_main_experiment_2_few_shot.txt"))
        backbone_list = ["RN50", "ViT-B/16"]
        shots_list = [1, 2, 4, 8, 16]
        for backbone in backbone_list:
            for shots in shots_list:
                self.experiment_one(shots=shots, backbone=backbone, log_txt_path=log_txt_path)
                pass
        pass

    def experiment_one(self, shots=16, backbone="RN50", train_epoch=50, has_ood=True, log_txt_path=None):
        results = []
        for dataset_name in self.datasets.split('/'):
            # Dataset
            if dataset_name == "imagenet":
                config = ConfigImageDomainShift(seed=self.seed, shots=shots, backbone=backbone,
                                                train_epoch=train_epoch, has_ood=has_ood)
            else:
                config = Config10Dataset(dataset_name=dataset_name, seed=self.seed, shots=shots,
                                         backbone=backbone, train_epoch=train_epoch)
                pass

            # Runner
            runner = Runner(config=config)
            acc_list = runner.train()
            results.append({"name": dataset_name, "acc": acc_list, "detail": config.get_detail()})

            Tools.print({"name": dataset_name, "acc": acc_list, "detail": config.get_detail()}, log_txt_path)
            pass

        # 计算平均结果
        acc_keys = ["clip_logits", "mlp_logits", "ada_logits", "tot_logits", "acc"]
        for key in acc_keys:
            avg_acc, count = 0, 0
            avg_acc += results[0]['acc'][0][key]  # ImageNet
            count += 1
            for result in results[1:]:
                avg_acc += sum([one[key] for one in result['acc']])
                count += len([one[key] for one in result['acc']])
                pass
            Tools.print(f"avg {key} acc={avg_acc / count}", log_txt_path)
            pass
        pass

    pass


if __name__ == '__main__':
    all_experiment = AllExperiments()
    all_experiment.main_experiment_1_zero_shot()
    all_experiment.main_experiment_2_few_shot()
    pass
