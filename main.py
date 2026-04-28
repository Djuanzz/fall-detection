#!/usr/bin/env python
from __future__ import print_function
import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
import copy
import platform


# ── DictAction: pengganti torchlight.DictAction ───────────────────────────────
class DictAction(argparse.Action):
    """
    argparse action untuk menerima argumen dict dari command line.
    Format: --arg key1=val1 key2=val2
    """
    @staticmethod
    def _parse_value(v):
        if v.lower() == 'true':
            return True
        if v.lower() == 'false':
            return False
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    def __call__(self, parser, namespace, values, option_string=None):
        result = getattr(namespace, self.dest, None) or {}
        if isinstance(values, list):
            for item in values:
                if '=' in item:
                    k, v = item.split('=', 1)
                    result[k.strip()] = self._parse_value(v.strip())
                else:
                    raise argparse.ArgumentTypeError(
                        "Format harus key=value, dapat: '{}'".format(item))
        setattr(namespace, self.dest, result)


# ── Fungsi bantu loss ──────────────────────────────────────────────────────────

def build_loss(loss_name: str, loss_args: dict) -> nn.Module:
    """
    Buat loss function dengan aman.
    Konversi 'weight' dari list ke torch.Tensor secara otomatis.
    """
    args = dict(loss_args) if loss_args else {}
    if 'weight' in args:
        w = args['weight']
        if w is not None and not isinstance(w, torch.Tensor):
            args['weight'] = torch.tensor(w, dtype=torch.float32)
        elif w is None:
            del args['weight']
    LossClass = getattr(nn, loss_name, None)
    if LossClass is None:
        raise ValueError("Loss '{}' tidak ditemukan di torch.nn".format(loss_name))
    return LossClass(**args)


def move_loss_to_device(loss: nn.Module, device) -> nn.Module:
    """
    [FIXED] Pindahkan SEMUA parameter/buffer loss ke device yang sama dengan model.
    Sebelumnya hanya handle CrossEntropyLoss — sekarang handle semua loss
    yang punya attribute 'weight'.
    """
    # Cara paling robust: panggil .to(device) langsung pada loss module
    # Ini akan memindahkan semua buffer (termasuk .weight) ke device yang tepat
    loss = loss.to(device)
    return loss


# ─────────────────────────────────────────────────────────────────────────────

# Set file descriptor limit (Linux/Mac only, skip on Windows)
if platform.system() != 'Windows':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def init_seed(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Per-worker seed so each DataLoader worker has independent RNG state."""
    np.random.seed(worker_id)
    random.seed(worker_id)


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (
            class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    parser = argparse.ArgumentParser(description='Fall Detection — BlockGCN')

    parser.add_argument('--work-dir', default='./work_dir/fall_detection',
                        help='folder untuk menyimpan hasil')
    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('--config', default='./config/fall_detection/balanced.yaml',
                        help='path ke file konfigurasi')

    # processor
    parser.add_argument('--phase', default='train', help='train atau test')
    parser.add_argument('--save-score', type=str2bool, default=True)

    # debug & log
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=1)
    parser.add_argument('--save-epoch', type=int, default=10)
    parser.add_argument('--eval-interval', type=int, default=5)
    parser.add_argument('--print-log', type=str2bool, default=True)
    parser.add_argument('--show-topk', type=int, default=[1], nargs='+')

    # feeder
    parser.add_argument('--feeder', default='feeders.fall_feeder.Feeder')
    parser.add_argument('--num-worker', type=int, default=4)
    parser.add_argument('--train-feeder-args', action=DictAction, default=dict())
    parser.add_argument('--test-feeder-args',  action=DictAction, default=dict())

    # model
    parser.add_argument('--model', default=None)
    parser.add_argument('--model-args', action=DictAction, default=dict())
    parser.add_argument('--weights', default=None)
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+')

    # loss
    parser.add_argument('--loss', default='CrossEntropyLoss')
    parser.add_argument('--loss-args', action=DictAction, default=dict())

    # optimizer
    parser.add_argument('--base-lr', type=float, default=0.1)
    parser.add_argument('--step', type=int, default=[30, 45], nargs='+')
    parser.add_argument('--device', type=int, default=[0], nargs='+')
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--test-batch-size', type=int, default=16)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--num-epoch', type=int, default=55)
    parser.add_argument('--weight-decay', type=float, default=0.0004)
    parser.add_argument('--lr-decay-rate', type=float, default=0.1)
    parser.add_argument('--warm-up-epoch', type=int, default=5)

    parser.add_argument('--alpha', type=str2bool, default=False)

    return parser


class Processor():
    """Processor untuk Fall Detection berbasis Skeleton."""

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()

        # ── Validasi CUDA ──────────────────────────────────────────────────────
        if torch.cuda.is_available():
            num_gpu = torch.cuda.device_count()
            for d in (arg.device if isinstance(arg.device, list) else [arg.device]):
                if d >= num_gpu:
                    raise RuntimeError(
                        "Device cuda:{} tidak ada. GPU tersedia: {} buah (id 0..{})".format(
                            d, num_gpu, num_gpu - 1))
            print("GPU tersedia: {}".format(num_gpu))
            for i in range(num_gpu):
                print("  cuda:{} → {}".format(i, torch.cuda.get_device_name(i)))
            self.use_cuda = True
        else:
            if arg.phase == 'train':
                raise RuntimeError(
                    "CUDA tidak tersedia! Training membutuhkan GPU.\n"
                    "Pastikan PyTorch GPU dan driver NVIDIA sudah terinstall.")
            print("[WARN] CUDA tidak tersedia — menjalankan di CPU (test/inference only)")
            self.use_cuda = False

        if arg.phase == 'train':
            is_debug = arg.train_feeder_args.get('debug', False)
            if not is_debug:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'sudah ada')
                    answer = input('Hapus? y/n: ')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir dihapus: ', arg.model_saved_name)
                    else:
                        print('Dir tidak dihapus: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(
                    os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(
                    os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(
                    os.path.join(arg.model_saved_name, 'test'), 'test')

        self.global_step = 0
        self.load_model()

        if self.arg.phase != 'model_size':
            self.load_optimizer()
            self.load_data()

        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.best_f1       = 0.0
        self.best_f1_epoch = 0

        if self.use_cuda:
            self.model = self.model.cuda(self.output_device)
            self.loss   = move_loss_to_device(self.loss, self.output_device)
            if type(self.arg.device) is list and len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)
                self.loss = move_loss_to_device(self.loss, self.output_device)
        else:
            self.output_device = 'cpu'

    # ── Load data ──────────────────────────────────────────────────────────────

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        num_worker = self.arg.num_worker

        use_pin = self.use_cuda

        if self.arg.phase == 'train':
            train_dataset = Feeder(**self.arg.train_feeder_args)
            loader_kwargs = dict(
                dataset=train_dataset,
                batch_size=self.arg.batch_size,
                shuffle=True,
                pin_memory=use_pin,
                num_workers=num_worker,
                drop_last=True,
                worker_init_fn=worker_init_fn,
            )
            if num_worker > 0:
                loader_kwargs['prefetch_factor'] = 2
            self.data_loader['train'] = torch.utils.data.DataLoader(**loader_kwargs)

        test_dataset = Feeder(**self.arg.test_feeder_args)
        test_kwargs = dict(
            dataset=test_dataset,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            pin_memory=use_pin,
            num_workers=num_worker,
            drop_last=False,
            worker_init_fn=worker_init_fn,
        )
        if num_worker > 0:
            test_kwargs['prefetch_factor'] = 2
        self.data_loader['test'] = torch.utils.data.DataLoader(**test_kwargs)

    # ── Load model ─────────────────────────────────────────────────────────────

    def load_model(self):
        if torch.cuda.is_available():
            output_device = self.arg.device[0] if type(self.arg.device) is list \
                            else self.arg.device
        else:
            output_device = 'cpu'
        self.output_device = output_device

        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)

        self.model = Model(**self.arg.model_args)
        print(self.model)

        self.loss = build_loss(self.arg.loss, self.arg.loss_args)

        if self.arg.weights:
            self.print_log('Load weights dari {}.'.format(self.arg.weights))
            map_loc = ('cuda:{}'.format(output_device)
                       if torch.cuda.is_available() else 'cpu')
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'rb') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights, map_location=map_loc,
                                     weights_only=False)

            weights = OrderedDict([
                [k.split('module.')[-1], v]
                for k, v in weights.items()
            ])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except Exception:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Weights tidak ditemukan:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    # ── Load optimizer ─────────────────────────────────────────────────────────

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=self.arg.momentum,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError('Optimizer tidak dikenal: {}'.format(self.arg.optimizer))

        self.print_log('Warm up epoch: {}'.format(self.arg.warm_up_epoch))

    # ── Utilities ──────────────────────────────────────────────────────────────

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write("# command line: {}\n\n".format(' '.join(sys.argv)))
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer in ('SGD', 'Adam', 'AdamW'):
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                    self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_value = []
        acc_value  = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        use_amp = self.use_cuda
        amp_device = 'cuda' if self.use_cuda else 'cpu'
        scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)

        for batch_idx, (data, label) in enumerate(process):
            self.global_step += 1

            with torch.no_grad():
                data  = data.float().to(self.output_device)
                label = label.long().to(self.output_device)
            timer['dataloader'] += self.split_time()

            amp_device = 'cuda' if self.use_cuda else 'cpu'
            with torch.amp.autocast(amp_device, enabled=use_amp):
                output = self.model(data)
                loss   = self.loss(output, label)

            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            _, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())

            self.train_writer.add_scalar('acc',  acc,              self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(
                np.mean(loss_value), np.mean(acc_value) * 100))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([
                [k.split('module.')[-1], v.cpu()]
                for k, v in state_dict.items()
            ])
            torch.save(weights,
                       self.arg.model_saved_name + '-' + str(epoch + 1) +
                       '-' + str(int(self.global_step)) + '.pt')

    # ── Evaluasi ───────────────────────────────────────────────────────────────

    def eval(self, epoch, save_score=False, loader_name=None,
             wrong_file=None, result_file=None):
        if loader_name is None:
            loader_name = ['test']

        f_w = open(wrong_file,  'w') if wrong_file  is not None else None
        f_r = open(result_file, 'w') if result_file is not None else None

        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        best_f1_this_eval = 0.0

        for ln in loader_name:
            loss_value  = []
            score_frag  = []
            label_list  = []
            pred_list   = []
            step = 0

            process = tqdm(self.data_loader[ln], ncols=40)

            for batch_idx, (data, label) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    data  = data.float().to(self.output_device)
                    label = label.long().to(self.output_device)

                    output = self.model(data)
                    loss   = self.loss(output, label)

                score_frag.append(output.data.cpu().numpy())
                loss_value.append(loss.data.item())

                _, predict_label = torch.max(output.data, 1)
                pred_list.append(predict_label.data.cpu().numpy())
                step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true    = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(step) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_frag)
            loss  = np.mean(loss_value)

            accuracy = self.data_loader[ln].dataset.top_k(score, 1)

            if accuracy > self.best_acc:
                self.best_acc       = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: {:.4f}  model: {}'.format(
                accuracy, self.arg.model_saved_name))

            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss,     self.global_step)
                self.val_writer.add_scalar('acc',  accuracy, self.global_step)

            sample_names = self.data_loader[ln].dataset.sample_name
            score_dict   = dict(zip(sample_names, score))

            self.print_log('\tMean {} loss ({} batches): {:.4f}'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))

            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            label_list = np.concatenate(label_list)
            pred_list  = np.concatenate(pred_list)
            confusion  = confusion_matrix(label_list, pred_list)
            list_diag    = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc     = list_diag / (list_raw_sum + 1e-9)

            if confusion.shape == (2, 2):
                tn, fp, fn, tp = confusion.ravel()
                prec   = tp / max(tp + fp, 1)
                rec    = tp / max(tp + fn, 1)
                spec   = tn / max(tn + fp, 1)
                f1     = 2 * prec * rec / max(prec + rec, 1e-9)
                best_f1_this_eval = max(best_f1_this_eval, f1)
                self.print_log(
                    '\tTN={} FP={} FN={} TP={} | '
                    'Prec={:.2f}% Rec={:.2f}% Spec={:.2f}% F1={:.2f}%'.format(
                        tn, fp, fn, tp,
                        prec * 100, rec * 100, spec * 100, f1 * 100))
                if self.arg.phase == 'train':
                    self.val_writer.add_scalar('precision', prec,  self.global_step)
                    self.val_writer.add_scalar('recall',    rec,   self.global_step)
                    self.val_writer.add_scalar('f1',        f1,    self.global_step)

            with open('{}/epoch{}_{}_each_class_acc.csv'.format(
                    self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

        if f_w is not None:
            f_w.close()
        if f_r is not None:
            f_r.close()
        return best_f1_this_eval

    # ── Start ──────────────────────────────────────────────────────────────────

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = int(self.arg.start_epoch *
                                   len(self.data_loader['train']))

            num_params = sum(p.numel() for p in self.model.parameters()
                             if p.requires_grad)
            self.print_log('# Parameters: {}'.format(num_params))

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (
                    ((epoch + 1) % self.arg.save_interval == 0) or
                    (epoch + 1 == self.arg.num_epoch)
                ) and (epoch + 1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)
                epoch_f1 = self.eval(epoch,
                                     save_score=self.arg.save_score,
                                     loader_name=['test'])
                if epoch_f1 > self.best_f1:
                    self.best_f1       = epoch_f1
                    self.best_f1_epoch = epoch + 1

            pattern = os.path.join(
                self.arg.work_dir,
                'runs-' + str(self.best_acc_epoch) + '*')
            best_files = glob.glob(pattern)
            if best_files:
                weights_path = best_files[0]
                weights = torch.load(weights_path,
                                     map_location=('cuda:{}'.format(self.output_device)
                                                   if self.use_cuda else 'cpu'),
                                     weights_only=False)
                if type(self.arg.device) is list and len(self.arg.device) > 1:
                    weights = OrderedDict([
                        ['module.' + k, v.to(self.output_device)]
                        for k, v in weights.items()
                    ])
                self.model.load_state_dict(weights)

                wf = weights_path.replace('.pt', '_wrong.txt')
                rf = weights_path.replace('.pt', '_right.txt')
                self.arg.print_log = False
                self.eval(epoch=0, save_score=True,
                          loader_name=['test'],
                          wrong_file=wf, result_file=rf)
                self.arg.print_log = True

            self.print_log('Best accuracy: {:.4f}  (epoch {})'.format(
                self.best_acc, self.best_acc_epoch))
            self.print_log('Best F1      : {:.4f}  (epoch {})'.format(
                self.best_f1, self.best_f1_epoch))
            self.print_log('Work dir     : {}'.format(self.arg.work_dir))
            self.print_log('Params       : {}'.format(num_params))
            self.print_log('Weight decay : {}'.format(self.arg.weight_decay))
            self.print_log('Base LR      : {}'.format(self.arg.base_lr))
            self.print_log('Batch size   : {}'.format(self.arg.batch_size))
            self.print_log('Seed         : {}'.format(self.arg.seed))

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Gunakan --weights untuk menentukan file model.')
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.print_log('Model  : {}'.format(self.arg.model))
            self.print_log('Weights: {}'.format(self.arg.weights))
            self.eval(epoch=0,
                      save_score=self.arg.save_score,
                      loader_name=['test'],
                      wrong_file=wf, result_file=rf)
            self.print_log('Done.')


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()

    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)

        valid_keys = set(vars(p).keys())
        normalized_arg = {}
        for k, v in default_arg.items():
            k_norm = k.replace('-', '_')
            if k_norm not in valid_keys:
                print('WRONG ARG (diabaikan): {}'.format(k))
                continue
            normalized_arg[k_norm] = v

        parser.set_defaults(**normalized_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()