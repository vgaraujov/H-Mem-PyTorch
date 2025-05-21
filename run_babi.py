try:
    import comet_ml
    has_comet = True
except (ImportError):
    has_comet = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils import data as data_utils
    
import logging
import yaml
import argparse
from box import *
from tqdm import tqdm
from pathlib import Path
from seed import set_seed

from model import HMemQA
from dataset import bAbIDataset

logger = logging.getLogger(__name__)

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer: optim.Optimizer, multiplier: float, steps: int, decay_factor: float = 1.0, decay_after_steps: bool = True):
        self.multiplier = multiplier
        self.steps = steps
        self.decay_factor = decay_factor
        self.decay_after_steps = decay_after_steps
        # It's important that initial_base_lrs is a copy of the original base_lrs
        # before _LRScheduler.__init__ might alter them or before they are used.
        # However, _LRScheduler stores the original LRs from optimizer.param_groups in self.base_lrs.
        # So, we capture them *after* super().__init__() to ensure they are correctly initialized.
        super(WarmupScheduler, self).__init__(optimizer=optimizer)
        # Capture the base LRs that _LRScheduler has set. These are the true "initial" LRs.
        self.initial_base_lrs = list(self.base_lrs)

    def get_lr(self):
        if self.last_epoch < self.steps:
            # Warmup phase: multiply the original base LRs by the multiplier
            return [initial_lr * self.multiplier for initial_lr in self.initial_base_lrs]
        
        # After warmup phase
        if self.decay_after_steps and self.decay_factor < 1.0:
            # Calculate number of decay steps that have occurred after warmup
            decay_steps = self.last_epoch - self.steps + 1
            decay_multiplier = self.decay_factor ** decay_steps
            # Apply decay to the original base LRs
            return [initial_lr * decay_multiplier for initial_lr in self.initial_base_lrs]
        
        # No decay after warmup, or decay_factor is 1.0 (no decay)
        # Return the original base LRs (or what they were set to post-warmup if multiplier was intended to be permanent)
        # Given the logic, it seems the multiplier is only for warmup. So return initial_base_lrs.
        return list(self.initial_base_lrs)

    # decay_lr method is removed

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, default=1)
parser.add_argument("--data_dir", type=str, metavar='PATH', default="/mnt/ialabnas/homes/vgaraujo/TPRBERT/tasks_1-20_v1-2/en-10k",
                    help="dataset directory path")
args = parser.parse_args()

config = box_from_file(Path('config.yaml'), file_type='yaml')

logging.basicConfig(level=config.training.logging_level)

if has_comet:
    experiment = comet_ml.Experiment(
        api_key="6oPfzhw55BucwZ4m8rV62ppyw",
        project_name="h-mem",
    )                      
    experiment.set_name("h-mem babi task {}".format(args.task_id))
    tags = [args.task_id, "pytorch"]
    if config.model.read_before_write:
        tags.append("read_before_write")
    experiment.add_tags(tags)

logging.info(f"Loading Dataset")
train_dataset = bAbIDataset(args.data_dir, args.task_id)
test_dataset = bAbIDataset(args.data_dir, args.task_id, train=False)
logging.info(f"Dataset size: {len(train_dataset)}")
logging.info(f"Vocab size: {train_dataset.num_vocab}")
logging.info(f"Sentence size: {train_dataset.sentence_size}")

train_loader = data_utils.DataLoader(
    train_dataset,
    batch_size=config.training.batch_size,
    shuffle = True)
test_loader = data_utils.DataLoader(
    test_dataset,
    batch_size=config.training.batch_size,
    shuffle = False)

config.model.vocab_size = train_dataset.num_vocab
config.model.max_seq = max(train_dataset.sentence_size, train_dataset.query_size)

# define if gpu or cpu
use_cuda = not config.training.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# set seed for reproducibility
set_seed(config.training.seed, use_cuda)

model = HMemQA(config.model).to(device)

optimizer = optim.Adam(model.parameters(),
                       lr=config.optimization.learning_rate,
                       weight_decay=1e-3)

loss_fn = nn.CrossEntropyLoss(reduction='none')
warm_up = config.optimization.get("warm_up", False)

scheduler = WarmupScheduler(optimizer=optimizer,
                            steps=config.optimization.warm_up_steps if warm_up else 0,
                            multiplier=config.optimization.warm_up_factor if warm_up else 1,
                            decay_factor=config.optimization.get("decay_factor", 1.0),
                            decay_after_steps=config.optimization.get("decay", False))

for i in range(config.training.epochs):
    logging.info(f"##### EPOCH: {i} #####")
    # Train
    model.train()
    correct = 0
    train_loss = 0
    for story, query, answer in tqdm(train_loader):
        optimizer.zero_grad()
        logits = model(story.to(device), query.to(device))
        answer = answer.to(device)
        correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
        correct += correct_batch.item()

        loss = loss_fn(logits, answer)
        train_loss += loss.sum().item()
        loss = loss.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.optimization.max_grad_norm)

        optimizer.step()

    train_acc = correct / len(train_loader.dataset)
    train_loss = train_loss / len(train_loader.dataset)
    
    # Validation
    model.eval()
    correct = 0
    valid_loss = 0
    with torch.no_grad():
        for story, query, answer in tqdm(test_loader):
            optimizer.zero_grad()
            logits = model(story.to(device), query.to(device))
            answer = answer.to(device)
            correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
            correct += correct_batch.item()

            loss = loss_fn(logits, answer)
            valid_loss += loss.sum().item()

        valid_acc = correct / len(test_loader.dataset)
        valid_loss = valid_loss / len(test_loader.dataset)
        
        if has_comet:
            with experiment.train():
                experiment.log_metrics({'loss': train_loss,
                                        'accuracy': train_acc})
                experiment.log_metrics({'val_loss': valid_loss,
                                        'val_accuracy': valid_acc})

    logging.info(f"\nTrain accuracy: {train_acc:.3f}, loss: {train_loss:.3f}"
                 f"\nValid accuracy: {valid_acc:.3f}, loss: {valid_loss:.3f}"
                 f"\nLR: {optimizer.param_groups[0]['lr']:.3f}")
        
    scheduler.step()
