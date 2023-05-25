import numpy as np
import torch
from torch import nn
from torch.backends import cudnn

from .models import ModelBuilder
from .utils import TransformData, get_vocabulary, postprocess, load_checkpoint
from .config import cfg

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
cuda = True if torch.cuda.is_available() else False


class Recognizer:
    def __init__(self):
        self.voc = get_vocabulary()
        voc_len = len(self.voc)
        self.char2id = dict(zip(self.voc, range(voc_len)))
        self.id2char = dict(zip(range(voc_len), self.voc))
        cfg["eos"] = self.char2id["EOS"]
        cfg["len_voc"] = len(self.voc)
        self.model = ModelBuilder(cfg)
        checkpoint = load_checkpoint(cfg["checkpoint"])
        self.model.load_state_dict(checkpoint["state_dict"])
        del checkpoint
        self.transform = TransformData()
        if cuda:
            self.device = torch.device("cuda")
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            self.model = self.model.to(self.device)
            self.model = nn.DataParallel(self.model)
        self.model.eval()
        print("Loaded STR model with {}".format("GPU" if cuda else "CPU"))

    def __call__(self, images):
        with torch.no_grad():
            rec_preds = self.model(self.transform(images))[0]
            outputs = postprocess(rec_preds, self.id2char, self.char2id)

            return outputs
