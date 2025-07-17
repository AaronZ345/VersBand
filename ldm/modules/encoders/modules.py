import torch
import torch.nn as nn
from functools import partial
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer
from importlib_resources import files
from ldm.modules.encoders.CLAP.utils import read_config_as_args
from ldm.modules.encoders.CLAP.clap import TextEncoder
from ldm.util import count_params
from ldm.modules.encoders.caption_generator import CaptionGenerator2


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]# (bsz,1)
        c = self.embedding(c)
        return c

class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class FrozenFLANEmbedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/flan-t5-large", device="cuda", max_length=77, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)# tango的flanT5是不定长度的batch，这里做成定长的batch
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

class FrozenCLAPEmbedder(AbstractEncoder):
    """Uses the CLAP transformer encoder for text from microsoft"""
    def __init__(self, weights_path, freeze=True, device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()

        model_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))['model']
        match_params = dict()
        for key in list(model_state_dict.keys()):
            if 'caption_encoder' in key:
                match_params[key.replace('caption_encoder.', '')] = model_state_dict[key]

        config_as_str = files('ldm').joinpath('modules/encoders/CLAP/config.yml').read_text()
        args = read_config_as_args(config_as_str, is_config_str=True)

        # To device
        self.tokenizer = AutoTokenizer.from_pretrained(args.text_model) # args.text_model
        self.caption_encoder = TextEncoder(
            args.d_proj, args.text_model, args.transformer_embed_dim
        )

        self.max_length = max_length
        self.device = device
        if freeze: self.freeze()

        print(f"{self.caption_encoder.__class__.__name__} comes with {count_params(self.caption_encoder) * 1.e-6:.2f} M params.")

    def freeze(self):# only freeze
        self.caption_encoder.base = self.caption_encoder.base.eval()
        for param in self.caption_encoder.base.parameters():
            param.requires_grad = False


    def encode(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)

        outputs = self.caption_encoder.base(input_ids=tokens)
        z = self.caption_encoder.projection(outputs.last_hidden_state)
        return z

class FrozenCLAPFLANEmbedder(AbstractEncoder):
    """Uses the CLAP transformer encoder for text from microsoft"""
    def __init__(self, weights_path,t5version="google/t5-v1_1-large", freeze=True, device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()

        model_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))['model']
        match_params = dict()
        for key in list(model_state_dict.keys()):
            if 'caption_encoder' in key:
                match_params[key.replace('caption_encoder.', '')] = model_state_dict[key]

        config_as_str = files('ldm').joinpath('modules/encoders/CLAP/config.yml').read_text()
        args = read_config_as_args(config_as_str, is_config_str=True)

        self.clap_tokenizer = AutoTokenizer.from_pretrained(args.text_model) # args.text_model
        self.caption_encoder = TextEncoder(
            args.d_proj, args.text_model, args.transformer_embed_dim
        )
    
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5version)
        self.t5_transformer = T5EncoderModel.from_pretrained(t5version)

        self.max_length = max_length
        self.to(device=device)
        if freeze: self.freeze()

        print(f"{self.caption_encoder.__class__.__name__} comes with {count_params(self.caption_encoder) * 1.e-6:.2f} M params.")

    def freeze(self):
        self.caption_encoder = self.caption_encoder.eval()
        for param in self.caption_encoder.parameters():
            param.requires_grad = False

    def to(self,device):
        self.t5_transformer.to(device)
        self.caption_encoder.to(device)
        self.device = device

    def encode(self, text):
        ori_caption = text['ori_caption']
        struct_caption = text['struct_caption']
        # print(ori_caption,struct_caption)
        clap_batch_encoding = self.clap_tokenizer(ori_caption, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        ori_tokens = clap_batch_encoding["input_ids"].to(self.device)
        t5_batch_encoding = self.t5_tokenizer(struct_caption, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        struct_tokens = t5_batch_encoding["input_ids"].to(self.device)
        outputs = self.caption_encoder.base(input_ids=ori_tokens)
        z = self.caption_encoder.projection(outputs.last_hidden_state)
        z2 = self.t5_transformer(input_ids=struct_tokens).last_hidden_state
        return torch.concat([z,z2],dim=1)


class FrozenTextVocalEmbedder(AbstractEncoder):
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):
        super().__init__()

        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version).to(device)
        self.device = device
        self.max_length = max_length  # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, cond, device=None, return_cuda=True):
        text = cond['caption']
        acoustic = cond['acoustic']
        batch_encoding = self.tokenizer(text=text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device if device is None else device)
        outputs = self.transformer(input_ids=tokens)
        if return_cuda:
            z = outputs.last_hidden_state.detach()
        else:
            z = outputs.last_hidden_state.cpu().detach()

        ret = {
            'caption': z,
            'acoustic': acoustic,
            'name': cond['name']
        }

        return ret

    def encode(self, text):
        return self(text)


class FrozenTextVocalMusicalEmbedder(AbstractEncoder):
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):
        super().__init__()

        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version).to(device)
        self.device = device
        self.max_length = max_length  # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, cond, device=None, return_cuda=True):
        # caption, prompt = cond['caption']
        caption = prompt = []
        for line in cond['caption']:
            caption.append(line.split('<csep>')[0])
            prompt.append(line.split('<csep>')[1])

        acoustic = cond['acoustic']
        batch_encoding = self.tokenizer(text=caption, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device if device is None else device)
        outputs = self.transformer(input_ids=tokens)
        if return_cuda:
            z = outputs.last_hidden_state.detach()
        else:
            z = outputs.last_hidden_state.cpu().detach()

        batch_encoding = self.tokenizer(text=prompt, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device if device is None else device)
        outputs = self.transformer(input_ids=tokens)
        if return_cuda:
            prompt = outputs.last_hidden_state.detach()
        else:
            prompt = outputs.last_hidden_state.cpu().detach()

        z = torch.concat([z, prompt], dim=1)    # [B, T, C]

        ret = {
            'caption': z,
            'acoustic': acoustic,
            'name': cond['name']
        }




        return ret

    def encode(self, text):
        return self(text)

