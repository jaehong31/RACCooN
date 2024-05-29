#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# modified from https://github.com/haotian-liu/LLaVA/blob/7ace501183c4bdec6052ec1a30039cdc3242a67c/llava/train/train.py

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from natsort import natsorted
import torch

import transformers
from torch.utils.data import Dataset
from llava.train.llava_trainer import VegasTrainer#, LLaVATrainer

from video_chatgpt import video_conversation as conversation_lib
from llava.model import *

from instruction_templete import get_instruction

from PIL import Image
import datetime
import torch.distributed as dist
dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400))
# TODO: import and use code from ../data/dataset.py

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
DEFAULT_TRANSCRIPT_START = "The noisy audio transcript of this video is:"
OBJ_START_TOKEN = "<obj_start>"
OBJ_END_TOKEN = "<obj_end>"

import io, base64, pickle, random
from tqdm import tqdm
import numpy as np

def b2f(b): return Image.open(io.BytesIO(base64.b64decode(b))).convert('RGB')
def resize(f):
    w, h = f.size
    if w>h:
        p = (w-h)//2
        f = f.crop([p, 0, p+h, h])
    elif h>w:
        p = (h-w)//2
        f = f.crop([0, p, w, p+w])
    f = f.resize([512, 512])
    return f
def img2npy(f): return (2.0*np.array(f)/255.0-1.0).transpose((2, 0, 1)).astype(np.float32)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_vid_start_end: bool = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                            metadata={"help": "Path to the training data."})
    inpainted_data_path: str = field(default=None,
                            metadata={"help": "Path to the training data."})
    inpainted_prompt_path: str = field(default='',
                            metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    sep_video_conv_front: bool = False
    video_token_len: int = 0
    video_folder: Optional[str] = field(default=None)
    inpainted_video_folder: Optional[str] = field(default=None)
    frame_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

@dataclass
class VegasArguments:
    vegas: bool = field(default=False)
    sam: bool = field(default=False)
    k_means: int = field(default=30)    
    max_aggregation_scales: int = field(default=1)
    projection_path: Optional[str] = field(default="")
    

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])        
        # lora_module_names.add(name)
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                    tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                                sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
        sources: Sequence[str],
        multimodal_cfg: dict,
) -> Dict:
    is_multimodal = multimodal_cfg['is_multimodal']
    video_token_len = multimodal_cfg['video_token_len']
    if not is_multimodal:
        return sources

    for source in sources:
        if multimodal_cfg['sep_video_conv_front']:
            assert DEFAULT_VIDEO_TOKEN in source[0]['value']
            source[0]['value'] = source[0]['value'].replace(DEFAULT_VIDEO_TOKEN, '').strip()
            source[0]['value'] = DEFAULT_VIDEO_TOKEN + conversation_lib.default_conversation.sep + \
                                    conversation_lib.default_conversation.roles[0] + ": " + source[0]['value']
        for sentence in source:
            replace_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
            if multimodal_cfg['use_vid_start_end']:
                replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, replace_token)

    return sources

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids) + len(tokenizer(conv.sep).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids)
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def uniform_samples(data_list, K):
    if K <= 0 or not data_list:
        return []  # Return an empty list if no samples needed or list is empty

    n = len(data_list)
    if K >= n:
        return data_list[:]  # Return the whole list if more samples requested than elements

    interval = n / K
    result = [data_list[int(i * interval)] for i in range(K)]
    return result

def load_mask(video_path, n_frames, mask_id, convert_to_box=False, normalize=True):
    WIDTH = 512
    HEIGHT = 320
    
    frame_files = list(sorted(os.listdir(video_path)))
    frame_files = [x for x in frame_files if not x.startswith('.')]  # Excludes files like .DS_Store
    # selected_frames = [frame_files[i] for i in indices]
    
    # list_dir=[file for file in os.listdir(full_path)]
    list_dir = natsorted(frame_files)        
    selected_frames = uniform_samples(list_dir, n_frames)        
    # frames = [Image.open(os.path.join(video_path, f)) for f in selected_samples]        
    
    frames = []
    
    for frame_name in selected_frames:
        image = Image.open(os.path.join(video_path, frame_name))
        all_mask = np.array(image)
        mask = (all_mask == int(mask_id)).astype(np.uint8) * 255
        if convert_to_box:
            # Find the bounding box of the mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            try:
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                if normalize:
                    box = '%.2f, %.2f, %.2f, %.2f'%(xmin/WIDTH, ymin/HEIGHT, xmax/WIDTH, ymax/HEIGHT)
                else:
                    box = [xmin, ymin, xmax, ymax]            
            except:
                box = "'', '', '', ''"
            frames.append(box)
        else:
            image = Image.fromarray(mask)
            image = image.resize((WIDTH, HEIGHT), resample=Image.BILINEAR)
            frames.append(image)
    
    if not convert_to_box:
        # Stack images and convert to a tensor
        frames = np.stack(frames, axis=2)
        frames = torch.from_numpy(frames).permute(2, 0, 1).contiguous().unsqueeze(1)
        frames = torch.where(frames > 0, torch.tensor(0.0), torch.tensor(1.0))
    
    return frames

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(sources, tokenizer)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                        tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                    tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

def load_video(video_path, sample_num=16, sample_type='random'):
    WIDTH = 512
    HEIGHT = 320
    
    frame_files = list(sorted(os.listdir(video_path)))
    # exclude .DS_Store
    frame_files = [x for x in frame_files if x[0]!='.']
    print(frame_files)
    vlen = len(frame_files)

    n_frms = min(sample_num, vlen)
    start, end = 0, vlen

    intervals = np.linspace(start=start, stop=end, num=n_frms + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1]))

    if sample_type == 'random':
        indices = []
        for x in ranges:
            if x[0] == x[1]:
                indices.append(x[0])
            else:
                indices.append(rnd.choice(range(x[0], x[1])))
    elif sample_type == 'uniform':
        indices = [(x[0] + x[1]) // 2 for x in ranges]
    
    selected_frames = [frame_files[i] for i in indices]
    if len(selected_frames) < sample_num:
        selected_frames += [frame_files[-1]] * (sample_num - len(selected_frames))
        indices += [indices[-1]] * (sample_num - len(indices))
    
    # [:max_num_frames]
    frames = []
    # print(len(selected_frames))
    for frame_name in selected_frames:
        image = Image.open(os.path.join(video_path, frame_name)).convert("RGB")
        image = image.resize((WIDTH, HEIGHT), resample=Image.BILINEAR)
        frames.append(image)

    frames = np.stack(frames, axis=2)
    frames = torch.from_numpy(frames).permute(2, 3, 0, 1).contiguous().unsqueeze(0)
    frames = frames.float().div(255).clamp(0, 1).half() * 2.0 - 1.0
    return frames, indices

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                    tokenizer: transformers.PreTrainedTokenizer,
                    multimodal_cfg: dict):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        # list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        # self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg
        
        # self.data_meta = '/nas-hdd/shoubin/videos/rovi/data/gpt4'
        self.data_meta = '/nas-hdd/shoubin/videos/rovi/data/gpt4_new_with_hint/'
        
        gt_files = list(os.listdir(self.data_meta))
        
        gt_files_feats = list(os.listdir(self.multimodal_cfg['video_folder']))
        _gt_files_feats = [gtf_pkl.replace('.pkl', '.json') for gtf_pkl in gt_files_feats]
        
        gt_files = list(set(_gt_files_feats) & set(gt_files))
        
        self.gt_lists = []
        self.gt_query_type = []
        self.video_file_names = []

        gt_files.remove('CcFVOzQ1Oqc.json')
        
        _gt_files_names = [gtf_pkl.replace('.json', '') for gtf_pkl in gt_files]
        # for gtf in gt_files:
        #     gt_path = os.path.join(self.data_meta, gtf)
        #     with open(gt_path, "rb") as f:
        #         gt_response = json.load(f)
                
        #         if not 'error' in gt_response.keys():
        #             # self.video_file_names.append(gtf.split(".")[0])
        #             pre_resp = gt_response['content']
        #             if pre_resp != '':
        #                 for obj_idx in range(5):
        #                     for templete in ['\n\n<OBJ_IDX>. ', '\n<OBJ_IDX>. ', '\n<OBJ_IDX>.\t', '\n\n<OBJ_IDX>) ', '\n\n**<OBJ_IDX>. ']:  
        #                         _target = templete.replace('<OBJ_IDX>', str(obj_idx+1))
        #                         if _target in pre_resp:
        #                             if obj_idx == 0:
        #                                 pre_resp = pre_resp.replace(_target, f'{_target} {OBJ_START_TOKEN}')
        #                             else:
        #                                 pre_resp = pre_resp.replace(_target, f'{OBJ_END_TOKEN}{_target}{OBJ_START_TOKEN}')
        #                             pre_resp = pre_resp.replace('<OBJ_IDX>', str(obj_idx+1))
        #                             break
                        
        #                 resp_nn_sentence_split = pre_resp.split('\n\n')                        
        #                 if len(resp_nn_sentence_split) > 2:
        #                     last_resp = resp_nn_sentence_split[-1]
        #                     if last_resp != '':
        #                         if not str(resp_nn_sentence_split[-1])[0].isnumeric():
        #                             out_resp = pre_resp.replace(f'\n\n{resp_nn_sentence_split[-1]}', OBJ_END_TOKEN)                                
        #                         else:
        #                             out_resp = pre_resp + OBJ_END_TOKEN                            
        #                     else:
        #                         out_resp = pre_resp[:-2] + OBJ_END_TOKEN
        #                 else:        
        #                     out_resp = pre_resp + OBJ_END_TOKEN
                        
        #                 if '<obj_end><obj_end>' in out_resp:
        #                     out_resp = out_resp.replace('<obj_end><obj_end>', '<obj_end>')
        #                 self.gt_lists.append(out_resp)
        #                 self.gt_query_type.append(0)
        #                 self.video_file_names.append(gtf.split(".")[0])
            
        # # self.data_meta_single_obj = '/nas-hdd/shoubin/videos/rovi/data/v2_unfilter_small_mask.json'
        # data_meta_single_obj = '/nas-hdd/shoubin/videos/rovi/data/v2_unfilter_small_mask_unique_video.json'
        # with open(data_meta_single_obj, "rb") as f:
        #     single_gt_response_dict = json.load(f)
        #     for _dct in single_gt_response_dict:
        #         if _dct["vid"] in _gt_files_names:
        #             self.video_file_names.append(_dct["vid"])
        #             self.gt_lists.append(f'{OBJ_START_TOKEN}{_dct["description"]}{OBJ_END_TOKEN}')
        #             self.gt_query_type.append(1)
        
        if self.multimodal_cfg['inpainted_prompt_path'] != '':
            with open(self.multimodal_cfg['inpainted_prompt_path'], "rb") as inpf:
                inpainted_gt_response_dict = json.load(inpf)
            
            # import pdb; pdb.set_trace()
            self.query_individual_objects = {}            
            # bounding box
            qwetmp=[]
            data_meta_single_obj_for_inpaint = '/nas-hdd/shoubin/videos/rovi/data/v4_test.json'
            with open(data_meta_single_obj_for_inpaint, "rb") as f:
                single_gt_response_dict = json.load(f)
                print(len(single_gt_response_dict))
                for _dct in tqdm(single_gt_response_dict):
                    if _dct["vid"] in _gt_files_names and _dct["task"] == 'adding':
                        temp_vid_name = os.path.join(_dct["vid"], _dct["mask_id"])
                        if os.path.exists(os.path.join(self.multimodal_cfg['inpainted_video_folder'], temp_vid_name)+'.pkl'):
                            if temp_vid_name not in inpainted_gt_response_dict.keys():
                                qwetmp.append(temp_vid_name)
                                # qwetmp=['i4CbI7HILpI/1', 'ed3291c3d6/1', '463c0f4fdd/1', '463c0f4fdd/2', '40N9GKNzkpE/1', 'mcdODLZPwiI/1', 'ee947ed771/3', 'aaff16c2db/1', '89b874547b/1', '03c95b4dae/1', '03c95b4dae/2', 'aKMEXYjIzMc/1']
                            else:
                                self.gt_lists.append(inpainted_gt_response_dict[temp_vid_name].replace('**', ''))
                                self.video_file_names.append([_dct["vid"], _dct["mask_id"]])
                                self.gt_query_type.append(2)                    
                                self.query_individual_objects[temp_vid_name] = _dct["description"].split(':')[0]
                        else:
                            # outliers_inpaint = ['0ckDwjPokW0/1', '0ckDwjPokW0/3', '76a75f4eee/2'] 
                            pass
        # Shuffle two lists with same order
        import pdb; pdb.set_trace()
        temp_zip = list(zip(self.video_file_names, self.gt_lists, self.gt_query_type))
        random.shuffle(temp_zip)
        res1, res2, res3 = zip(*temp_zip)
        # res1 and res2 come out as tuples, and so must be converted to lists.
        self.video_file_names, self.gt_lists, self.gt_query_type = list(res1), list(res2), list(res3)        
        self.query = get_instruction()
        
    def __len__(self):
        return len(self.video_file_names)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        video_file = self.video_file_names[i]
        
        video_folder = self.multimodal_cfg['video_folder']
        inpainted_video_folder = self.multimodal_cfg['inpainted_video_folder']
        if isinstance(video_file, list):
            video_path = os.path.join(inpainted_video_folder, *video_file)+'.pkl'            
        else:
            video_path = os.path.join(video_folder, f'{video_file}.pkl')
        with open(video_path, "rb") as f:
            features = pickle.load(f).cpu()
        
        gt_response = self.gt_lists[i]
        if self.gt_query_type[i] < 2:
            _query = self.query[self.gt_query_type[i]]
        else:
            video_file_combined = os.path.join(*video_file)
            _query = self.query[self.gt_query_type[i]].replace('<TARGET_OBJ>', self.query_individual_objects[video_file_combined])
            
        _sources = [[
            {'from': 'human', 
            'value': f'<video>\n{_query}'}, 
            {'from': 'gpt', 
            'value': gt_response}
        ]]
        sources = preprocess_multimodal(
            copy.deepcopy(_sources), self.multimodal_cfg)
        
        data_dict = preprocess(
            sources,
            self.tokenizer)
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                                labels=data_dict["labels"][0])
        
        data_dict["video"] = features
        # TODO for end-to-end training
        # p2p_inp, p2p_ans = img2npy(resize(b2f(b[0])).resize([256, 256])), img2npy(resize(b2f(b[1])).resize([256, 256]))
        # data_dict['p2p_inp'], data_dict['p2p_ans'] = p2p_inp, p2p_ans
        
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                    for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'video' in instances[0]:
            features = [torch.tensor(instance['video']) for instance in instances]
            if all(x is not None and x.shape == features[0].shape for x in features):
                batch['video_spatio_temporal_features'] = torch.stack(features)
            else:
                batch['video_spatio_temporal_features'] = features

        # TODO for end-to-end training
        # batch['p2p_inp'], batch['p2p_ans'] = [
        #             torch.cat([torch.from_numpy(d['p2p_inp']).unsqueeze(dim=0) for d in instances], dim=0),
        #             torch.cat([torch.from_numpy(d['p2p_ans']).unsqueeze(dim=0) for d in instances], dim=0)
        #             ]

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, video_token_len) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # dataset_cls = (LazySupervisedDataset_Video
    #                if data_args.lazy_preprocess else SupervisedDataset)
    dataset_cls = (LazySupervisedDataset
                    if data_args.lazy_preprocess else SupervisedDataset)
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                multimodal_cfg=dict(
                                    video_token_len=video_token_len,
                                    is_multimodal=data_args.is_multimodal,
                                    sep_video_conv_front=data_args.sep_video_conv_front,
                                    video_folder=data_args.video_folder,
                                    inpainted_video_folder=data_args.inpainted_video_folder,
                                    inpainted_data_path=data_args.inpainted_data_path,
                                    inpainted_prompt_path=data_args.inpainted_prompt_path,
                                    frame_aspect_ratio=data_args.frame_aspect_ratio,
                                    use_vid_start_end=getattr(data_args, 'mm_use_vid_start_end', False)))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, VegasArguments))
    model_args, data_args, training_args, vegas_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    print(model_args)
    print(data_args)
    print(training_args)
    print(vegas_args)
    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_int8_training
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    model = VideoChatGPTLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        # torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float,
    )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    
    lora_target_modules = [
        "q_proj",
        # "k_proj",
        "v_proj",
    ]
    
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        logging.warning("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if "mpt" in model_args.model_name_or_path:
            conversation_lib.default_conversation = conversation_lib.conv_templates["mpt"]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]
    
    model_vision_dict = model.get_model().initialize_vision_modules(
        pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter
    )
    vision_config = model_vision_dict['vision_config']

    data_args.is_multimodal = True

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    # if model_args.tune_mm_mlp_adapter:
    #     model.requires_grad_(False)
    #     for p in model.get_model().mm_projector.parameters():
    #         p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    # if training_args.freeze_mm_mlp_adapter:
    #     for p in model.get_model().mm_projector.parameters():
    #         p.requires_grad = False
    
    model.config.mm_use_vid_start_end = data_args.mm_use_vid_start_end = model_args.mm_use_vid_start_end
    vision_config.use_vid_start_end = training_args.use_vid_start_end = model_args.mm_use_vid_start_end
    model.config.sep_video_conv_front = data_args.sep_video_conv_front
    model.initialize_vision_tokenizer(mm_use_vid_start_end=model_args.mm_use_vid_start_end, tokenizer=tokenizer,
                                        device=training_args.device, tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
                                        pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter)
    
    os.makedirs('_log', exist_ok=True)
    # tokenizer.add_tokens(['[OBJ0]', '[OBJ1]', '[OBJ2]', '[OBJ3]', '[OBJ4]', '[OBJ5]', '[OBJ6]', '[OBJ7]'], special_tokens=True)
    tokenizer.add_tokens([OBJ_START_TOKEN, OBJ_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    print(tokenizer), json.dump(tokenizer.get_vocab(), open('_log/vocabs.json', 'w'), indent=2)

    # for n, p in model.named_parameters():
    #     if 'embed_tokens' in n or 'lm_head' in n or 'edit_head' in n or 'unet' in n: p.requires_grad = True
    #     else: p.requires_grad = False
    
    for n, p in model.named_parameters():
        if p.requires_grad == False:
            if 'embed_tokens' in n or 'lm_head' in n or 'mm_projector' in n: p.requires_grad = True
            else: p.requires_grad = False
        
    with open('_log/parameters.txt', 'w') as F:
        for n, p in model.named_parameters(): F.write('%s %s %s\n'%(n, str(p.shape), str(p.requires_grad)))

    with open('_log/args_train.txt', 'w') as F:
        for key in vars(training_args): F.write('%s: %s\n'%(str(key), str(vars(training_args)[key])))
    
    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
            else:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
            print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)
                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    # video_token_len = 592 # 588 + 30
    print(f'data_args.video_token_len: {data_args.video_token_len}')
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, video_token_len=data_args.video_token_len)
    
    # import pdb; pdb.set_trace()
    trainer = VegasTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    if training_args.lora_enable:
        NotImplementedError("lora_enable")
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                        output_dir=training_args.output_dir)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    train()
