import random
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utils import neg_sample

class AttrTensorDict(nn.Module):
    def __init__(self, tensor_dict):
        super().__init__()
        # 将字典中的张量注册为 Buffer 或 Parameter
        for k, v in tensor_dict.items():
            if isinstance(v, torch.Tensor):
                self.register_buffer(k, v)  # 或者用 Parameter，看是否需要梯度


class S3RecDataset(torch.utils.data.Dataset):
    """
    Build Dataset for RQ-VAE Training

    Args:
        data_dir = os.environ.get('TRAIN_DATA_PATH')
        feature_id = MM emb ID
    """

    def __init__(self, data_dir, feature_id):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.mm_emb_id = [feature_id]
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_id)

        self.mm_emb = self.mm_emb_dict[self.mm_emb_id[0]]
        self.tid_list, self.emb_list = list(self.mm_emb.keys()), list(self.mm_emb.values())
        self.emb_list = [torch.tensor(emb, dtype=torch.float32) for emb in self.emb_list]

        assert len(self.tid_list) == len(self.emb_list)
        self.item_cnt = len(self.tid_list)

    def __getitem__(self, index):
        tid = torch.tensor(self.tid_list[index], dtype=torch.long)
        emb = self.emb_list[index]
        return tid, emb

    def __len__(self):
        return self.item_cnt

    @staticmethod
    def collate_fn(batch):
        tid, emb = zip(*batch)

        tid_batch, emb_batch = torch.stack(tid, dim=0), torch.stack(emb, dim=0)
        return tid_batch, emb_batch


class PretrainDataset(Dataset):

    def __init__(self, args, user_seq, long_sequence, mask_p, mask_id, item_size, attribute_size, item2attribute):
        self.args = args
        self.mask_p = mask_p
        self.mask_id = mask_id
        self.item_size = item_size
        self.attribute_size = attribute_size
        self.item2attribute = item2attribute
        self.attr_name = self.attribute_size.keys()
        self.user_seq = user_seq
        self.long_sequence = long_sequence
        self.max_len = args.max_seq_length
        self.part_sequence = []
        self.split_sequence()

    def split_sequence(self):
        for seq in self.user_seq:
            input_ids = seq[-(self.max_len+2):-2] # keeping same as train set, last 2 for test?
            for i in range(len(input_ids)):
                self.part_sequence.append(input_ids[:i+1])

    def __len__(self):
        return len(self.part_sequence)

    def __getitem__(self, index):

        sequence = self.part_sequence[index] # pos_items
        # sample neg item for every masked item
        masked_item_sequence = []
        neg_items = []
        # Masked Item Prediction
        item_set = set(sequence)
        for item in sequence[:-1]:
            prob = random.random()
            if prob < self.mask_p:
                masked_item_sequence.append(self.mask_id)
                neg_items.append(neg_sample(item_set, self.item_size))
            else:
                masked_item_sequence.append(item)
                neg_items.append(item)

        # add mask at the last position
        masked_item_sequence.append(self.mask_id)
        neg_items.append(neg_sample(item_set, self.item_size))

        # Segment Prediction
        if len(sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            sample_length = random.randint(1, len(sequence) // 2)
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)
            pos_segment = sequence[start_id: start_id + sample_length]
            neg_segment = self.long_sequence[neg_start_id:neg_start_id + sample_length]
            masked_segment_sequence = sequence[:start_id] + [self.mask_id] * sample_length + sequence[
                                                                                      start_id + sample_length:]
            pos_segment = [self.mask_id] * start_id + pos_segment + [self.mask_id] * (
                        len(sequence) - (start_id + sample_length))
            neg_segment = [self.mask_id] * start_id + neg_segment + [self.mask_id] * (
                        len(sequence) - (start_id + sample_length))

        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)

        # padding sequence
        pad_len = self.max_len - len(sequence)
        masked_item_sequence = [0] * pad_len + masked_item_sequence
        pos_items = [0] * pad_len + sequence
        neg_items = [0] * pad_len + neg_items
        masked_segment_sequence = [0]*pad_len + masked_segment_sequence
        pos_segment = [0]*pad_len + pos_segment
        neg_segment = [0]*pad_len + neg_segment

        masked_item_sequence = masked_item_sequence[-self.max_len:]
        pos_items = pos_items[-self.max_len:]
        neg_items = neg_items[-self.max_len:]

        masked_segment_sequence = masked_segment_sequence[-self.max_len:]
        pos_segment = pos_segment[-self.max_len:]
        neg_segment = neg_segment[-self.max_len:]

        # Associated Attribute Prediction
        # Masked Attribute Prediction
        # 有很多个attr，用dict保存
        attributes = defaultdict(list)
        for attr in self.attr_name:
            for item in pos_items:
                attribute = [0] * self.attribute_size[attr]
                try:
                    now_attribute = self.item2attribute[str(item)][attr]
                    if isinstance(now_attribute, list):
                        for a in now_attribute:
                            attribute[a] = 1
                    elif isinstance(now_attribute, int):
                        attribute[now_attribute] = 1
                    else:
                        assert 0, "wrong instance"
                except:
                    pass
                attributes[attr].append(list(attribute))


        for attr in self.attr_name:
            assert len(attributes[attr]) == self.max_len, f"{attr} do not match the max_len"
        assert len(masked_item_sequence) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len
        assert len(masked_segment_sequence) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len

        for attr, value in attributes.items():
            attributes[attr] = torch.tensor(value, dtype=torch.long)


        cur_tensors = (attributes,  # 这是一个dict
                       torch.tensor(masked_item_sequence, dtype=torch.long),
                       torch.tensor(pos_items, dtype=torch.long),
                       torch.tensor(neg_items, dtype=torch.long),
                       torch.tensor(masked_segment_sequence, dtype=torch.long),
                       torch.tensor(pos_segment, dtype=torch.long),
                       torch.tensor(neg_segment, dtype=torch.long),)
        return cur_tensors
    
    def collate_fn(batch):
        attributes, masked_item_sequence, pos_items, neg_items, masked_segment_sequence, pos_segment, neg_segment = zip(*batch)
        collated_attributes = {}
        keys = attributes[0].keys()
        for k in keys:
            collated_attributes[k] = torch.stack([d[k] for d in attributes])
        
        masked_item_sequence = torch.stack(masked_item_sequence)
        pos_items = torch.stack(pos_items)
        neg_items = torch.stack(neg_items)
        masked_segment_sequence = torch.stack(masked_segment_sequence)
        pos_segment = torch.stack(pos_segment)
        neg_segment = torch.stack(neg_segment)
        return collated_attributes, masked_item_sequence, pos_items, neg_items, masked_segment_sequence, pos_segment, neg_segment

class SASRecDataset(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0] # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]


        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long), # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)
    

def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('part-*'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import argparse
    from utils import parse_user_seqs, parse_item_attr

    parser = argparse.ArgumentParser()

    # parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    # parser.add_argument('--data_name', default='Beauty', type=str)

    # model args
    parser.add_argument("--model_name", default='Pretrain', type=str)

    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    # pre train args
    parser.add_argument("--pre_epochs", type=int, default=300, help="number of pre_train epochs")
    parser.add_argument("--pre_batch_size", type=int, default=100)

    parser.add_argument("--mask_p", type=float, default=0.2, help="mask probability")
    parser.add_argument("--aap_weight", type=float, default=0.2, help="aap loss weight")
    parser.add_argument("--mip_weight", type=float, default=1.0, help="mip loss weight")
    parser.add_argument("--map_weight", type=float, default=1.0, help="map loss weight")
    parser.add_argument("--sp_weight", type=float, default=0.5, help="sp loss weight")

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    parser.add_argument("--local_test", type=str, default=False, help="if local test, use the fixed dataset path")


    args = parser.parse_args()

    data_dir = "F:\\Work\\202508_TencentAd\\TencentGR_1k\\TencentGR_1k\\"
    data_file = Path(data_dir, 'seq.jsonl')
    item2attribute_file = Path(data_dir, 'item_feat_dict.json')
    user_seq, max_item, long_sequence = parse_user_seqs(data_file)
    item2attribute, attribute_size = parse_item_attr(item2attribute_file)
    item_size = max_item + 2
    mask_id = max_item + 1
    attribute_size = {k: v + 1 for k, v in attribute_size.items()}
    
    train_dataset = PretrainDataset(args, user_seq, long_sequence, args.mask_p, mask_id, item_size, attribute_size, item2attribute)
    train_loader = DataLoader(train_dataset, batch_size=1)
    print(next(iter(train_loader)))