import os
import random
from torch.utils.data import Dataset
import torch
import numpy as np
import json
from transformers import AutoTokenizer, LlamaForCausalLM
import copy
import pickle

#from build.lib.timesformer.datasets.cv2_transform import pad_image

IGNORE_INDEX = -100
def pad_tensor(tensor, target_shape=(8,3072)):  # llama3.2-3B: 3072
    current_shape = tensor.shape
    if current_shape [0] < target_shape[0]:
        padding_rows = target_shape[0] - current_shape[0]
        padding_tensor = torch.zeros((padding_rows, target_shape[1]), dtype=tensor.dtype)
        tensor = torch.cat([tensor, padding_tensor], dim=0)
    return tensor

#
def pad_tensor_action(tensor, target_shape=(4,3072)):  # llama3.2-3B: 3072
    current_shape = tensor.shape
    if current_shape [0] < target_shape[0]:
        padding_rows = target_shape[0] - current_shape[0]
        padding_tensor = torch.zeros((padding_rows, target_shape[1]), dtype=tensor.dtype)
        tensor = torch.cat([tensor, padding_tensor], dim=0)
    return tensor

class LLM_dataset(Dataset):
    def __init__(self, feature_root, video_info_path, full_top3_path, max_frames, max_words,
                 tokenizer_name='meta-llama/Meta-Llama-3-1B'):

        self.video_info = video_info_path  # 训练用的json文件
        self.feature_root = feature_root  # 视频特征
        self.full_top3 = full_top3_path   # TLD_players
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.predict_model = LlamaForCausalLM.from_pretrained(tokenizer_name, torch_dtype=torch.bfloat16)
        self.max_frames = max_frames
        self.max_words = max_words

        # load files
        #self.video_feature = os.listdir(feature_root)  # npy files --dim 768
        self.video_info_dict = json.load(open(self.video_info, 'r'))  # "Video100228": {"source_path": "/home/xzy/xzy_nba/VG_NBA_2024/20221111-Phoenix Suns-Orlando Magic/20", "caption": "C.Payne makes 2-pt jump shot from 18 ft", "save_path": "/home/xzy/xzy_nba/VG_NBA_videos_train/Video100228", "game_id": "20221111-Phoenix Suns-Orlando Magic"}

        self.video_ID = list(self.video_info_dict.keys())


        self.tokenizer.pad_token_id = 128001
        #self.tokenizer.add_tokens(["[PLAYER]", "[TEAM]", "[COACH]", "[REFEREE]", "([TEAM])"], special_tokens=True)


    def __len__(self):
        #print(len(self.video_feature))
        return len(self.video_ID)

    def __getitem__(self, index):

        # video
        video_id = self.video_ID[index]  # get the video_id
        #print("video_id: ", video_id)

        # video_mask = np.zeros((1, self.max_frames), dtype=np.int64)
        video = np.zeros((1, self.max_frames, 196, 768), dtype=np.float32)
        # print("video: ", video.shape)  # video:  (1, 16, 196, 768)
        video_feature = torch.from_numpy(np.load(os.path.join(self.feature_root, video_id, 'out.npy')))
        T, _, _ = video_feature.shape
        # print("video_feature: ", video_feature.shape)  # torch.Size([16, 196, 768])
        if video_feature.shape[0] > self.max_frames:
            if video_feature.shape[0] >= 2 * self.max_frames:
                step = video_feature.shape[0] // self.max_frames
                indices = torch.arange(0, video_feature.shape[0], step)[:self.max_frames]
                video[0] = video_feature[indices]
            # #print("video_feature:", video_feature.shape[0])
            # video_index = video_feature.shape[0] - self.max_frames
            # #print("video_index:", video_index)
            # video[0] = video_feature[video_index:]
            # video_mask[0][:self.max_frames] = [1] * self.max_frames
            else:
                video[0] = video_feature[-self.max_frames:]
        else:
            video[0][:T, :, :] = video_feature

        # print("video: ", video.shape)  # video:  (1, 16, 196, 768)

        p_pkl = str(video_id) + '.pkl'
        p_path = os.path.join(self.full_top3, p_pkl)
        video_player_dict = pickle.load(open(p_path, 'rb'))
        # entity prompt
        with torch.no_grad():

            entity_dict = video_player_dict[video_id]
            #print("entity_dict", entity_dict)
            entity_list = entity_dict['identity_list']
            #print("entity_list: ", entity_list)
            entity_feature_list = entity_dict['identity_feature_list']
            action_list = entity_dict['action_list']
            #print("action_list: ", action_list)
            action_feature_list = entity_dict['action_feature_list']

            #print("------")

            # identity 0
            try:
                entity_one = entity_list[0]
                entity_one_feature = entity_feature_list[0]
                #print("entity_one: ", entity_one_feature.shape)
                name_one_tokens = self.tokenizer(
                    text=entity_one,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True
                ).input_ids[0]
                name_one_embeds = self.predict_model.model.embed_tokens(name_one_tokens)
                name_one_embeds = name_one_embeds[1:]
                name_one = pad_tensor(name_one_embeds)
            except:
                entity_one = 'M.Morris'
                entity_one_feature = np.random.rand(1,768).astype(np.float32)
                name_one_tokens = self.tokenizer(
                    text=entity_one,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True
                ).input_ids[0]
                name_one_embeds = self.predict_model.model.embed_tokens(name_one_tokens)
                name_one_embeds = name_one_embeds[1:]
                name_one = pad_tensor(name_one_embeds)

            # identity 1
            try:
                entity_two = entity_list[1]
                entity_two_feature = entity_feature_list[1]
                name_two_tokens = self.tokenizer(
                    text=entity_two,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True
                ).input_ids[0]
                name_two_embeds = self.predict_model.model.embed_tokens(name_two_tokens)
                name_two_embeds = name_two_embeds[1:]
                name_two = pad_tensor(name_two_embeds)
            except:
                entity_two_feature = entity_one_feature
                name_two = name_one

            # # identity 2
            # entity_three = entity_list[2]
            # entity_three_feature = entity_feature_list[2]
            # name_three_tokens = self.tokenizer(
            #     text=entity_three,
            #     return_tensors="pt",
            #     max_length=128,
            #     truncation=True
            # ).input_ids[0]
            # name_three_embeds = self.predict_model.model.embed_tokens(name_three_tokens)
            # name_three_embeds = name_three_embeds[1:]
            # name_three = pad_tensor(name_three_embeds)

            # action 0
            try:
                action_one = action_list[0]
                action_one_feature = action_feature_list[0]
                action_one_tokens = self.tokenizer(
                    text=action_one,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True
                ).input_ids[0]
                action_one_embeds = self.predict_model.model.embed_tokens(action_one_tokens)

                action_one_embeds = action_one_embeds[1:]
                action_one = pad_tensor_action(action_one_embeds)
            except:
                action_one = 'Shot'
                action_one_feature = np.random.rand(1,768).astype(np.float32)
                action_one_tokens = self.tokenizer(
                    text=action_one,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True
                ).input_ids[0]
                action_one_embeds = self.predict_model.model.embed_tokens(action_one_tokens)

                action_one_embeds = action_one_embeds[1:]
                action_one = pad_tensor_action(action_one_embeds)
            #print("action_one: ", action_one.shape)

            # action 1
            try:
                action_two = action_list[1]
                action_two_feature = action_feature_list[1]
                action_two_tokens = self.tokenizer(
                    text=action_two,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True
                ).input_ids[0]
                action_two_embeds = self.predict_model.model.embed_tokens(action_two_tokens)
                action_two_embeds = action_two_embeds[1:]
                action_two = pad_tensor_action(action_two_embeds)
                #print("action_two: ", action_two.shape)
            except:
                action_two_feature = action_one_feature
                action_two = action_one
                #print("action_two: ", action_two.shape)

            # # action 2
            # action_three = action_list[2]
            # action_three_feature = action_feature_list[2]
            # action_three_tokens = self.tokenizer(
            #     text=action_three,
            #     return_tensors="pt",
            #     max_length=128,
            #     truncation=True
            # ).input_ids[0]
            # action_three_embeds = self.predict_model.model.embed_tokens(action_three_tokens)
            # action_three_embeds = action_three_embeds[1:]
            # action_three = pad_tensor_action(action_three_embeds)

            # try:
            #     entity_two = entity_list[1]
            #     entity_two_feature = entity_dict[entity_two]
            #     name_two_tokens = self.tokenizer(
            #         text=entity_two,
            #         return_tensors="pt",
            #         max_length=128,
            #         truncation=True
            #     ).input_ids[0]
            #     name_two_embeds = self.predict_model.model.embed_tokens(name_two_tokens)
            #     name_two_embeds = name_two_embeds[1:]
            #     name_two = pad_tensor(name_two_embeds)
            #     #print("name_two: ", name_two.shape)
            #
            #     #name_two_embeds = torch.mean(name_two_embeds, dim=0)
            # except:
            #     name_two_tokens = name_one_tokens
            #     #name_two_embeds = name_one_embeds
            #     name_two = name_one
            #     #print("name_two: ", name_two.shape)
            #
            #     entity_two_feature = entity_one_feature

        # caption
        caption = self.video_info_dict[video_id]["caption"] + "<|end_of_text|>"
        caption_tokens = self.tokenizer(
            text=caption,
            return_tensors="pt",
            max_length=128,
            truncation=True
        ).input_ids[0]
        labels = copy.deepcopy(caption_tokens)

        # 补0直到长度达到了30
        while len(caption_tokens) < 30:  # 补0直到长度达到了30
            caption_tokens = torch.cat((
                caption_tokens,
                torch.tensor([self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")])))

            labels = torch.cat((
                labels,
                torch.tensor([-100])))

        attention_mask = caption_tokens.ne(self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"))

        # print("video_id: ", video_id)
        # print("video: ", video.shape)
        # print("video_mask: ", video_mask.shape)
        # print("name_one: ", name_one.shape)
        # print("entity_one_feature: ", entity_one_feature.shape)
        # print("name_two: ", name_two.shape)
        # print("entity_two_feature: ", entity_two_feature.shape)
        # print("action_one: ", action_one.shape)
        # print("action_one_feature: ", action_one_feature.shape)
        # print("action_two: ", action_two.shape)
        # print("action_two_feature: ", action_two_feature.shape)
        # print("---------")

        return (video, caption_tokens.detach(), labels.detach(), attention_mask.detach(),
                name_one.detach(), entity_one_feature,
                name_two.clone().detach(), entity_two_feature,
                action_one.detach(), action_one_feature,
                action_two.detach(), action_two_feature,
                self.video_info_dict[video_id]["caption"], video_id)


class LLM_dataset_test(Dataset):
    def __init__(self, name_file, feature_root, video_info_path, full_top3_path, max_frames, max_words,
                 tokenizer_name='meta-llama/Meta-Llama-3-1B'):

        self.name_file = name_file
        self.video_info = video_info_path  # 训练用的json文件
        self.feature_root = feature_root  # 视频特征
        self.full_top3 = full_top3_path   # TLD_players
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.predict_model = LlamaForCausalLM.from_pretrained(tokenizer_name, torch_dtype=torch.bfloat16)
        self.max_frames = max_frames
        self.max_words = max_words

        # load files
        #self.video_feature = os.listdir(feature_root)  # npy files --dim 768
        self.video_info_dict = json.load(open(self.video_info, 'r'))  # "Video100228": {"source_path": "/home/xzy/xzy_nba/VG_NBA_2024/20221111-Phoenix Suns-Orlando Magic/20", "caption": "C.Payne makes 2-pt jump shot from 18 ft", "save_path": "/home/xzy/xzy_nba/VG_NBA_videos_train/Video100228", "game_id": "20221111-Phoenix Suns-Orlando Magic"}

        self.video_ID = list(self.video_info_dict.keys())
        self.name_info = json.load(open(self.name_file, 'r'))


        self.tokenizer.pad_token_id = 128001
        #self.tokenizer.add_tokens(["[PLAYER]", "[TEAM]", "[COACH]", "[REFEREE]", "([TEAM])"], special_tokens=True)


    def __len__(self):
        #print(len(self.video_feature))
        return len(self.video_ID)

    def __getitem__(self, index):

        # video
        video_id = self.video_ID[index]  # get the video_id
        name_list = self.name_info[video_id]['gt_list']
        act_list = self.name_info[video_id]['action_list']
        #print("video_id: ", video_id)

        # video_mask = np.zeros((1, self.max_frames), dtype=np.int64)
        video = np.zeros((1, self.max_frames, 196, 768), dtype=np.float32)
        # print("video: ", video.shape)  # video:  (1, 16, 196, 768)
        video_feature = torch.from_numpy(np.load(os.path.join(self.feature_root, video_id, 'out.npy')))
        T, _, _ = video_feature.shape
        # print("video_feature: ", video_feature.shape)  # torch.Size([16, 196, 768])
        if video_feature.shape[0] > self.max_frames:
            if video_feature.shape[0] >= 2 * self.max_frames:
                step = video_feature.shape[0] // self.max_frames
                indices = torch.arange(0, video_feature.shape[0], step)[:self.max_frames]
                video[0] = video_feature[indices]
            # #print("video_feature:", video_feature.shape[0])
            # video_index = video_feature.shape[0] - self.max_frames
            # #print("video_index:", video_index)
            # video[0] = video_feature[video_index:]
            # video_mask[0][:self.max_frames] = [1] * self.max_frames
            else:
                video[0] = video_feature[-self.max_frames:]
        else:
            video[0][:T, :, :] = video_feature

        # print("video: ", video.shape)  # video:  (1, 16, 196, 768)

        p_pkl = str(video_id) + '.pkl'
        p_path = os.path.join(self.full_top3, p_pkl)
        video_player_dict = pickle.load(open(p_path, 'rb'))
        # entity prompt
        with torch.no_grad():

            entity_dict = video_player_dict[video_id]
            #print("entity_dict", entity_dict)
            entity_list = entity_dict['identity_list']
            #print("entity_list: ", entity_list)
            entity_feature_list = entity_dict['identity_feature_list']
            action_list = entity_dict['action_list']
            #print("action_list: ", action_list)
            action_feature_list = entity_dict['action_feature_list']

            #print("------")

            # identity 0
            try:
                entity_one = entity_list[0]
                entity_one_feature = entity_feature_list[0]
                #print("entity_one: ", entity_one_feature.shape)
                name_one_tokens = self.tokenizer(
                    text=entity_one,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True
                ).input_ids[0]
                name_one_embeds = self.predict_model.model.embed_tokens(name_one_tokens)
                name_one_embeds = name_one_embeds[1:]
                name_one = pad_tensor(name_one_embeds)
            except:
                entity_one = 'M.Morris'
                entity_one_feature = np.random.rand(1,768).astype(np.float32)
                name_one_tokens = self.tokenizer(
                    text=entity_one,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True
                ).input_ids[0]
                name_one_embeds = self.predict_model.model.embed_tokens(name_one_tokens)
                name_one_embeds = name_one_embeds[1:]
                name_one = pad_tensor(name_one_embeds)

            # identity 1
            try:
                entity_two = entity_list[1]
                entity_two_feature = entity_feature_list[1]
                name_two_tokens = self.tokenizer(
                    text=entity_two,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True
                ).input_ids[0]
                name_two_embeds = self.predict_model.model.embed_tokens(name_two_tokens)
                name_two_embeds = name_two_embeds[1:]
                name_two = pad_tensor(name_two_embeds)
            except:
                entity_two_feature = entity_one_feature
                name_two = name_one

            # # identity 2
            # entity_three = entity_list[2]
            # entity_three_feature = entity_feature_list[2]
            # name_three_tokens = self.tokenizer(
            #     text=entity_three,
            #     return_tensors="pt",
            #     max_length=128,
            #     truncation=True
            # ).input_ids[0]
            # name_three_embeds = self.predict_model.model.embed_tokens(name_three_tokens)
            # name_three_embeds = name_three_embeds[1:]
            # name_three = pad_tensor(name_three_embeds)

            # action 0
            try:
                action_one = action_list[0]
                action_one_feature = action_feature_list[0]
                action_one_tokens = self.tokenizer(
                    text=action_one,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True
                ).input_ids[0]
                action_one_embeds = self.predict_model.model.embed_tokens(action_one_tokens)

                action_one_embeds = action_one_embeds[1:]
                action_one = pad_tensor_action(action_one_embeds)
            except:
                action_one = 'Shot'
                action_one_feature = np.random.rand(1,768).astype(np.float32)
                action_one_tokens = self.tokenizer(
                    text=action_one,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True
                ).input_ids[0]
                action_one_embeds = self.predict_model.model.embed_tokens(action_one_tokens)

                action_one_embeds = action_one_embeds[1:]
                action_one = pad_tensor_action(action_one_embeds)
            #print("action_one: ", action_one.shape)

            # action 1
            try:
                action_two = action_list[1]
                action_two_feature = action_feature_list[1]
                action_two_tokens = self.tokenizer(
                    text=action_two,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True
                ).input_ids[0]
                action_two_embeds = self.predict_model.model.embed_tokens(action_two_tokens)
                action_two_embeds = action_two_embeds[1:]
                action_two = pad_tensor_action(action_two_embeds)
                #print("action_two: ", action_two.shape)
            except:
                action_two_feature = action_one_feature
                action_two = action_one
                #print("action_two: ", action_two.shape)

            # # action 2
            # action_three = action_list[2]
            # action_three_feature = action_feature_list[2]
            # action_three_tokens = self.tokenizer(
            #     text=action_three,
            #     return_tensors="pt",
            #     max_length=128,
            #     truncation=True
            # ).input_ids[0]
            # action_three_embeds = self.predict_model.model.embed_tokens(action_three_tokens)
            # action_three_embeds = action_three_embeds[1:]
            # action_three = pad_tensor_action(action_three_embeds)

            # try:
            #     entity_two = entity_list[1]
            #     entity_two_feature = entity_dict[entity_two]
            #     name_two_tokens = self.tokenizer(
            #         text=entity_two,
            #         return_tensors="pt",
            #         max_length=128,
            #         truncation=True
            #     ).input_ids[0]
            #     name_two_embeds = self.predict_model.model.embed_tokens(name_two_tokens)
            #     name_two_embeds = name_two_embeds[1:]
            #     name_two = pad_tensor(name_two_embeds)
            #     #print("name_two: ", name_two.shape)
            #
            #     #name_two_embeds = torch.mean(name_two_embeds, dim=0)
            # except:
            #     name_two_tokens = name_one_tokens
            #     #name_two_embeds = name_one_embeds
            #     name_two = name_one
            #     #print("name_two: ", name_two.shape)
            #
            #     entity_two_feature = entity_one_feature

        # caption
        caption = self.video_info_dict[video_id]["caption"] + "<|end_of_text|>"
        caption_tokens = self.tokenizer(
            text=caption,
            return_tensors="pt",
            max_length=128,
            truncation=True
        ).input_ids[0]
        labels = copy.deepcopy(caption_tokens)

        # 补0直到长度达到了30
        while len(caption_tokens) < 30:  # 补0直到长度达到了30
            caption_tokens = torch.cat((
                caption_tokens,
                torch.tensor([self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")])))

            labels = torch.cat((
                labels,
                torch.tensor([-100])))

        attention_mask = caption_tokens.ne(self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"))

        # print("video_id: ", video_id)
        # print("video: ", video.shape)
        # print("video_mask: ", video_mask.shape)
        # print("name_one: ", name_one.shape)
        # print("entity_one_feature: ", entity_one_feature.shape)
        # print("name_two: ", name_two.shape)
        # print("entity_two_feature: ", entity_two_feature.shape)
        # print("action_one: ", action_one.shape)
        # print("action_one_feature: ", action_one_feature.shape)
        # print("action_two: ", action_two.shape)
        # print("action_two_feature: ", action_two_feature.shape)
        # print("---------")

        return (video, video_id, name_list, act_list, caption_tokens.detach(), labels.detach(), attention_mask.detach(),
                name_one.detach(), entity_one_feature,
                name_two.clone().detach(), entity_two_feature,
                action_one.detach(), action_one_feature,
                action_two.detach(), action_two_feature,
                self.video_info_dict[video_id]["caption"])

