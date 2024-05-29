from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from .multimodal_projector.builder import build_vision_projector 
from transformers import CLIPVisionConfig
import os


DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"


class VisionConfig:
    def __init__(self, 
                 frame_size=224,
                 patch_size=14,
                 hidden_size=1024,
                 ):
        # from CLIP config
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        # video_chatgpt config
        self.use_vid_start_end = None
        self.vid_start_token = None
        self.vid_end_token = None
        self.vid_patch_token = None


class VideoChatGPTConfig(LlamaConfig):
    model_type = "VideoChatGPT"


class VideoChatGPTLlamaModel(LlamaModel):
    config_class = VideoChatGPTConfig

    def __init__(self, config: LlamaConfig):
        super(VideoChatGPTLlamaModel, self).__init__(config)
        
        if hasattr(config, "mm_vision_tower"):
            clip_cfg = CLIPVisionConfig.from_pretrained(config.mm_vision_tower)
            self.vision_config = VisionConfig(
                frame_size=clip_cfg.image_size,
                patch_size = clip_cfg.patch_size,
                hidden_size=clip_cfg.hidden_size
            )

        if hasattr(config, "use_mm_proj"):
            if self.vision_config.frame_size==224: # using LLaVA-v1.1-Lightning
                self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
            else: # using LLaVA-v1.5
                self.mm_projector = build_vision_projector(config)

    def initialize_vision_modules(self, pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
        vision_config = self.vision_config
        num_patches = (vision_config.frame_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size

        if not hasattr(self, 'mm_projector'):
            if self.vision_config.frame_size==224: # using LLaVA-v1.1-Lightning
                self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)
            else: # using LLaVA-v1.5
                self.mm_projector = build_vision_projector(vision_config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword): 
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

        return dict(
            num_patches=num_patches,
            vision_config=vision_config
        )

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if (input_ids.shape[1] != 1 or self.training) and video_spatio_temporal_features is not None:

            video_features = self.mm_projector(video_spatio_temporal_features)
            dummy_video_features = torch.zeros(video_features.shape[1], 1024, device=inputs_embeds.device,
                                               dtype=inputs_embeds.dtype)
            dummy_video_features = self.mm_projector(dummy_video_features)

            new_input_embeds = []
            cur_video_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == self.vision_config.vid_patch_token).sum() == 0:
                    # Multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_video_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_video_idx += 1
                    continue
                if self.vision_config.use_vid_start_end:
                    if (cur_input_ids == self.vision_config.vid_start_token).sum() != (
                            cur_input_ids == self.vision_config.vid_end_token).sum():
                        raise ValueError("The number of video start tokens and video end tokens should be the same.")
                    video_start_tokens = torch.where(cur_input_ids == self.vision_config.vid_start_token)[0]
                    for video_start_token_pos in video_start_tokens:
                        cur_video_features = video_features[cur_video_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_video_features.shape[0]
                        
                        if cur_input_ids[video_start_token_pos + num_patches + 1] != self.vision_config.vid_end_token:
                            raise ValueError("The video end token should follow the video start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:video_start_token_pos].detach(),
                                                                cur_input_embeds[
                                                                video_start_token_pos:video_start_token_pos + 1],
                                                                cur_video_features, cur_input_embeds[
                                                                                    video_start_token_pos + num_patches
                                                                                    + 1:video_start_token_pos
                                                                                    + num_patches + 2],
                                                                cur_input_embeds[
                                                                video_start_token_pos + num_patches + 2:].detach()),
                                                                dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:video_start_token_pos + 1],
                                                                cur_video_features,
                                                                cur_input_embeds[video_start_token_pos
                                                                                + num_patches + 1:]), dim=0)
                        cur_video_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_video_features = video_features[cur_video_idx]
                    num_patches = cur_video_features.shape[0]
                    if (cur_input_ids == self.vision_config.vid_patch_token).sum() != num_patches:
                        raise ValueError(
                            "The number of video patch tokens should be the same as the number of video patches.")
                    masked_indices = torch.where(cur_input_ids == self.vision_config.vid_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start + num_patches,
                                                        device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The video patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(),
                                                            cur_video_features,
                                                            cur_input_embeds[mask_index_start + num_patches:].detach()),
                                                            dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_video_features,
                                                            cur_input_embeds[mask_index_start + num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_video_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(VideoChatGPTLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class EditMapper(nn.Module):
    def __init__(self):
        super().__init__()

        self.llm2hid = nn.Linear(4096, 512)
        self.query = nn.Parameter(torch.randn(1, 77, 512))
        self.mapper = nn.Transformer(batch_first=True, norm_first=True,
                                        d_model=512, nhead=4, num_encoder_layers=4, num_decoder_layers=4,
                                        dim_feedforward=2048, dropout=0.0)
        self.hid2feat = nn.Linear(512, 768)

    def forward(self, llm, emb):
        hid = self.llm2hid(llm+emb)
        hid = self.mapper(hid, self.query.repeat(llm.shape[0], 1, 1))
        feat = self.hid2feat(hid)

        return feat


class VideoChatGPTLlamaForCausalLM(LlamaForCausalLM):
    config_class = VideoChatGPTConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = VideoChatGPTLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # TODO for end-to-end training
        # self.edit_head = EditMapper()
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            p2p_inp=None, p2p_ans=None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            video_spatio_temporal_features=video_spatio_temporal_features
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels#.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # TODO From special tokens to UNET. thanks to ml-mgie paper
        # if labels is not None:
        #     llm = []
        #     for i in range(labels.shape[0]):
        #         try: p = labels[i].data.cpu().tolist().index(32003)-1
        #         except: p = len(labels[i])-9
        #         p = min(len(hidden_states[i])-9, p)
        #         llm.append(hidden_states[i][p:p+8].unsqueeze(0))
        #     llm = torch.cat(llm, dim=0)
        #     hid_edit = self.edit_head(llm, self.model.embed_tokens.weight[-8:].unsqueeze(dim=0).repeat(labels.shape[0], 1, 1))

        #     B, DROP = labels.shape[0], 0.05

        #     hid_null = self.edit_head(torch.zeros(B, 8, 4096, device=labels.device),
        #                                 self.model.embed_tokens.weight[-8:].unsqueeze(dim=0).repeat(labels.shape[0], 1, 1))

        #     with torch.no_grad():
        #         lat_ans, lat_inp = self.vae.encode(p2p_ans).latent_dist.sample()*self.vae.config.scaling_factor, self.vae.encode(p2p_inp).latent_dist.mode()
        #         lat_ans, lat_inp = [torch.from_numpy(lat_ans.data.cpu().float().numpy()).to(lat_ans.device),
        #                             torch.from_numpy(lat_inp.data.cpu().float().numpy()).to(lat_inp.device)]

            # noise = torch.randn_like(lat_ans)
            # ts = torch.randint(0, self.scheduler.config.num_train_timesteps, (B, ), device=noise.device).long()
            # lat_noise = self.scheduler.add_noise(lat_ans, noise, ts)

            # prob = torch.rand(B, device=lat_ans.device)
            # mask = (prob<(DROP*2)).reshape(B, 1, 1)
            # hid_edit = torch.where(mask, hid_null, hid_edit)
            # mask = (1.0-((prob>=DROP).to(lat_inp.dtype)*(prob<(DROP*3)).to(lat_inp.dtype))).reshape(B, 1, 1, 1)
            # lat_inp *= mask

            # out = self.unet(torch.cat([lat_noise, lat_inp], dim=1), ts, hid_edit).sample

            # loss_ce, loss_edit = loss, nn.functional.mse_loss(out, noise, reduction='mean')
            # if int(os.environ['LOCAL_RANK'])==0: print('loss_ce:', loss_ce, '/', 'loss_edit:', loss_edit)
            # loss = loss_ce+loss_edit*0.5

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "video_spatio_temporal_features": kwargs.get("video_spatio_temporal_features", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, mm_use_vid_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        vision_config = self.get_model().vision_config
        vision_config.use_vid_start_end = mm_use_vid_start_end
        tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_vid_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.get_model().orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. "
                        f"Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]

AutoConfig.register("VideoChatGPT", VideoChatGPTConfig)
AutoModelForCausalLM.register(VideoChatGPTConfig, VideoChatGPTLlamaForCausalLM)
