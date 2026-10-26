import copy
import itertools
import math
from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .mmencoder_withlora import MMTransformer_withlora
from .mmencoder_with_duallora import MMTransformer_with_duallora
from .mmencoder_with_duallora_ada_A import MMTransformer_with_duallora_A
from .mmencoder_with_dual_lora_gate_a import MMTransformer_with_duallora_gate
from .mmencoder_with_dynamic_weight_dual_lora import MMTransformer_with_duallora_wight
from .mmencoder_with_ada_dual_lora import MMTransformer_with_adaptive_dual_lora
from .mmencoder_with_dual_lora_gate_a import text_Transformer_with_duallora_gate
# from .uncentainy_vib import MultiModalPoEFusion

class VisionTokenizer(nn.Module):
    def __init__(self, conv1, class_embedding, positional_embedding,ln_pre):
        super(VisionTokenizer, self).__init__()
        self.conv1 = conv1
        self.class_embedding = class_embedding
        self.positional_embedding = positional_embedding
        self.ln_pre = ln_pre

    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        return x
    

class MultimodalVisionEncoder(nn.Module):
    def __init__(self, transformer, ln_post, proj):
        super(MultimodalVisionEncoder, self).__init__()
        self.transformer = transformer
        self.ln_post = ln_post
        self.proj = proj

    def forward(self, x,modality='RGB'):
        # x = x.type(self.transformer.resblocks[0]._modules[0].attn.lora_A_in[0].weight.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x,modality)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = x @ self.proj
        return x
    

class CLIPTextEncoder(nn.Module):
    def __init__(self, token_embedding, positional_embedding,transformer,ln_final,text_projection):
        super(CLIPTextEncoder, self).__init__()
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.transformer = transformer
        self.ln_final = ln_final
        self.text_projection = text_projection

    def forward(self, text, dtype, modality='TEXT'):
        x = self.token_embedding(text).type(dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x, modality)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x @ self.text_projection
        eot_embed = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        embeds = x
        #x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return eot_embed,embeds

class UNIReid_lora(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.test_setting = args.test_setting

        self._set_task()

        base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.width = base_cfg['vision_width']  # 768
        self.image_resolution = base_cfg['image_resolution']
        self.encoder_layers = base_cfg['vision_layers']
        # self.heads = self.embed_dim //64 fixed this bug
        self.heads = self.width//64

        self.lora_r1= args.lora_r1
        self.lora_r2 = args.lora_r2

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / args.temperature))  # 0.07
        # self.logit_scale = torch.ones([]) * np.log(1 / args.temperature)  # 0.07

        self.mmt_depth = args.mmt_depth

        self.clip_text_encoder = self.build_clip_pretrained_text_encoder(base_model,args)
        self.rgb_tokenizer = self.build_vision_tokenizer(base_model)
        self.nir_tokenizer = self.build_vision_tokenizer(base_model)
        self.cp_tokenizer = self.build_vision_tokenizer(base_model)
        self.sk_tokenizer = self.build_vision_tokenizer(base_model)
        # self.vision_tokenizer = self.build_vision_tokenizer(base_model)
        self.vision_encoder = self.build_vision_encoder(base_model,args)

        for name, p in self.clip_text_encoder.named_parameters():
            if "lora" in name.lower():
                p.requires_grad = True
            else:
                p.requires_grad = False
        # for p in self.clip_text_encoder.parameters():
        #     p.requires_grad = False
        for p in self.rgb_tokenizer.parameters():
            p.requires_grad = True
        for p in self.nir_tokenizer.parameters():
            p.requires_grad = True
        for p in self.cp_tokenizer.parameters():
            p.requires_grad = True
        for p in self.sk_tokenizer.parameters():
            p.requires_grad = True
        # for p in self.vision_encoder.parameters():
        #     p.requires_grad = True
        if 'mm' in args.loss_names:
            # self.create_mm_fusion_module()
            self.mm_two_adapter = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.embed_dim // 4, self.embed_dim // 16),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.embed_dim // 16, 1)
                ) for _ in range(4)
            ])  
            self.mm_three_adapter = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.embed_dim // 4, self.embed_dim // 16),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.embed_dim // 16, 1)
                ) for _ in range(4)
            ])
            self.mm_four_adapter = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.embed_dim // 4, self.embed_dim // 16),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.embed_dim // 16, 1)
                ) for _ in range(4)
            ])          
            self._init_mm_adapter(self.mm_two_adapter)
            self._init_mm_adapter(self.mm_three_adapter)
            self._init_mm_adapter(self.mm_four_adapter)
            # self.fusion_module = MultiModalPoEFusion(in_dim=self.embed_dim, latent_dim=self.embed_dim, hidden_dim=self.embed_dim, dropout=0.1)

        if 'id' in args.loss_names:
            self.create_id_classifier()

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _init_mm_adapter(self, module_list):
        for adapter in module_list:
            for m in adapter.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

            # 最后一层单独缩小一点，更稳
            last_linear = adapter[-1]
            nn.init.normal_(last_linear.weight, std=1e-3)
            if last_linear.bias is not None:
                nn.init.constant_(last_linear.bias, 0.0)

    def create_id_classifier(self):
        print('num_classes:{}'.format(self.num_classes))
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        nn.init.normal_(self.classifier.weight.data, std=0.001)
        nn.init.constant_(self.classifier.bias.data, val=0.0)

    def create_mm_fusion_module(self):
        self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                self.embed_dim // 64,
                                                batch_first=True)
        self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                   layers=self.mmt_depth,
                                                   heads=self.embed_dim //
                                                         64)
        scale = self.cross_modal_transformer.width ** -0.5

        self.ln_pre = LayerNorm(self.embed_dim)
        self.ln_post = LayerNorm(self.embed_dim)

        proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # init cross attn
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

        self.mm_head = nn.Sequential(
            OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                         ('gelu', QuickGELU()),
                         ('ln', LayerNorm(self.embed_dim)),
                         ('fc', nn.Linear(self.embed_dim, self.embed_dim))]))
        # init mlm head
        nn.init.normal_(self.mm_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mm_head.fc.weight, std=proj_std)

    def mm_fusion(self, q, k, v):
        x = self.cross_attn(self.ln_pre(q),self.ln_pre(k),self.ln_pre(v),need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = self.mm_head(x)
        x =  torch.mean(x,dim=1).float()
        return x

    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x    

    @property
    def dtype(self):
        return self.nir_tokenizer.conv1.weight.dtype

    def build_vision_tokenizer(self,base_model):
        conv1 = copy.deepcopy(base_model.visual.conv1)
        cls = copy.deepcopy(base_model.visual.class_embedding)
        pe = copy.deepcopy(base_model.visual.positional_embedding)
        ln_pre = copy.deepcopy(base_model.visual.ln_pre)
        return VisionTokenizer(conv1, cls, pe, ln_pre)

    def build_vision_encoder(self,base_model,args):
        if args.add_lora:
            transformer = MMTransformer_with_duallora_gate(width=self.width,layers=self.encoder_layers,heads=self.heads, lora_r1=self.lora_r1, lora_r2=self.lora_r2, num_loras=args.num_loras,lora_layers=args.lora_layers,lora_mode=args.lora_mode)
            stat = copy.deepcopy(base_model.visual.transformer).state_dict()
            transformer.load_state_dict(stat,strict=False)
            print('Pretrained Multimodal Encoder with LoRAs Loaded, with LoRA_r1={} Lora_r2={}, LoRA_layers={}'.format(args.lora_r1,args.lora_r2,args.lora_layers))
            # msg = transformer.load_state_dict(stat, strict=False)
            # print("missing keys:", msg.missing_keys)
            # print("unexpected keys:", msg.unexpected_keys)
        else:
            transformer = copy.deepcopy(base_model.visual.transformer)
        ln_post = copy.deepcopy(base_model.visual.ln_post)
        proj = copy.deepcopy(base_model.visual.proj)
        encoder = MultimodalVisionEncoder(transformer,ln_post,proj)
        return encoder

    def build_clip_pretrained_text_encoder(self,base_model,args):
        # transformer = copy.deepcopy(base_model.transformer)
        if args.text_add_lora:
            transformer = text_Transformer_with_duallora_gate(width=base_model.token_embedding.weight.shape[1],layers=self.encoder_layers,heads=base_model.token_embedding.weight.shape[1]//64, lora_r1=self.lora_r1, lora_r2=self.lora_r2, num_loras=args.text_num_loras,lora_layers=args.text_lora_layers,lora_mode=args.lora_mode)
            stat = copy.deepcopy(base_model.transformer).state_dict()
            transformer.load_state_dict(stat,strict=False)
            print('Pretrained Text Encoder with LoRAs Loaded, with LoRA_r1={} Lora_r2={}, LoRA_layers={}'.format(args.lora_r1,args.lora_r2,args.text_lora_layers))
            # msg = transformer.load_state_dict(stat, strict=False)
            # print("missing keys:", msg.missing_keys)
            # print("unexpected keys:", msg.unexpected_keys)
        else:
            transformer = copy.deepcopy(base_model.transformer)
        token_embedding = copy.deepcopy(base_model.token_embedding)
        positional_embedding = copy.deepcopy(base_model.positional_embedding)
        ln_final = copy.deepcopy(base_model.ln_final)
        text_projection = copy.deepcopy(base_model.text_projection)
        return CLIPTextEncoder(token_embedding, positional_embedding,transformer,ln_final,text_projection)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_rgb_cls(self,x):
        x = self.rgb_tokenizer(x.type(self.dtype))
        # print(f'X type:{x.dtype} vit type:{self.vision_encoder.transformer.resblocks[0].attn.in_proj_weight.dtype}')
        x = self.vision_encoder(x,'RGB')
        # return x[:, 0, :].float()
        return x[:, 0, :]

    def encode_nir_cls(self,x):
        x = self.nir_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'NIR')
        # return x[:, 0, :].float()
        return x[:, 0, :]


    def encode_cp_cls(self, x):
        x = self.cp_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'CP')
        # return x[:, 0, :].float()
        return x[:, 0, :]

    def encode_sk_cls(self,x):
        x = self.sk_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'SK')
        # return x[:, 0, :].float()
        return x[:, 0, :]

    def encode_text_cls(self,x):
        # x,_ = self.clip_text_encoder(x,self.dtype,'TEXT')
        x,_ = self.clip_text_encoder(x,self.dtype)
        # x = x.float()
        return x

    def encode_rgb_embeds(self,x):
        x = self.rgb_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'RGB')
        return x

    def encode_nir_embeds(self,x):
        x = self.nir_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'NIR')
        return x

    def encode_cp_embeds(self,x):
        x = self.cp_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'CP')
        return x

    def encode_sk_embeds(self,x):
        x = self.sk_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'SK')
        return x

    def encode_text_embeds(self,x):
        eot, x = self.clip_text_encoder(x,self.dtype,'TEXT')
        return eot, x

    def _get_adapter_by_group_size(self, group_size):
        if group_size == 2:
            return self.mm_two_adapter
        elif group_size == 3:
            return self.mm_three_adapter
        elif group_size == 4:
            return self.mm_four_adapter
        else:
            raise ValueError(f"Unsupported group_size: {group_size}")

    def fuse_by_indices(self, modal_feats, idxs):
        group_size = len(idxs)
        adapter = self._get_adapter_by_group_size(group_size)

        feats = [modal_feats[i] for i in idxs]
        logits = [adapter[i](modal_feats[i]) for i in idxs]   # 每个 [B, 1]

        weight_logits = torch.cat(logits, dim=1)              # [B, K]
        weight_prob = F.softmax(weight_logits, dim=1)         # [B, K]

        fused_feat = 0.0
        for j, feat in enumerate(feats):
            fused_feat = fused_feat + feat * weight_prob[:, j:j+1]

        return fused_feat, weight_logits, weight_prob

    def get_four_modal_fusion_statistics(self, batch):
        """Return four-modal weights and entropy for a fixed input sample."""
        modal_feats = [
            self.encode_nir_cls(batch['nirs']),
            self.encode_cp_cls(batch['cps']),
            self.encode_sk_cls(batch['sks']),
            self.encode_text_cls(batch['caption_ids']),
        ]
        _, _, weight_prob = self.fuse_by_indices(
            modal_feats, (0, 1, 2, 3)
        )
        prob = weight_prob.float().clamp_min(1e-12)
        prob = prob / prob.sum(dim=1, keepdim=True)
        entropy_terms = -prob * prob.log()
        entropy = entropy_terms.sum(dim=1)
        normalized_entropy = entropy / math.log(4.0)
        return {
            'weights': prob,
            'entropy_terms': entropy_terms,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
        }

    def compute_group_contribution_kl(self, rgb_feat, modal_feats, idxs, logit_scale, temperature=0.07):
        fused_full, weight_logits, _ = self.fuse_by_indices(modal_feats, idxs)
        full_loss = objectives.compute_itc_per_sample(rgb_feat, fused_full, logit_scale)  # [B]

        contrib_scores = []
        for leave_out_idx in idxs:
            remain_idxs = tuple([i for i in idxs if i != leave_out_idx])

            if len(remain_idxs) == 1:
                fused_wo = modal_feats[remain_idxs[0]]
            else:
                fused_wo, _, _ = self.fuse_by_indices(modal_feats, remain_idxs)

            loss_wo = objectives.compute_itc_per_sample(rgb_feat, fused_wo, logit_scale)  # [B]
            contrib_i = loss_wo - full_loss
            contrib_scores.append(contrib_i)

        contrib_scores = torch.stack(contrib_scores, dim=1)   # [B, K]

        target_prob = F.softmax(contrib_scores.detach() / temperature, dim=1)
        pred_log_prob = F.log_softmax(weight_logits, dim=1)

        kl_per_sample = F.kl_div(pred_log_prob, target_prob, reduction='none').sum(dim=1)  # [B]
        kl_loss = kl_per_sample.mean()

        return kl_loss

    def init_router_multimodal_embeds(self,rgb,nir,cp,sk,text):
        rgb_embeds = self.encode_rgb_embeds(rgb)
        nir_embeds = self.encode_nir_embeds(nir)
        cp_embeds = self.encode_cp_embeds(cp)
        sk_embeds = self.encode_sk_embeds(sk)
        text_eot,text_embeds = self.encode_text_embeds(text)
        mm_embeds = [nir_embeds,cp_embeds,sk_embeds,text_embeds]
        combined_embeds_for_one = [rgb_embeds[:,0,:].float(),nir_embeds[:,0,:].float(),cp_embeds[:,0,:].float(),sk_embeds[:,0,:].float(),text_eot.float()]

        combinations_for_two = itertools.combinations(mm_embeds, 2)
        combined_embeds_for_two = []
        for combo in combinations_for_two:
            combined = torch.cat(combo, dim=1)
            combined_embeds_for_two.append(combined)

        combinations_for_three = itertools.combinations(mm_embeds, 3)
        combined_embeds_for_three = []
        for combo in combinations_for_three:
            combined = torch.cat(combo, dim=1)
            combined_embeds_for_three.append(combined)

        combined_embeds_for_four = [torch.cat(mm_embeds,dim=1)]

        return combined_embeds_for_one,combined_embeds_for_two,combined_embeds_for_three,combined_embeds_for_four

    # def new_router_multimodal_embeds(self, nir_feat, cp_feat, sk_feat, text_feat):
    #     """
    #     输入:
    #         nir_feat, cp_feat, sk_feat, text_feat: [B, D] 或 [B, L, D]
    #     输出:
    #         combined_embeds_for_two:   所有2模态组合的融合特征
    #         combined_embeds_for_three: 所有3模态组合的融合特征
    #         combined_embeds_for_four:  4模态融合特征
    #     """

    #     # 每个模态先得到一个权重logit
    #     nir_weight = self.mm_adapter[0](nir_feat)   # [B, 1] 或 [B, L, 1]
    #     cp_weight = self.mm_adapter[1](cp_feat)
    #     sk_weight = self.mm_adapter[2](sk_feat)
    #     text_weight = self.mm_adapter[3](text_feat)

    #     # 把特征和对应权重放在一起
    #     modal_feats = [nir_feat, cp_feat, sk_feat, text_feat]
    #     modal_weights = [nir_weight, cp_weight, sk_weight, text_weight]

    #     combined_embeds_for_two = []
    #     combined_embeds_for_three = []

    #     # 2模态融合
    #     for idxs in itertools.combinations(range(4), 2):
    #         feats = [modal_feats[i] for i in idxs]
    #         weights = [modal_weights[i] for i in idxs]

    #         # 拼接权重后做 softmax
    #         weight_tensor = torch.cat(weights, dim=-1)   # [B, 2] 或 [B, L, 2]
    #         norm_weight = F.softmax(weight_tensor, dim=-1)

    #         # 加权融合
    #         fused_feat = 0
    #         for j, feat in enumerate(feats):
    #             fused_feat = fused_feat + feat * norm_weight[..., j:j+1]

    #         combined_embeds_for_two.append(fused_feat)

    #     # 3模态融合
    #     for idxs in itertools.combinations(range(4), 3):
    #         feats = [modal_feats[i] for i in idxs]
    #         weights = [modal_weights[i] for i in idxs]

    #         weight_tensor = torch.cat(weights, dim=-1)   # [B, 3] 或 [B, L, 3]
    #         norm_weight = F.softmax(weight_tensor, dim=-1)

    #         fused_feat = 0
    #         for j, feat in enumerate(feats):
    #             fused_feat = fused_feat + feat * norm_weight[..., j:j+1]

    #         combined_embeds_for_three.append(fused_feat)

    #     # 4模态融合
    #     weight_tensor_four = torch.cat(modal_weights, dim=-1)  # [B, 4] 或 [B, L, 4]
    #     norm_weight_four = F.softmax(weight_tensor_four, dim=-1)

    #     combined_embeds_for_four = 0
    #     for j, feat in enumerate(modal_feats):
    #         combined_embeds_for_four = combined_embeds_for_four + feat * norm_weight_four[..., j:j+1]

    #     return combined_embeds_for_two, combined_embeds_for_three, [combined_embeds_for_four], modal_weights

    # def new_router_multimodal_embeds(self, nir_feat, cp_feat, sk_feat, text_feat):
    #     ## 两个模态融合调用mm_two_adapter，三个模态融合调用mm_three_adapter，四个模态融合调用mm_four_adapter
    #     nir_two_weight = self.mm_two_adapter[0](nir_feat)   # [B, 1] 或 [B, L, 1]
    #     cp_two_weight = self.mm_two_adapter[1](cp_feat)
    #     sk_two_weight = self.mm_two_adapter[2](sk_feat)
    #     text_two_weight = self.mm_two_adapter[3](text_feat)

    #     nir_three_weight = self.mm_three_adapter[0](nir_feat)   # [B, 1] 或 [B, L, 1]
    #     cp_three_weight = self.mm_three_adapter[1](cp_feat)
    #     sk_three_weight = self.mm_three_adapter[2](sk_feat)
    #     text_three_weight = self.mm_three_adapter[3](text_feat)

    #     nir_four_weight = self.mm_four_adapter[0](nir_feat)   # [B, 1] 或 [B, L, 1]
    #     cp_four_weight = self.mm_four_adapter[1](cp_feat)
    #     sk_four_weight = self.mm_four_adapter[2](sk_feat)
    #     text_four_weight = self.mm_four_adapter[3](text_feat)

    #     modal_feats = [nir_feat, cp_feat, sk_feat, text_feat]

    #     # 2模态融合, 3模态融合, 4模态融合的权重分别不同，需要先进行softmax
    #     combined_embeds_for_two = []
    #     for idxs in itertools.combinations(range(4), 2):
    #         feats = [modal_feats[i] for i in idxs]
    #         weights = [nir_two_weight, cp_two_weight, sk_two_weight, text_two_weight]
    #         weight_tensor = torch.cat([weights[i] for i in idxs], dim=-1)
    #         norm_weight = F.softmax(weight_tensor, dim=-1)
    #         fused_feat = 0
    #         for j, feat in enumerate(feats):
    #             fused_feat = fused_feat + feat * norm_weight[..., j:j+1]
    #         combined_embeds_for_two.append(fused_feat)
    #     combined_embeds_for_three = []
    #     for idxs in itertools.combinations(range(4), 3):
    #         feats = [modal_feats[i] for i in idxs]
    #         weights = [nir_three_weight, cp_three_weight, sk_three_weight, text_three_weight]
    #         weight_tensor = torch.cat([weights[i] for i in idxs], dim=-1)
    #         norm_weight = F.softmax(weight_tensor, dim=-1)
    #         fused_feat = 0
    #         for j, feat in enumerate(feats):
    #             fused_feat = fused_feat + feat * norm_weight[..., j:j+1]
    #         combined_embeds_for_three.append(fused_feat)
    #     combined_embeds_for_four = []
    #     weight_tensor_four = torch.cat([nir_four_weight, cp_four_weight, sk_four_weight, text_four_weight], dim=-1)
    #     norm_weight_four = F.softmax(weight_tensor_four, dim=-1)
    #     fused_feat = 0
    #     for j, feat in enumerate(modal_feats):
    #         fused_feat = fused_feat + feat * norm_weight_four[..., j:j+1]
    #     combined_embeds_for_four.append(fused_feat)
    #     return combined_embeds_for_two, combined_embeds_for_three, combined_embeds_for_four#, modal_weights

    def new_router_multimodal_embeds(self, nir_feat, cp_feat, sk_feat, text_feat):
        modal_feats = [nir_feat, cp_feat, sk_feat, text_feat]

        combined_embeds_for_two = []
        combined_embeds_for_three = []
        combined_embeds_for_four = []

        for idxs in itertools.combinations(range(4), 2):
            fused_feat, _, _ = self.fuse_by_indices(modal_feats, idxs)
            combined_embeds_for_two.append(fused_feat)

        for idxs in itertools.combinations(range(4), 3):
            fused_feat, _, _ = self.fuse_by_indices(modal_feats, idxs)
            combined_embeds_for_three.append(fused_feat)

        fused_feat, _, _ = self.fuse_by_indices(modal_feats, (0, 1, 2, 3))
        combined_embeds_for_four.append(fused_feat)

        return combined_embeds_for_two, combined_embeds_for_three, combined_embeds_for_four
    
    def new_router_multimodal_embeds_two(self, sk_feat, text_feat):
        modal_feats = [sk_feat, text_feat]

        combined_embeds_for_two = []

        for idxs in itertools.combinations(range(2), 2):
            fused_feat, _, _ = self.fuse_by_indices(modal_feats, idxs)
            combined_embeds_for_two.append(fused_feat)

        return combined_embeds_for_two

    def forward(self, batch):
        ret = dict()
        rgbs = batch['rgbs']
        nirs = batch['nirs']
        cps = batch['cps']
        sks = batch['sks']
        texts = batch['caption_ids']
        # logit_scale = self.logit_scale
        logit_scale = self.logit_scale.exp().clamp(max=100)
        ret.update({'temperature': self.args.temperature})

        if 'mlm' in self.current_task:
            image_feats = self.encode_rgb_embeds(rgbs)
            mlm_tokens = batch['mlm_tokens']
            _, mlm_feats = self.encode_text_embeds(mlm_tokens)  # [batch_size, text_len, embed_dim]
            # mlm_labels = batch['mlm_labels']

            # mlm_feats = self.base_model.encode_text(mlm_labes)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        if 'mm_sdm' in self.current_task:
            rgb_feat = self.encode_rgb_cls(rgbs)
            nir_feat = self.encode_nir_cls(nirs)
            cp_feat = self.encode_cp_cls(cps)
            sk_feat = self.encode_sk_cls(sks)
            text_feat = self.encode_text_cls(texts)
            cone_losses, ctwo_losses, cthree_losses, cfour_losses = [], [], [], []
            all_kls = []

            out = self.fusion_module(feat_list=[nir_feat, cp_feat, sk_feat, text_feat], sample=False, mode="custom", custom_sizes=(1, 2, 3, 4))
            cone_feats = out['fused_feats_for_one']
            ctwo_feats = out['fused_feats_for_two']
            cthree_feats = out['fused_feats_for_three']
            cfour_feats = out['fused_feats_for_four']

            cone_dists = out['fused_dists_for_one']
            ctwo_dists = out['fused_dists_for_two']
            cthree_dists = out['fused_dists_for_three']
            cfour_dists = out['fused_dists_for_four']

            for cone_feat in cone_feats:
                cone_losses.append(objectives.compute_sdm(rgb_feat, cone_feat, batch['pids'], logit_scale))

            for ctwo_feat in ctwo_feats:
                ctwo_losses.append(objectives.compute_sdm(rgb_feat, ctwo_feat, batch['pids'], logit_scale))

            for cthree_feat in cthree_feats:
                cthree_losses.append(objectives.compute_sdm(rgb_feat, cthree_feat, batch['pids'], logit_scale))

            for cfour_feat in cfour_feats:
                cfour_losses.append(objectives.compute_sdm(rgb_feat, cfour_feat, batch['pids'], logit_scale))

            # for dist in cone_dists:
            #     all_kls.append(objectives.compute_kl_to_standard_normal(
            #         dist["fused_mu"].float(), dist["fused_logvar"].float()
            #     ))

            # for dist in ctwo_dists:
            #     all_kls.append(objectives.compute_kl_to_standard_normal(
            #         dist["fused_mu"].float(), dist["fused_logvar"].float()
            #     ))

            # for dist in cthree_dists:
            #     all_kls.append(objectives.compute_kl_to_standard_normal(
            #         dist["fused_mu"].float(), dist["fused_logvar"].float()
            #     ))

            # for dist in cfour_dists:
            #     all_kls.append(objectives.compute_kl_to_standard_normal(
            #         dist["fused_mu"].float(), dist["fused_logvar"].float()
            #     ))

            # vib_kl_loss = torch.mean(torch.stack(all_kls))
            # ret.update({'vib_kl_loss': self.args.vib_beta * vib_kl_loss})
            
            cone_aver_loss = torch.mean(torch.stack(cone_losses))
            ctwo_aver_loss = torch.mean(torch.stack(ctwo_losses))
            cthree_aver_loss = torch.mean(torch.stack(cthree_losses))
            cfour_aver_loss = torch.mean(torch.stack(cfour_losses))

            ret.update({'cone_mmsdm_loss': cone_aver_loss})
            ret.update({'ctwo_mmsdm_loss': ctwo_aver_loss})
            ret.update({'cthree_mmsdm_loss': cthree_aver_loss})
            ret.update({'cfour_mmsdm_loss': cfour_aver_loss})

            # rgb_feat, modal_embeds = self.new_router_multimodal_embeds(
            #     rgbs, nirs, cps, sks, texts
            # )

            # import itertools

            # cone_loss = 0.0
            # ctwo_loss = 0.0
            # cthree_loss = 0.0
            # cfour_loss = 0.0

            # cone_count = 0
            # ctwo_count = 0
            # cthree_count = 0
            # cfour_count = 0

            # # -------------------------------------------------
            # # 1️⃣ 单模态
            # # -------------------------------------------------
            # for embed in modal_embeds:

            #     feat = embed[:, 0, :].float()

            #     cone_loss += objectives.compute_sdm(
            #         rgb_feat, feat, batch['pids'], logit_scale
            #     )

            #     del feat
            #     torch.cuda.empty_cache()

            #     cone_count += 1

            # # -------------------------------------------------
            # # 2️⃣ 两模态组合
            # # -------------------------------------------------
            # for combo in itertools.combinations(modal_embeds, 2):

            #     combined = torch.cat(combo, dim=1)

            #     feat = self.mm_fusion(combined, combined, combined)

            #     ctwo_loss += objectives.compute_sdm(
            #         rgb_feat, feat, batch['pids'], logit_scale
            #     )

            #     del combined, feat
            #     torch.cuda.empty_cache()

            #     ctwo_count += 1

            # # -------------------------------------------------
            # # 3️⃣ 三模态
            # # -------------------------------------------------
            # for combo in itertools.combinations(modal_embeds, 3):

            #     combined = torch.cat(combo, dim=1)

            #     feat = self.mm_fusion(combined, combined, combined)

            #     cthree_loss += objectives.compute_sdm(
            #         rgb_feat, feat, batch['pids'], logit_scale
            #     )

            #     del combined, feat
            #     torch.cuda.empty_cache()

            #     cthree_count += 1

            # # -------------------------------------------------
            # # 4️⃣ 四模态
            # # -------------------------------------------------
            # combined = torch.cat(modal_embeds, dim=1)

            # feat = self.mm_fusion(combined, combined, combined)

            # cfour_loss += objectives.compute_sdm(
            #     rgb_feat, feat, batch['pids'], logit_scale
            # )

            # del combined, feat
            # torch.cuda.empty_cache()

            # cfour_count += 1

            # # -------------------------------------------------
            # # 5️⃣ 平均
            # # -------------------------------------------------
            # ret.update({'cone_mmsdm_loss': cone_loss / cone_count})
            # ret.update({'ctwo_mmsdm_loss': ctwo_loss / ctwo_count})
            # ret.update({'cthree_mmsdm_loss': cthree_loss / cthree_count})
            # ret.update({'cfour_mmsdm_loss': cfour_loss / cfour_count})

            # if 'id' in self.current_task:
            #     # all_feats = [rgb_feat] + cone_feats + ctwo_feats + cthree_feats + cfour_feats
            #     cone_all_feats = [rgb_feat] + cone_feats
            #     # assert len(all_feats) == 5 # 16
            #     cone_logits_list = []
            #     ctwo_logits_list = []
            #     cthree_logits_list = []
            #     cfour_logits_list = []
            #     for feat in cone_all_feats:
            #         logits = self.classifier(feat.half()).float()
            #         cone_logits_list.append(logits)
            #     for feat in ctwo_feats:
            #         logits = self.classifier(feat.half()).float()
            #         ctwo_logits_list.append(logits)
            #     for feat in cthree_feats:
            #         logits = self.classifier(feat.half()).float()
            #         cthree_logits_list.append(logits)
            #     for feat in cfour_feats:
            #         logits = self.classifier(feat.half()).float()
            #         cfour_logits_list.append(logits)
            #     ret.update({'cone_id_loss': objectives.compute_id(cone_logits_list,batch['pids']) * self.args.id_loss_weight})
            #     ret.update({'ctwo_id_loss': objectives.compute_id(ctwo_logits_list,batch['pids']) * self.args.id_loss_weight})
            #     ret.update({'cthree_id_loss': objectives.compute_id(cthree_logits_list,batch['pids']) * self.args.id_loss_weight})
            #     ret.update({'cfour_id_loss': objectives.compute_id(cfour_logits_list,batch['pids']) * self.args.id_loss_weight})
            #     return ret
            if 'id' in self.current_task:
                all_feats = [rgb_feat] + cone_feats + ctwo_feats + cthree_feats + cfour_feats
                assert len(all_feats) == 16
                logits_list = []
                for feat in all_feats:
                    logits = self.classifier(feat.half()).float()
                    logits_list.append(logits)
                ret.update({'id_loss': objectives.compute_id(logits_list,batch['pids']) * self.args.id_loss_weight})
            return ret

        if 'mm_itc' in self.current_task:
            # cone_embeds, ctwo_embeds, cthree_embeds, cfour_embeds = self.init_router_multimodal_embeds(rgbs,nirs,cps,sks,texts)

            # cone_feats = cone_embeds
            # ctwo_feats,cthree_feats,cfour_feats = [],[],[]

            # for i in range(len(ctwo_embeds)):
            #     ctwo_embed = ctwo_embeds[i]
            #     ctwo_feat = self.mm_fusion(ctwo_embed,ctwo_embed,ctwo_embed)
            #     ctwo_feats.append(ctwo_feat)

            # for i in range(len(cthree_embeds)):
            #     cthree_embed = cthree_embeds[i]
            #     cthree_feat = self.mm_fusion(cthree_embed, cthree_embed, cthree_embed)
            #     cthree_feats.append(cthree_feat)

            # for i in range(len(cfour_embeds)):
            #     cfour_embed = cfour_embeds[i]
            #     cfour_feat = self.mm_fusion(cfour_embed, cfour_embed, cfour_embed)
            #     cfour_feats.append(cfour_feat)

            # rgb_feat = cone_feats[0]
            # cone_feats = cone_feats[1:]

            rgb_feat = self.encode_rgb_cls(rgbs)
            nir_feat = self.encode_nir_cls(nirs)
            cp_feat = self.encode_cp_cls(cps)
            sk_feat = self.encode_sk_cls(sks)
            text_feat = self.encode_text_cls(texts)
            cone_losses, ctwo_losses, cthree_losses, cfour_losses = [], [], [], []
            cone_losses, ctwo_losses = [], []
            all_kls = []

            # 每个模态融合之前，经过adapter映射到同一维度
            # nir_feat_mm = self.mm_adapter[0](nir_feat)
            # cp_feat_mm = self.mm_adapter[1](cp_feat)
            # sk_feat_mm = self.mm_adapter[2](sk_feat)
            # text_feat_mm = self.mm_adapter[3](text_feat)

            # out = self.fusion_module(feat_list=[nir_feat_mm, cp_feat_mm, sk_feat_mm, text_feat_mm], sample=False, mode="custom", custom_sizes=(2, 3, 4))
            # out = self.fusion_module(feat_list=[nir_feat, cp_feat, sk_feat, text_feat], sample=False, mode="custom", custom_sizes=(2, 3, 4))
            # cone_feats = out['fused_feats_for_one']
            # ctwo_feats = out['fused_feats_for_two']
            # cthree_feats = out['fused_feats_for_three']
            # cfour_feats = out['fused_feats_for_four']

            # cone_dists = out['fused_dists_for_one']
            # ctwo_dists = out['fused_dists_for_two']
            # cthree_dists = out['fused_dists_for_three']
            # cfour_dists = out['fused_dists_for_four']
 
            # ctwo_feats, cthree_feats, cfour_feats, fuse_weights = self.new_router_multimodal_embeds(nir_feat, cp_feat, sk_feat, text_feat)
            ctwo_feats, cthree_feats, cfour_feats = self.new_router_multimodal_embeds(nir_feat, cp_feat, sk_feat, text_feat)
            ctwo_feats = self.new_router_multimodal_embeds_two(sk_feat, text_feat)

            # cone_losses,ctwo_losses,cthree_losses,cfour_losses = [],[],[],[]

            # for cone_feat in cone_feats:
            #     cone_losses.append(objectives.compute_itc(rgb_feat, cone_feat, logit_scale))

            for ctwo_feat in ctwo_feats:
                ctwo_losses.append(objectives.compute_itc(rgb_feat, ctwo_feat, logit_scale))

            for cthree_feat in cthree_feats:
                cthree_losses.append(objectives.compute_itc(rgb_feat, cthree_feat, logit_scale))

            for cfour_feat in cfour_feats:
                cfour_losses.append(objectives.compute_itc(rgb_feat, cfour_feat, logit_scale))

            # for dist in cone_dists:
            #     all_kls.append(objectives.compute_kl_to_standard_normal(
            #         dist["fused_mu"].float(), dist["fused_logvar"].float()
            #     ))

            # for dist in ctwo_dists:
            #     all_kls.append(objectives.compute_kl_to_standard_normal(
            #         dist["fused_mu"].float(), dist["fused_logvar"].float()
            #     ))

            # for dist in cthree_dists:
            #     all_kls.append(objectives.compute_kl_to_standard_normal(
            #         dist["fused_mu"].float(), dist["fused_logvar"].float()
            #     ))

            # for dist in cfour_dists:
            #     all_kls.append(objectives.compute_kl_to_standard_normal(
            #         dist["fused_mu"].float(), dist["fused_logvar"].float()
            #     ))

            # vib_kl_loss = torch.mean(torch.stack(all_kls))
            # ret.update({'vib_kl_loss': self.args.vib_beta * vib_kl_loss})

            # cone_aver_loss = torch.mean(torch.stack(cone_losses))
            ctwo_aver_loss = torch.mean(torch.stack(ctwo_losses))
            cthree_aver_loss = torch.mean(torch.stack(cthree_losses))
            cfour_aver_loss = torch.mean(torch.stack(cfour_losses))

            # ret.update({'cone_mmitc_loss': cone_aver_loss})
            ret.update({'ctwo_mmitc_loss': ctwo_aver_loss * self.args.mm_itc_loss_weight})
            ret.update({'cthree_mmitc_loss': cthree_aver_loss * self.args.mm_itc_loss_weight})
            ret.update({'cfour_mmitc_loss': cfour_aver_loss * self.args.mm_itc_loss_weight})

            # nir_feat, cp_feat, sk_feat, text_feat = cone_feats[0], cone_feats[1], cone_feats[2], cone_feats[3]

            ret.update({'nir_itc_loss': objectives.compute_itc(rgb_feat, nir_feat, logit_scale) * self.args.single_itc_loss_weight})
            ret.update({'cp_itc_loss': objectives.compute_itc(rgb_feat, cp_feat, logit_scale) * self.args.single_itc_loss_weight})
            ret.update({'sk_itc_loss': objectives.compute_itc(rgb_feat, sk_feat, logit_scale) * self.args.single_itc_loss_weight})
            ret.update({'txt_itc_loss': objectives.compute_itc(rgb_feat, text_feat, logit_scale) * self.args.single_itc_loss_weight})

            # if 'mm_kl' in self.current_task and self.args.use_weight:
            #     cone_feats = [nir_feat, cp_feat, sk_feat, text_feat]
            #     similarity_scores = []

            #     for feat in cone_feats:
            #         sim_score = F.cosine_similarity(rgb_feat, feat, dim=-1)  # [B]
            #         similarity_scores.append(sim_score)

            #     similarity_scores = torch.stack(similarity_scores, dim=1)    # [B, 4]

            #     # fuse_weights: list of 4 tensors, each [B, 1]
            #     fuse_weights = torch.cat(fuse_weights, dim=1)                # [B, 4]
            #     ret.update({'adapter_kl_loss': objectives.compute_adapter_kl_loss(fuse_weights, similarity_scores) })#* self.args.adapter_kl_loss_weight})

            if 'mm_kl' in self.current_task and self.args.use_weight:
                modal_feats = [nir_feat, cp_feat, sk_feat, text_feat]
                # modal_feats = [sk_feat, text_feat]

                two_kl_losses = []
                three_kl_losses = []

                for idxs in itertools.combinations(range(4), 2):
                    two_kl_losses.append(
                        self.compute_group_contribution_kl(
                            rgb_feat, modal_feats, idxs, logit_scale,
                            temperature=self.args.adapter_temperature
                        )
                    )

                for idxs in itertools.combinations(range(4), 3):
                    three_kl_losses.append(
                        self.compute_group_contribution_kl(
                            rgb_feat, modal_feats, idxs, logit_scale,
                            temperature=self.args.adapter_temperature
                        )
                    )

                four_kl_loss = self.compute_group_contribution_kl(
                    rgb_feat, modal_feats, (0, 1, 2, 3), logit_scale,
                    temperature=self.args.adapter_temperature
                )

                ret.update({
                    'ctwo_adapter_kl_loss': torch.mean(torch.stack(two_kl_losses)) * self.args.adapter_kl_loss_weight,
                    'cthree_adapter_kl_loss': torch.mean(torch.stack(three_kl_losses)) * self.args.adapter_kl_loss_weight,
                    'cfour_adapter_kl_loss': four_kl_loss * self.args.adapter_kl_loss_weight,
                })

            if 'id' in self.current_task:
                all_feats = [rgb_feat, nir_feat, cp_feat, sk_feat, text_feat] + ctwo_feats + cthree_feats + cfour_feats
                assert len(all_feats) == 16
                logits_list = []
                for feat in all_feats:
                    logits = self.classifier(feat.half()).float()
                    logits_list.append(logits)
                ret.update({'id_loss': objectives.compute_id(logits_list,batch['pids']) * self.args.id_loss_weight})

            return ret

        if 'mm_supitc' in self.current_task:
            cone_embeds, ctwo_embeds, cthree_embeds, cfour_embeds = self.router_multimodal_embeds(rgbs,nirs,cps,sks,texts)

            cone_feats = cone_embeds
            ctwo_feats,cthree_feats,cfour_feats = [],[],[]

            for i in range(len(ctwo_embeds)):
                ctwo_embed = ctwo_embeds[i]
                ctwo_feat = self.mm_fusion(ctwo_embed,ctwo_embed,ctwo_embed)
                ctwo_feats.append(ctwo_feat)

            for i in range(len(cthree_embeds)):
                cthree_embed = cthree_embeds[i]
                cthree_feat = self.mm_fusion(cthree_embed, cthree_embed, cthree_embed)
                cthree_feats.append(cthree_feat)

            for i in range(len(cfour_embeds)):
                cfour_embed = cfour_embeds[i]
                cfour_feat = self.mm_fusion(cfour_embed, cfour_embed, cfour_embed)
                cfour_feats.append(cfour_feat)

            rgb_feat = cone_feats[0]
            cone_feats = cone_feats[1:]

            cone_losses,ctwo_losses,cthree_losses,cfour_losses = [],[],[],[]

            for cone_feat in cone_feats:
                cone_losses.append(objectives.compute_supitc(rgb_feat, cone_feat, batch['pids']))

            for ctwo_feat in ctwo_feats:
                ctwo_losses.append(objectives.compute_supitc(rgb_feat, ctwo_feat, batch['pids']))

            for cthree_feat in cthree_feats:
                cthree_losses.append(objectives.compute_supitc(rgb_feat, cthree_feat, batch['pids']))

            for cfour_feat in cfour_feats:
                cfour_losses.append(objectives.compute_supitc(rgb_feat, cfour_feat, batch['pids']))

            cone_aver_loss = torch.mean(torch.stack(cone_losses))
            ctwo_aver_loss = torch.mean(torch.stack(ctwo_losses))
            cthree_aver_loss = torch.mean(torch.stack(cthree_losses))
            cfour_aver_loss = torch.mean(torch.stack(cfour_losses))

            ret.update({'cone_mmsupitc_loss': cone_aver_loss})
            ret.update({'ctwo_mmsupitc_loss': ctwo_aver_loss})
            ret.update({'cthree_mmsupitc_loss': cthree_aver_loss})
            ret.update({'cfour_mmsupitc_loss': cfour_aver_loss})
            return ret

        if 'mm_cmpm' in self.current_task:
            cone_embeds, ctwo_embeds, cthree_embeds, cfour_embeds = self.router_multimodal_embeds(rgbs,nirs,cps,sks,texts)

            cone_feats = cone_embeds
            ctwo_feats,cthree_feats,cfour_feats = [],[],[]

            for i in range(len(ctwo_embeds)):
                ctwo_embed = ctwo_embeds[i]
                ctwo_feat = self.mm_fusion(ctwo_embed,ctwo_embed,ctwo_embed)
                ctwo_feats.append(ctwo_feat)

            for i in range(len(cthree_embeds)):
                cthree_embed = cthree_embeds[i]
                cthree_feat = self.mm_fusion(cthree_embed, cthree_embed, cthree_embed)
                cthree_feats.append(cthree_feat)

            for i in range(len(cfour_embeds)):
                cfour_embed = cfour_embeds[i]
                cfour_feat = self.mm_fusion(cfour_embed, cfour_embed, cfour_embed)
                cfour_feats.append(cfour_feat)

            rgb_feat = cone_feats[0]
            cone_feats = cone_feats[1:]

            cone_losses,ctwo_losses,cthree_losses,cfour_losses = [],[],[],[]

            for cone_feat in cone_feats:
                cone_losses.append(objectives.compute_cmpm(rgb_feat, cone_feat, batch['pids']))

            for ctwo_feat in ctwo_feats:
                ctwo_losses.append(objectives.compute_cmpm(rgb_feat, ctwo_feat, batch['pids']))

            for cthree_feat in cthree_feats:
                cthree_losses.append(objectives.compute_cmpm(rgb_feat, cthree_feat, batch['pids']))

            for cfour_feat in cfour_feats:
                cfour_losses.append(objectives.compute_cmpm(rgb_feat, cfour_feat, batch['pids']))

            cone_aver_loss = torch.mean(torch.stack(cone_losses))
            ctwo_aver_loss = torch.mean(torch.stack(ctwo_losses))
            cthree_aver_loss = torch.mean(torch.stack(cthree_losses))
            cfour_aver_loss = torch.mean(torch.stack(cfour_losses))

            ret.update({'cone_mmcmpm_loss': cone_aver_loss})
            ret.update({'ctwo_mmcmpm_loss': ctwo_aver_loss})
            ret.update({'cthree_mmcmpm_loss': cthree_aver_loss})
            ret.update({'cfour_mmcmpm_loss': cfour_aver_loss})
            return ret

        rgb_feats = self.encode_rgb_cls(rgbs)
        nir_feats = self.encode_nir_cls(nirs)
        cp_feats = self.encode_cp_cls(cps)
        sk_feats = self.encode_sk_cls(sks)
        text_feats = self.encode_text_cls(texts)

        if 'itc' in self.current_task:
            ret.update({'nir_itc_loss': objectives.compute_itc(rgb_feats, nir_feats, logit_scale)})
            ret.update({'cp_itc_loss': objectives.compute_itc(rgb_feats, cp_feats, logit_scale)})
            ret.update({'sk_itc_loss': objectives.compute_itc(rgb_feats, sk_feats, logit_scale)})
            ret.update({'txt_itc_loss': objectives.compute_itc(rgb_feats, text_feats, logit_scale)})
            # return ret

        if 'sdm' in self.current_task:
            ret.update({'nir_sdm_loss': objectives.compute_sdm(rgb_feats, nir_feats, batch['pids'], logit_scale)})
            ret.update({'cp_sdm_loss': objectives.compute_sdm(rgb_feats, cp_feats, batch['pids'], logit_scale)})
            ret.update({'sk_sdm_loss': objectives.compute_sdm(rgb_feats, sk_feats, batch['pids'], logit_scale)})
            ret.update({'txt_sdm_loss': objectives.compute_sdm(rgb_feats, text_feats, batch['pids'], logit_scale)})
            # return ret

        if 'text_itc' in self.current_task:
            ret.update({'text_itc_loss': objectives.compute_itc(text_feats, text_feats, logit_scale)})
        return ret

    @torch.no_grad()
    def visualize_two_modal_contribution(
        self,
        data_loader,
        save_path="./two_modal_contribution.png",
        temperature=0.07,
        combos=None,
        max_batches=None,
        device=None,
        dpi=300,
    ):
        """
        Visualize modal contribution for two-modal fusion.

        Modal order:
            0: IR/NIR
            1: CP
            2: SK
            3: Text

        The figure compares:
            1. Estimated modal contribution:
            computed by ITC loss increase after removing each modality.
            2. Learned fusion weight:
            predicted by MSF two-modal adapters.
        """
        import os
        import math
        import numpy as np
        import torch
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        from model import objectives

        self.eval()

        if device is None:
            device = next(self.parameters()).device

        modal_name = {
            0: "IR",
            1: "CP",
            2: "SK",
            3: "Text",
        }

        # 默认画 Text 与三个视觉辅助模态的组合
        if combos is None:
            combos = [
                (3, 0),  # Text + IR
                (3, 2),  # Text + SK
                (3, 1),  # Text + CP
            ]

        stats = {
            tuple(idxs): {
                "contrib": [],
                "weight": [],
            }
            for idxs in combos
        }

        logit_scale = self.logit_scale.exp().clamp(max=100)

        for batch_idx, batch in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # ========= 1. parse batch =========
            if isinstance(batch, dict):
                rgbs = batch["rgbs"].to(device)
                nirs = batch["nirs"].to(device)
                cps = batch["cps"].to(device)
                sks = batch["sks"].to(device)
                texts = batch["caption_ids"].to(device)
            elif isinstance(batch, (list, tuple)):
                # 这里需要改成你 test/val dataloader 的真实顺序
                # 如果不确定，先打印 batch 每个元素的 shape
                rgbs = batch[0].to(device)
                nirs = batch[1].to(device)
                cps = batch[2].to(device)
                sks = batch[3].to(device)
                texts = batch[4].to(device)
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            # ========= 2. extract features =========
            rgb_feat = self.encode_rgb_cls(rgbs)
            nir_feat = self.encode_nir_cls(nirs)
            cp_feat = self.encode_cp_cls(cps)
            sk_feat = self.encode_sk_cls(sks)
            text_feat = self.encode_text_cls(texts)

            modal_feats = [nir_feat, cp_feat, sk_feat, text_feat]

            # ========= 3. compute contribution and fusion weight =========
            for idxs in combos:
                idxs = tuple(idxs)

                fused_full, weight_logits, weight_prob = self.fuse_by_indices(
                    modal_feats, idxs
                )  # weight_prob: [B, 2]

                full_loss = objectives.compute_itc_per_sample(
                    rgb_feat, fused_full, logit_scale
                )  # [B]

                contrib_scores = []

                for leave_out_idx in idxs:
                    remain_idxs = tuple(i for i in idxs if i != leave_out_idx)

                    # two-modal leave-one-out 后只剩一个模态
                    fused_wo = modal_feats[remain_idxs[0]]

                    loss_wo = objectives.compute_itc_per_sample(
                        rgb_feat, fused_wo, logit_scale
                    )  # [B]

                    contrib_i = loss_wo - full_loss
                    contrib_scores.append(contrib_i)

                contrib_scores = torch.stack(contrib_scores, dim=1)  # [B, 2]

                target_prob = F.softmax(
                    contrib_scores.detach() / temperature,
                    dim=1
                )  # [B, 2]

                stats[idxs]["contrib"].append(target_prob.mean(dim=0).cpu())
                stats[idxs]["weight"].append(weight_prob.mean(dim=0).cpu())

        # ========= 4. aggregate =========
        results = {}
        for idxs in combos:
            idxs = tuple(idxs)
            if len(stats[idxs]["contrib"]) == 0:
                raise RuntimeError("No batches were processed. Please check data_loader or max_batches.")

            avg_contrib = torch.stack(stats[idxs]["contrib"], dim=0).mean(dim=0).numpy()
            avg_weight = torch.stack(stats[idxs]["weight"], dim=0).mean(dim=0).numpy()

            results[idxs] = {
                "contrib": avg_contrib,
                "weight": avg_weight,
            }

        # ========= 5. plot =========
        n = len(combos)
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.0 * ncols, 3.2 * nrows),
            squeeze=False
        )

        axes_flat = axes.flatten()

        for ax, idxs in zip(axes_flat, combos):
            idxs = tuple(idxs)
            names = [modal_name[i] for i in idxs]

            contrib = results[idxs]["contrib"]
            weight = results[idxs]["weight"]

            x = np.arange(len(names))
            bar_width = 0.35

            ax.bar(
                x - bar_width / 2,
                contrib,
                bar_width,
                label="Estimated contribution"
            )
            ax.bar(
                x + bar_width / 2,
                weight,
                bar_width,
                label="Learned fusion weight"
            )

            ax.set_xticks(x)
            ax.set_xticklabels(names)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Weight")
            ax.set_title(" + ".join(names))

            for i, v in enumerate(contrib):
                ax.text(i - bar_width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
            for i, v in enumerate(weight):
                ax.text(i + bar_width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

        for ax in axes_flat[len(combos):]:
            ax.axis("off")

        axes_flat[0].legend(fontsize=8)

        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        print(f"Saved two-modal contribution visualization to: {save_path}")

        return results

def build_lora_model(args, num_classes=11003):
    model = UNIReid_lora(args, num_classes)
    # convert model to fp16
    convert_weights(model)
    return model
