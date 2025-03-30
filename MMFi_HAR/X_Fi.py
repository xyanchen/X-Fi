import torch
from torch import nn
from einops import rearrange
import os

from backbones.RGB_benchmark.RGB_ResNet import *
from backbones.depth_benchmark.depth_ResNet18 import *
from backbones.mmwave_benchmark.mmwave_point_transformer_TD import *
from backbones.lidar_benchmark.lidar_point_transformer import *
from backbones.lidar_benchmark.pointnet_util import *

"map each modality num of features to 32 "
" Use modality fusion transforemr"

class rgb_feature_extractor(nn.Module):
    def __init__(self, rgb_model):
        super(rgb_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(rgb_model.children())[:-2])
    def forward(self, x):
        x = self.part(x).view(x.size(0), 512, -1)
        x = x.permute(0, 2, 1)
        return x

class depth_feature_extractor(nn.Module):
    def __init__(self, depth_model):
        super(depth_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(depth_model.children())[:-2])
    def forward(self, x):
        x = self.part(x).view(x.size(0), 512, -1)
        x = x.permute(0, 2, 1)
        return x

class mmwave_feature_extractor(nn.Module):
    def __init__(self, mmwave_model):
        super(mmwave_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(mmwave_model.children())[:-1])
    def forward(self, x):
        x, _ = self.part(x)
        return x

class lidar_feature_extractor(nn.Module):
    def __init__(self, lidar_model):
        super(lidar_feature_extractor, self).__init__()
        npoints, nblocks, nneighbor, n_c, d_points = 1024, 5, 16, 51, 3
        self.fc1 = lidar_model.backbone.fc1
        self.transformer1 = lidar_model.backbone.transformer1
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks - 4):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(lidar_model.backbone.transition_downs[i])
            self.transformers.append(lidar_model.backbone.transformers[i])
        self.nblocks = nblocks
    
    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks - 4):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        points = points.view(points.size(0), -1, 512)
        return points

def selective_pos_enc(xyz, npoint):
    """
    Input:
        xyz: input points position data, [B, N, 3]
    Return:
        new_xyz: sampled points position data, [B, S, 3]
        out: new features of sampled points, [B, S, C]
    """
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    return new_xyz

class feature_extrator(nn.Module):
    def __init__(self):
        super(feature_extrator, self).__init__()
        
        rgb_model = RGB_ResNet18()
        rgb_model.load_state_dict(torch.load('./backbones/RGB_benchmark/RGB_Resnet18.pt'))
        rgb_extractor = rgb_feature_extractor(rgb_model)
        rgb_extractor.eval()

        depth_model = Depth_ResNet18()
        depth_model.load_state_dict(torch.load('./backbones/depth_benchmark/depth_Resnet18.pt'))
        depth_extractor = depth_feature_extractor(depth_model)
        depth_extractor.eval()
        
        mmwave_model = mmwave_PointTransformerReg()
        mmwave_model.load_state_dict(torch.load('./backbones/mmwave_benchmark/mmwave_all_random_TD.pt'))
        mmwave_extractor = mmwave_feature_extractor(mmwave_model)
        mmwave_extractor.eval()

        lidar_model = lidar_PointTransformer_cls(root=os.getcwd())
        lidar_model.load_state_dict(torch.load('./backbones/lidar_benchmark/lidar_all_random.pt'))
        lidar_extractor = lidar_feature_extractor(lidar_model)
        lidar_extractor.eval()

        self.rgb_extractor = rgb_extractor
        self.depth_extractor = depth_extractor
        self.mmwave_extractor = mmwave_extractor
        self.lidar_extractor = lidar_extractor

    def forward(self, rgb_data, depth_data, mmwave_data, lidar_data, modality_list):
        if sum(modality_list) == 0:
            raise ValueError("At least one modality should be selected")
        else:
            real_feature_list = []
            if modality_list[0] == True:
                rgb_feature = self.rgb_extractor(rgb_data)
                "shape b x 49 x 512"
                real_feature_list.append(rgb_feature)
            if modality_list[1] == True:
                depth_feature = self.depth_extractor(depth_data)
                "shape b x 49 x 512"
                real_feature_list.append(depth_feature)
            if modality_list[2] == True:
                mmwave_feature = self.mmwave_extractor(mmwave_data)
                "shape b x 32 x 512"
                real_feature_list.append(mmwave_feature)
            if modality_list[3] == True:
                lidar_feature = self.lidar_extractor(lidar_data)
                "shape b x 32 x 512"
                real_feature_list.append(lidar_feature)
        
        return real_feature_list


class linear_projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear_projector, self).__init__()
        '''Conv 1d layer for each modality'''
        self.rgb_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(49, 32),
            nn.ReLU()
        )
        self.depth_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(49, 32),
            nn.ReLU()
        )
        self.mmwave_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.lidar_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.pos_enc_layer = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
    def forward(self, feature_list, lidar_points, modality_list):
        # example:
        # feature_list = [rgb_feature, mmwave_feature, lidar_feature]
        # modality_list = [True, False, True, True, False]
        feature_flag = 0
        for i in range(len(modality_list)):
            if modality_list[i] == True:
                if i == 0:
                    rgb_feature = feature_list[feature_flag]
                elif i == 1:
                    depth_feature = feature_list[feature_flag]
                elif i == 2:
                    mmwave_feature = feature_list[feature_flag]
                elif i == 3:
                    lidar_feature = feature_list[feature_flag]
                feature_flag += 1
            else:
                continue
        if sum (modality_list) == 0:
            raise ValueError("At least one modality should be selected")
        else:
            projected_feature_list = []
            if modality_list[0] == True:
                projected_feature_list.append(self.rgb_linear_projection(rgb_feature.permute(0, 2, 1)))
            if modality_list[1] == True:
                projected_feature_list.append(self.depth_linear_projection(depth_feature.permute(0, 2, 1)))
            if modality_list[2] == True:
                projected_feature_list.append(self.mmwave_linear_projection(mmwave_feature.permute(0, 2, 1)))
            if modality_list[3] == True:
                projected_feature_list.append(self.lidar_linear_projection(lidar_feature.permute(0, 2, 1)))
            projected_feature = torch.cat(projected_feature_list, dim=2).permute(0, 2, 1)
            "projected_feature shape: B, 32*n, 512"
            if modality_list[3] == True:
                feature_shape = projected_feature.shape
                new_xyz = selective_pos_enc(lidar_points, feature_shape[1])
                'new_xyz shape: B, 32, 3'
                pos_enc = self.pos_enc_layer(new_xyz.permute(0, 2, 1)).permute(0, 2, 1)
                'pos_enc shape: B, 32, 512'
                projected_feature += pos_enc
            else:
                pass
        return projected_feature

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size = 512, num_heads = 8, dropout = 0.0):
        super(MultiHeadAttention,self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.pool = nn.AdaptiveAvgPool2d((32,None))
    
    def forward(self, x, mask = None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        out = self.pool(out)
        return out

class qkv_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super(qkv_Attention,self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, qkv):
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward,self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class kv_projection(nn.Module):
    def __init__(self, dim, expension = 2):
        super(kv_projection,self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.MLP = nn.Sequential(
            nn.Linear(dim, dim * expension),
            nn.LayerNorm(dim * expension),
            nn.ReLU(),
            nn.Linear(dim * expension, dim)
        )
        # self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)
    
    def forward(self, x):
        x = self.MLP(x)
        # q = self.to_q(self.norm(x))
        k = self.to_k(self.norm(x))
        v = self.to_v(self.norm(x))
        kv = [k, v]
        return kv

class cross_modal_transformer(nn.Module):
    def __init__(self, num_feature=32, max_num_modality=4, dim_expansion=2, emb_size = 512, num_heads = 8, dropout=0.3):
        super(cross_modal_transformer,self).__init__()
        self.attention = MultiHeadAttention(emb_size, num_heads, dropout)
        self.ffw = FeedForward(emb_size, emb_size*dim_expansion, dropout)
        self.pool = nn.AdaptiveAvgPool2d((32,None))
    
    def forward(self, feature_embedding, modality_list):
        feature_embedding_ = self.attention(feature_embedding) + self.pool(feature_embedding)
        out_feature_embedding = self.ffw(feature_embedding_) + feature_embedding_
        return out_feature_embedding

class fusion_transformer(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads, dim_heads, dropout):
        super(fusion_transformer,self).__init__()
        self.mutihead_attention = qkv_Attention(dim, num_heads, dim_heads, dropout)
        self.feed_forward = FeedForward(dim, hidden_dim, dropout)
    
    def forward(self, feature_embedding, kv):
    # def forward(self, x, modality_list):
        qkv = (feature_embedding, kv[0], kv[1])
        x = self.mutihead_attention(qkv) + feature_embedding
        new_feature_embedding = self.feed_forward(x) + x
        # x = self.mutihead_attention(x) + x
        # x = self.feed_forward(x) + x
        return new_feature_embedding


class cross_attention_transformer_block(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads, dim_heads, num_modality, dropout):
        super(cross_attention_transformer_block,self).__init__()
        self.transformer_layers = nn.ModuleList([])
        for _ in range(num_modality):
            self.transformer_layers.append(fusion_transformer(dim, hidden_dim, num_heads, dim_heads, dropout))

    def forward(self, feature_embedding, kv_list, modality_list):
        "each layer fusion"
        transformer_layer_idx = 0
        feature_idx_ = 0
        features_list = []
        for layer in self.transformer_layers:
            # print("transformer_layer_idx", transformer_layer_idx)
            # print("feature_idx_", feature_idx_)
            if modality_list[transformer_layer_idx] == True:
                # print(qkv_list[feature_idx_][1:])
                new_feature_embedding = layer(feature_embedding, kv_list[feature_idx_])
                features_list.append(new_feature_embedding)
                transformer_layer_idx += 1
                feature_idx_ += 1
            else:
                transformer_layer_idx += 1
        feature_embedding = torch.cat(features_list, dim=1)
        return feature_embedding

class classification_Head(nn.Sequential):
    def __init__(self, emb_size=512, num_classes=27):
        super(classification_Head,self).__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, num_classes)
    
    def forward(self, x):
        # print(x.shape)
        x = torch.mean(x, dim=1)
        # print(x.shape)
        x = self.norm(x)
        x = self.fc(x)
        # x = x.view(x.size(0), 17, 3)
        return x
    
class X_Fusion(nn.Module):
    def __init__(self, num_modalities, dim, qkv_hidden_expansion, hidden_dim, num_feature, num_heads, dim_heads, model_depth, dropout):
        super(X_Fusion,self).__init__()
        self.kv_layers = nn.ModuleList([])
        self.cross_attention_transformer = nn.ModuleList([])
        for _ in range(num_modalities):
            self.kv_layers.append(kv_projection(dim, qkv_hidden_expansion))
        self.cross_attention_transformer = cross_attention_transformer_block(dim, hidden_dim, num_heads, dim_heads, num_modalities, dropout)
        self.cross_modal_transformer = cross_modal_transformer(num_feature, num_modalities, qkv_hidden_expansion, dim, num_heads, dropout)
        self.depth = model_depth
        self.classification_head = classification_Head(dim, 27)

    def forward(self, feature, modality_list):
        num_modalities = sum(modality_list)
        feature_list = list(feature.chunk(num_modalities, dim = 1))

        kv_list = []
        kv_layer_idx = 0
        feature_idx = 0
        for kv_layer in self.kv_layers:
            if modality_list[kv_layer_idx] == True:
                kv = kv_layer(feature_list[feature_idx])
                kv_list.append(kv)
                feature_idx += 1
                kv_layer_idx += 1
            else:
                kv_layer_idx += 1
                continue
        "output kv_list: [[k_1, v_1], [k_2, v_2], ...]"
        feature_embedding = self.cross_modal_transformer(feature, modality_list)
        "feature_embedding shape: B, 32, 512 (averaged sum of all features)"

        for i in range(self.depth):
            "feature_embedding shape: B, 32, 512"
            feature_embedding = self.cross_attention_transformer(feature_embedding, kv_list, modality_list)
            "feature_embedding shape: B, 32*n, 512"
            feature_embedding = self.cross_modal_transformer(feature_embedding, modality_list)
            "feature_embedding shape: B, 32, 512"

        x = self.classification_head(feature_embedding)
        return x
        


class X_Fi(nn.Module):
    def __init__(self, model_depth = 2):
        super(X_Fi, self).__init__()
        self.feature_extractor = feature_extrator()
        self.linear_projector = linear_projector(512, 512)
        self.X_Fusion_block = X_Fusion(
            num_modalities = 4,
            dim = 512,
            qkv_hidden_expansion = 2,
            hidden_dim = 512,
            num_feature = 32,
            num_heads = 8,
            dim_heads = 64,
            model_depth = model_depth,
            dropout = 0.
        )
    def forward(self, rgb_data, depth_data, mmwave_data, lidar_data, modality_list):
        feature_list = self.feature_extractor(rgb_data, depth_data, mmwave_data, lidar_data, modality_list)
        projected_features = self.linear_projector(feature_list, lidar_data, modality_list)
        out = self.X_Fusion_block(projected_features, modality_list)
        return out



