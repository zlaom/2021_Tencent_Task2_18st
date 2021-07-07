import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import BertTokenizer, AdamW, AutoModel 

class Concat(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(2048 + 1024,82),
        )
        
    def forward(self, video_feature, text_feature):
        x = torch.cat([video_feature, text_feature], 1)
        x = self.classifier(x)
        
        return x
    
    
class NVSENet(nn.Module):
    def __init__(self, d_text=4096, d_video=2048, reduce=8):
        super().__init__()
        self.se_text = nn.Sequential(
            nn.Linear(d_text, d_text//reduce),
            nn.BatchNorm1d(d_text//reduce),
            nn.ReLU(inplace=True),
            nn.Linear(d_text//reduce, d_video),
            nn.BatchNorm1d(d_video),
            nn.Sigmoid()
        )
        self.se_video = nn.Sequential(
            nn.Linear(d_video, d_video//reduce),
            nn.BatchNorm1d(d_video//reduce),
            nn.ReLU(inplace=True),
            nn.Linear(d_video//reduce, d_text),
            nn.BatchNorm1d(d_text),
            nn.Sigmoid()
        )
        d_all = d_video + d_text
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_all, 82),
        )
        
    def forward(self, video, text):
        a_tv = self.se_text(text)
        a_vt = self.se_video(video)
        x = torch.cat([text * a_vt, video * a_tv], 1)
        x = self.classifier(x)
        
        return x

    
class TVSENet(nn.Module):
    def __init__(self, d_text=1024, d_video=2048, reduce=8):
        super().__init__()
        self.se_text = nn.Sequential(
            nn.Linear(d_text, d_text//reduce),
            nn.BatchNorm1d(d_text//reduce),
            nn.ReLU(inplace=True),
            nn.Linear(d_text//reduce, d_video),
            nn.BatchNorm1d(d_video),
            nn.Sigmoid()
        )
        self.se_video = nn.Sequential(
            nn.Linear(d_video, d_video//reduce),
            nn.BatchNorm1d(d_video//reduce),
            nn.ReLU(inplace=True),
            nn.Linear(d_video//reduce, d_text),
            nn.BatchNorm1d(d_text),
            nn.Sigmoid()
        )
        d_all = d_video + d_text
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_all, 82),
        )
        
    def forward(self, video, text):
        a_tv = self.se_text(text)
        a_vt = self.se_video(video)
        x = torch.cat([text * a_vt, video * a_tv], 1)
        x = self.classifier(x)
        
        return x

    
class TNSENet(nn.Module):
    def __init__(self, d_text=1024, d_video=4096, reduce=8):
        super().__init__()
        self.se_text = nn.Sequential(
            nn.Linear(d_text, d_text//reduce),
            nn.BatchNorm1d(d_text//reduce),
            nn.ReLU(inplace=True),
            nn.Linear(d_text//reduce, d_video),
            nn.BatchNorm1d(d_video),
            nn.Sigmoid()
        )
        self.se_video = nn.Sequential(
            nn.Linear(d_video, d_video//reduce),
            nn.BatchNorm1d(d_video//reduce),
            nn.ReLU(inplace=True),
            nn.Linear(d_video//reduce, d_text),
            nn.BatchNorm1d(d_text),
            nn.Sigmoid()
        )
        d_all = d_video + d_text
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_all, 82),
        )
        
    def forward(self, video, text):
        a_tv = self.se_text(text)
        a_vt = self.se_video(video)
        x = torch.cat([text * a_vt, video * a_tv], 1)
        x = self.classifier(x)
        
        return x

    
class TNVSENet(nn.Module):
    def __init__(self, d_text=1024, d_video=2048, d_nextvlad=4096, reduce=8):
        super().__init__()
        self.se_t2v = nn.Sequential(
            nn.Linear(d_text, d_text//reduce),
            nn.BatchNorm1d(d_text//reduce),
            nn.ReLU(inplace=True),
            nn.Linear(d_text//reduce, d_video),
            nn.BatchNorm1d(d_video),
            nn.Sigmoid()
        )
        self.se_v2n = nn.Sequential(
            nn.Linear(d_video, d_video//reduce),
            nn.BatchNorm1d(d_video//reduce),
            nn.ReLU(inplace=True),
            nn.Linear(d_video//reduce, d_nextvlad),
            nn.BatchNorm1d(d_nextvlad),
            nn.Sigmoid()
        )
        self.se_n2t = nn.Sequential(
            nn.Linear(d_nextvlad, d_nextvlad//reduce),
            nn.BatchNorm1d(d_nextvlad//reduce),
            nn.ReLU(inplace=True),
            nn.Linear(d_nextvlad//reduce, d_text),
            nn.BatchNorm1d(d_text),
            nn.Sigmoid()
        )
        
        d_all = d_video + d_text + d_nextvlad
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_all, 82),
        )
        
    def forward(self, text, video, nextvlad):
        tv = self.se_t2v(text)
        vn = self.se_v2n(video)
        nt = self.se_n2t(nextvlad)
        
        x = torch.cat([video * tv, text * nt, nextvlad * vn], 1)
        x = self.classifier(x)
        
        return x

    
class TNV2SENet(nn.Module):
    def __init__(self, d_text=1024, d_video=2048, d_nextvlad=4096, reduce=8):
        super().__init__()
        d_nv = d_nextvlad + d_video
        self.se_nv2t = nn.Sequential(
            nn.Linear(d_nv, d_nv//reduce),
            nn.BatchNorm1d(d_nv//reduce),
            nn.ReLU(inplace=True),
            nn.Linear(d_nv//reduce, d_text),
            nn.BatchNorm1d(d_text),
            nn.Sigmoid()
        )
        d_tn = d_text + d_nextvlad
        self.se_tn2v = nn.Sequential(
            nn.Linear(d_tn, d_tn//reduce),
            nn.BatchNorm1d(d_tn//reduce),
            nn.ReLU(inplace=True),
            nn.Linear(d_tn//reduce, d_video),
            nn.BatchNorm1d(d_video),
            nn.Sigmoid()
        )
        d_tv = d_text + d_video
        self.se_tv2n = nn.Sequential(
            nn.Linear(d_tv, d_tv//reduce),
            nn.BatchNorm1d(d_tv//reduce),
            nn.ReLU(inplace=True),
            nn.Linear(d_tv//reduce, d_nextvlad),
            nn.BatchNorm1d(d_nextvlad),
            nn.Sigmoid()
        )
        
        d_all = d_video + d_text + d_nextvlad
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_all, 82),
        )
        
    def forward(self, text, video, nextvlad):
        nextvlad_video = torch.cat([nextvlad, video], 1)
        text_video = torch.cat([text, video], 1)
        text_nextvlad = torch.cat([text, nextvlad], 1)
        nvt = self.se_nv2t(nextvlad_video)
        tnv = self.se_tn2v(text_nextvlad)
        tvn = self.se_tv2n(text_video)
        
        x = torch.cat([video * tnv, text * nvt, nextvlad * tvn], 1)
        x = self.classifier(x)
        
        return x
    
class StyleCompare(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(2048 * 2, 82),
        )
        
    def forward(self, video_feature, scene_feature, people_feature, display_feature, global_feature):
        x = torch.cat([video_feature, display_feature], 1)
        x = self.classifier(x)
        return x


class StyleCompare2(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_s = nn.Sequential(nn.Dropout(0.1),nn.Linear(2048, 24),)
        self.l_p = nn.Sequential(nn.Dropout(0.1),nn.Linear(2048, 19),)
        self.l_d = nn.Sequential(nn.Dropout(0.1),nn.Linear(2048, 27),)
        self.l_g = nn.Sequential(nn.Dropout(0.1),nn.Linear(2048, 12),)
        self.l_v = nn.Sequential(nn.Dropout(0.1),nn.Linear(2048, 82),)

    def forward(self, video_feature, scene_feature, people_feature, display_feature, global_feature):
        s = self.l_s(scene_feature)
        p = self.l_p(people_feature)
        d = self.l_d(display_feature)
        g = self.l_g(global_feature)
        x = torch.cat([s,p,d,g], 1)
        
        v = self.l_v(video_feature)
        x = x + v
        return x


# class StyleCompare2(nn.Module):
#     def __init__(self, d_video=2048, reduce=16):
#         super().__init__()
        
#         self.l_s = nn.Sequential(nn.Dropout(0.1),nn.Linear(2048, 24),)
#         self.l_p = nn.Sequential(nn.Dropout(0.1),nn.Linear(2048, 19),)
#         self.l_d = nn.Sequential(nn.Dropout(0.1),nn.Linear(2048, 27),)
#         self.l_g = nn.Sequential(nn.Dropout(0.1),nn.Linear(2048, 12),)
#         self.l_v = nn.Sequential(nn.Dropout(0.1),nn.Linear(2048, 82),)
        
#         self.se_s = nn.Sequential(
#             nn.Linear(d_video, d_video//reduce),
#             nn.BatchNorm1d(d_video//reduce),
#             nn.ReLU(inplace=True),
#             nn.Linear(d_video//reduce, d_video),
#             nn.BatchNorm1d(d_video),
#             nn.Sigmoid()
#         )
#         self.se_p = nn.Sequential(
#             nn.Linear(d_video, d_video//reduce),
#             nn.BatchNorm1d(d_video//reduce),
#             nn.ReLU(inplace=True),
#             nn.Linear(d_video//reduce, d_video),
#             nn.BatchNorm1d(d_video),
#             nn.Sigmoid()
#         )
#         self.se_d = nn.Sequential(
#             nn.Linear(d_video, d_video//reduce),
#             nn.BatchNorm1d(d_video//reduce),
#             nn.ReLU(inplace=True),
#             nn.Linear(d_video//reduce, d_video),
#             nn.BatchNorm1d(d_video),
#             nn.Sigmoid()
#         )
#         self.se_g = nn.Sequential(
#             nn.Linear(d_video, d_video//reduce),
#             nn.BatchNorm1d(d_video//reduce),
#             nn.ReLU(inplace=True),
#             nn.Linear(d_video//reduce, d_video),
#             nn.BatchNorm1d(d_video),
#             nn.Sigmoid()
#         )
        
#     def forward(self, video_f, scene_f, people_f, display_f, global_f):
#         scene_f = self.se_s(video_f) * scene_f
#         people_f = self.se_p(video_f) * people_f
#         display_f = self.se_d(video_f) * display_f
#         global_f = self.se_g(video_f) * global_f
        
#         s = self.l_s(scene_f)
#         p = self.l_p(people_f)
#         d = self.l_d(display_f)
#         g = self.l_g(global_f)
        
#         x = torch.cat([s,p,d,g], 1)
        
#         v = self.l_v(video_f)
#         x = x + v
#         return x
    


    
# class StyleConcat(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.1),
#             nn.Linear(1024+768, 82),
#         )

#     def forward(self, text_feature, scene_feature, people_feature, display_feature, global_feature):
#         x = torch.cat([text_feature, people_feature], 1)
#         x = self.classifier(x)
        
#         return x


# class StyleConcat(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.1),
#             nn.Linear(768 * 4 + 1024, 82),
#         )

#     def forward(self, text_feature, scene_feature, people_feature, display_feature, global_feature):
#         x = torch.cat([text_feature, scene_feature, people_feature, display_feature, global_feature], 1)
#         x = self.classifier(x)
        
#         return x

    
# class StyleConcat(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.attention = nn.Sequential(
#             nn.Linear(768 * 4, 800),
#             nn.Dropout(0.5),
#             nn.ReLU(),
#             nn.Linear(800, 4),
#             nn.Softmax()
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.1),
#             nn.Linear(768 * 4, 82),
#         )
        
#     def forward(self, video_feature, scene_feature, people_feature, display_feature, global_feature):
#         a = torch.cat([scene_feature, people_feature, display_feature, global_feature], 1)
#         a = self.attention(a)
#         w1 = a[:,0].unsqueeze(1).expand(a.shape[0], 768)
#         w2 = a[:,1].unsqueeze(1).expand(a.shape[0], 768)
#         w3 = a[:,2].unsqueeze(1).expand(a.shape[0], 768)
#         w4 = a[:,3].unsqueeze(1).expand(a.shape[0], 768)
#         x = torch.cat([w1 * scene_feature, w2 * people_feature, w3 * display_feature, w4 * global_feature], 1)
#         x = self.classifier(x)
        
#         return x


# class StyleConcat(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.dropout = nn.Dropout(0.5)
        
#         self.s_p = nn.Linear(768 * 2, 82)
#         self.s_d = nn.Linear(768 * 2, 82)
#         self.s_g = nn.Linear(768 * 2, 82)
#         self.p_d = nn.Linear(768 * 2, 82)
#         self.p_g = nn.Linear(768 * 2, 82)
#         self.d_g = nn.Linear(768 * 2, 82)
        
#         self.final = nn.Linear(82 * 6, 82)

#     def forward(self, video_feature, scene_feature, people_feature, display_feature, global_feature):
#         s = self.dropout(scene_feature)
#         p = self.dropout(people_feature)
#         d = self.dropout(display_feature)
#         g = self.dropout(global_feature)
        
#         s_p = self.dropout(self.s_p(torch.cat([s, p], 1)))
#         s_d = self.dropout(self.s_d(torch.cat([s, d], 1)))
#         s_g = self.dropout(self.s_g(torch.cat([s, g], 1)))
#         p_d = self.dropout(self.p_d(torch.cat([p, d], 1)))
#         p_g = self.dropout(self.p_g(torch.cat([p, g], 1)))
#         d_g = self.dropout(self.d_g(torch.cat([d, g], 1)))
        
#         x = torch.cat([s_p, s_d, s_g, p_d, p_g, d_g], 1)
#         x = self.final(x)

#         return x
    
    
class Mutan(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_dim = 1024
        self.video_dim = 2048
        self.out_dim = 82
        self.num_layers = 10
#         self.predict_linear = nn.Linear(self.out_dim, 82)
#         self.dropout = nn.Dropout(0.1)

        hv = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(self.text_dim, self.out_dim)
            hv.append(nn.Sequential(do, lin, nn.tanh()))
        self.text_layers = nn.ModuleList(hv)

        hq = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(self.video_dim, self.out_dim)
            hq.append(nn.Sequential(do, lin, nn.tanh()))
        self.video_layers = nn.ModuleList(hq)
        
    def forward(self, vf, tf):
        batch_size = tf.size()[0]
        x_mm = []
        for i in range(self.num_layers):
            x_hv = tf
            x_hv = self.text_layers[i](x_hv)

            x_hq = vf
            x_hq = self.video_layers[i](x_hq)
            x_mm.append(torch.mul(x_hq, x_hv))
        #
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.out_dim)
        
#         x_mm = F.tanh(x_mm)
#         x_mm = self.dropout(x_mm)
#         x_mm = self.predict_linear(x_mm)
        return x_mm


class MFB(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_video = nn.Linear(2048, 5000)
        self.linear_text = nn.Linear(768, 5000)
        self.linear_predict = nn.Linear(1000, 82)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, video_feature, text_feature):
        video_feature = self.linear_video(video_feature)
        text_feature = self.linear_text(text_feature)
        fs = torch.mul(video_feature, text_feature)
        fs = self.dropout(fs)
        fs = fs.view(-1, 1000, 5) # [B, dim_output, dim_sum_pool]
        fs = torch.sum(fs, 2)
        fs = torch.sqrt(F.relu(fs)) - torch.sqrt(F.relu(-fs))
        fs = F.normalize(fs)
        fs = self.linear_predict(fs)
        
        return fs

# 直接采用large bert版
class MFH(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_dim = 2000
        self.num = 5
        median = self.out_dim * self.num
        self.video_l1 = nn.Linear(2048, median)
        self.text_l1 = nn.Linear(1024, median)
        
        self.video_l2 = nn.Linear(2048, median)
        self.text_l2 = nn.Linear(1024, median)
        
        self.predict_l = nn.Linear(2*self.out_dim, 82)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, video_feature, text_feature):
        v1 = self.video_l1(video_feature)
        t1 = self.text_l1(text_feature)
        
        fs1 = torch.mul(v1, t1)
        fs1_drop = self.dropout(fs1)
        fs1 = fs1_drop.view(-1, self.out_dim, self.num) # [B, dim_output, dim_sum_pool]
        fs1 = torch.sum(fs1, 2)
        fs1 = torch.sqrt(F.relu(fs1)) - torch.sqrt(F.relu(-fs1))
        fs1 = F.normalize(fs1)
        
        v2 = self.video_l2(video_feature)
        t2 = self.text_l2(text_feature)
        
        fs2 = torch.mul(v2, t2)
        fs2 = torch.mul(fs1_drop, fs2)
        fs2 = self.dropout(fs2)
        fs2 = fs2.view(-1, self.out_dim, self.num) # [B, dim_output, dim_sum_pool]
        fs2 = torch.sum(fs2, 2)
        fs2 = torch.sqrt(F.relu(fs2)) - torch.sqrt(F.relu(-fs2))
        fs2 = F.normalize(fs2)
        
        fusion = torch.cat((fs1, fs2), 1)
        result = self.predict_l(fusion)
        
        return result
    
    
class TextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("hfl/chinese-macbert-base")
        self.classifier = nn.Sequential(nn.Dropout(0.1))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooler = outputs.pooler_output
        x = self.classifier(pooler)

        return x

    
# bert + video feature 由于保存模型的类名字要一致，这个名字就暂时不再做修改了
class Fusion(nn.Module):
    def __init__(self, text_ckpt=None):
        super().__init__()
        self.text_classifier = TextClassifier()
        if text_ckpt:
            self.text_classifier.load_state_dict(torch.load(text_ckpt), strict=False)
            print('text model match successful!')
        # 将上面的模型参数固定
#         for p in self.parameters():
#             p.requires_grad = False
        self.mfb = MFB()

    def forward(self, input_ids, attention_mask, video_feature):
        
        text_feature = self.text_classifier(input_ids, attention_mask)
        fusion = self.mfb(video_feature, text_feature)
        
        return fusion 

# tfvf_mfb模型用的
class LargeText_MFB(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_video = nn.Linear(2048, 5000)
        self.linear_text = nn.Linear(1024, 5000)
        self.linear_predict = nn.Linear(1000, 82)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, video_feature, text_feature):
        video_feature = self.linear_video(video_feature)
        text_feature = self.linear_text(text_feature)
        fs = torch.mul(video_feature, text_feature)
        fs = self.dropout(fs)
        fs = fs.view(-1, 1000, 5) # [B, dim_output, dim_sum_pool]
        fs = torch.sum(fs, 2)
        fs = torch.sqrt(F.relu(fs)) - torch.sqrt(F.relu(-fs))
        fs = F.normalize(fs)
        fs = self.linear_predict(fs)
        
        return fs