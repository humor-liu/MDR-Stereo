import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from core.update import BasicMultiUpdateBlock
from core.extractor import MultiBasicEncoder, depth_any_fea
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import *
import time
from core.ASPP import ASPP
from core.FFT_att import ffnatt
try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
class Pred_Layer(nn.Module):
    def __init__(self, in_c=256):
        super(Pred_Layer, self).__init__()
        self.enlayer = BasicConv(96, 96, kernel_size=3, padding=1, stride=1)
        self.outlayer = nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.enlayer(x)
        x1 = self.outlayer(x)
        return x, x1
class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv0 = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*8, is_3d=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*8, in_channels*8, is_3d=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*8, in_channels*4, deconv=True, is_3d=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, in_channels, deconv=True, is_3d=True, IN=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))


        self.feature_att_4 = FeatureAtt(in_channels, 96)
        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*8, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        conv0 = self.conv0(x)
        conv0 = self.feature_att_4(conv0, features[0])

        conv1 = self.conv1(conv0)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv



class MDRStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.depth_any_coder = depth_any_fea(args.dinov2_encoder)

        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn="batch", downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )
        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x(64, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(96, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64), nn.ReLU()
            )

        self.ffnatt = ffnatt(48,8,8)
        self.aspp = ASPP(48,[4,12,18])
        self.group = BasicConv(96+48,48,kernel_size=3,padding=1,stride=1)

        self.spx_2_gru = Conv2x(64, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.conv = BasicConv(96, 96, kernel_size=3, padding=1, stride=1)
        self.pred = Pred_Layer(96)

        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)
        self.cost_agg0 = hourglass(8)
        self.cost_agg1 = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp*4., spx_pred)
        return up_disp

    def forward(self, image1, image2, iters=12, test_mode=False):
        """ Estimate disparity between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            features_left,  features_right = self.depth_any_coder(image1, image2)

            stem_2x = self.stem_2(image1)
            stem_4x = self.stem_4(stem_2x)
            stem_2y = self.stem_2(image2)
            stem_4y = self.stem_4(stem_2y)

            stem_4x_aspp = self.aspp(stem_4x)
            stem_4y_aspp = self.aspp(stem_4y)
            stem_4x_satt = self.ffnatt(stem_4x)
            stem_4y_satt = self.ffnatt(stem_4y)
            stem_4x = torch.cat((stem_4x_satt,stem_4x_aspp,stem_4x),dim=1)
            stem_4y = torch.cat((stem_4y_satt,stem_4y_aspp,stem_4y),dim=1)

            stem_4x = self.group(stem_4x)
            stem_4y = self.group(stem_4y)

            features_left[0] = torch.cat((features_left[0], stem_4x), 1)
            features_right[0] = torch.cat((features_right[0], stem_4y), 1)

            features_left[0],pred1 = self.pred(features_left[0])
            match_left_1 = self.desc(features_left[0] * pred1)
            match_left_2 = self.desc(features_left[0] * (1-pred1))
            features_right[0], pred2 = self.pred(features_right[0])
            match_right_1 = self.desc(features_right[0] * pred2)
            match_right_2 = self.desc(features_right[0] * (1 - pred2))
            all_disp_voulme_1 = build_gwc_volume(match_left_1,match_right_1, self.args.max_disp//16,8)
            all_disp_voulme_2 = build_gwc_volume(match_left_2,match_right_2, self.args.max_disp//16,8)

            geo_encoding_volume0 = self.cost_agg0(all_disp_voulme_1,features_left)
            geo_encoding_volume1 = self.cost_agg1(all_disp_voulme_2, features_left)
            del all_disp_voulme_1,all_disp_voulme_2
            cost_volume0 = self.classifier(geo_encoding_volume0)
            prob_volume0 = F.softmax(cost_volume0.squeeze(1), dim=1)
            agg_disp0 = disparity_regression(prob_volume0, self.args.s_disp_range, self.args.s_disp_interval)
            cost_volume1 = self.classifier(geo_encoding_volume1)
            prob_volume1 = F.softmax(cost_volume1.squeeze(1), dim=1)
            agg_disp1 = disparity_regression(prob_volume1, self.args.s_disp_range, self.args.s_disp_interval)

            disp_feature1 = agg_disp0 * pred1 + agg_disp1 *(1-pred1)
            disp_feature = disp_feature1

            cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        geo_block = Combined_Geo_Encoding_Volume

        geo_fn = geo_block(geo_encoding_volume0.float(), geo_encoding_volume1.float() , match_left_1.float(), match_right_1.float(), radius=self.args.corr_radius)

        b, c, h, w = match_left_1.shape
        coords = torch.arange(w).float().to(match_left_1.device).reshape(1,1,w,1).repeat(b, h, 1, 1)

        disp = disp_feature
        iter_preds = []

        # GRUs iterations to update disparity
        for itr in range(iters):
            disp = disp.detach()
            geo_feat0, geo_feat1, init_corr = geo_fn(disp, coords)
            with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat0, geo_feat1, init_corr,pred1, disp, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)

            disp = disp + delta_disp
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            iter_preds.append(disp_up)

        if test_mode:
            return disp_up

        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            xspx = self.spx_4(features_left[0])
            xspx = self.spx_2(xspx, stem_2x)
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, 1)
        agg_disp0 = context_upsample(agg_disp0*4., spx_pred.float())
        agg_disp1 = context_upsample(agg_disp1*4., spx_pred.float())
        disp_feature1 = context_upsample(disp_feature1*4., spx_pred.float())
        return [agg_disp0, agg_disp1, disp_feature1], iter_preds