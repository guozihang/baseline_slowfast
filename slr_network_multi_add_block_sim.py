import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer , TemporalSlowFastFuse , gloss_encoder
import slowfast_modules.slowfast as slowfast
import importlib


class Identity ( nn.Module ) :
    def __init__ ( self ) :
        super ( Identity , self ).__init__ ( )

    def forward ( self , x ) :
        return x


class NormLinear ( nn.Module ) :
    def __init__ ( self , in_dim , out_dim ) :
        super ( NormLinear , self ).__init__ ( )
        self.weight = nn.Parameter ( torch.Tensor ( in_dim , out_dim ) )
        nn.init.xavier_uniform_ ( self.weight , gain = nn.init.calculate_gain ( 'relu' ) )

    def forward ( self , x ) :
        outputs = torch.matmul ( x , F.normalize ( self.weight , dim = 0 ) )
        return outputs


class Attention ( nn.Module ) :
    def __init__ ( self , dim=1024 , num_heads=1 , qkv_bias=False , qk_scale=None , attn_drop=0. , proj_drop=0. ) :
        super ( ).__init__ ( )
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear ( dim , dim * 3 , bias = qkv_bias )
        self.attn_drop = nn.Dropout ( attn_drop )
        self.proj = nn.Linear ( dim , dim )
        self.proj_drop = nn.Dropout ( proj_drop )

    def forward ( self , x ) :
        B , N , C = x.shape
        qkv = self.qkv ( x ).reshape ( B , N , 3 , self.num_heads , C // self.num_heads ).permute ( 2 , 0 , 3 , 1 , 4 )
        q , k , v = qkv [ 0 ] , qkv [ 1 ] , qkv [ 2 ]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose ( -2 , -1 )) * self.scale
        attn = attn.softmax ( dim = -1 )
        attn = self.attn_drop ( attn )

        x = (attn @ v).transpose ( 1 , 2 ).reshape ( B , N , C )
        x = self.proj ( x )
        x = self.proj_drop ( x )
        return x


class SLRModel ( nn.Module ) :
    def __init__ (
            self , num_classes , c2d_type , conv_type , load_pkl , slowfast_config , slowfast_args=None ,
            use_bn=False , hidden_size=1024 , gloss_dict=None , loss_weights=None ,
            weight_norm=True , share_classifier=1
    ) :
        super ( SLRModel , self ).__init__ ( )
        self.decoder = None
        self.loss = dict ( )
        self.criterion_init ( )
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr ( slowfast , c2d_type ) ( slowfast_config = slowfast_config ,
                                                        slowfast_args = slowfast_args ,
                                                        load_pkl = load_pkl , multi = True )
        self.gloss_dict = gloss_dict
        self.conv1d = TemporalSlowFastFuse ( fast_input_size = 256 , slow_input_size = 2048 ,
                                             hidden_size = hidden_size , conv_type = conv_type , use_bn = use_bn ,
                                             num_classes = num_classes )
        self.decoder = utils.Decode ( gloss_dict , num_classes , 'beam' )

        self.temporal_model = nn.ModuleList (
            [ BiLSTMLayer ( rnn_type = 'LSTM' , input_size = hidden_size , hidden_size = hidden_size ,
                            num_layers = 2 , bidirectional = True ) for i in range ( 3 ) ] )
        if weight_norm :
            self.classifier = nn.ModuleList ( [ NormLinear ( hidden_size , self.num_classes ) for i in range ( 3 ) ] )
            self.conv1d.fc = nn.ModuleList ( [ NormLinear ( hidden_size , self.num_classes ) for i in range ( 3 ) ] )
        else :
            self.classifier = nn.ModuleList ( [ nn.Linear ( hidden_size , self.num_classes ) for i in range ( 3 ) ] )
            self.conv1d.fc = nn.ModuleList ( [ nn.Linear ( hidden_size , self.num_classes ) for i in range ( 3 ) ] )
        if share_classifier == 1 :
            self.conv1d.fc = self.classifier
        elif share_classifier == 2 :
            classifier = self.classifier [ 0 ]
            self.classifier = nn.ModuleList ( [ classifier for i in range ( 3 ) ] )
            self.conv1d.fc = nn.ModuleList ( [ classifier for i in range ( 3 ) ] )
        # self.register_backward_hook(self.backward_hook)
        self.video_gloss_attn_a = Attention ( 1024 )
        self.video_gloss_attn_b = Attention ( 1024 )
        self.video_gloss_attn_c = Attention ( 1024 )
        self.video_gloss_attn = [ self.video_gloss_attn_a , self.video_gloss_attn_b , self.video_gloss_attn_c ]
        self.cosine_sim = torch.nn.CosineSimilarity ( dim = -1 , eps = 1e-6 )
        self.gloss_encoder = gloss_encoder ( )

    def backward_hook ( self , module , grad_input , grad_output ) :
        for g in grad_input :
            g [ g != g ] = 0

    def masked_bn ( self , inputs , len_x ) :
        def pad ( tensor , length ) :
            return torch.cat (
                [ tensor , tensor.new ( length - tensor.size ( 0 ) , *tensor.size ( ) [ 1 : ] ).zero_ ( ) ] )

        x = torch.cat ( [ inputs [ len_x [ 0 ] * idx :len_x [ 0 ] * idx + lgt ] for idx , lgt in enumerate ( len_x ) ] )
        x = self.conv2d ( x )
        x = torch.cat ( [ pad ( x [ sum ( len_x [ :idx ] ) :sum ( len_x [ :idx + 1 ] ) ] , len_x [ 0 ] )
                          for idx , lgt in enumerate ( len_x ) ] )
        return x

    def forward ( self , x , len_x , label=None , label_lgt=None ) :
        if len ( x.shape ) == 5 :
            framewise = self.conv2d ( x.permute ( 0 , 2 , 1 , 3 , 4 ) )
        else :
            # frame-wise features
            framewise = x
        conv1d_outputs = self.conv1d ( framewise , len_x )
        pooled_features = [ ]
        for i in range ( len ( conv1d_outputs [ 'visual_feat' ] ) ) :
            conv1d_output = conv1d_outputs [ 'visual_feat' ] [ i ]
            conv1d_output = self.video_gloss_attn [ i ] ( conv1d_output )
            r_conv1d_output = conv1d_output.roll ( 1 , 0 )
            similarity = self.cosine_sim ( conv1d_output , r_conv1d_output ).squeeze ( -1 )
            topk_similarity , topk_indices = similarity.topk ( label_lgt [ 0 ] , sorted = True )
            paired = list ( zip ( topk_similarity.tolist ( ) , topk_indices.tolist ( ) ) )
            sorted_paired = sorted ( paired , key = lambda x : x [ 1 ] )
            se = [ ]
            for j in range ( len ( sorted_paired ) ) :
                if j == 0 :
                    se.append ( [ 0 , sorted_paired [ j ] [ 1 ] ] )
                elif j == len ( sorted_paired ) - 1 :
                    se.append ( [ sorted_paired [ -1 ] [ 1 ] , len ( conv1d_output ) - 1 ] )
                else :
                    se.append ( [ sorted_paired [ j - 1 ] [ 1 ] , sorted_paired [ j ] [ 1 ] ] )
            pooled_feature = [ ]
            for start , end in se :
                if end != start :
                    b = torch.nn.functional.avg_pool1d ( conv1d_output [ start :end ].squeeze ( 1 ).T ,
                                                         kernel_size = end - start )
                    pooled_feature.append ( b.T )
                else :
                    pooled_feature.append ( conv1d_output [ start ].squeeze ( 1 ) )

            pooled_features.append ( torch.concat ( pooled_feature , dim = 0 ) )
        gloss_feature = self.gloss_encoder ( label )
        lgt = conv1d_outputs [ 'feat_len' ]
        outputs = [ ]
        for i in range ( len ( conv1d_outputs [ 'visual_feat' ] ) ) :
            tm_outputs = self.temporal_model [ i ] ( conv1d_outputs [ 'visual_feat' ] [ i ] , lgt )
            outputs.append ( self.classifier [ i ] ( tm_outputs [ 'predictions' ] ) )

        pred = None if self.training \
            else self.decoder.decode ( outputs [ 0 ] , lgt , batch_first = False , probs = False )
        conv_pred = None if self.training \
            else self.decoder.decode ( conv1d_outputs [ 'conv_logits' ] [ 0 ] , lgt , batch_first = False ,
                                       probs = False )

        return {
            # "framewise_features": framewise,
            # "visual_features": conv1d_outputs['visual_feat'],
            "feat_len" : lgt ,
            "conv_logits" : conv1d_outputs [ "conv_logits" ] ,
            "sequence_logits" : outputs ,
            "conv_sents" : conv_pred ,
            "recognized_sents" : pred ,
            "pooled_features" : pooled_features ,
            "gloss_feature" : gloss_feature ,
            "label" : label
        }

    def criterion_calculation ( self , ret_dict , label , label_lgt ) :
        loss = 0
        total_loss = {}
        for k , weight in self.loss_weights.items ( ) :
            if k == 'SeqCTC' :
                total_loss [ 'SeqCTC' ] = weight * self.loss [ 'CTCLoss' ] (
                    ret_dict [ "sequence_logits" ] [ 0 ].log_softmax ( -1 ) ,
                    label.cpu ( ).int ( ) ,
                    ret_dict [ "feat_len" ].cpu ( ).int ( ) ,
                    label_lgt.cpu ( ).int ( ) ).mean ( )
                loss += total_loss [ 'SeqCTC' ]
            elif k == 'Slow' or k == 'Fast' :
                i = 1 if k == 'Slow' else 2
                total_loss [ f"{k}_{i}" ] = weight * self.loss_weights [ 'SeqCTC' ] * self.loss [ 'CTCLoss' ] (
                    ret_dict [ "sequence_logits" ] [ i ].log_softmax ( -1 ) ,
                    label.cpu ( ).int ( ) , ret_dict [ "feat_len" ].cpu ( ).int ( ) ,
                    label_lgt.cpu ( ).int ( ) ).mean ( )
                loss += total_loss [ f"{k}_{i}" ]
                if 'ConvCTC' in self.loss_weights :
                    total_loss [ f'ConvCTC_{i}' ] = weight * self.loss_weights [ 'ConvCTC' ] * self.loss [
                        'CTCLoss' ] (
                        ret_dict [ "conv_logits" ] [ i ].log_softmax ( -1 ) ,
                        label.cpu ( ).int ( ) , ret_dict [ "feat_len" ].cpu ( ).int ( ) ,
                        label_lgt.cpu ( ).int ( ) ).mean ( )
                    loss += total_loss [ f'ConvCTC_{i}' ]
                if 'Dist' in self.loss_weights :
                    # loss += weight * self.loss_weights['Dist'] * self.loss['distillation'](ret_dict["conv_intra_logits"][i],
                    #                                                                     ret_dict["sequence_logits"].detach(),
                    #                                                                     use_blank=False)
                    total_loss [ f'Dist_{i}' ] = weight * self.loss_weights [ 'Dist' ] * self.loss [ 'distillation' ] (
                        ret_dict [ "conv_logits" ] [ i ] ,
                        ret_dict [ "sequence_logits" ] [ i ].detach ( ) ,
                        use_blank = False )
                    loss += total_loss [ f'Dist_{i}' ]
            elif k == 'ConvCTC' :
                total_loss [ 'ConvCTC' ] = weight * self.loss [ 'CTCLoss' ] (
                    ret_dict [ "conv_logits" ] [ 0 ].log_softmax ( -1 ) ,
                    label.cpu ( ).int ( ) ,
                    ret_dict [ "feat_len" ].cpu ( ).int ( ) ,
                    label_lgt.cpu ( ).int ( ) ).mean ( )
                loss += total_loss [ 'ConvCTC' ]
            elif k == 'Dist' :
                total_loss [ 'Dist' ] = weight * self.loss [ 'distillation' ] ( ret_dict [ "conv_logits" ] [ 0 ] ,
                                                                                ret_dict [ "sequence_logits" ] [
                                                                                    0 ].detach ( ) ,
                                                                                use_blank = False )
                loss += total_loss [ 'Dist' ]
            elif k == 'Cosine' :
                for i in range ( 3 ) :
                    # total_loss [ f'Cosine_{i}' ] = weight * self.loss['CosineLoss'](ret_dict["pooled_features"][i], ret_dict["gloss_feature"], ret_dict["label"])
                    # loss += total_loss [ f'Cosine_{i}' ]
                    total_loss [ f'Cosine_{i}' ] = weight * self.loss [ 'CosineLoss' ] (
                        ret_dict [ "pooled_features" ] [ i ] ,
                        ret_dict [ "gloss_feature" ] )
                    loss += total_loss [ f'Cosine_{i}' ]
        return loss , total_loss

    def criterion_init ( self ) :
        self.loss [ 'CTCLoss' ] = torch.nn.CTCLoss ( reduction = 'none' , zero_infinity = False )
        self.loss [ 'distillation' ] = SeqKD ( T = 8 )
        # self.loss['CosineLoss'] = torch.nn.CosineEmbeddingLoss()
        self.loss [ 'CosineLoss' ] = CosineSimilarityLoss ( )
        return self.loss


class CosineSimilarityLoss ( nn.Module ) :
    def __init__ ( self ) :
        super ( CosineSimilarityLoss , self ).__init__ ( )

    def forward ( self , input1 , input2 ) :
        cosine_similarity = nn.functional.cosine_similarity ( input1 , input2 , dim = 1 )
        loss = 1 - cosine_similarity
        return loss.mean ( )
