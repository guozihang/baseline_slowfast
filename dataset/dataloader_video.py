import os
import pickle
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings

sys.path.append ( os.path.dirname ( os.path.abspath ( os.path.dirname ( __file__ ) ) ) )
warnings.simplefilter ( action = 'ignore' , category = FutureWarning )

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
from utils import video_augmentation
from torch.utils.data.sampler import Sampler

sys.path.append ( ".." )
global kernel_sizes


class BaseFeeder ( data.Dataset ) :
    def __init__ ( self , prefix , gloss_dict , dataset='phoenix2014' , drop_ratio=1 , num_gloss=-1 , mode="train" ,
                   transform_mode=True ,
                   datatype="lmdb" , frame_interval=1 , image_scale=1.0 , kernel_size=1 , input_size=224 ) :
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.dataset = dataset
        self.input_size = input_size
        global kernel_sizes
        kernel_sizes = kernel_size
        self.frame_interval = frame_interval  # not implemented for read_features()
        self.image_scale = image_scale  # not implemented for read_features()
        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load ( f"./preprocess/{dataset}/{mode}_info.npy" , allow_pickle = True ).item ( )
        print ( mode , len ( self ) )
        self.data_aug = self.transform ( )
        print ( "" )

    def __getitem__ ( self , idx ) :
        if self.data_type == "video" :
            input_data , label , fi = self.read_video ( idx )
            input_data , label = self.normalize ( input_data , label )
            # input_data, label = self.normalize(input_data, label, fi['fileid'])
            return input_data , torch.LongTensor ( label ) , self.inputs_list [ idx ] [ 'original_info' ]
        elif self.data_type == "lmdb" :
            input_data , label , fi = self.read_lmdb ( idx )
            input_data , label = self.normalize ( input_data , label )
            return input_data , torch.LongTensor ( label ) , self.inputs_list [ idx ] [ 'original_info' ]
        elif self.data_type == "numpy" :
            input_data , label , fi = self.read_numpy ( idx )
            input_data , label = self.normalize ( input_data , label )
            return input_data , torch.LongTensor ( label ) , self.inputs_list [ idx ] [ 'original_info' ]
        elif self.data_type == "memmap" :
            if hasattr(self, 'mem') == False:
                self.init_memmap()
            input_data , label , fi = self.read_memmap ( idx )
            input_data , label = self.normalize ( input_data , label )
            return input_data , torch.LongTensor ( label ) , self.inputs_list [ idx ] [ 'original_info' ]
        else :
            input_data , label = self.read_features ( idx )
            return input_data , label , self.inputs_list [ idx ] [ 'original_info' ]

    def init_memmap ( self ) :
        if self.dataset == 'phoenix2014' :
            with open (
                    f"/share/huaiwen_group/guozihang/phoenix2014-release_memmap/phoenix2014-bigarray-map-{self.mode}" ,
                    mode = "rb" ) as f :
                self.info = pickle.load ( f )
            T = self.info [ -1 ] [ 'end' ]
            self.info = {i [ "path" ].split ( "/" ) [ -1 ] : [ i [ "start" ] , i [ "end" ] ] for i in self.info}
            self.mem = np.memmap (
                f"/share/huaiwen_group/guozihang/phoenix2014-release_memmap/phoenix2014-{self.mode}.pickle" ,
                mode = "r" ,
                shape = (T , 256 , 256 , 3) )

    def read_memmap ( self , index ) :
        fi = self.inputs_list [ index ]
        label_list = [ ]
        for phase in fi [ 'label' ].split ( " " ) :
            if phase == '' :
                continue
            if phase in self.dict.keys ( ) :
                label_list.append ( self.dict [ phase ] [ 0 ] )
        images = self.mem [ self.info [ fi [ 'fileid' ] + ".npy" ] [ 0 ] :self.info [ fi [ 'fileid' ] + ".npy" ] [ 1 ] ]
        images = np.split ( images , images.shape [ 0 ] , axis = 0 )
        images = [ np.squeeze ( im , axis = 0 ) for im in images ]
        return images , label_list , fi

    def read_numpy ( self , index ) :
        # load file info
        fi = self.inputs_list [ index ]
        if self.dataset in [ "phoenix2014" , "phoenix2014-noback" ] :
            # img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'])
            img_folder = os.path.join ( self.prefix , self.mode , fi [ 'fileid' ] + ".npy" )
        elif self.dataset == 'phoenix2014-T' :
            img_folder = os.path.join ( self.prefix , self.mode , fi [ 'fileid' ] + ".npy" )
        elif self.dataset == 'CSL' :
            img_folder = os.path.join ( self.prefix , "features/fullFrame-256x256px/" + fi [ 'folder' ] + "/*.jpg" )
        elif self.dataset == 'CSL-Daily' :
            img_folder = os.path.join ( self.prefix , fi [ 'fileid' ] + ".npy" )
        label_list = [ ]
        for phase in fi [ 'label' ].split ( " " ) :
            if phase == '' :
                continue
            if phase in self.dict.keys ( ) :
                label_list.append ( self.dict [ phase ] [ 0 ] )
        images = np.load ( img_folder )
        images = np.split ( images , images.shape [ 0 ] , axis = 0 )
        images = [ np.squeeze ( im , axis = 0 ) for im in images ]
        return images , label_list , fi

    def read_video ( self , index ) :
        # load file info
        fi = self.inputs_list [ index ]
        if 'phoenix' in self.dataset :
            img_folder = os.path.join ( self.prefix , "features/fullFrame-256x256px/" + fi [ 'folder' ] )
        elif self.dataset == 'CSL' :
            img_folder = os.path.join ( self.prefix , "features/fullFrame-256x256px/" + fi [ 'folder' ] + "/*.jpg" )
        elif self.dataset == 'CSL-Daily' :
            img_folder = os.path.join ( self.prefix , fi [ 'folder' ] )
        img_list = sorted ( glob.glob ( img_folder ) )
        img_list = img_list [ int ( torch.randint ( 0 , self.frame_interval , [ 1 ] ) ) : :self.frame_interval ]
        label_list = [ ]
        for phase in fi [ 'label' ].split ( " " ) :
            if phase == '' :
                continue
            if phase in self.dict.keys ( ) :
                label_list.append ( self.dict [ phase ] [ 0 ] )
        if self.dataset != 'CSL-Daily' :
            return [ cv2.cvtColor ( cv2.imread ( img_path ) , cv2.COLOR_BGR2RGB ) for img_path in
                     img_list ] , label_list , fi
        else :
            return [
                cv2.cvtColor ( cv2.resize ( cv2.imread ( img_path ) [ 40 : , ... ] , (256 , 256) ) , cv2.COLOR_BGR2RGB )
                for img_path in img_list ] , label_list , fi

    def read_features ( self , index ) :
        # load file info
        fi = self.inputs_list [ index ]
        data = np.load ( f"./features/{self.mode}/{fi [ 'fileid' ]}_features.npy" , allow_pickle = True ).item ( )
        return data [ 'features' ] , data [ 'label' ]

    def normalize ( self , video , label , file_id=None ) :
        video , label = self.data_aug ( video , label , file_id )
        # video = video.float() / 127.5 - 1
        mean = [ 0.45 , 0.45 , 0.45 ]
        std = [ 0.225 , 0.225 , 0.225 ]
        video = ((video.float ( ) / 255.) - 0.45) / 0.225

        return video , label

    def transform ( self ) :
        if self.transform_mode == "train" :
            print ( "Apply training transform." )
            return video_augmentation.Compose ( [
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomCrop ( self.input_size ) ,
                video_augmentation.RandomHorizontalFlip ( 0.5 ) ,
                video_augmentation.Resize ( self.image_scale ) ,
                video_augmentation.ToTensor ( ) ,
                video_augmentation.TemporalRescale ( 0.2 , self.frame_interval ) ,
            ] )
        else :
            print ( "Apply testing transform." )
            return video_augmentation.Compose ( [
                video_augmentation.CenterCrop ( self.input_size ) ,
                video_augmentation.Resize ( self.image_scale ) ,
                video_augmentation.ToTensor ( ) ,
            ] )

    def byte_to_img ( self , byteflow ) :
        unpacked = pa.deserialize ( byteflow )
        imgbuf = unpacked [ 0 ]
        buf = six.BytesIO ( )
        buf.write ( imgbuf )
        buf.seek ( 0 )
        img = Image.open ( buf ).convert ( 'RGB' )
        return img

    @staticmethod
    def collate_fn ( batch ) :
        batch = [ item for item in sorted ( batch , key = lambda x : len ( x [ 0 ] ) , reverse = True ) ]
        video , label , info = list ( zip ( *batch ) )

        left_pad = 0
        last_stride = 1
        total_stride = 1
        global kernel_sizes
        for layer_idx , ks in enumerate ( kernel_sizes ) :
            if ks [ 0 ] == 'K' :
                left_pad = left_pad * last_stride
                left_pad += int ( (int ( ks [ 1 ] ) - 1) / 2 )
            elif ks [ 0 ] == 'P' :
                last_stride = int ( ks [ 1 ] )
                total_stride = total_stride * last_stride
        if len ( video [ 0 ].shape ) > 3 :
            max_len = len ( video [ 0 ] )
            video_length = torch.LongTensor (
                [ np.ceil ( len ( vid ) / total_stride ) * total_stride + 2 * left_pad for vid in video ] )
            right_pad = int ( np.ceil ( max_len / total_stride ) ) * total_stride - max_len + left_pad
            max_len = max_len + left_pad + right_pad
            padded_video = [ torch.cat (
                (
                    vid [ 0 ] [ None ].expand ( left_pad , -1 , -1 , -1 ) ,
                    vid ,
                    vid [ -1 ] [ None ].expand ( max_len - len ( vid ) - left_pad , -1 , -1 , -1 ) ,
                )
                , dim = 0 )
                for vid in video ]
            padded_video = torch.stack ( padded_video )
        else :
            max_len = len ( video [ 0 ] )
            video_length = torch.LongTensor ( [ len ( vid ) for vid in video ] )
            padded_video = [ torch.cat (
                (
                    vid ,
                    vid [ -1 ] [ None ].expand ( max_len - len ( vid ) , -1 ) ,
                )
                , dim = 0 )
                for vid in video ]
            padded_video = torch.stack ( padded_video ).permute ( 0 , 2 , 1 )
        label_length = torch.LongTensor ( [ len ( lab ) for lab in label ] )
        if max ( label_length ) == 0 :
            return padded_video , video_length , [ ] , [ ] , info
        else :
            padded_label = [ ]
            for lab in label :
                padded_label.extend ( lab )
            padded_label = torch.LongTensor ( padded_label )
            return padded_video , video_length , padded_label , label_length , info

    def __len__ ( self ) :
        return len ( self.inputs_list ) - 1

    def record_time ( self ) :
        self.cur_time = time.time ( )
        return self.cur_time

    def split_time ( self ) :
        split_time = time.time ( ) - self.cur_time
        self.record_time ( )
        return split_time


if __name__ == "__main__" :
    gloss_dict = np.load ( './preprocess/phoenix2014/gloss_dict.npy' , allow_pickle = True ).item ( )
    feeder = BaseFeeder (
        prefix = './dataset/phoenix2014/phoenix-2014-multisigner' ,
        gloss_dict = gloss_dict ,
        dataset = 'phoenix2014' ,
        datatype = 'video' ,
        kernel_size = [ 'K5' , 'P2' , 'K5' , 'P2' ]
    )
    dataloader = torch.utils.data.DataLoader (
        dataset = feeder ,
        batch_size = 1 ,
        shuffle = True ,
        drop_last = True ,
        num_workers = 0 ,
        collate_fn = feeder.collate_fn ,
    )
    for data in dataloader :
        pdb.set_trace ( )
