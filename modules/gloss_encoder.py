import pickle

import torch
import torch.nn as nn


class gloss_encoder(nn.Module):
    def __init__(self):
        super(gloss_encoder, self).__init__()
        with open ( "./preprocess/gloss_bert_feature.pkl" , "rb" ) as f :
            self.gloss_feature_dict = pickle.load (f)
        self.linear = nn.Linear ( 768 , 1024 )

    def forward(self, labels):
        gloss_feature_list = []
        for i in labels:
            gloss = self.gloss_feature_dict[int(i)]
            gloss_feature_list.append(gloss)
        return self.linear(torch.concat(gloss_feature_list))