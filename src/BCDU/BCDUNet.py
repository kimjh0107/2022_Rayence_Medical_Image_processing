import numpy as np
import torch 
import torch.nn as nn
from .ConvLSTM import ConvBLSTM, ConvLSTM


class BCDUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, num_filter=64, frame_size=(256, 256), bidirectional=False, norm='instance'):
        super(BCDUNet, self).__init__()
        self.num_filter = num_filter
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.frame_size = np.array(frame_size)
        self.conv1_0 = nn.Conv2d(n_channels, num_filter, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding=1)
        self.conv2_0 = nn.Conv2d(num_filter, num_filter*2, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(num_filter*2, num_filter*2, kernel_size=3, stride=1, padding=1)
        self.conv3_0 = nn.Conv2d(num_filter*2, num_filter*4, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(num_filter*4, num_filter*4, kernel_size=3, stride=1, padding=1)
        self.conv4_0 = nn.Conv2d(num_filter*4, num_filter*8, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(num_filter*8, num_filter*8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(num_filter*8, num_filter*8, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(num_filter*8, num_filter*8, kernel_size=3, stride=1, padding=1)
        self.conv4_4 = nn.Conv2d(num_filter*16, num_filter*8, kernel_size=3, stride=1, padding=1)
        self.conv4_5 = nn.Conv2d(num_filter*8, num_filter*8, kernel_size=3, stride=1, padding=1)
        
        self.conv6_0 = nn.Conv2d(num_filter*2, num_filter*4, kernel_size=3, stride=1, padding=1)
        self.conv6_1 = nn.Conv2d(num_filter*4, num_filter*4, kernel_size=3, stride=1, padding=1)

        self.conv7_0 = nn.Conv2d(num_filter, num_filter*2, kernel_size=3, stride=1, padding=1)
        self.conv7_1 = nn.Conv2d(num_filter*2, num_filter*2, kernel_size=3, stride=1, padding=1)

        self.conv8_0 = nn.Conv2d(num_filter//2, num_filter, kernel_size=3, stride=1, padding=1)
        self.conv8_1 = nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(num_filter, num_filter//2, kernel_size=3, stride=1, padding=1)

        self.conv9_0 = nn.Conv2d(num_filter//2, n_classes, kernel_size=1, stride=1)

        self.convt1 = nn.ConvTranspose2d(num_filter*8, num_filter*4, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(num_filter*4)
        self.convt2 = nn.ConvTranspose2d(num_filter*4, num_filter*2, kernel_size=2, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(num_filter*2)
        self.convt3 = nn.ConvTranspose2d(num_filter*2, num_filter, kernel_size=2, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(num_filter)

        if bidirectional:
            self.clstm1 = ConvBLSTM(num_filter*4, num_filter*2, (3, 3), (1,1), 'tanh', list(self.frame_size//4))
            self.clstm2 = ConvBLSTM(num_filter*2, num_filter, (3, 3), (1,1), 'tanh', list(self.frame_size//2))
            self.clstm3 = ConvBLSTM(num_filter, num_filter//2, (3, 3), (1,1), 'tanh', list(self.frame_size))
        else:
            self.clstm1 = ConvLSTM(num_filter*4, num_filter*2, (3, 3), (1,1), 'tanh', list(self.frame_size//4))
            self.clstm2 = ConvLSTM(num_filter*2, num_filter, (3, 3), (1,1), 'tanh', list(self.frame_size//2))
            self.clstm3 = ConvLSTM(num_filter, num_filter//2, (3, 3), (1,1), 'tanh', list(self.frame_size))
        
        
    def forward(self, x):
        N = self.frame_size
        conv1 = self.conv1_0(x)
        conv1 = self.conv1_1(conv1)
        pool1 = self.maxpool(conv1)
        conv2 = self.conv2_0(pool1)
        conv2 = self.conv2_1(conv2)
        pool2 = self.maxpool(conv2)
        conv3 = self.conv3_0(pool2)
        conv3 = self.conv3_1(conv3)
        drop3 = self.dropout(conv3)
        pool3 = self.maxpool(conv3)
        # D1
        conv4 = self.conv4_0(pool3)
        conv4_1 = self.conv4_1(conv4)
        drop4_1 = self.dropout(conv4_1)
        # D2
        conv4_2 = self.conv4_2(drop4_1)
        conv4_2 = self.conv4_3(conv4_2)
        conv4_2 = self.dropout(conv4_2)
        # D3
        merge_dense = torch.cat((conv4_2, drop4_1), 1)
        conv4_3 = self.conv4_4(merge_dense)
        conv4_3 = self.conv4_5(conv4_3)
        drop4_3 = self.dropout(conv4_3)
 
        up6 = self.convt1(drop4_3)
        up6 = self.bn1(up6)
        up6 = nn.ReLU()(up6)

        x1 = drop3.view(-1,1,self.num_filter*4,*(N//4))
        x2 = up6.view(-1,1,self.num_filter*4,*(N//4))

        merge6 = torch.cat((x1, x2), 1)
        merge6 = self.clstm1(merge6)
        
        conv6 = self.conv6_0(merge6)
        conv6 = self.conv6_1(conv6)

        up7 = self.convt2(conv6)
        up7 = self.bn2(up7)
        up7 = nn.ReLU()(up7)

        x1 = conv2.view(-1,1,self.num_filter*2,*(N//2))
        x2 = up7.view(-1,1,self.num_filter*2,*(N//2))
        merge7 = torch.cat((x1, x2), 1)
        merge7 = self.clstm2(merge7)

        conv7 = self.conv7_0(merge7)
        conv7 = self.conv7_1(conv7)

        up8 = self.convt3(conv7)
        up8 = self.bn3(up8)
        up8 = nn.ReLU()(up8)

        x1 = conv1.view(-1,1,self.num_filter,*N)
        x2 = up8.view(-1,1,self.num_filter,*N)
        merge8 = torch.cat((x1, x2), 1)
        merge8 = self.clstm3(merge8)

        conv8 = self.conv8_0(merge8)
        conv8 = self.conv8_1(conv8)
        conv8 = self.conv8_2(conv8)

        conv9 = self.conv9_0(conv8)
        
        return torch.softmax(conv9, dim = 1)
