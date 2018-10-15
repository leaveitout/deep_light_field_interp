#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017.
Contact sbruton[á]tcd.ie.
"""
import torch


class BasicNet(torch.nn.Module):
    def __init__(self, num_in_channels, num_out_channels):
        super(BasicNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=num_in_channels,
                                     out_channels=num_out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        # self.batch_norm1 = torch.nn.InstanceNorm2d(num_out_channels)

        self.conv2 = torch.nn.Conv2d(in_channels=num_out_channels,
                                     out_channels=num_out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        # self.batch_norm2 = torch.nn.InstanceNorm2d(num_out_channels)

        self.conv3 = torch.nn.Conv2d(in_channels=num_out_channels,
                                     out_channels=num_out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        # self.batch_norm3 = torch.nn.InstanceNorm2d(num_out_channels)

        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, input_tensor):
        output = self.conv1(input_tensor)
        # output = self.batch_norm1(output)
        output = self.relu(output)
        output = self.conv2(output)
        # output = self.batch_norm2(output)
        output = self.relu(output)
        output = self.conv3(output)
        # output = self.batch_norm3(output)
        output = self.relu(output)

        return output


class SubNet(torch.nn.Module):
    def __init__(self):
        super(SubNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=64,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.conv2 = torch.nn.Conv2d(in_channels=64,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.conv3 = torch.nn.Conv2d(in_channels=64,
                                     out_channels=51,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.upsample = torch.nn.Upsample(scale_factor=2,
                                          mode='bilinear')

        self.conv4 = torch.nn.Conv2d(in_channels=51,
                                     out_channels=4,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, input_tensor):
        output = self.conv1(input_tensor)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.conv3(output)
        output = self.relu(output)
        output = self.upsample(output)
        output = self.conv4(output)

        return output


class LFRankVolBaseNet(torch.nn.Module):
    def __init__(self):
        super(LFRankVolBaseNet, self).__init__()

        self.conv1 = BasicNet(16, 32)
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = BasicNet(32, 64)
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = BasicNet(64, 128)
        self.pool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv4 = BasicNet(128, 256)
        self.pool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv5 = BasicNet(256, 512)
        self.pool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.deconv5 = BasicNet(512, 512)
        self.upsample5 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(inplace=False)
        )

        self.deconv4 = BasicNet(512, 256)
        self.upsample4 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(inplace=False)
        )

        self.deconv3 = BasicNet(256, 128)
        self.upsample3 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(inplace=False)
        )

        self.deconv2 = BasicNet(128, 64)
        self.upsample2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(inplace=False)
        )

        self.conv1_vol = BasicNet(16, 32)
        self.pool1_vol = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2_vol = BasicNet(32, 64)
        self.pool2_vol = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3_vol = BasicNet(64, 128)
        self.pool3_vol = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv4_vol = BasicNet(128, 256)
        self.pool4_vol = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv5_vol = BasicNet(256, 512)
        self.pool5_vol = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.deconv5_vol = BasicNet(512, 512)
        self.upsample5_vol = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(inplace=False)
        )

        self.deconv4_vol = BasicNet(512, 256)
        self.upsample4_vol = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(inplace=False)
        )

        self.deconv3_vol = BasicNet(256, 128)
        self.upsample3_vol = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(inplace=False)
        )

        self.deconv2_vol = BasicNet(128, 64)
        self.upsample2_vol = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(inplace=False)
        )

        self.final_subnet = SubNet()

    def forward(self, inputs):
        # concat_inputs = torch.reshape(
        #     inputs,
        #     (inputs.shape[0] * inputs.shape[1],
        #      inputs.shape[2],
        #      inputs.shape[3],
        #      inputs.shape[4])
        # )
        # concat_inputs = torch.cat([input_1, input_2, input_3, input_4], 1)
        # print(type(inputs))
        # print(inputs.shape)
        inputs_image = inputs[:, :16]
        vol_rep = inputs[:, 16:]

        conv1_output = self.conv1(inputs_image)
        pool1_output = self.pool1(conv1_output)

        conv2_output = self.conv2(pool1_output)
        pool2_output = self.pool2(conv2_output)

        conv3_output = self.conv3(pool2_output)
        pool3_output = self.pool3(conv3_output)

        conv4_output = self.conv4(pool3_output)
        pool4_output = self.pool4(conv4_output)

        conv5_output = self.conv5(pool4_output)
        pool5_output = self.pool5(conv5_output)

        conv1_vol_output = self.conv1_vol(vol_rep)
        pool1_vol_output = self.pool1_vol(conv1_vol_output)

        conv2_vol_output = self.conv2_vol(pool1_vol_output)
        pool2_vol_output = self.pool2_vol(conv2_vol_output)

        conv3_vol_output = self.conv3_vol(pool2_vol_output)
        pool3_vol_output = self.pool3_vol(conv3_vol_output)

        conv4_vol_output = self.conv4_vol(pool3_vol_output)
        pool4_vol_output = self.pool4_vol(conv4_vol_output)

        conv5_vol_output = self.conv5_vol(pool4_vol_output)
        pool5_vol_output = self.pool5_vol(conv5_vol_output)


        deconv5_output = self.deconv5(pool5_output)
        upsample5_output = self.upsample5(deconv5_output)

        combined_output = upsample5_output + conv5_output

        deconv4_output = self.deconv4(combined_output)
        upsample4_output = self.upsample4(deconv4_output)

        combined_output = upsample4_output + conv4_output

        deconv3_output = self.deconv3(combined_output)
        upsample3_output = self.upsample3(deconv3_output)

        combined_output = upsample3_output + conv3_output

        deconv2_output = self.deconv2(combined_output)
        upsample2_output = self.upsample2(deconv2_output)

        combined_output = upsample2_output + conv2_output

        # Volume net
        deconv5_vol_output = self.deconv5_vol(pool5_vol_output)
        upsample5_vol_output = self.upsample5_vol(deconv5_vol_output)

        combined_vol_output = upsample5_vol_output + conv5_output

        deconv4_vol_output = self.deconv4_vol(combined_vol_output)
        upsample4_vol_output = self.upsample4_vol(deconv4_vol_output)

        combined_vol_output = upsample4_vol_output + conv4_output

        deconv3_vol_output = self.deconv3_vol(combined_vol_output)
        upsample3_vol_output = self.upsample3_vol(deconv3_vol_output)

        combined_vol_output = upsample3_vol_output + conv3_output

        deconv2_vol_output = self.deconv2_vol(combined_vol_output)
        upsample2_vol_output = self.upsample2_vol(deconv2_vol_output)

        combined_vol_output = upsample2_vol_output + conv2_output

        combined_vol_img_output = combined_output + combined_vol_output

        return self.final_subnet(combined_vol_img_output)

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data)
