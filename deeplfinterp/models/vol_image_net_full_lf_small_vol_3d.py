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

        # self.conv3 = torch.nn.Conv2d(in_channels=num_out_channels,
        #                              out_channels=num_out_channels,
        #                              kernel_size=3,
        #                              stride=1,
        #                              padding=1)

        # self.batch_norm3 = torch.nn.InstanceNorm2d(num_out_channels)

        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, input_tensor):
        output = self.conv1(input_tensor)
        # output = self.batch_norm1(output)
        output = self.relu(output)
        output = self.conv2(output)
        # output = self.batch_norm2(output)
        output = self.relu(output)
        # output = self.conv3(output)
        # # output = self.batch_norm3(output)
        # output = self.relu(output)

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
                                          mode='bilinear',
                                          align_corners=True)

        self.conv4 = torch.nn.Conv2d(in_channels=51,
                                     out_channels=48,
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


class LFVolBaseNetFullLFSmall(torch.nn.Module):
    def __init__(self):
        super(LFVolBaseNetFullLFSmall, self).__init__()
        print("LFVolBaseNetFullLFSmall")

        self.conv1 = BasicNet(16, 32)
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = BasicNet(32, 64)
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = BasicNet(64, 128)
        self.pool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv4 = BasicNet(128, 256)
        self.pool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        # self.conv5 = BasicNet(256, 512)
        # self.pool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        #
        # self.deconv5 = BasicNet(512, 512)
        # self.upsample5 = torch.nn.Sequential(
        #     torch.nn.Upsample(
        #         scale_factor=2,
        #         mode='bilinear',
        #         align_corners=True
        #     ),
        #     torch.nn.Conv2d(
        #         in_channels=512,
        #         out_channels=512,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1
        #     ),
        #     torch.nn.ReLU(inplace=False)
        # )

        self.deconv4 = BasicNet(256, 256)
        self.upsample4 = torch.nn.Sequential(
            torch.nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            ),
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
            torch.nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            ),
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
            torch.nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            ),
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

    def forward(self, inputs, x_mean, x_std_dev):
        # concat_inputs = torch.reshape(
        #     inputs,
        #     (inputs.shape[0] * inputs.shape[1],
        #      inputs.shape[2],
        #      inputs.shape[3],
        #      inputs.shape[4])
        # )
        # concat_inputs = torch.cat([input_1, input_2, input_3, input_4], 1)

        # Need to do this so that we can load the data as np.uint8.
        # It is extremely slow to load a light field as floats.
        # It is also not possible to load as fp16 as pytorch does not have
        # concatenation defined for cpu fp16.
        inputs = inputs.float()
        inputs = inputs - x_mean
        inputs = inputs / x_std_dev

        conv1_output = self.conv1(inputs)
        pool1_output = self.pool1(conv1_output)

        conv2_output = self.conv2(pool1_output)
        pool2_output = self.pool2(conv2_output)

        conv3_output = self.conv3(pool2_output)
        pool3_output = self.pool3(conv3_output)

        conv4_output = self.conv4(pool3_output)
        pool4_output = self.pool4(conv4_output)

        # conv5_output = self.conv5(pool4_output)
        # pool5_output = self.pool5(conv5_output)
        #
        # deconv5_output = self.deconv5(pool5_output)
        # upsample5_output = self.upsample5(deconv5_output)
        #
        # combined_output = upsample5_output + conv5_output

        deconv4_output = self.deconv4(pool4_output)
        upsample4_output = self.upsample4(deconv4_output)

        combined_output = upsample4_output + conv4_output

        deconv3_output = self.deconv3(combined_output)
        upsample3_output = self.upsample3(deconv3_output)

        combined_output = upsample3_output + conv3_output

        deconv2_output = self.deconv2(combined_output)
        upsample2_output = self.upsample2(deconv2_output)

        combined_output = upsample2_output + conv2_output

        outputs = self.final_subnet(combined_output)

        outputs = outputs * x_std_dev

        outputs = outputs + x_mean

        return outputs

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data)
