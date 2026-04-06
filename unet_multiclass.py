import torch
from torch import nn
from torchvision.transforms import functional,ToTensor

def crop_tensor(enc_feat, target_tensor):
    """
    Crop encoder feature map (enc_feat) to match spatial size of target_tensor.
    Similar to slicing a matrix to center-crop.
    
    Do only if padding is valid(0) must crop the encoder to decoder

    Args:
        enc_feat (Tensor): Encoder feature map (B, C, H_enc, W_enc)
        target_tensor (Tensor): Decoder feature map (B, C, H_target, W_target)
    
    Returns:
        Tensor: Cropped encoder feature map with same HxW as target_tensor
    """
    # target height and width
    H, W = target_tensor.shape[2], target_tensor.shape[3]

    # compute starting indices for center crop
    start_h = (enc_feat.shape[2] - H) // 2
    start_w = (enc_feat.shape[3] - W) // 2

    # slice like a matrix
    cropped = enc_feat[:, :, start_h:start_h + H, start_w:start_w + W]

    return cropped

class DoubleConv(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(DoubleConv,self).__init__()     # older we can write like this with param
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),    # Extracts features (edges, textures, shapes) , Often reduces spatial size - Downsampling 
            ## if padding is 1 no crop need while in upsampling ,But for practical training on 256×256 → use padded U-Net.
            nn.BatchNorm2d(out_channels),   # So we save parameters and slightly speed up computation by setting bias=False.
            nn.ReLU(inplace=True) ,   # it can change the input tensor with directly without creating the new memory or object
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False), # we do next step as a batch norm The convolution bias becomes redundant because BatchNorm can already shift the activation using β.
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
    
    def forward(self,x):
        return self.conv(x)
    
class UNet_Multiclass(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_Multiclass, self).__init__()

        # Encoder (Contracting path)
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)   # shrink by factor of 2

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(256, 512)    
        self.dropout_enc = nn.Dropout2d(0.2)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        self.dropout = nn.Dropout2d(p=0.3) 
        '''
        reduce overfitting -> use dropout 
        [1, 0, 1] --> 1 is alive and 0 is dropout 
        Channel 0 → keep, Channel 1 → ❌ drop, Channel 2 → keep
        Some feature maps are removed → forces network to:
            Learn redundant representations
            Not depend on specific channels
        '''
        # Decoder (Expanding path)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)   # Increases spatial resolution --> Upsampling ,grow by factor of 2  
        self.dropout_dec = nn.Dropout2d(0.1)
        self.conv4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        # Final 1x1 Conv
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        d4 = self.down4(p3)
        d4 = self.dropout_enc(d4)
        p4 = self.pool4(d4)

        # Bottleneck
        bottleneck = self.bottleneck(p4)
        bottleneck = self.dropout(bottleneck)  # kept + scaled => output = input * mask / (1 - p)

        # Decoder with Skip Connections
        up4 = self.up4(bottleneck)   # batch, channels, H, W
        d4_cropped = crop_tensor(d4, up4)  # crop encoder feature map to match
        ''' | Order in `torch.cat`         | Effect on network                                                   |
            | ---------------------------- | ------------------------------------------------------------------- |
            | `[up, skip]` (decoder first) | Standard, matches U-Net paper                                       |
            | `[skip, up]` (encoder first) | Works, but channel order is reversed; weights may learn differently |'''
        up4 = torch.cat([up4, d4_cropped], dim=1)   # concatenate along channels so dim=1 
        up4 = self.conv4(up4)
        up4 = self.dropout_dec(up4)


        up3 = self.up3(up4)
        d3_cropped = crop_tensor(d3, up3)   # padding is 1 so no need to crop we directly apply up3,d3 in concatination
        up3 = torch.cat([up3, d3_cropped], dim=1)
        up3 = self.conv3(up3)

        up2 = self.up2(up3)
        d2_cropped = crop_tensor(d2, up2)
        up2 = torch.cat([up2, d2_cropped], dim=1)
        up2 = self.conv2(up2)

        up1 = self.up1(up2)
        d1_cropped = crop_tensor(d1, up1)
        up1 = torch.cat([up1, d1_cropped], dim=1)
        up1 = self.conv1(up1)

        return self.final(up1)


