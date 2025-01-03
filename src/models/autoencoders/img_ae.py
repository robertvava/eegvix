class ImageEncoder(nn.Module):
    def __init__(self, input_channels = 3, latent_dim=128):
        super(ImageEncoder, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        c_hid = 64

        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, c_hid,  kernel_size=3, padding=1, stride=2),  # 64x64 -> 32x32
            nn.BatchNorm2d(c_hid),
            nn.GELU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hid),
            nn.Conv2d(c_hid, c_hid*2, kernel_size=3, padding=1, stride=2), # 32x32 -> 16x16
            nn.BatchNorm2d(c_hid*2),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*c_hid),
            nn.GELU(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            nn.Dropout(0.25),
            nn.Conv2d(2*c_hid,2*c_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*c_hid),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            nn.GELU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(2*16*c_hid, latent_dim)

    def forward(self, x):
        enc = self.encoder(x)
        x_fc = self.fc(enc)
        return x_fc


class ImageDecoder(nn.Module):
    def __init__(self, latent_dim, resolution = 64):
        super(ImageDecoder, self).__init__()
        self.resolution = resolution
        c_hid = 64
        self.fc = nn.Linear(latent_dim, 2*16*c_hid)

        self.conv1 = nn.Sequential(nn.ConvTranspose2d(c_hid*2, c_hid*2,kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.GELU(),
            nn.BatchNorm2d(c_hid*2),
            nn.ConvTranspose2d(c_hid*2, c_hid*2, kernel_size=3, padding = 1)
            )
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.BatchNorm2d(c_hid),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.GELU()
            )
        self.conv3 = nn.Sequential(nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.GELU(),
            nn.BatchNorm2d(c_hid),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1)
            )
        self.conv4 = nn.ConvTranspose2d(c_hid, 3, kernel_size=3, output_padding=1, padding=1, stride=2)
        self.tanh = nn.Tanh() 


    def forward(self, x):
        x_fc = self.fc(x)
        x_fc = x_fc.reshape(-1, 128, 4, 4)
        conv1 = self.conv1(x_fc)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        x = self.tanh(conv4)
        return x

class ReconLoss(nn.Module):
    def __init__(self, alpha=0.2):
        super(ReconLoss, self).__init__()
        self.alpha = alpha

    def forward(self, outputs, targets):
        mse_loss = F.mse_loss(outputs, targets)
        ssim_value = self.ssim(outputs, targets)
        loss = self.alpha * mse_loss + (1 - self.alpha) * (1 - ((ssim_value+1)/2))
        return loss
    
    @staticmethod
    def ssim(img1, img2, window_size=11, size_average=True, C1=0.01**2, C2=0.03**2):
        """
        Compute the SSIM between two batches of images.

        Args:
        - img1: First batch of images
        - img2: Second batch of images
        - window_size: Size of the window used to compute SSIM
        - size_average: If True, average the SSIM over the image; otherwise, return SSIM map
        - C1, C2: Stability constants
        """

        window = torch.ones((1, 1, window_size, window_size))
        window /= window.sum()

        window = window.repeat(img1.shape[1], 1, 1, 1)

        window = window.to(img1.device)

        def _ssim(img1, img2, window, C1, C2):
            mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
            mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

            return ssim_map

        ssim_map = _ssim(img1, img2, window, C1, C2)

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map
