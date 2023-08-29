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
#         x_fourier = self.fourier(x)
#         x_combined = torch.cat([x, x_fourier], dim=1)
        enc = self.encoder(x)
        x_fc = self.fc(enc)
        return x_fc
    
    
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        c_hid = 64

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, c_hid, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c_hid, c_hid * 2, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.LayerNorm([16, 16]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c_hid * 2, c_hid * 4, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.LayerNorm([8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c_hid * 4, c_hid * 8, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.LayerNorm([4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c_hid * 8, 1, kernel_size=4, stride=1, padding=0),  # 4x4 -> 1x1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


    
class EEGEncoder(nn.Module):
    def __init__(self, input_channels= 17, latent_dim=128):
        super(EEGEncoder, self).__init__()

        self.latent_dim = latent_dim
        channels_list = [input_channels, 64, 128, 128]
        self.layers = nn.ModuleList()
        self.shapes = []

        current_channels = input_channels
        kernel_size = 4
        stride = 2
        padding = 0

        for i in range(len(channels_list)-1):
            if i == len(channels_list):
                self.layers.append(nn.Conv1d(current_channels, current_channels, kernel_size = 1))
                break
            current_channels = channels_list[i+1]
            self.layers.append(nn.Sequential(
                nn.Conv1d(channels_list[i], channels_list[i+1], kernel_size, stride, padding),
                nn.BatchNorm1d(channels_list[i+1])
            ))
            self.shapes.append(self.compute_output_shape(channels_list[i+1], kernel_size, stride, padding))
        self.layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*self.layers)
        self.fc = nn.Linear(1280, self.latent_dim)

    def compute_output_shape(self, input_size, kernel_size, stride, padding):
        output_size = ((input_size - kernel_size + 2 * padding) / stride) + 1
        return output_size

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x_fc = self.fc(x)
        return x_fc
    
class ImageDecoder(nn.Module):
    def __init__(self, latent_dim = 32, resolution = 64):
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