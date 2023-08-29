class EEGEncoder(nn.Module):
    def __init__(self, input_channels, latent_dim=128):
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

class EEGDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(EEGDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.rec_dim = 1280

        channels_list = [17, 64, 128, 128]
        self.reconstruct = nn.Linear(self.latent_dim, self.rec_dim)

        self.layers = nn.ModuleList()
        self.shapes = []
        kernel_size = 4
        stride = 2
        padding = 0


        for i in range(len(channels_list)-1, 0, -1):
            self.layers.append(nn.Sequential(
                nn.ConvTranspose1d(channels_list[i], channels_list[i-1], kernel_size, stride, padding),
                nn.BatchNorm1d(channels_list[i-1])
            ))
            current_channels = channels_list[i-1]

        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose1d(17, 17, kernel_size = 5, stride = 1, padding = 1))
        self.layers.append(nn.ConvTranspose1d(17, 17, kernel_size = 5, stride = 1, padding = 1))
        self.layers.append(nn.ConvTranspose1d(17, 17, kernel_size = 5, stride = 1, padding = 1))
        self.layers.append(nn.Sigmoid())
        self.decode = nn.Sequential(*self.layers)

    def compute_output_shape(self, input_size, kernel_size, stride, padding):
        output_size = ((input_size - 1) * stride) - 2 * padding + kernel_size
        return output_size

    def forward(self, x):
        x = self.reconstruct(x)
        x = x.view(x.size(0), 128, -1)
        x = self.decode(x)
        return x


# ===============================================================================================================


class EEGAutoEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(EEGAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = EEGEncoder(17, self.latent_dim)
        self.decoder = EEGDecoder(self.latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class EEGAELoss(nn.Module):
    ''' This loss will align the original and reconstructed eegs on three aspects:

        1. Amplitude preservation preserved by MSELoss
        2. Robustness to outliers preserved by HuberLoss
        3. Shape preservation using CosineSimilarity for each of the 17 channels individually.
    '''
    def __init__(self, lambda_mse = 0.7, lambda_huber=0.05, lambda_cosine = 0.25):
        super(EEGAELoss, self).__init__()
        self.lambda_mse = lambda_mse
        self.lambda_huber = lambda_huber
        self.lambda_cosine = lambda_cosine

        self.mseloss = nn.MSELoss()
        self.uber_loss = nn.SmoothL1Loss()
        self.cosine_sim = nn.CosineSimilarity(dim=2)

    def forward(self, original, reconstructed):
        loss1 = self.mseloss(original, reconstructed)
        loss2 = self.uber_loss(original, reconstructed)
        loss3 = self.cosine_sim(original, reconstructed)
        loss3 = 1.0 - loss3.mean(dim = 1)


        return (self.lambda_mse * loss1 + self.lambda_huber * loss2 + self.lambda_cosine * loss3.mean()) * 10