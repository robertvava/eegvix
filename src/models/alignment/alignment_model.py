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

class EEGEncoder(nn.Module):
    def __init__(self, input_channels = 17, latent_dim=128):
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
    
class TripletMarginLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  
        distance_negative = (anchor - negative).pow(2).sum(1)  
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss
    
    
def get_semi_hard_negatives(anchor_latent, img_latent, margin = 0.2):
    distances = torch.cdist(anchor_latent.unsqueeze(0), img_latent)[0]
    
    positive_distances = torch.norm(anchor_latent - img_latent, dim=1)
    
    mask = (distances > positive_distances.unsqueeze(1)) & (distances < positive_distances.unsqueeze(1) + margin)
    
    if mask.sum() > 0:
        distances_masked = distances * mask.float()
        hardest_negative_index = distances_masked.max(1).indices
        return img_latent[hardest_negative_index]
    else:
        # If no semi-hard negatives, fall back to random sampling or another strategy
        random_idx = torch.randint(0, img_latent.size(0), (1,))
        return img_latent[random_idx]


    
class AlignmentLoss(nn.Module):
    """
    Omitted Triplet Loss. 
    Cosine Similarity: Ensures alignment in direction.
    MSE: Penalizes large deviations.
    MMD: Matches the distributions of the two latent spaces.
    """
    def __init__(self,  lambda_mmd=0.25, lambda_mse=0.25, lambda_cosinesim = 0.25, lambda_triplet = 0.25):
        super(AlignmentLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.mse_loss = nn.MSELoss()
        self.mmd_loss = MMDLoss()
        self.lambda_mmd = lambda_mmd
        self.lambda_mse = lambda_mse
        self.lambda_cosinesim = lambda_cosinesim
        self.lambda_triplet = lambda_triplet

    def forward(self, eeg_latent, image_latent, negative_latent):
        mmd = self.mmd_loss(eeg_latent, image_latent)
        mse = self.mse_loss(eeg_latent, image_latent)
        cosinesim = self.cosine_similarity(eeg_latent, image_latent)
        triplet = 
        total_loss =  self.lambda_mmd * mmd + self.lambda_mse * mse + (1.0 - (self.lambda_cosinesim * cosinesim.mean()))
        return total_loss

def generate_numbers():
    numbers = torch.rand(3)
    
    numbers = torch.sort(numbers).values
    
    result = torch.cat([numbers[0], numbers[1] - numbers[0], numbers[2] - numbers[1], 1 - numbers[2]])
    
    return result

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