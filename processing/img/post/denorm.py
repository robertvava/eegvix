def denormalize(tensor, mean, std):
    
    tensor_copy = tensor.clone().detach()  # make a copy of the tensor
    for t, m, s in zip(tensor_copy, mean, std):
        t.mul_(s).add_(m)  # denormalize
    return tensor_copy

# Normalization mean and std for ImageNet
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
