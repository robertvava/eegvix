from torchvision import transforms

def transformation():

    return  transforms.Compose([
                transforms.Resize((244,244)),
                transforms.ToTensor(), 
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                ])