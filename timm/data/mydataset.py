import torch
from PIL import Image

from torchvision import models, transforms


####################################################################### 
## Dataset Init
class MyDataset(torch.utils.data.Dataset):
    """
    Attributes
    ----------
    img_list : 리스트
        이미지의 경로를 저장한 리스트
    label_list : 리스트
        label의 경로를 저장한 리스트
    phase : 'train' or 'val'
        학습 또는 테스트 여부 결정
    transform : object
        전처리 클래스의 인스턴스
    """

    def __init__(self, is_train, root, split, batch_size, repeats):
        self.img_list = img_list
        self.label_list = label_list
        self.phase = phase  # train 또는 val을 지정
        self.transform = transform  # 이미지의 변형

    def __len__(self):
        '''이미지의 갯수를 반환'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        전처리한 이미지 및 라벨 return
        '''
        # img_path = self.img_list[index]
        # img = Image.open(img_path).convert('RGB')
        img = self.img_list[index]
        
        transformed_img = self.transform(img, self.phase)
        label = self.label_list[index]
        
        return transformed_img, label
    

####################################################################### 
## Transform
class MyTransform():
    """
    Attributes
    ----------
    resize : int
        Transform 수행 후 변경될 width / height 값.
    mean : (R, G, B)
        각 색상 채널의 평균값.
    std : (R, G, B)
        각 색상 채널의 표준 편차.
    """

    def __init__(self, resize=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    (resize, resize), scale=(0.5, 1.0)),  
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),  # 텐서로 변환
                transforms.Normalize(mean, std)  # 표준화
            ]),
            'val': transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),  # 텐서로 변환
                transforms.Normalize(mean, std)  # 표준화
            ])
        }

    def __call__(self, img, phase='train'):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            전처리 모드를 지정.
        """
        return self.data_transform[phase](img)