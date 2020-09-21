from torchvision.datasets import VOCDetection

root = './test'

VOCDetection(root, year='2012', image_set='val', download=True)
VOCDetection(root, year='2007', image_set='val', download=True)