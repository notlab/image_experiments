from models.cifar10 import train, evaluate
from models.stylenet import vgg16

#train.train_stock_cifar10()
#evaluate.eval_stock_cifar10()



vgg16.vgg16([], '../checkpoints/stylenet/vgg16_weights.npz', 1)
