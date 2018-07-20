
# coding: utf-8

# In[2]:


import numpy as np
import mxnet as mx
import mxnet.ndarray as F
import mxnet.gluon as gluon
from mxnet.gluon import nn
from mxnet import autograd
from mxnet.ndarray import one_hot
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
from skimage import io
import numpy as np
import time


# In[6]:


n_labels = 2
dataset_name = 'TOKYO'
label_colours = np.array([[0, 0, 0], [255, 255, 255]])
lr = 0.01
batch_size = 32

class Net(gluon.Block):
    def __init__(self):
        super(Net, self).__init__()
        ####### encoding layers #######
        self.d0 = nn.Conv2D(in_channels=3, channels=64, kernel_size=3, strides=2, padding=1)
        self.d1 = nn.Conv2D(in_channels=64, channels=128, kernel_size=3, strides=2, padding=1)
        self.d1_norm = nn.BatchNorm(in_channels=128, momentum=0.9)
        self.d2 = nn.Conv2D(in_channels=128, channels=256, kernel_size=3, strides=2, padding=1)
        self.d2_norm = nn.BatchNorm(in_channels=256, momentum=0.9)
        self.d3 = nn.Conv2D(in_channels=256, channels=512, kernel_size=3, strides=2, padding=1)
        self.d3_norm = nn.BatchNorm(in_channels=512, momentum=0.9)
        self.d4 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=2, padding=1)
        self.d4_norm = nn.BatchNorm(in_channels=512, momentum=0.9)
        self.d5 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=2, padding=1)
        self.d5_norm = nn.BatchNorm(in_channels=512, momentum=0.9)
        self.d6 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=2, padding=1)
        self.d6_norm = nn.BatchNorm(in_channels=512, momentum=0.9)
        self.d7 = nn.Conv2D(in_channels=512, channels=512, kernel_size=4, strides=2, padding=1)
        self.d7_norm = nn.BatchNorm(in_channels=512, momentum=0.9)

        ####### decoding layers #######
        self.u0 = nn.Conv2DTranspose(in_channels=512, channels=512, kernel_size=4, strides=2, padding=1)
        self.u0_norm = nn.BatchNorm(in_channels=512, momentum=0.9)
        self.u1 = nn.Conv2DTranspose(in_channels=1024, channels=512, kernel_size=4, strides=2, padding=1)
        self.u1_norm = nn.BatchNorm(in_channels=512, momentum=0.9)
        self.u2 = nn.Conv2DTranspose(in_channels=1024, channels=512, kernel_size=4, strides=2, padding=1)
        self.u2_norm = nn.BatchNorm(in_channels=512, momentum=0.9)
        self.u3 = nn.Conv2DTranspose(in_channels=1024, channels=512, kernel_size=4, strides=2, padding=1)
        self.u3_norm = nn.BatchNorm(in_channels=512, momentum=0.9)
        self.u4 = nn.Conv2DTranspose(in_channels=1024, channels=256, kernel_size=4, strides=2, padding=1)
        self.u4_norm = nn.BatchNorm(in_channels=256, momentum=0.9)
        self.u5 = nn.Conv2DTranspose(in_channels=512, channels=128, kernel_size=4, strides=2, padding=1)
        self.u5_norm = nn.BatchNorm(in_channels=128, momentum=0.9)
        self.u6 = nn.Conv2DTranspose(in_channels=256, channels=64, kernel_size=4, strides=2, padding=1)
        self.u6_norm = nn.BatchNorm(in_channels=64, momentum=0.9)
        self.u7 = nn.Conv2DTranspose(in_channels=128, channels=n_labels, kernel_size=4, strides=2, padding=1)

    def forward(self, x):
        ####### encoding layers #######
        x_d0 = F.LeakyReLU(self.d0(x), slope=0.2)
        x_d1 = F.LeakyReLU(self.d1_norm(self.d1(x_d0)), slope=0.2)
        x_d2 = F.LeakyReLU(self.d2_norm(self.d2(x_d1)), slope=0.2)
        x_d3 = F.LeakyReLU(self.d3_norm(self.d3(x_d2)), slope=0.2)
        x_d4 = F.LeakyReLU(self.d4_norm(self.d4(x_d3)), slope=0.2)
        x_d5 = F.LeakyReLU(self.d5_norm(self.d5(x_d4)), slope=0.2)
        x_d6 = F.LeakyReLU(self.d6_norm(self.d6(x_d5)), slope=0.2)
        x_d7 = F.relu(self.d7_norm(self.d7(x_d6)))

        ####### decoding layers #######
        x = F.relu(F.Dropout(self.u0_norm(self.u0(x_d7))))
        xcat = F.concat(x, x_d6, dim=1)
        x = F.relu(F.Dropout(self.u1_norm(self.u1(xcat))))
        xcat = F.concat(x, x_d5, dim=1)
        x = F.relu(F.Dropout(self.u2_norm(self.u2(xcat))))
        xcat = F.concat(x, x_d4, dim=1)
        x = F.relu(F.Dropout(self.u3_norm(self.u3(xcat))))
        xcat = F.concat(x, x_d3, dim=1)
        x = F.relu(F.Dropout(self.u4_norm(self.u4(xcat))))
        xcat = F.concat(x, x_d2, dim=1)
        x = F.relu(self.u5_norm(self.u5(xcat)))
        xcat = F.concat(x, x_d1, dim=1)
        x = F.relu(self.u6_norm(self.u6(xcat)))
        xcat = F.concat(x, x_d0, dim=1)
        x = self.u7(xcat)
        return x

def test_model(model, test_loader, epoch):
    for i, (images, labels) in enumerate(test_loader):
        outputs = model(images.astype('float32'))
        images = images.transpose((0,2,3,1))
        r, c = 3, 3
        fig, axs = plt.subplots(r, c)
        images = images.astype('int')
        for i in range(c):
            #axs[0, i].imshow(images[i, :, :, :])
            axs[0, i].imshow(visualize(labels[i, :, :]))
            axs[0, i].axis('off')
            axs[1, i].imshow(visualize(labels[i, :, :]))
            axs[1, i].axis('off')
            filled_in = visualize(np.argmax(outputs[i], axis=0))
            axs[2, i].imshow(filled_in)
            axs[2, i].axis('off')
        fig.savefig('./imgs_results/testing_epoch_%s.png' % (epoch))
        plt.close()
        break

def save_model(model, epoch):
    model.save('model_', epoch)

def train_model(model, train_loader, test_loader, num_epochs=10, batch_size=32, save_interval=100):

    # Loss and optimizer
   
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr})
    soft_loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1, sparse_label=False)
    
    epoch = 0
    while (epoch < num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            print('Epoch %s of %s' % (epoch, num_epochs))

            images = images.as_in_context(mx.gpu())
            labels = labels.as_in_context(mx.gpu())
        
            labels = one_hot(labels, 2)
            labels = labels.transpose((0,3,1,2))

            # Forward pass
            with autograd.record():
                outputs = model(images.astype('float32'))
                loss = soft_loss(outputs, labels)
                loss.backward()
            trainer.step(images.shape[0])

#        if (epoch) % save_interval == 0:
#            print ('Epoch [{}/{}], Loss: {}'
#                   .format(epoch, num_epochs, loss.mean()))
            #test_model(model, test_loader, epoch)
            #save_model(model, epoch)

            epoch +=1
            if (epoch > num_epochs):
                break

    #test_model(model, test_loader, epoch)
    #save_model(model, epoch + 1)

def visualize(temp):
    r, g, b = temp.copy(), temp.copy(), temp.copy()
    for l in range(0, n_labels):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = (r / 255.0)  # [:,:,0]
    rgb[:, :, 1] = (g / 255.0)  # [:,:,1]
    rgb[:, :, 2] = (b / 255.0)  # [:,:,2]

    return rgb


def load_data(is_testing=False, batch_size=32, shuffle=True, drop_last=True):
    data_type = "Train"
    batch_images = glob('./%s_data/*' % (data_type))

    images, labels = [], []
    for img_path in batch_images:
        img_name = img_path.split('/')[-1]
        img = io.imread(img_path)
        label = io.imread('./%s_label/%s' % (data_type, img_name))

        if not is_testing:
            transform = np.random.randint(0, 1)
            if transform == 1:
                img = np.fliplr(img)
                label = np.fliplr(label)

        images.append(img)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    images = images.transpose((0, 3, 1, 2))
    
    dataset = gluon.data.ArrayDataset(images, labels)
    dataloader = gluon.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

if __name__ == '__main__':
    time_start = time.time()
    num_epochs = 1000
    save_interval = 50
    batch_size = 32

    net = Net()
    net.initialize(mx.init.Xavier(), ctx=mx.gpu())

    print('Loading training data')
    train_loader = load_data(is_testing=False, batch_size=batch_size, shuffle=True, drop_last=True)
    print('Loading testing data')
    test_loader = load_data(is_testing=True, batch_size=3, shuffle=True, drop_last=True)
    print('Training...')
    train_model(net, train_loader, test_loader, num_epochs, batch_size, save_interval)

    time_elapsed = (time.time() - time_start)
    with open("./time_metrics.txt", "w") as text_file:
        text_file.write("Time Runnning: {} Epochs: {}".format(time_elapsed, num_epochs))

