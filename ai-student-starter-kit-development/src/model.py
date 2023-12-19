from tabnanny import check
import torch
import torchmetrics
import torchvision
import pytorch_lightning as pl


class SimpleClassifier(pl.LightningModule):
    """
    Classifier Model written in pytorch_lightning

    ...

    Attributes
    ----------
    numm_classes : int
        number of classes (size of the output layer)
    zero_prob : float
        dropout probability that a neuron is set to 0
    class_weights : numpy array
        arrays of weights to be used as class weights for the loss
    learning_rate : float
        learning rate for the optimizer
    weight_decay : float
        weight regularization parameter

    Methods
    -------
    
    """
    def __init__(self, 
                 num_classes=7,
                 model_name='resnet101',
                 class_weights=None,
                 use_optimal_lr=True,
                 learning_rate=5e-5,
                 check_early_stop=False):
        super().__init__()
#find the best model from the literature
#log training accurcy per class
#if training doesnot work make the classification head smaller
        self.num_classes = num_classes
        self.zero_prob = 0.3 #check with lower values
        self.learning_rate = learning_rate
        self.use_optimal_lr = use_optimal_lr
        self.check_early_stop=check_early_stop
        if model_name == 'resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=True, progress=True)
            self.backbone.fc = torch.nn.Sequential(
                torch.nn.Linear(in_features=2048, out_features=512, bias=True),
                torch.nn.BatchNorm1d(512),
                torch.nn.SiLU(),
                torch.nn.Dropout(p=self.zero_prob, inplace=False),
                torch.nn.Linear(in_features=512, out_features=128, bias=True),
                torch.nn.BatchNorm1d(128),
                torch.nn.SiLU(),
                torch.nn.Linear(in_features=128, out_features=num_classes, bias=True))
            print(self.backbone)
#pick images of each class and make gradcam for every classes

#fusion paper resnet50 ~74%
#align with that architecture
#batch 32 lr 3e-5 
            self.lr_dict = [
                {'params':
                    list(self.backbone.conv1.parameters()) +
                    list(self.backbone.bn1.parameters()) +
                    list(self.backbone.layer1.parameters()) +
                    list(self.backbone.layer2.parameters()) +
                    list(self.backbone.layer3.parameters()) +
                    list(self.backbone.layer4.parameters()) +
                    list(self.backbone.avgpool.parameters())
                    , 'lr': 1e-4},
                {'params': self.backbone.fc.parameters(), 'lr': 1e-3},
            ]
        if model_name == 'efficientnet':
            self.backbone = torchvision.models.efficientnet_b0(pretrained=True, progress=True)
            self.backbone.classifier = torch.nn.Sequential(

            torch.nn.Dropout(p=0.3, inplace=False),
            torch.nn.Linear(in_features=1280, out_features=512, bias=True),
            torch.nn.BatchNorm1d(512),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=512, out_features=128, bias=True),
            torch.nn.BatchNorm1d(128),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=128, out_features=num_classes, bias=True))
            print(self.backbone)
            self.lr_dict = [
                {'params':
                    list(self.backbone.features.parameters()) +
                    list(self.backbone.avgpool.parameters())
                    , 'lr': 1e-4},
                {'params': self.backbone.classifier.parameters(), 'lr': 1e-3},
            ]

        if model_name == 'resnet101':
            self.backbone = torchvision.models.resnet101(pretrained=True, progress=True)
            self.backbone.fc = torch.nn.Sequential(
                torch.nn.Linear(in_features=2048, out_features=512, bias=True),
                torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=self.zero_prob, inplace=False),
                torch.nn.Linear(in_features=512, out_features=128, bias=True),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=128, out_features=num_classes, bias=True))
            print(self.backbone)

            self.lr_dict = [
                {'params':
                    list(self.backbone.conv1.parameters()) +
                    list(self.backbone.bn1.parameters()) +
                    list(self.backbone.layer1.parameters()) +
                    list(self.backbone.layer2.parameters()) +
                    list(self.backbone.layer3.parameters()) +
                    list(self.backbone.layer4.parameters()) +
                    list(self.backbone.avgpool.parameters())
                    , 'lr': 1e-4},
                {'params': self.backbone.fc.parameters(), 'lr': 1e-3},
            ]
        if model_name == 'vgg16':
            self.backbone = torchvision.models.vgg16(pretrained=True, progress=True)
            self.backbone.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=self.zero_prob, inplace=False),
                torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=self.zero_prob, inplace=False),
                torch.nn.Linear(in_features=4096, out_features=num_classes, bias=True))
            print(self.backbone)
#swish
            self.lr_dict = [
                {'params':
                    list(self.backbone.features.parameters()) +
                    list(self.backbone.avgpool.parameters())
                    , 'lr': 1e-4},
                {'params': self.backbone.classifier.parameters(), 'lr': 1e-3},
            ]

        #self.val_acc = torchmetrics.Accuracy(num_classes=num_classes, average='macro') #sample
        self.val_acc = torchmetrics.Accuracy(num_classes=num_classes, average='none') #sample
        #to see validation per class
        self.test_acc = torchmetrics.Accuracy(num_classes=num_classes, average='none')

        if class_weights is not None:
            self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
        else:
            self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.backbone(x)

    def configure_optimizers(self):

        if self.use_optimal_lr:
          
            optimizer = torch.optim.Adam(self.lr_dict, 
                                    lr=self.learning_rate)
        else:
            optimizer = torch.optim.Adam(self.backbone.parameters(), 
                                    lr=self.learning_rate)
        return optimizer
    

    def training_step(self, train_batch, batch_idx):
        _, x, _, y = train_batch

        logits = self.forward(x)
        loss = self.loss(logits, y)

        self.log('train_loss',
                loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True)

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        _, x, _, y = val_batch
        logits = self.forward(x)

        loss = self.loss(logits, y)
        self.val_acc(logits, y)
        self.log('val_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def predict_step(self, test_batch, batch_idx):
        _, x, _, y = test_batch
        return self.forward(x), y
    
    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_acc.compute() , on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_epoch_end(self, outputs):
        self.logger.experiment.add_scalars('test_acc', {
            idx : torch.mean(self.val_test_accuracy[idx].compute()) for idx in range(self.num_classes)
        }, global_step=self.global_step)
        #copy to val epoch end to see acc for each class
