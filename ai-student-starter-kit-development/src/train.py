import os
from argparse import ArgumentParser
from gc import callbacks
from subprocess import call
import PIL
import random
import matplotlib.image as im
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

#from pytorch_gradcam import gradcam
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.svm import SVC
from torchsummary import summary
from PIL import Image
from matplotlib import cm
from datasets.seven_point_dataset import SevenPointDataset
from datasets.isic_2019 import ISIC2019Dataset

from gradcam import *
from model import SimpleClassifier
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

def test(batch_size=16,
          model_name='resnet101',
          learning_rate=1e-3,
          max_epochs=100):
 
    dataset = SevenPointDataset('/space/derma-data/seven_point', batch_size=batch_size, normalize_weights=True, use_metadata=0)
    #ISIC dataset
    # dataset = ISIC2019Dataset('/space/derma-data/isic-2019', batch_size=batch_size, normalize_weights=True, md_choice='all', sampling_rate=-1)

    dataset.setup()
    early_stopping=False #flag
    #optimizer=adam
    model = SimpleClassifier(model_name='resnet50',learning_rate=learning_rate,
                            class_weights=list(dataset.class_weights),
                            num_classes=dataset.num_classes,
                            check_early_stop=early_stopping,
                            #optimizer=optimizer                            
                            )

    #checkpoint = torch.load('saved_model.ckpt') #vgg16, 1e-3, batch 32, 350epoch
    
    checkpoint = torch.load('/u/home/javadov/ai_kit/ai-student-starter-kit/checkpoints/melornotresnet50_50epochs.ckpt')
    model.load_state_dict(checkpoint['state_dict'])
    if early_stopping:
        callbacks=[EarlyStopping(min_delta=0.001,monitor="val_loss", patience=10,strict=False, verbose=True,mode="min")]
    else:
        callbacks=None
    print(model)
    trainer = pl.Trainer(devices=1,
                         accelerator='gpu',
                         logger=False,
                         callbacks=callbacks,
                         checkpoint_callback=False,
                        )

    predictions = trainer.predict(model, dataset.test_dataloader())


    features, labels = zip(*predictions)

    ''' 
        zip():
        Makes an iterator that aggregates elements from each of the iterables.
        Merges by appending the elements one to the other, by grouping elements with the same index.

        about *:
        It is an unpacking operator.
        about **:
        It is an unpacking operator for dictionaries.
    '''
    features = torch.vstack(features)

    '''  
        torch.vstack(): 
        Stacks tensors in sequence vertically (row wise).
        
    '''

    # features is a batch_size X output_size tensor
    labels = torch.cat(labels)

    print("\n\n############\nCNN Classification\n############")

    clf_report = pd.DataFrame(classification_report(labels, torch.argmax(features, dim=1), output_dict=True, zero_division=0.0))
    print("Classification report:\n{}".format(clf_report))


 
    (input_original, input, _, label) = next(iter(dataset.test_dataloader()))
    output = model.forward(input)

    fig, ax = plt.subplots(4,4)
    idx = 0
    for i in range(4):
        for j in range(4):
            img = input_original.cpu().detach().numpy()[idx,:,:,:]
            ax[i,j].imshow(np.transpose(img, (1,2,0)))
            ax[i,j].set_title(f'Pred {torch.argmax(output[idx,:])} : Real {labels[idx]}')
            idx += 1
    fig.set_size_inches(25,25)


    # plt.savefig('test_results', bbox_inches='tight', dpi=300) #vgg16, 1e-3, batch 32, 350epoch
    plt.savefig('/u/home/javadov/ai_kit/ai-student-starter-kit/images/test_results_melornotresnet50', bbox_inches='tight', dpi=300)

    disp = ConfusionMatrixDisplay.from_predictions(labels, torch.argmax(features, dim=1), normalize='true', display_labels=list(range(features.shape[1])))
    font = {'family' : 'Dejavu Sans',
        'weight' : 'bold',
        'size'   : 25}

    plt.rc('font', **font)
    disp.plot()
    fig = plt.gcf()
    fig.set_size_inches(25,25)
    # plt.savefig('confusion_matrix', bbox_inches='tight', dpi=300) #vgg16, 1e-3, batch 32, 350epoch
    plt.savefig('/u/home/javadov/ai_kit/ai-student-starter-kit/images/confusion_matrix_melornotresnet50', bbox_inches='tight', dpi=300)

    #GRAD_CAM
    iterator= iter(dataset.test_dataloader())
    (input_original, input, _, label) = next(iterator)
    images=[]
    fig, ax = plt.subplots(5,5)

    idx = 0
    for i in range(5):
        for j in range(5): 
            img = np.transpose(input_original.cpu().detach().numpy()[idx,:,:,:],(1,2,0))*255
            img = img.astype(np.uint8)
            pil_img = Image.fromarray(img).convert('RGB')
            torch_img = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])(pil_img)
            normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
            gradcam = GradCAM.from_config(model_type='resnet', arch=model, layer_name='backbone.layer4')
            # extended_input = (input[idx])[None,:]
            mask, logit = gradcam(normed_torch_img, class_idx=None)

            # make heatmap from mask and synthesize saliency map using heatmap and img
            heatmap, cam_result = visualize_cam(mask, torch_img)
            # heatmap, cam_result = visualize_cam(mask, np.transpose(torch.unsqueeze(input[idx].permute(1,2,0),dim=-1)))

            images=[]
            images.extend([torch_img, heatmap, cam_result])
            # images.extend([np.transpose(torch.unsqueeze(input[idx].permute(1,2,0),dim=-1)).cpu(), heatmap, cam_result])

            grid_image = make_grid(images, nrow=1)
            result=transforms.ToPILImage()(grid_image)
            result = result.save("/u/home/javadov/ai_kit/ai-student-starter-kit/images/melornot_gradcam.jpg")
            grd_img= im.imread("/u/home/javadov/ai_kit/ai-student-starter-kit/images/melornot_gradcam.jpg")
            ax[i,j].imshow(grd_img)
            # # create an axes on the right side of ax. The width of cax will be 5%
            # # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            # divider = make_axes_locatable(ax[i,j])
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # color_bar = plt.colorbar(im, cax=cax)
            # cb.ax[i,j].yaxis.set_tick_params(labelright=False)
            divider = make_axes_locatable(ax[i,j])
            cax = divider.append_axes('bottom', size='20%', pad=0.2)
            pos = ax[i,j].imshow(grd_img, cmap='jet', interpolation='none')
            fig.colorbar(pos, cax=cax,orientation='horizontal')
            #ax.set_xticks(ax.get_xticks()[::2])
            # labels = [item.get_text() for item in cax.get_xticklabels()]
            # labels[0] = 'Testing'
            ax[i,j].tick_params(
            axis='both',          # changes apply to the both-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
            cax.tick_params(
            axis='both',          # changes apply to the both-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
            # cax.set_xticklabels(labels)
            cax.set_xlabel('Relevance Scale',fontdict={'fontsize':16})
            cax.margins(0.8)
            # ax[i,j].imshow()
            logitList= logit.tolist()
            flat_logit= [x for xs in logitList for x in xs]
            logitList= [ '%.0f' % elem for elem in flat_logit ]
            # ax[i,j].set_title(f'Pred {torch.argmax(output[idx,:])} : Real {labels[idx]}, \n Logit {logitList}',fontdict=
            # {'fontsize':14},rotation=0)
            ax[i,j].set_title(f'Pred {torch.argmax(output[idx,:])} : Real {labels[idx]}',fontdict=
            {'fontsize':18},rotation=0,pad=16)
    #tighter spacing #add color bar #to see only red marks and no blue on the overlay
    #rise - main step
    #gradcam + binary classification (grouping of classes) - have a look at google doc to properly group the classes.
    #like melamoma/nevus
    #from your report choose a couple of images with the gradcam
    #and put them on a page nicely, with some accuracy and sensitivy info
    #about the model so Theo can show them to Melia when he meets with her (make up a Report)
    #read how to assess AI Models in medical domain
    #use slurm 


            idx+=1
    fig.set_size_inches(25,25)


    # plt.savefig('test_results', bbox_inches='tight', dpi=300) #vgg16, 1e-3, batch 32, 350epoch
    plt.savefig('/u/home/javadov/ai_kit/ai-student-starter-kit/images/test_results_melornot_collage_resnet50', bbox_inches='tight', dpi=300)

    #GRAD_CAM++
    iterator= iter(dataset.test_dataloader())
    (input_original, input, _, label) = next(iterator)
    images=[]
    fig, ax = plt.subplots(5,5)
    idx = 0
    for i in range(5):
        for j in range(5):
            img = np.transpose(input_original.cpu().detach().numpy()[idx,:,:,:],(1,2,0))*255
            img = img.astype(np.uint8)
            pil_img = Image.fromarray(img).convert('RGB')
            torch_img = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])(pil_img)
            normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
            gradcam = GradCAMpp.from_config(model_type='resnet', arch=model, layer_name='backbone.layer4')
            mask, logit = gradcam(normed_torch_img, class_idx=None) 
            
                # make heatmap from mask and synthesize saliency map using heatmap and img
            heatmap, cam_result = visualize_cam(mask, torch_img)
            images=[]
            images.extend([torch_img.cpu(), heatmap, cam_result])
            grid_image = make_grid(images, nrow=1)
            result=transforms.ToPILImage()(grid_image)
            result = result.save("/u/home/javadov/ai_kit/ai-student-starter-kit/images/gradcamPP_melornotresnet50.jpg")
            grd_img= im.imread("/u/home/javadov/ai_kit/ai-student-starter-kit/images/gradcamPP_melornotresnet50.jpg")
            ax[i,j].imshow(grd_img)

            logitList= logit.tolist()
            flat_logit= [x for xs in logitList for x in xs]
            logitList= [ '%.2f' % elem for elem in flat_logit ]
            ax[i,j].set_title(f'Pred {torch.argmax(output[idx,:])} : Real {labels[idx]}, \n Logit {logitList}',pad=40)

            idx+=1
    fig.set_size_inches(25,25)


    # plt.savefig('test_results', bbox_inches='tight', dpi=300) #vgg16, 1e-3, batch 32, 350epoch
    plt.savefig('/u/home/javadov/ai_kit/ai-student-starter-kit/images/test_results_PlusPlus_collage_melornotresnet50', bbox_inches='tight', dpi=300)

    # grid_image = make_grid(images, nrow=4)
    # result=transforms.ToPILImage()(grid_image)
    # result = result.save("gradcam_resnet5.jpg") #for resnet5.ckpt

    # # lbl=label.cpu().detach()[idx]
    #         model.eval()
    #         ax[i,j].imshow(img)
    #         idx += 1

    

    # print((input_original, input, _, label))
    # output = model.forward(input)

    # fig= plt.figure()
    # idx = 0
    
    # img = np.transpose(input_original.cpu().detach().numpy()[idx,:,:,:],(1,2,0))

    # fig.set_size_inches(25,25)
    # plt.imshow(img)

    # # # apply_gradcam(img,lbl,model)
    # # # image2  = np.add(np.multiply(image.numpy(), np.array(norm_std)) ,np.array(norm_mean))
    # # # print("True Class: ",label[0].cpu())

    # plt.savefig('sample_resnet5_for_gradcam.png', bbox_inches='tight',dpi=300)

    # # ############## just for testing #####################
    # # alexnet = models.alexnet(pretrained=True)           #
    # # vgg = models.vgg16(pretrained=True)                 #
    # # resnet = models.resnet101(pretrained=True)          #
    # # densenet = models.densenet161(pretrained=True)      #
    # # squeezenet = models.squeezenet1_1(pretrained=True)  #
    # # #####################################################

    # pil_img = PIL.Image.open('sample_resnet5_for_gradcam.png').convert('RGB')
    # torch_img = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.ToTensor()
    # ])(pil_img)
    # normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
    # # lbl=label.cpu().detach()[idx]
    # model.eval()



    # gradcam = GradCAM.from_config(model_type='resnet', arch=model, layer_name='backbone.layer4')
    # mask, logit = gradcam(normed_torch_img, class_idx=None) 
    #     # make heatmap from mask and synthesize saliency map using heatmap and img
    # heatmap, cam_result = visualize_cam(mask, torch_img)
    # images.extend([torch_img.cpu(), heatmap, cam_result])


    # grid_image = make_grid(images, nrow=1)
    # result=transforms.ToPILImage()(grid_image)
    # result = result.save("gradcam_resnet5.jpg") #for resnet5.ckpt


    # (input_original, input, _, label) = next(iterator)
    # print("torch_img",torch_img)
    # print("type of img",type(torch_img))
  



# def apply_gradcam(image, label, model):
#     """
#     Generate Grad-CAM
#     """

#     model = model


#     model.eval()

#     # The layers
#     # target_layers = ["layer4"]
#     target_layers= ["backbone.layer4"]
#     target_class = label

#     # Images
#     # image = image.unsqueeze(0)
#     gcam = GradCAM(model=model)
#     probs, ids = gcam.forward(np.array(image))
#     ids_ = torch.LongTensor([[target_class]] * len(image))
#     gcam.backward(ids=ids_)

#     for target_layer in target_layers:
#         print("Generating Grad-CAM @{}".format(target_layer))

#         # Grad-CAM
#         regions = gcam.generate(target_layer=target_layer)
#         for j in range(len(image)):
#             print(
#                 "\t#{}: {} ({:.5f})".format(
#                     j, target_class, float(probs[ids == target_class])
#                 )
#             )
            
#             gcam=regions[j, 0]
#             plt.imshow(gcam.cpu())
#             plt.show()
#             fig = plt.gcf()
#             fig.set_size_inches(25,25)
#             # plt.savefig('confusion_matrix', bbox_inches='tight', dpi=300) #vgg16, 1e-3, batch 32, 350epoch
#             plt.savefig('gradcam', dpi=300)
    

            



def train(batch_size=16,
          model_name='resnet101',
          learning_rate=1e-3,
          max_epochs=100):

    dataset = SevenPointDataset('/space/derma-data/seven_point', batch_size=batch_size, normalize_weights=True, use_metadata=0)
    # dataset = ISIC2019Dataset('/space/derma-data/isic-2019', batch_size=batch_size, normalize_weights=True, md_choice='all', sampling_rate=-1)

    dataset.setup()

    print(dataset.class_weights)
    early_stopping=False #flag
    model = SimpleClassifier(model_name='resnet50',learning_rate=learning_rate,
                            class_weights=list(dataset.class_weights),
                            num_classes=dataset.num_classes,
                            check_early_stop=early_stopping)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_acc',
                                                       save_top_k=3,
                                                       filename= model_name + '-val_acc_max-{val_acc:.2f}' + '-epoch-{epoch}',
                                                       mode='max',
                                                       dirpath='/space/javadov/starter-kit/') ######
#efficient net
    if early_stopping:
        callbacks=[checkpoint_callback,EarlyStopping(min_delta=0.001,monitor="val_loss", patience=10,strict=False, verbose=True,mode="min")]
    else:
        callbacks=[checkpoint_callback]
#nvidia smi
    trainer = pl.Trainer(devices=1,
                         accelerator='gpu',
                         callbacks=callbacks,
                         max_epochs=max_epochs,
                        )

    trainer.fit(model, dataset)

    #trainer.save_checkpoint('saved_model.ckpt') vgg16, 1e-3, batch 32, 350epoch
    # trainer.save_checkpoint('saved_model2.ckpt')  #resnet50, 10 epoch  
    trainer.save_checkpoint('/u/home/javadov/ai_kit/ai-student-starter-kit/checkpoints/melornotresnet50_50epochs.ckpt') #resnet50, max100epoch
def main():
    #seed
    def set_seed(seed=15):
        
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] =str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    set_seed()
    parser = ArgumentParser()
    parser.add_argument('-t', '--train', type=str, dest='do_training', default='train', help="Do training (train) or testing (test)")
    parser.add_argument('-g', '--gradcam', type=str, dest='do_gradcam', default='gradcam', help="Do gradcam")
    parser.add_argument('-e', '--max_epochs', type=int, dest='max_epochs', default=10, help="Number of training epochs")
    parser.add_argument('-b', '--batch_size', type=int, dest='batch_size', default=16, help="Batch size")
    parser.add_argument('-lr', '--learning_rate', type=float, dest='learning_rate', default=1e-3, help="Learning rate")
    # parser.add_argument('-i', '--image', type=int, dest='image', default='idk', help="image")
    # parser.add_argument('-b', '--label', type=int, dest='label', default='melanoma', help="label")
    # parser.add_argument('-lr', '--model', type=float, dest='model', default='resnet50', help="model")


    args = parser.parse_args()

    (train if args.do_training == 'train' else test)(batch_size=args.batch_size, max_epochs=args.max_epochs, learning_rate=args.learning_rate)
    #(apply_gradcam if args.do_gradcam == 'gradcam')(image=args.image, label=args.label, model=args.model)

    
if __name__ == "__main__":
    main()
    # test(16,10,1e-4)

    # apply_gradcam()
