# Explainable AI for Clinical Decision Support in Dermatology

Continuous advancements and remarkable results in the Artificial Intelligence (AI) sphere inspire a number of domains to incorporate AI techniques into their routines as well. The concept of Explainable AI (XAI) becomes especially vital when the domains involve sensible topics, such as cases where important decisions for humans must be taken. The medical context, in our case the dermatological domain, is a suitable example of a situation when the performance of AI should be justifiable with a fair enough XAI performance, for a number of reasons, such as ethical and legal perspectives. On the other hand, skin lesion diagnosis, which may potentially save lives, especially if there is melanoma involved, is of crucial importance in dermatology. This work first presents common XAI concepts and methodologies with an extensive literature review, followed by an appropriate projection to our problem definition. After that, the work focuses on the training and explanation of skin lesion image classification algorithms via several state-of-the-art explainable AI methods, some of which are pioneering methods for the given context.

## What has been obtained?
Throughout the paper, we have done an extensive literature review and mapped the common conventions and new trends to our task of skin lesion classification. After the literature review, the overall process can be roughly divided into two parts: learning and explaining. In our work, as part of data, two public skin lesion datasets, Seven-Point-Checklist and ISIC-2019 have been used. Then as part of modelling, we relied on transfer learning of ResNet and VGG models, and we documented the retrieved performances. Afterwards, we divided explanation into two parts: explaining deep learning models and explaining any black-box models. For deep learning model explanations, we have used and compared Grad-CAM++ (for the first time with skin lesions, to the best of our knowledge) and Grad-CAM. According to our experiments, Grad-CAM++ was inclined to cover the class regions more than Grad-CAM.
In the next part, we examined LIME, SHAP, and Anchors. Likely to be connected to its stochastic nature, LIME was sometimes generating unintuitive explanations, but was faster than SHAP. SHAP took significantly longer than any other method mentioned, but its explanations may make more sense to domain experts. Anchors were the most promising of the methods among those in Section 2.3.2. Also, due to its scalable and easy-to-use nature, Anchors might outweigh Grad-CAM(++).
It is also important to mention that there were limited time and GPU resources, hence the work could not have been extended for more enhanced results. Undoubtedly, there are many other XAI methods already existing and appearing day-by-day. Other versions of Grad-CAM ( XGrad-CAM), LIME (DLIME,QLIME,MPSLIME), and SHAP (KernelSHAP, BSHAP, TreeExplainer) are the variations where even better answers to our questions may be hidden. The incorporation of various contemporary XAI dashboards, such as InterpretML (available at [32]), DrWhy (available at [33]), DeepExplain [34], IML [35], etc., may be another route to take, particularly to make things simpler for non-IT people.
Last but not least, the datasets ISIC-2019 and Seven-Point-Checklist both include meta-data (patient information). Future research may combine such meta-data with image data to retrieve even better models and explanations using the techniques described.
## Getting started (Conda based approach)

Install miniconda in your home.

Set the `PATH` to point to the miniconda installation folder.
If it's installed in your home dir then it should look like this:
```
export PATH="~/miniconda3/bin:$PATH"
```

```
conda create --name <env_name> python=3.8 -y
conda activate <env_name>
```

## Package installation

```
source activate <env_name>
python3 -m pip install -r requirements.txt
```

## Acknowledgements
Work has been done with the (compute resource as well as scientific) support of the Chair of Computational Imaging and Inverse Problems (PD Dr. Tobias Lasser), Department of Computer Science
Munich Institute of Biomedical Engineering, Technical University of Munich 