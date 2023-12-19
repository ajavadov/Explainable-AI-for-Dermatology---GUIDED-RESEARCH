import numpy as np

#label_list
nevus_list = ['blue nevus','clark nevus','combined nevus','congenital nevus','dermal nevus','recurrent nevus','reed or spitz nevus']
basal_cell_carcinoma_list = ['basal cell carcinoma']
melanoma_list = ['melanoma','melanoma (in situ)','melanoma (less than 0.76 mm)','melanoma (0.76 to 1.5 mm)','melanoma (more than 1.5 mm)','melanoma metastasis']
miscellaneous_list = ['dermatofibroma','lentigo','melanosis','miscellaneous','vascular lesion']
SK_list = ['seborrheic keratosis']
label_list = [nevus_list,basal_cell_carcinoma_list,melanoma_list,miscellaneous_list,SK_list]

#New Group Label List
label_list_2 = [["MELANOMA"], ["NOT MELANOMA"]]



#seven-point
pigment_network_label_list = [['absent'],['typical'],['atypical']]#'typical:1,atypical:2
streaks_label_list = [['absent'],['regular'],['irregular']]#regular:1, irregular:2
pigmentation_label_list = [['absent'],
                      ['diffuse regular','localized regular'],
                      ['localized irregular','diffuse irregular']]#regular:1, irregular:2
regression_structures_label_list = [['absent'],
                               ['blue areas','combinations','white areas']]# present:1
dots_and_globules_label_list = [['absent'],['regular'],['irregular']]#regular:1, irregular:2
blue_whitish_veil_label_list = [['absent'],['present']]#present:1
vascular_structures_label_list = [['absent'],
                             ['within regression','arborizing','comma','hairpin','wreath'],
                             ['linear irregular','dotted']]

#num of each label
num_label = len(label_list)
num_pigment_network_label = len(pigment_network_label_list)
num_streaks_label = len(streaks_label_list)
num_pigmentation_label = len(pigmentation_label_list)
num_regression_structures_label = len(regression_structures_label_list)
num_dots_and_globules_label = len(dots_and_globules_label_list)
num_blue_whitish_veil_label = len(blue_whitish_veil_label_list)
num_vascular_structures_label = len(vascular_structures_label_list)

#for new grouping (melanoma vs. nonmelanoma)
num_label_2=len(label_list_2)

#metadata information
meta_label_categorical_no_nan = {
    'level_of_diagnostic_difficulty' : ['low','medium','high'],
    'elevation' : ['flat','palpable','nodular'],
    'location' : ['back','lower limbs','abdomen','upper limbs','chest',
                    'head neck','acral','buttocks','genital areas'],
    'sex' : ['female','male'],
    'management' : ['excision','clinical follow up','no further examination']
}

level_of_diagnostic_difficulty_label_list = ['low','medium','high']
evaluation_list = ['flat','palpable','nodular']
location_list = ['back','lower limbs','abdomen','upper limbs','chest',
                'head neck','acral','buttocks','genital areas']
sex_list = ['female','male']
management_list = ['excision','clinical follow up','no further examination']

num_level_of_diagnostic_difficulty_label_list = len(level_of_diagnostic_difficulty_label_list)
num_evaluation_list = len(evaluation_list)
num_location_list = len(location_list)
num_sex_list = len(sex_list)
num_management_list = len(management_list)

meta_data_labels = ['level_of_diagnostic_difficulty','elevation','location','sex','management']

meta_data_sizes = [num_level_of_diagnostic_difficulty_label_list,
                    num_evaluation_list,
                    num_location_list,
                    num_sex_list,
                    num_management_list]

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

#for grouping (mel/nonmel):
def encode_label(img_info):#, index_num):
    # Encode the diagnositic label
    diagnosis_label = img_info['diagnosis']#[index_num]
    for index, label in enumerate(label_list_2):
        if diagnosis_label in label:
            diagnosis_index = index
            diagnosis_label_one_hot = to_categorical(diagnosis_index, num_label_2)
        # print(index_num,diagnosis_index,diagnosis_label,diagnosis_label_one_hot)
        else:
            continue
    #Encode the Seven-point label
    # 1
    pigment_network_label = img_info['pigment_network']#[index_num]
    for index, label in enumerate(pigment_network_label_list):
        if pigment_network_label in label:
            pigment_network_index = index
            pigment_network_label_one_hot = to_categorical(pigment_network_index, num_pigment_network_label)
        else:
            continue
    # 2
    streaks_label = img_info['streaks']#[index_num]
    for index, label in enumerate(streaks_label_list):
        if streaks_label in label:
            streaks_index = index
            streaks_label_one_hot = to_categorical(streaks_index, num_streaks_label)
        else:
            continue
    # 3
    pigmentation_label = img_info['pigmentation']#[index_num]
    for index, label in enumerate(pigmentation_label_list):
        if pigmentation_label in label:
            pigmentation_index = index
            pigmentation_label_one_hot = to_categorical(pigmentation_index, num_pigmentation_label)
        else:
            continue
    # 4
    regression_structures_label = img_info['regression_structures']#[index_num]
    for index, label in enumerate(regression_structures_label_list):
        if regression_structures_label in label:
            regression_structures_index = index
            regression_structures_label_one_hot = to_categorical(regression_structures_index,
                                                                 num_regression_structures_label)
        else:
            continue
    # 5
    dots_and_globules_label = img_info['dots_and_globules']#[index_num]
    for index, label in enumerate(dots_and_globules_label_list):
        if dots_and_globules_label in label:
            dots_and_globules_index = index
            dots_and_globules_label_one_hot = to_categorical(dots_and_globules_index, num_dots_and_globules_label)
        else:
            continue
    # 6
    blue_whitish_veil_label = img_info['blue_whitish_veil']#[index_num]
    for index, label in enumerate(blue_whitish_veil_label_list):
        if blue_whitish_veil_label in label:
            blue_whitish_veil_index = index
            blue_whitish_veil_label_one_hot = to_categorical(blue_whitish_veil_index, num_blue_whitish_veil_label)
        else:
            continue
    # 7
    vascular_structures_label = img_info['vascular_structures']#[index_num]
    for index, label in enumerate(vascular_structures_label_list):
        if vascular_structures_label in label:
            vascular_structures_index = index
            vascular_structures_label_one_hot = to_categorical(vascular_structures_index, num_vascular_structures_label)
        else:
            continue

    return diagnosis_index



# def encode_label(img_info):#, index_num):
#     # Encode the diagnositic label
#     diagnosis_label = img_info['diagnosis']#[index_num]
#     for index, label in enumerate(label_list):
#         if diagnosis_label in label:
#             diagnosis_index = index
#             diagnosis_label_one_hot = to_categorical(diagnosis_index, num_label)
#         # print(index_num,diagnosis_index,diagnosis_label,diagnosis_label_one_hot)
#         else:
#             continue
#     #Encode the Seven-point label
#     # 1
#     pigment_network_label = img_info['pigment_network']#[index_num]
#     for index, label in enumerate(pigment_network_label_list):
#         if pigment_network_label in label:
#             pigment_network_index = index
#             pigment_network_label_one_hot = to_categorical(pigment_network_index, num_pigment_network_label)
#         else:
#             continue
#     # 2
#     streaks_label = img_info['streaks']#[index_num]
#     for index, label in enumerate(streaks_label_list):
#         if streaks_label in label:
#             streaks_index = index
#             streaks_label_one_hot = to_categorical(streaks_index, num_streaks_label)
#         else:
#             continue
#     # 3
#     pigmentation_label = img_info['pigmentation']#[index_num]
#     for index, label in enumerate(pigmentation_label_list):
#         if pigmentation_label in label:
#             pigmentation_index = index
#             pigmentation_label_one_hot = to_categorical(pigmentation_index, num_pigmentation_label)
#         else:
#             continue
#     # 4
#     regression_structures_label = img_info['regression_structures']#[index_num]
#     for index, label in enumerate(regression_structures_label_list):
#         if regression_structures_label in label:
#             regression_structures_index = index
#             regression_structures_label_one_hot = to_categorical(regression_structures_index,
#                                                                  num_regression_structures_label)
#         else:
#             continue
#     # 5
#     dots_and_globules_label = img_info['dots_and_globules']#[index_num]
#     for index, label in enumerate(dots_and_globules_label_list):
#         if dots_and_globules_label in label:
#             dots_and_globules_index = index
#             dots_and_globules_label_one_hot = to_categorical(dots_and_globules_index, num_dots_and_globules_label)
#         else:
#             continue
#     # 6
#     blue_whitish_veil_label = img_info['blue_whitish_veil']#[index_num]
#     for index, label in enumerate(blue_whitish_veil_label_list):
#         if blue_whitish_veil_label in label:
#             blue_whitish_veil_index = index
#             blue_whitish_veil_label_one_hot = to_categorical(blue_whitish_veil_index, num_blue_whitish_veil_label)
#         else:
#             continue
#     # 7
#     vascular_structures_label = img_info['vascular_structures']#[index_num]
#     for index, label in enumerate(vascular_structures_label_list):
#         if vascular_structures_label in label:
#             vascular_structures_index = index
#             vascular_structures_label_one_hot = to_categorical(vascular_structures_index, num_vascular_structures_label)
#         else:
#             continue

#     return diagnosis_index
    # return diagnosis_label_one_hot
    # return np.array([diagnosis_index,
    #                  pigment_network_index,
    #                  streaks_index,
    #                  pigmentation_index,
    #                  regression_structures_index,
    #                  dots_and_globules_index,
    #                  blue_whitish_veil_index,
    #                  vascular_structures_index])



def encode_meta_label(img_info):

    level_of_diagnostic_difficulty_label = img_info['level_of_diagnostic_difficulty']#[index_num]
    #print(level_of_diagnostic_difficulty_label)


    level_of_diagnostic_difficulty_label_one_hot = to_categorical(level_of_diagnostic_difficulty_label_list.index(level_of_diagnostic_difficulty_label),
                                                len(level_of_diagnostic_difficulty_label_list))

    for index,label in enumerate(level_of_diagnostic_difficulty_label_list):

        if level_of_diagnostic_difficulty_label in label:
            level_of_diagnostic_difficulty_index = index
            level_of_diagnostic_difficulty_label_one_hot = to_categorical(level_of_diagnostic_difficulty_index,num_level_of_diagnostic_difficulty_label_list)
        else:
            continue

    evaluation_label = img_info['elevation']#[index_num]
    for index,label in enumerate(evaluation_list):
        if evaluation_label in label:
            evaluation_label_index = index
            evaluation_label_one_hot = to_categorical(evaluation_label_index,num_evaluation_list)
        else:
            continue

    sex_label = img_info['sex']#[index_num]
    for index,label in enumerate(sex_list):
        if sex_label in label:
            sex_label_index = index
            sex_label_one_hot = to_categorical(sex_label_index,num_sex_list)
        else:
            continue

    location_label = img_info['location']#[index_num]
    for index,label in enumerate(location_list):
        if location_label in label:
            location_label_index = index
            location_label_one_hot = to_categorical(location_label_index,num_location_list)
        else:
            continue

    management_label = img_info['management']#[index_num]
    for index,label in enumerate(management_list):
        if management_label in label:
            management_label_index = index
            management_label_one_hot = to_categorical(management_label_index,num_management_list)
        else:
            continue

    meta_vector = [
        level_of_diagnostic_difficulty_label_one_hot,
        evaluation_label_one_hot,
        location_label_one_hot,
        sex_label_one_hot,
        management_label_one_hot
    ]
    meta_vector = np.hstack(meta_vector)

    return meta_vector


def encode_meta_label(img_info, use_metadata=0):

    meta_vector = []

    idx = 0

    for key, value in meta_label_categorical_no_nan.items():
        meta_vector.append(to_categorical(value.index(str(img_info[key])), len(value)))
        idx += 1
        if idx == use_metadata:
            break

    meta_vector = np.hstack(meta_vector)

    return meta_vector