import matplotlib.pyplot as plt
import random
import numpy as np
import os
import pandas as pd
from .losses import dice_coefficient

titles = ["image", "mask", "prediction", "threshholded prediction", "img vs pred", "mask vs pred"]

def plot_predictions_final(generator, model, slice=13):
  img, mask = next(iter(generator))
  img = img[0:2]
  mask = mask[0:2]
  pred = model.predict(img)

  p_temp = pred
  p_temp[p_temp >= 0.5] = 1
  p_temp[p_temp < 0.5] = 0

  print(dice_coefficient(mask, p_temp))
  print(generator.filenames)


  img = img.reshape(2, 251, 251, 64)
  mask = mask.reshape(2, 251, 251, 64)
  pred = pred.reshape(2, 251, 251, 64)

  fig, axs = plt.subplots(2, 6)
  fig.set_size_inches(19.5, 12.5)

  for i, ax in enumerate(axs.flatten()[:6]):
    ax.set_title(titles[i], fontweight='bold')
    

  for i in range(2):
    slices = []
    slices_p = []
    for j in range(32):
      if mask[i][:,:,j].any():
        slices.append(j)
      sl = pred[i][:,:,j]
      if len(sl[sl > 0.5]) > 0:
        slices_p.append(j)
    if len(slices) > 0: 
      slice = random.choice(slices)
    elif len(slices_p) > 0:
      slice = random.choice(slices_p)
    else:
      slice = 13

    p_temp = pred[i]
    p_temp[p_temp >= 0.5] = 1
    p_temp[p_temp < 0.5] = 0


    axs[i, 0].imshow(img[i][:,:,slice])
    axs[i, 1].imshow(mask[i][:,:,slice])
    axs[i, 2].imshow(pred[i][:,:,slice])
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    axs[i, 3].imshow(pred[i][:,:,slice])
    axs[i, 4].imshow(img[i][:,:,slice], cmap='gray')
    axs[i, 4].imshow(pred[i][:,:,slice], cmap='jet', alpha=0.5)
    axs[i, 5].imshow(mask[i][:,:,slice], cmap='gray')
    axs[i, 5].imshow(pred[i][:,:,slice], cmap='PuBuGn', alpha=0.5)


def plot_predictions_train(generator, model, slice=13):
  img, mask = next(iter(generator))
  pred = model.predict(img)

  p_temp = pred
  p_temp[p_temp >= 0.5] = 1
  p_temp[p_temp < 0.5] = 0

  print(dice_coefficient(mask, p_temp))
  print(generator.filenames)


  img = img.reshape(4, 251, 251, 32)
  mask = mask.reshape(4, 251, 251, 32)
  pred = pred.reshape(4, 251, 251, 32)

  fig, axs = plt.subplots(4, 6)
  fig.set_size_inches(19.5, 12.5)

  for i, ax in enumerate(axs.flatten()[:6]):
    #ax.axis("off")
    ax.set_title(titles[i], fontweight='bold')
    

  for i in range(4):
    slices = []
    slices_p = []
    for j in range(32):
      if mask[i][:,:,j].any():
        slices.append(j)
      sl = pred[i][:,:,j]
      if len(sl[sl > 0.5]) > 0:
        slices_p.append(j)
    if len(slices) > 0: 
      slice = random.choice(slices)
    elif len(slices_p) > 0:
      slice = random.choice(slices_p)
    else:
      slice = 13

    p_temp = pred[i]
    p_temp[p_temp >= 0.5] = 1
    p_temp[p_temp < 0.5] = 0


    axs[i, 0].imshow(img[i][:,:,slice])
    axs[i, 1].imshow(mask[i][:,:,slice])
    axs[i, 2].imshow(pred[i][:,:,slice])
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    axs[i, 3].imshow(pred[i][:,:,slice])
    axs[i, 4].imshow(img[i][:,:,slice], cmap='gray')
    axs[i, 4].imshow(pred[i][:,:,slice], cmap='jet', alpha=0.5)
    axs[i, 5].imshow(mask[i][:,:,slice], cmap='gray')
    axs[i, 5].imshow(pred[i][:,:,slice], cmap='PuBuGn', alpha=0.5)

def predict_final(u_net, images):

    images_split = np.zeros((len(images) * 2, 251, 251, 32, 1))
    for i in range(len(images)):
      images_split[i*2,:,:,0:16] = images[i,:,:,0:16]
      images_split[i*2,:,:,16:32] = images[i,:,:,42:58]
      images_split[i*2+1,:,:,0:32] = images[i,:,:,13:45]
    
    predictions_split = u_net.predict(images_split)
    predictions = np.zeros((len(images) , 251, 251, 58, 1))
    for i in range(len(predictions)):
      predictions[i,:,:,0:13] = predictions_split[i*2,:,:,0:13]
      predictions[i,:,:,13] = predictions_split[i*2,:,:,13] * 0.8 + predictions_split[i*2+1,:,:,0] * 0.2
      predictions[i,:,:,14] = predictions_split[i*2,:,:,14] * 0.5 + predictions_split[i*2+1,:,:,1] * 0.5
      predictions[i,:,:,15] = predictions_split[i*2,:,:,15] * 0.2 + predictions_split[i*2+1,:,:,2] * 0.8
      predictions[i,:,:,16:42] = predictions_split[i*2+1,:,:,3:29]
      predictions[i,:,:,42] = predictions_split[i*2,:,:,16] * 0.2 + predictions_split[i*2+1,:,:,29] * 0.8
      predictions[i,:,:,43] = predictions_split[i*2,:,:,17] * 0.5 + predictions_split[i*2+1,:,:,30] * 0.5
      predictions[i,:,:,44] = predictions_split[i*2,:,:,18] * 0.8 + predictions_split[i*2+1,:,:,31] * 0.2
      predictions[i,:,:,45:58] = predictions_split[i*2,:,:,19:32]

    return predictions

def dice_test_statistics(test_complete_generator, model):
    imgs, masks = next(iter(test_complete_generator))
    dice = []
    for i in range(int(imgs.shape[0] / 3)):
        im = imgs[i*3:i*3+3]
        ma = masks[i*3:i*3+3]
        predictions = model.predict(im)
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0
        dice.append(dice_coefficient(ma, predictions))
    print(dice)
    print("test gen dice " + str(np.average(dice)))

def dice_train_statistics(train_complete_generator, model):
    imgs, masks = next(iter(train_complete_generator))
    dice = []
    for i in range(int(imgs.shape[0] / 4)):
        im = imgs[i*4:i*4+4]
        ma = masks[i*4:i*4+4]
        predictions = model.predict(im)
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0
        dice.append(dice_coefficient(ma, predictions))
    print(dice)
    print("train gen dice " + str(np.average(dice)))

def dice_final_statistics(final_complete_generator, model):
    imgs, masks = next(iter(final_complete_generator))
    dice = []
    for i in range(int(imgs.shape[0] / 3)):
        im = imgs[i*3:i*3+3]
        ma = masks[i*3:i*3+3]
        predictions = predict_final(model, im)
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0
        predictions = predictions[:,:,:,:]
        dice.append(dice_coefficient(ma, predictions))
    print(dice)
    print("train gen dice " + str(np.average(dice)))


def subtype_statistics(generator, model, root_dir, img_dir_3d, mask_dir_3d):
    subtypes = {'Intraventricular': [0, 0.0], 'Intraparenchymal': [0, 0.0], 'Subarachnoid': [0, 0.0], 'Epidural': [0, 0.0], 'Subdural': [0, 0.0], 'No_Hemorrhage': [0, 0.0]}

    diagnoses = pd.read_csv(root_dir + '/hemorrhage_diagnosis_raw_ct.csv')
    print(generator.filenames)
    for i in range(len(generator.filenames)):

        img_path = os.path.join(img_dir_3d, generator.filenames[i])
        img = np.load(img_path)
        mask_path = os.path.join(mask_dir_3d, generator.filenames[i])
        mask = np.load(mask_path)

        diagnosis = diagnoses.loc[diagnoses['PatientNumber'] == int(generator.filenames[i][:-8])]
        prediction = predict_final(model, img[np.newaxis,:,:,:,np.newaxis])
        dice = dice_coefficient(mask[np.newaxis,:,:,:,np.newaxis], prediction)
        type_found = False
        for subtype in subtypes.keys():
            if diagnosis[subtype].any() and subtype != "No_Hemorrhage":
                subtypes[subtype][0] = subtypes[subtype][0] + 1
                subtypes[subtype][1] = subtypes[subtype][1] + dice   
                type_found = True
            elif (not type_found) and subtype == "No_Hemorrhage":
                subtypes[subtype][0] = subtypes[subtype][0] + 1
                subtypes[subtype][1] = subtypes[subtype][1] + dice   

    print(subtypes)

    print("Dice by type")
    for sub in subtypes.keys():
      if subtypes[sub][0] > 0:
        avg = (subtypes[sub][1] / subtypes[sub][0]).numpy()
        print(sub + ": " + str(avg))

  