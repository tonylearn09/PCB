import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import random
import scipy
import glob
import cv2
from skimage import io
from skimage import transform


if __name__ == '__main__':
    train_path = '20170613-54-o1'
    train_sideA_mother_path = os.path.join(train_path, 'Board_SideA_01.tif')
    train_sideB_mother_path = os.path.join(train_path, 'Board_SideB_01.tif')
    train_sideA_path = os.path.join(train_path, 'SideA')
    train_sideB_path = os.path.join(train_path, 'SideB')
    train_sideA_OK_path = os.path.join(train_sideA_path, 'Image', 'OK')
    train_sideA_NG_path = os.path.join(train_sideA_path, 'Image', 'C+NG')
    train_sideB_OK_path = os.path.join(train_sideB_path, 'Image', 'OK')
    train_sideB_NG_path = os.path.join(train_sideB_path, 'Image', 'C+NG')
    train_sideA_mother = io.imread(train_sideA_mother_path)
    train_sideB_mother = io.imread(train_sideB_mother_path)

    train_images = []
    train_images_context = []
    train_targets = []

    # Read sideA defect txt
    defect_sideA_txt_path = train_sideA_path + '/Defect_00*.txt'
    defect_sideA_txt = glob.glob(defect_sideA_txt_path)
    #dict = {}
    i = 1
    num_ok = 0
    num_ng = 0
    for file in sorted(defect_sideA_txt):
        with open(file, 'r') as f:
            start_defect = f.readlines()[11:]
            start_defect = [s.split() for s in start_defect]
            prefix = '00' + '0'* (3 - len(str(i))) + str(i)
            print(prefix)
            for n in range(len(start_defect) - 1):
                small_board_id_str = prefix
                small_board_postfix = start_defect[n][0]
                if len(small_board_postfix) < 5:
                    small_board_id_str += ('_' + '0'*(5-len(small_board_postfix)) + small_board_postfix)
                else:
                    small_board_id_str += ('_' + small_board_postfix)
                #print small_board_id_str
                # The actual defect image
                #print(train_sideA_mother[int(start_defect[n][6])/2: int(start_defect[n][8])/2, int(start_defect[n][5])/2: int(start_defect[n][7])/2, :].shape)
                # defect image
                train_defect = transform.resize(
                    train_sideA_mother[int(start_defect[n][6])/2: int(start_defect[n][8])/2, int(start_defect[n][5])/2: int(start_defect[n][7])/2, :], (32, 32, 3))
                #plt.imshow(train_defect)
                #plt.show()
                train_images.append(train_defect)
                #train_images.append(cv2.flip(train_defect, 0))
                #train_images.append(cv2.flip(train_defect, 1))
                # context
                train_sideA_context_row = int(start_defect[n][8]) / 2 - int(start_defect[n][6]) / 2
                train_sideA_context_col = int(start_defect[n][7]) / 2 - int(start_defect[n][5]) / 2
                #print(train_sideA_context_row)
                #print(train_sideA_context_col)
                '''train_defect_context = transform.resize(
                    train_sideA_mother[max(0, int(start_defect[n][4]) / 2 - train_sideA_context_row): 
                                       min(train_sideA_mother.shape[0], int(start_defect[n][4]) / 2 + train_sideA_context_row),
                                       max(0, int(start_defect[n][3]) / 2 - train_sideA_context_col): 
                                       min(train_sideA_mother.shape[1], int(start_defect[n][3]) / 2 + train_sideA_context_col), :], (32, 32, 3))'''
                '''print(i)
                print(int(start_defect[n][4]) / 2 - train_sideA_context_row)
                print(int(start_defect[n][4]) / 2 + train_sideA_context_row)
                print(int(start_defect[n][3]) / 2 - train_sideA_context_col)
                print(int(start_defect[n][3]) / 2 + train_sideA_context_col)'''
                train_defect_context = transform.resize(
                    train_sideA_mother[int(start_defect[n][4]) / 2 - train_sideA_context_row: 
                                       int(start_defect[n][4]) / 2 + train_sideA_context_row,
                                       int(start_defect[n][3]) / 2 - train_sideA_context_col: 
                                       int(start_defect[n][3]) / 2 + train_sideA_context_col, :], (32, 32, 3))
                #train_images.append(train_sideA_mother[start_defect[n][6]: start_defect[n][8], start_defect[n][5]: start_defect[n][7], :])
                #plt.imshow(train_defect_context)
                #plt.show()
                train_images_context.append(train_defect_context)
                #train_images_context.append(cv2.flip(train_defect_context, 0))
                #train_images_context.append(cv2.flip(train_defect_context, 1))
                possible_ok_path = os.path.join(train_sideA_OK_path, small_board_id_str + '.bmp')
                print(possible_ok_path)
                if os.path.exists(possible_ok_path):
                    # OK defects
                    #print('ok')
                    train_targets.append(np.array([1]))
                    #train_targets.append(np.array([1]))
                    #train_targets.append(np.array([1]))
                    num_ok += 1
                else:
                    # NG defects
                    # More generate 6 images
                    train_images.append(train_defect)
                    train_images_context.append(train_defect_context)
                    train_images.append(scipy.ndimage.interpolation.shift(train_defect, [random.randrange(-2, 2), random.randrange(-2, 2), 0]))
                    train_images_context.append(scipy.ndimage.interpolation.shift(train_defect_context, [random.randrange(-2, 2), random.randrange(-2, 2), 0]))
                    train_images.append(scipy.ndimage.interpolation.shift(train_defect, [random.randrange(-2, 2), random.randrange(-2, 2), 0]))
                    train_images_context.append(scipy.ndimage.interpolation.shift(train_defect_context, [random.randrange(-2, 2), random.randrange(-2, 2), 0]))
                    train_images.append(scipy.ndimage.interpolation.rotate(train_defect, random.randrange(-10, 10), reshape=False))
                    train_images_context.append(scipy.ndimage.interpolation.rotate(train_defect_context, random.randrange(-10, 10), reshape=False))
                    train_images.append(scipy.ndimage.interpolation.rotate(train_defect, random.randrange(-10, 10), reshape=False))
                    train_images_context.append(scipy.ndimage.interpolation.rotate(train_defect_context, random.randrange(-10, 10), reshape=False))
                    train_images.append(cv2.flip(train_defect, 0))
                    train_images_context.append(cv2.flip(train_defect_context, 0))
                    train_images.append(cv2.flip(train_defect, 1))
                    train_images_context.append(cv2.flip(train_defect_context, 1))
                    #print('ng')
                    train_targets.append(np.array([0]))
                    train_targets.append(np.array([0]))
                    train_targets.append(np.array([0]))
                    train_targets.append(np.array([0]))
                    train_targets.append(np.array([0]))
                    train_targets.append(np.array([0]))
                    train_targets.append(np.array([0]))
                    train_targets.append(np.array([0]))
                    #train_targets.append(np.array([0]))
                    #train_targets.append(np.array([0]))
                    num_ng += 8


            #dict[i] = start_defect
            i = i+1
            #print(start_defect)

    # Finsih SideA
    '''train_images_arr = np.array(train_images)
    train_images_context_arr = np.array(train_images_context)
    train_targets_arr = np.array(train_targets)'''

    # Read sideB defect txt
    defect_sideB_txt_path = train_sideB_path + '/Defect_00*.txt'
    defect_sideB_txt = glob.glob(defect_sideB_txt_path)
    #dict = {}
    i = 1
    for file in sorted(defect_sideB_txt):
        with open(file, 'r') as f:
            start_defect = f.readlines()[11:]
            start_defect = [s.split() for s in start_defect]
            prefix = '00' + '0'* (3 - len(str(i))) + str(i)
            #print(prefix)
            for n in range(len(start_defect) - 1):
                small_board_id_str = prefix
                small_board_postfix = start_defect[n][0]
                if len(small_board_postfix) < 5:
                    small_board_id_str += ('_' + '0'*(5-len(small_board_postfix)) + small_board_postfix)
                else:
                    small_board_id_str += ('_' + small_board_postfix)
                #print small_board_id_str
                # The actual defect image
                #print(train_sideA_mother[int(start_defect[n][6])/2: int(start_defect[n][8])/2, int(start_defect[n][5])/2: int(start_defect[n][7])/2, :].shape)
                # defect image
                train_defect = transform.resize(
                    train_sideB_mother[int(start_defect[n][6])/2: int(start_defect[n][8])/2, int(start_defect[n][5])/2: int(start_defect[n][7])/2, :], (32, 32, 3))
                train_images.append(train_defect)
                #train_images.append(cv2.flip(train_defect, 0))
                #train_images.append(cv2.flip(train_defect, 1))
                # context
                train_sideA_context_row = int(start_defect[n][8]) / 2 - int(start_defect[n][6]) / 2
                train_sideA_context_col = int(start_defect[n][7]) / 2 - int(start_defect[n][5]) / 2
                train_defect_context = transform.resize(
                    train_sideA_mother[int(start_defect[n][4]) / 2 - train_sideA_context_row: 
                                       int(start_defect[n][4]) / 2 + train_sideA_context_row,
                                       int(start_defect[n][3]) / 2 - train_sideA_context_col: 
                                       int(start_defect[n][3]) / 2 + train_sideA_context_col, :], (32, 32, 3))
                '''train_defect_context = transform.resize(
                    train_sideB_mother[min(0, int(start_defect[n][4]) / 2 - train_sideB_context_row): 
                                       max(train_sideB_mother.shape[0], int(start_defect[n][4]) / 2 + train_sideB_context_row),
                                       min(0, int(start_defect[n][3]) / 2 - train_sideB_context_col): 
                                       max(train_sideB_mother.shape[1], int(start_defect[n][3]) / 2 + train_sideB_context_col), :], (32, 32, 3))'''
                #train_images.append(train_sideA_mother[start_defect[n][6]: start_defect[n][8], start_defect[n][5]: start_defect[n][7], :])
                train_images_context.append(train_defect_context)
                #train_images_context.append(cv2.flip(train_defect_context, 0))
                #train_images_context.append(cv2.flip(train_defect_context, 1))
                possible_ok_path = os.path.join(train_sideB_OK_path, small_board_id_str + '.bmp')
                print(possible_ok_path)
                if os.path.exists(possible_ok_path):
                    # OK defects
                    #print('ok')
                    train_targets.append(np.array([1]))
                    #train_targets.append(np.array([1]))
                    #train_targets.append(np.array([1]))
                    num_ok += 1
                else:
                    # NG defects
                    train_images.append(train_defect)
                    train_images_context.append(train_defect_context)
                    train_images.append(scipy.ndimage.interpolation.shift(train_defect, [random.randrange(-2, 2), random.randrange(-2, 2), 0]))
                    train_images_context.append(scipy.ndimage.interpolation.shift(train_defect_context, [random.randrange(-2, 2), random.randrange(-2, 2), 0]))
                    train_images.append(scipy.ndimage.interpolation.shift(train_defect, [random.randrange(-2, 2), random.randrange(-2, 2), 0]))
                    train_images_context.append(scipy.ndimage.interpolation.shift(train_defect_context, [random.randrange(-2, 2), random.randrange(-2, 2), 0]))
                    train_images.append(scipy.ndimage.interpolation.rotate(train_defect, random.randrange(-10, 10), reshape=False))
                    train_images_context.append(scipy.ndimage.interpolation.rotate(train_defect_context, random.randrange(-10, 10), reshape=False))
                    train_images.append(scipy.ndimage.interpolation.rotate(train_defect, random.randrange(-10, 10), reshape=False))
                    train_images_context.append(scipy.ndimage.interpolation.rotate(train_defect_context, random.randrange(-10, 10), reshape=False))
                    train_images.append(cv2.flip(train_defect, 0))
                    train_images_context.append(cv2.flip(train_defect_context, 0))
                    train_images.append(cv2.flip(train_defect, 1))
                    train_images_context.append(cv2.flip(train_defect_context, 1))
                    #print('ng')
                    train_targets.append(np.array([0]))
                    train_targets.append(np.array([0]))
                    train_targets.append(np.array([0]))
                    train_targets.append(np.array([0]))
                    train_targets.append(np.array([0]))
                    train_targets.append(np.array([0]))
                    train_targets.append(np.array([0]))
                    train_targets.append(np.array([0]))
                    #train_targets.append(np.array([0]))
                    #train_targets.append(np.array([0]))
                    num_ng += 8

            #dict[i] = start_defect
            i = i+1
            #print(start_defect)

    print(num_ok)
    print(num_ng)

    # Finsih SideA and SideB
    train_images_arr = np.array(train_images, dtype=np.float32)
    np.save('train_images',train_images_arr)
    #train_images_arr.tofile('train_images')

    #print train_images_arr.shape
    train_images_context_arr = np.array(train_images_context, dtype=np.float32)
    np.save('train_images_context', train_images_context_arr)
    #train_images_context_arr.tofile('train_images_context')

    train_targets_arr = np.array(train_targets, dtype=np.int16)
    np.save('train_targets', train_targets_arr)
    #train_targets_arr.tofile('train_targets')
    #print train_targets_arr.shape
