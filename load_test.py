import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import scipy
import glob
import cv2
from skimage import io
from skimage import transform


if __name__ == '__main__':
    test_path = '20170626-TEST-O2'
    test_sideA_mother_path = os.path.join(test_path, 'Board_SideA_01.tif')
    test_sideB_mother_path = os.path.join(test_path, 'Board_SideB_01.tif')
    test_sideA_path = os.path.join(test_path, 'SideA')
    test_sideB_path = os.path.join(test_path, 'SideB')
    test_sideA_OK_path = os.path.join(test_sideA_path, 'Image', 'OK')
    test_sideA_NG_path = os.path.join(test_sideA_path, 'Image', 'C+NG')
    test_sideB_OK_path = os.path.join(test_sideB_path, 'Image', 'OK')
    test_sideB_NG_path = os.path.join(test_sideB_path, 'Image', 'C+NG')
    test_sideA_mother = io.imread(test_sideA_mother_path)
    test_sideB_mother = io.imread(test_sideB_mother_path)

    test_images = []
    test_images_context = []
    test_targets = []

    # Read sideA defect txt
    defect_sideA_txt_path = test_sideA_path + '/Defect_00*.txt'
    defect_sideA_txt = glob.glob(defect_sideA_txt_path)
    #dict = {}
    i = 1
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

                test_defect = transform.resize(
                    test_sideA_mother[int(start_defect[n][6])/2: int(start_defect[n][8])/2, int(start_defect[n][5])/2: int(start_defect[n][7])/2, :], (32, 32, 3))
                #plt.imshow(test_defect)
                #plt.show()
                test_images.append(test_defect)
                # context
                test_sideA_context_row = int(start_defect[n][8]) / 2 - int(start_defect[n][6]) / 2
                test_sideA_context_col = int(start_defect[n][7]) / 2 - int(start_defect[n][5]) / 2
                test_defect_context = transform.resize(
                    test_sideA_mother[int(start_defect[n][4]) / 2 - test_sideA_context_row: 
                                       int(start_defect[n][4]) / 2 + test_sideA_context_row,
                                       int(start_defect[n][3]) / 2 - test_sideA_context_col: 
                                       int(start_defect[n][3]) / 2 + test_sideA_context_col, :], (32, 32, 3))
                #test_images.append(test_sideA_mother[start_defect[n][6]: start_defect[n][8], start_defect[n][5]: start_defect[n][7], :])
                #plt.imshow(test_defect_context)
                #plt.show()
                test_images_context.append(test_defect_context)
                possible_ok_path = os.path.join(test_sideA_OK_path, small_board_id_str + '.bmp')
                print(possible_ok_path)
                if os.path.exists(possible_ok_path):
                    # OK defects
                    #print('ok')
                    test_targets.append(np.array([1]))
                else:
                    # NG defects
                    #print('ng')
                    test_targets.append(np.array([0]))


            #dict[i] = start_defect
            i = i+1
            #print(start_defect)

    # Finsih SideA
    '''test_images_arr = np.array(test_images)
    test_images_context_arr = np.array(test_images_context)
    test_targets_arr = np.array(test_targets)'''

    # Read sideB defect txt
    defect_sideB_txt_path = test_sideB_path + '/Defect_00*.txt'
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
                #print(test_sideA_mother[int(start_defect[n][6])/2: int(start_defect[n][8])/2, int(start_defect[n][5])/2: int(start_defect[n][7])/2, :].shape)
                # defect image
                test_defect = transform.resize(
                    test_sideB_mother[int(start_defect[n][6])/2: int(start_defect[n][8])/2, int(start_defect[n][5])/2: int(start_defect[n][7])/2, :], (32, 32, 3))
                test_images.append(test_defect)
                # context
                test_sideA_context_row = int(start_defect[n][8]) / 2 - int(start_defect[n][6]) / 2
                test_sideA_context_col = int(start_defect[n][7]) / 2 - int(start_defect[n][5]) / 2
                test_defect_context = transform.resize(
                    test_sideA_mother[int(start_defect[n][4]) / 2 - test_sideA_context_row: 
                                       int(start_defect[n][4]) / 2 + test_sideA_context_row,
                                       int(start_defect[n][3]) / 2 - test_sideA_context_col: 
                                       int(start_defect[n][3]) / 2 + test_sideA_context_col, :], (32, 32, 3))
                #test_images.append(test_sideA_mother[start_defect[n][6]: start_defect[n][8], start_defect[n][5]: start_defect[n][7], :])
                test_images_context.append(test_defect_context)
                possible_ok_path = os.path.join(test_sideB_OK_path, small_board_id_str + '.bmp')
                print(possible_ok_path)
                if os.path.exists(possible_ok_path):
                    # OK defects
                    #print('ok')
                    test_targets.append(np.array([1]))
                else:
                    # NG defects
                    #print('ng')
                    test_targets.append(np.array([0]))


            #dict[i] = start_defect
            i = i+1
            #print(start_defect)

    # Finsih SideA and SideB
    test_images_arr = np.array(test_images, dtype=np.float32)
    np.save('test_images',test_images_arr)
    #test_images_arr.tofile('test_images')

    #print test_images_arr.shape
    test_images_context_arr = np.array(test_images_context, dtype=np.float32)
    np.save('test_images_context', test_images_context_arr)
    #test_images_context_arr.tofile('test_images_context')

    test_targets_arr = np.array(test_targets, dtype=np.int16)
    np.save('test_targets', test_targets_arr)
    #test_targets_arr.tofile('test_targets')
    #print test_targets_arr.shape
