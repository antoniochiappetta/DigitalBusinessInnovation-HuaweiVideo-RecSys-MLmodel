#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import scipy.sparse as sps
import numpy as np
import zipfile

from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import downloadFromURL



class URM5Fold_WarmCold_Reader(DataReader):

    DATASET_SUBFOLDER_URM = "MMTF14K/Final_MMTF14K_Web/Data/"
    DATASET_SUBFOLDER_ICM = "MMTF14K/Final_MMTF14K_Web"

    AVAILABLE_URMS = ["URM_train_1","URM_train_2","URM_train_3","URM_train_4","URM_train_5","URM_test_1","URM_test_2","URM_test_3","URM_test_4","URM_test_5"]
    AVAILABLE_ICM = ["ICM_genre","ICM_year","ICM_AVF","ICM_deep","ICM_ivector_1","ICM_ivector_2","ICM_ivector_3","ICM_ivector_4","ICM_ivector_5"]

    IS_IMPLICIT = True


    def __init__(self):
        super(URM5Fold_WarmCold_Reader, self).__init__()



    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER_URM


    def get_loaded_URM_names(self):
        return self.AVAILABLE_URMS.copy()


    def _load_from_original_file(self):

        # ICM

        print("URM5Fold_WarmCold_Reader: Loading original data ICM")

        zipFile_path_ICM = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER_ICM

        dataFile_ICM = zipfile.ZipFile(zipFile_path_ICM + ".zip")

        # METADATA
        print("Generating ICMs for Metadata features")

        ICM_genre_path = dataFile_ICM.extract("Final_MMTF14K_Web/Metadata/GenreFeatures.csv", path=zipFile_path_ICM + "decompressed")
        ICM_year_path = dataFile_ICM.extract("Final_MMTF14K_Web/Metadata/YearOfProd.csv", path=zipFile_path_ICM + "decompressed")

        self.tokenToFeatureMapper_ICM_genre = {}
        self.tokenToFeatureMapper_ICM_year = {}

        self.ICM_genre, self.tokenToFeatureMapper_ICM_genre, _ = self._loadICM(ICM_genre_path, header=True, separator=",")
        self.ICM_year, self.tokenToFeatureMapper_ICM_year, _ = self._loadICM(ICM_year_path, header=True, separator=",")

        # AUDIO
        print("Generating ICMs for Audio features")

        # ICM_BLF_path = dataFile_ICM.extract("Final_MMTF14K_Web/Audio/BLF/All/blf_sim_matrix.csv", path=zipFile_path_ICM + "decompressed")
        ICM_ivector_1_path = dataFile_ICM.extract("Final_MMTF14K_Web/Audio/ivector/IVec_splitItem_fold_1_gmm_512_tvDim_100.csv", path=zipFile_path_ICM + "decompressed")
        ICM_ivector_2_path = dataFile_ICM.extract("Final_MMTF14K_Web/Audio/ivector/IVec_splitItem_fold_2_gmm_512_tvDim_100.csv", path=zipFile_path_ICM + "decompressed")
        ICM_ivector_3_path = dataFile_ICM.extract("Final_MMTF14K_Web/Audio/ivector/IVec_splitItem_fold_3_gmm_512_tvDim_100.csv", path=zipFile_path_ICM + "decompressed")
        ICM_ivector_4_path = dataFile_ICM.extract("Final_MMTF14K_Web/Audio/ivector/IVec_splitItem_fold_4_gmm_512_tvDim_100.csv", path=zipFile_path_ICM + "decompressed")
        ICM_ivector_5_path = dataFile_ICM.extract("Final_MMTF14K_Web/Audio/ivector/IVec_splitItem_fold_5_gmm_512_tvDim_100.csv", path=zipFile_path_ICM + "decompressed")

        self.tokenToFeatureMapper_ICM_BLF = {}
        self.tokenToFeatureMapper_ICM_ivector_1 = {}
        self.tokenToFeatureMapper_ICM_ivector_2 = {}
        self.tokenToFeatureMapper_ICM_ivector_3 = {}
        self.tokenToFeatureMapper_ICM_ivector_4 = {}
        self.tokenToFeatureMapper_ICM_ivector_5 = {}

        # self.ICM_BLF, self.tokenToFeatureMapper_ICM_BLF, _ = self._loadICM(ICM_BLF_path, header=True, separator=",")
        self.ICM_ivector_1, self.tokenToFeatureMapper_ICM_ivector_1, self.item_original_ID_to_index_1 = self._loadICM(ICM_ivector_1_path, header=True, separator=",")
        self.ICM_ivector_2, self.tokenToFeatureMapper_ICM_ivector_2, self.item_original_ID_to_index_2 = self._loadICM(ICM_ivector_2_path, header=True, separator=",")
        self.ICM_ivector_3, self.tokenToFeatureMapper_ICM_ivector_3, self.item_original_ID_to_index_3 = self._loadICM(ICM_ivector_3_path, header=True, separator=",")
        self.ICM_ivector_4, self.tokenToFeatureMapper_ICM_ivector_4, self.item_original_ID_to_index_4 = self._loadICM(ICM_ivector_4_path, header=True, separator=",")
        self.ICM_ivector_5, self.tokenToFeatureMapper_ICM_ivector_5, self.item_original_ID_to_index_5 = self._loadICM(ICM_ivector_5_path, header=True, separator=",")

        # VISUAL
        print("Generating ICMs for Visual features")
        
        ICM_AVF_path = dataFile_ICM.extract("Final_MMTF14K_Web/Visual/Aesthetic_features/Avg/AestheticFeatures-AVG-All.csv", path=zipFile_path_ICM + "decompressed")
        ICM_deep_path = dataFile_ICM.extract("Final_MMTF14K_Web/Visual/AlexNet_features/Avg/AlexNetFeatures -AVG-fc7.csv", path=zipFile_path_ICM + "decompressed")
        
        self.tokenToFeatureMapper_ICM_AVF = {}
        self.tokenToFeatureMapper_ICM_deep = {}

        self.ICM_AVF, self.tokenToFeatureMapper_ICM_AVF, _ = self._loadICM(ICM_AVF_path, header=True, separator=",")
        self.ICM_deep, self.tokenToFeatureMapper_ICM_deep, _ = self._loadICM(ICM_deep_path, header=True, separator=",")

        print("URM5Fold_WarmCold_Reader: cleaning temporary files ICM")

        import shutil

        shutil.rmtree(zipFile_path_ICM + "decompressed", ignore_errors=True)

        print("URM5Fold_WarmCold_Reader: saving ICM")

        # URM

        print("URM5Fold_WarmCold_Reader: Loading original data URM")

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER_URM
            
        dataFile = zipfile.ZipFile(zipFile_path + "itemids_splitted_5foldCV.zip")

        URM_train_1_path = dataFile.extract("itemids_splitted_5foldCV/useritemIds_train_itemsplit_fold1of5.csv", path=zipFile_path + "decompressed")
        URM_train_2_path = dataFile.extract("itemids_splitted_5foldCV/useritemIds_train_itemsplit_fold2of5.csv", path=zipFile_path + "decompressed")
        URM_train_3_path = dataFile.extract("itemids_splitted_5foldCV/useritemIds_train_itemsplit_fold3of5.csv", path=zipFile_path + "decompressed")
        URM_train_4_path = dataFile.extract("itemids_splitted_5foldCV/useritemIds_train_itemsplit_fold4of5.csv", path=zipFile_path + "decompressed")
        URM_train_5_path = dataFile.extract("itemids_splitted_5foldCV/useritemIds_train_itemsplit_fold5of5.csv", path=zipFile_path + "decompressed")
        URM_test_1_path = dataFile.extract("itemids_splitted_5foldCV/useritemIds_test_itemsplit_fold1of5.csv", path=zipFile_path + "decompressed")
        URM_test_2_path = dataFile.extract("itemids_splitted_5foldCV/useritemIds_test_itemsplit_fold2of5.csv", path=zipFile_path + "decompressed")
        URM_test_3_path = dataFile.extract("itemids_splitted_5foldCV/useritemIds_test_itemsplit_fold3of5.csv", path=zipFile_path + "decompressed")
        URM_test_4_path = dataFile.extract("itemids_splitted_5foldCV/useritemIds_test_itemsplit_fold4of5.csv", path=zipFile_path + "decompressed")
        URM_test_5_path = dataFile.extract("itemids_splitted_5foldCV/useritemIds_test_itemsplit_fold5of5.csv", path=zipFile_path + "decompressed")

        print("URM5Fold_WarmCold_Reader: loading URM")

        self.URM_train_1, _, _ = self._loadURM(URM_train_1_path, separator=",", header = True, if_new_user = "add", if_new_item = "ignore", col_mapper = self.item_original_ID_to_index_1)
        self.URM_train_2, _, _ = self._loadURM(URM_train_2_path, separator=",", header = True, if_new_user = "add", if_new_item = "ignore", col_mapper = self.item_original_ID_to_index_2)
        self.URM_train_3, _, _ = self._loadURM(URM_train_3_path, separator=",", header = True, if_new_user = "add", if_new_item = "ignore", col_mapper = self.item_original_ID_to_index_3)
        self.URM_train_4, _, _ = self._loadURM(URM_train_4_path, separator=",", header = True, if_new_user = "add", if_new_item = "ignore", col_mapper = self.item_original_ID_to_index_4)
        self.URM_train_5, _, _ = self._loadURM(URM_train_5_path, separator=",", header = True, if_new_user = "add", if_new_item = "ignore", col_mapper = self.item_original_ID_to_index_5)
        self.URM_test_1, _, _ = self._loadURM(URM_test_1_path, separator=",", header = True, if_new_user = "add", if_new_item = "ignore", col_mapper = self.item_original_ID_to_index_1)
        self.URM_test_2, _, _ = self._loadURM(URM_test_2_path, separator=",", header = True, if_new_user = "add", if_new_item = "ignore", col_mapper = self.item_original_ID_to_index_2)
        self.URM_test_3, _, _ = self._loadURM(URM_test_3_path, separator=",", header = True, if_new_user = "add", if_new_item = "ignore", col_mapper = self.item_original_ID_to_index_3)
        self.URM_test_4, _, _ = self._loadURM(URM_test_4_path, separator=",", header = True, if_new_user = "add", if_new_item = "ignore", col_mapper = self.item_original_ID_to_index_4)
        self.URM_test_5, _, _ = self._loadURM(URM_test_5_path, separator=",", header = True, if_new_user = "add", if_new_item = "ignore", col_mapper = self.item_original_ID_to_index_5)

        print("URM5Fold_WarmCold_Reader: cleaning temporary files URM")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        print("URM5Fold_WarmCold_Reader: saving URM")

    def _loadURM (self, filePath, header = False, separator="::", if_new_user = "add", if_new_item = "add", col_mapper = {}):

        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = col_mapper, on_new_col = if_new_item,
                                                        preinitialized_row_mapper = None, on_new_row = if_new_user)


        fileHandle = open(filePath, "r")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 1000000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

            user_id = line[0]
            item_id = line[1]
            URM_builder.add_data_lists([user_id], [item_id], [1])

        fileHandle.close()


        return  URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()

    
    def _loadICM(self, file_path, header=True, separator=','):

        # Genres
        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                        preinitialized_row_mapper = None, on_new_row = "add")


        fileHandle = open(file_path, "r", encoding="latin1")
        numCells = 0

        if header:
            headerLine = fileHandle.readline()
            headerLine = headerLine.split(separator)

        for line in fileHandle:
            numCells += 1
            if (numCells % 1000000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)

                movie_id = line[0]

                featureList = []
                featureValues = []
                for col in range(1,len(line)):
                    featureList.append(headerLine[col].replace("\n", ""))
                    featureValues.append(line[col].replace("\n", ""))
                        
                # Rows movie ID
                # Cols features
                ICM_builder.add_data_lists(row_list_to_add = [movie_id]*len(featureList), col_list_to_add = featureList, data_list_to_add = featureValues)

        fileHandle.close()

        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()
