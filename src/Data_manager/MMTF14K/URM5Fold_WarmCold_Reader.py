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
    DATASET_SUBFOLDER_URM_COMPLETE = "Movielens_20m/"

    AVAILABLE_URMS = ["URM_train_1","URM_train_2","URM_train_3","URM_train_4","URM_train_5","URM_test_1","URM_test_2","URM_test_3","URM_test_4","URM_test_5"]

    IS_IMPLICIT = True


    def __init__(self):
        super(URM5Fold_WarmCold_Reader, self).__init__()



    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER_URM


    def get_loaded_URM_names(self):
        print("HERE")
        return self.AVAILABLE_URMS.copy()


    def _load_from_original_file(self):
        # Load data from original

        print("URM5Fold_WarmCold_Reader: Loading original data")

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
        self.URM_train_1, _, _ = self._loadURM(URM_train_1_path, separator=",", header = True, if_new_user = "add", if_new_item = "add")
        self.URM_train_2, _, _ = self._loadURM(URM_train_2_path, separator=",", header = True, if_new_user = "add", if_new_item = "add")
        self.URM_train_3, _, _ = self._loadURM(URM_train_3_path, separator=",", header = True, if_new_user = "add", if_new_item = "add")
        self.URM_train_4, _, _ = self._loadURM(URM_train_4_path, separator=",", header = True, if_new_user = "add", if_new_item = "add")
        self.URM_train_5, _, _ = self._loadURM(URM_train_5_path, separator=",", header = True, if_new_user = "add", if_new_item = "add")
        self.URM_test_1, _, _ = self._loadURM(URM_test_1_path, separator=",", header = True, if_new_user = "add", if_new_item = "add")
        self.URM_test_2, _, _ = self._loadURM(URM_test_2_path, separator=",", header = True, if_new_user = "add", if_new_item = "add")
        self.URM_test_3, _, _ = self._loadURM(URM_test_3_path, separator=",", header = True, if_new_user = "add", if_new_item = "add")
        self.URM_test_4, _, _ = self._loadURM(URM_test_4_path, separator=",", header = True, if_new_user = "add", if_new_item = "add")
        self.URM_test_5, _, _ = self._loadURM(URM_test_5_path, separator=",", header = True, if_new_user = "add", if_new_item = "add")

        print("Movielens20MReader: Loading original data")

        print("URM5Fold_WarmCold_Reader: cleaning temporary files")

        import shutil

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        print("URM5Fold_WarmCold_Reader: saving URM")


    def _loadURM (self, filePath, header = False, separator="::", if_new_user = "add", if_new_item = "add"):

        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = if_new_item,
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

