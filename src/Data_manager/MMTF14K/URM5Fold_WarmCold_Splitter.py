#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/01/18

@author: Maurizio Ferrari Dacrema
"""

import scipy.sparse as sps
import numpy as np
import pickle

from Data_manager.DataSplitter import DataSplitter


class URM5Fold_WarmCold_Splitter(DataSplitter):
    """
    The splitter tries to load from the specific folder related to a dataset, a split in the format corresponding to
    the splitter class. Basically each split is in a different subfolder
    - The "original" subfolder contains the whole dataset, is composed by a single URM with all data and may contain
        ICMs as well, either one or many, depending on the dataset
    - The other subfolders "warm", "cold" ecc contains the splitted data.

    The dataReader class involvement is limited to the following cased:
    - At first the dataSplitter tries to load from the subfolder corresponding to that split. Say "warm"
    - If the dataReader is succesful in loading the files, then a split already exists and the loading is complete
    - If the dataReader raises a FileNotFoundException, then no split is available.
    - The dataSplitter then creates a new instance of dataReader using default parameters, so that the original data will be loaded
    - At this point the chosen dataSplitter takes the URM_all and selected ICM to perform the split
    - The dataSplitter saves the splitted data in the appropriate subfolder.
    - Finally, the dataReader is instantiated again with the correct parameters, to load the data just saved
    """

    """
     - It exposes the following functions
        - load_data(save_folder_path = None, force_new_split = False)   loads the data or creates a new split
    
    
    """


    def __init__(self, dataReader_object, n_folds = 5, forbid_new_split = False, force_new_split = False):
        """

        :param dataReader_object:
        :param n_folds:
        :param force_new_split:
        :param forbid_new_split:
        :param save_folder_path:    path in which to save the loaded dataset
                                    None    use default "dataset_name/split_name/"
                                    False   do not save
        """


        assert n_folds>1, "URM5Fold_WarmCold_Splitter: Number of folds must be  greater than 1"

        self.n_folds = n_folds

        # Create empty URM for each fold
        self.fold_split = {}

        super(URM5Fold_WarmCold_Splitter, self).__init__(dataReader_object, forbid_new_split=forbid_new_split, force_new_split=force_new_split)




    def get_statistics_URM(self):

        # This avoids the fixed bit representation of numpy preventing
        # an overflow when computing the product
        n_items = int(self.n_items)
        n_users = int(self.n_users)

        print("URM5Fold_WarmCold_Splitter for DataReader: {}\n"
              "\t Num items: {}\n"
              "\t Num users: {}\n".format(self.dataReader_object._get_dataset_name(), n_items, n_users))


        n_global_interactions = 0

        for fold_index in range(self.n_folds):
            URM_fold_object_train = self.fold_split[fold_index]["URM_train"]
            URM_fold_object_test = self.fold_split[fold_index]["URM_test"]
            n_global_interactions += URM_fold_object_train.nnz
            n_global_interactions += URM_fold_object_test.nnz


        for fold_index in range(self.n_folds):
            URM_fold_object_train = self.fold_split[fold_index]["URM_train"]
            URM_fold_object_test = self.fold_split[fold_index]["URM_test"]
            items_in_fold = self.fold_split[fold_index]["items_in_fold"]


            print("\t Statistics for fold {}: n_interactions {} ( {:.2f}%), n_items {} ( {:.2f}%), density: {:.2E}".format(
                fold_index,
                URM_fold_object_train.nnz + URM_fold_object_test.nnz, (URM_fold_object_train.nnz + URM_fold_object_test.nnz)/n_global_interactions*100,
                items_in_fold, items_in_fold/n_items*100,
                (URM_fold_object_train.nnz + URM_fold_object_test.nnz)/(int(n_items)*int(n_users))
            ))

        print("\n")

    def get_statistics_ICM(self):

        print("No ICM here!!")



    def get_URM_train_for_test_fold(self, n_test_fold):
        """
        The train set is defined as all data except the one of that fold, which is the test
        :param n_fold:
        :return:
        """


        URM_test = self.fold_split[n_test_fold]["URM_test"].copy()
        URM_train = self.fold_split[n_test_fold]["URM_train"].copy()

        return URM_train, URM_test

    
    def _split_data_from_original_dataset(self, save_folder_path):

        self.dataReader_object.load_data()

        URM_train_1 = self.dataReader_object.get_URM_from_name("URM_train_1")
        URM_train_1 = sps.csr_matrix(URM_train_1)
        URM_train_2 = self.dataReader_object.get_URM_from_name("URM_train_2")
        URM_train_2 = sps.csr_matrix(URM_train_2)
        URM_train_3 = self.dataReader_object.get_URM_from_name("URM_train_3")
        URM_train_3 = sps.csr_matrix(URM_train_3)
        URM_train_4 = self.dataReader_object.get_URM_from_name("URM_train_4")
        URM_train_4 = sps.csr_matrix(URM_train_4)
        URM_train_5 = self.dataReader_object.get_URM_from_name("URM_train_5")
        URM_train_5 = sps.csr_matrix(URM_train_5)

        URM_test_1 = self.dataReader_object.get_URM_from_name("URM_test_1")
        URM_test_1 = sps.csr_matrix(URM_test_1)
        URM_test_2 = self.dataReader_object.get_URM_from_name("URM_test_2")
        URM_test_2 = sps.csr_matrix(URM_test_2)
        URM_test_3 = self.dataReader_object.get_URM_from_name("URM_test_3")
        URM_test_3 = sps.csr_matrix(URM_test_3)
        URM_test_4 = self.dataReader_object.get_URM_from_name("URM_test_4")
        URM_test_4 = sps.csr_matrix(URM_test_4)
        URM_test_5 = self.dataReader_object.get_URM_from_name("URM_test_5")
        URM_test_5 = sps.csr_matrix(URM_test_5)

        self.n_users, self.n_items = URM_train_1.shape
    
        self.fold_split[0] = {}
        self.fold_split[0]["URM_train"] = URM_train_1
        self.fold_split[0]["URM_test"] = URM_test_1
        self.fold_split[0]["items_in_fold"] = URM_train_1.shape[1]
        self.fold_split[1] = {}
        self.fold_split[1]["URM_train"] = URM_train_2
        self.fold_split[1]["URM_test"] = URM_test_2
        self.fold_split[1]["items_in_fold"] = URM_train_2.shape[1]
        self.fold_split[2] = {}
        self.fold_split[2]["URM_train"] = URM_train_3
        self.fold_split[2]["URM_test"] = URM_test_3
        self.fold_split[2]["items_in_fold"] = URM_train_3.shape[1]
        self.fold_split[3] = {}
        self.fold_split[3]["URM_train"] = URM_train_4
        self.fold_split[3]["URM_test"] = URM_test_4
        self.fold_split[3]["items_in_fold"] = URM_train_4.shape[1]
        self.fold_split[4] = {}
        self.fold_split[4]["URM_train"] = URM_train_5
        self.fold_split[4]["URM_test"] = URM_test_5
        self.fold_split[4]["items_in_fold"] = URM_train_5.shape[1]

        fold_dict_to_save = {"fold_split": self.fold_split,
                    "n_folds": self.n_folds,
                    "n_items": self.n_items,
                    "n_users": self.n_users,
                    }

        pickle.dump(fold_dict_to_save,
                    open(save_folder_path + "URM_{}_fold_split".format(self.n_folds), "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("URM5Fold_WarmCold_Splitter: Split complete")


    def _load_previously_built_split_and_attributes(self, save_folder_path):
        """
        Loads all URM and ICM
        :return:
        """


        data_dict = pickle.load(open(save_folder_path + "URM_{}_fold_split".format(self.n_folds), "rb"))

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])


    def __iter__(self):

        self.__iterator_current_fold = 0
        return self


    def __next__(self):

        fold_to_return = self.__iterator_current_fold

        if self.__iterator_current_fold >= self.n_folds:
            raise StopIteration

        self.__iterator_current_fold += 1

        return fold_to_return, self[fold_to_return]


    def _get_split_subfolder_name(self):
        """

        :return: {n_folds}_fold/
        """
        return "{}_fold/".format(self.n_folds)


    def __getitem__(self, n_test_fold):
        """
        :param index:
        :return:
        """

        return self.get_URM_train_for_test_fold(n_test_fold)


    def __len__(self):

        return self.n_folds







