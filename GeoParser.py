#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:37:22 2017

@author: simonpelletier


"""

# valid for github
import os
import pandas as pd
import numpy as np
import GEOparse as Geo
from utils.utils import remove_all_unique, remove_all_same, create_missing_folders, select_meta_number, \
    rename_labels, get_user_int, rename


def search_meta_file_name(filename, list_meta_files, ask=False):
    """
    :param filename:
    :param list_meta_files:
    :param ask:
    :return:

    filename="GSE33000_GSE22845_GSE12417"
    list_meta_files=["GSE12417","GSE22845","GSE33000","GSE22845_GSE12417","GSE12417_GSE22845_GSE33000"]
    ask=False
    result = search_meta_file_name(filename="GSE33000_GSE22845_GSE12417",list_meta_files=list_meta_files)
    result = search_meta_file_name(filename="GSE33000_GSE12417_GSE22845",list_meta_files=list_meta_files)

    print(result)

    """
    import numpy as np
    query_list = filename.split("_")
    filename_to_return = None
    for existing_file in list_meta_files:
        list_meta_file = existing_file.split("_")
        number_datasets_found = len(np.intersect1d(query_list, list_meta_file))
        if number_datasets_found == len(query_list):
            print("The file you were looking for is there")
            if ask:
                answer = input("Do you want to use it? [y/n]")
                if answer == "y" or answer == "yes":
                    use_file = True
                else:
                    use_file = False
            else:
                use_file = True

            if filename == existing_file and use_file:
                print("File found!")
                filename_to_return = filename
                break
            elif use_file:
                correct_order = [list_meta_file.index(x) for x in query_list]
                filename_to_return = "_".join(np.array(list_meta_file)[correct_order])
                break
    return filename_to_return


class GeoParser:
    def __init__(self, home_path, geo_ids, unlabelled_geo_ids=None, bad_geo_ids=None, results_folder='results', data_folder='data',
                 destination_folder='annleukemia', dataframes_folder="dataframes", is_translate=True, silent=False):
        """
        # initial_lr because it can change automatically as the epochs go with a SCHEDULER
        # for example : ReduceLROnPlateau reduces the learning rate (lr) when the results are not improved for
        # a number of iterations specified by the user.
        #
        Advantages:
            1- Can start learning very fast and then does fine tuning;
                -   Too high:
                        The accuracy will most likely reach its optimum faster, but might not be
                        as good as with a smaller LR

                -   Too small:
                        The accuracy will most likely reach a better optimum, but might not be quite long (if too low
                        might seem like its not learning. Too low and it might not learn anything at all)

        Pitfalls:
            1- LR reduction too frequent or too large               (same problem as LR too SMALL)
            2- LR reduction not fast enough or not large enough     (same problem as LR too HIGH)


        Examples:
        destination_folder = "/Users/simonpelletier/data/annleukemia"
        initial_lr=1e-3
        init="he_uniform"
        n_epochs=2
        batch_size=128,
        hidden_size = 128
        translate=True
        silent=False

        """
        self.is_translate = is_translate
        self.silent = silent
        self.df = {}
        self.meta_df = {True: None, False:None}
        self.unlabelled_df = {}
        self.files_path = {}
        self.meta_file_path = None
        self.dataframes_folder = None

        # DATASETS IDS
        self.geo_ids = geo_ids

        # FOLDER NAMES

        self.data_folder = data_folder
        self.dataframes_folder = dataframes_folder
        self.destination_folder = destination_folder
        self.results_folder = results_folder
        self.unlabelled_geo_ids = unlabelled_geo_ids
        self.bad_geo_ids = bad_geo_ids

        #if self.unlabelled_geo_ids is not None:
        #    self.labelled_dict = dict(zip(zip(geo_ids, unlabelled_geo_ids),
        #                                  zip([[True] * len(geo_ids)], [[False] * len(unlabelled_geo_ids)])))

        # PATHS
        self.home_path = home_path
        self.data_folder_path = "/".join([self.home_path, self.destination_folder, self.data_folder]) + "/"
        self.results_folder_path = "/".join([self.home_path, self.destination_folder, self.results_folder]) + "/"
        self.dataframes_path = "/".join([self.data_folder_path, self.dataframes_folder]) + "/"
        self.translation_dict_path = "/".join([self.results_folder_path, "dictionaries"]) + "/"
        self.soft_path = self.data_folder_path + "/softs/"
        create_missing_folders(self.translation_dict_path)
        self.translation_results = {}
        # Hyperparameters

        #LAST ADDED
        self.meta_destination_folder = None
        self.meta_data_folder_path = None

    def get_geo(self, automatic_attribute_list, save_to_disk=True, load_from_disk=True):

        """

        Function that get informations on Gene Expression Omnibus (GEO)
        :param save_to_disk:
        :param load_from_disk:
        :return:

        """
        if len(self.geo_ids) == 0:
            print('WARNING! You must give at least one Geo id! The object created will be empty')

        # The key is the geo_id and values the matrices
        flag = True
        for geo_id in self.geo_ids:
            print('\nRunning:', geo_id)

            if load_from_disk:
                flag = self.load_geo(geo_id, labelled=True, bad_example=False)
                if flag:
                    print("The dataset will have to be built")
            if flag:
                print('Building the dataset ...')
                self.df[geo_id] = self.build_dataframe(geo_id, bad_example=False, labelled=True, save_to_disk=save_to_disk, automatic_attribute=automatic_attribute_list)
                if self.df[geo_id] is not None:
                    print("The file containing the data for ", geo_id, " was loaded successfully!")
        if self.unlabelled_geo_ids is not None:
            for geo_id in self.unlabelled_geo_ids:
                print('\nRunning unlabelled:', geo_id)

                if load_from_disk:
                    flag = self.load_geo(geo_id, labelled=False)
                    if flag:
                        print("The dataset will have to be built")
                if flag:
                    print('Building the dataset ...')

                    self.unlabelled_df[geo_id] = self.build_dataframe(geo_id, labelled=False, bad_example=False,
                                                                      save_to_disk=save_to_disk,
                                                                      automatic_attribute=automatic_attribute_list)
                    if self.unlabelled_df[geo_id] is not None:
                        print("The file containing the data for ", geo_id, " was loaded successfully!")
        if self.bad_geo_ids is not None:
            for geo_id in self.bad_geo_ids:
                print("\nRUNNING BAD EXAMPLE", geo_id)
                if load_from_disk and geo_id is not "":
                    flag = self.load_geo(geo_id, labelled=False, bad_example=True)
                    if flag:
                        print("The dataset will have to be built")
                if flag and geo_id is not "":
                    print('Building the dataset ...')

                    self.df[geo_id] = self.build_dataframe(geo_id, bad_example=True, labelled=False,
                                                           save_to_disk=save_to_disk,
                                                           automatic_attribute=automatic_attribute_list)
                    if self.df[geo_id] is not None:
                        print("BAD EXAMPLE: ", geo_id, "The file containing the data for ", geo_id, " was loaded successfully!")

    def load_geo(self, geo_id, labelled, bad_example=False):
        """

        :param geo_id:
        :return:

        Example:
        from debug.get_parameters import *

        dataframes_folder = "/Users/simonpelletier/data/annleukemia"
        g = GeoParser(destination_folder=data_destination)
        g.get_geo(geo_ids,load_from_disk)

        """
        flag = False
        print('Loading ' + geo_id + ", labelled: " + str(labelled) + ' ...')

        if not bad_example:
            self.df_file_name = geo_id + "_labelled" + str(labelled) + '_dataframe.pickle.npy'
        else:
            self.df_file_name = geo_id + "_labelled" + str(labelled) + '_bad_dataframe.pickle.npy'

        create_missing_folders(self.dataframes_path)
        current_directory_list = os.listdir(self.dataframes_path)
        if self.df_file_name in current_directory_list:
            print("File found at location:", self.data_folder_path + "/" + self.df_file_name)
            if labelled or bad_example:
                self.df[geo_id] = pd.read_pickle(self.dataframes_path + "/" + self.df_file_name)

                if sum(sum(np.isnan(self.df[geo_id].values)).tolist()) > 0:
                    print("Nans found. They are all replaced by 0")
                    self.df[geo_id][np.isnan(self.df[geo_id])] = 0
                print("self.df[geo_id]", self.df[geo_id].shape)
            else:
                self.unlabelled_df[geo_id] = pd.read_pickle(self.dataframes_path + "/" + self.df_file_name)

                if sum(sum(np.isnan(self.unlabelled_df[geo_id].values)).tolist()) > 0:
                    print("Nans found. They are all replaced by 0")
                    self.unlabelled_df[geo_id][np.isnan(self.unlabelled_df[geo_id])] = 0
                print("self.unlabelled_df[geo_id]", self.unlabelled_df[geo_id].shape)

        else:
            print(self.df_file_name, ' NOT FOUND in ', self.dataframes_path)
            flag = True
        return flag

    @staticmethod
    def make_metadata_matrix(gse, merged_values):
        """
        function to build a 2d matrix of the metadata (used to see which not to display)
        :return:
        example
        import GEOparse as Geo
        import numpy as np
        from GeoParser import GeoParser
        from utils import remove_all_unique, remove_all_same, create_missing_folders

        gse = Geo.GEOparse.get_GEO(Geo="GSE12417",destdir="/home/simon/data/annleukemia/",silent=True)
        g = GeoParser("GSE12417")

        geo_id='GSE33000'
        meta_dict = g.make_metadata_matrix(gse)
        meta_dict = remove_all_unique(meta_dict)
        meta_dict = remove_all_same(meta_dict)
        print(meta_dict)
        """
        # print("Make_metadata_matrix")
        all_infos = np.empty(shape=(len(list(gse.gsms[list(gse.gsms.keys())[0]].metadata.values())),
                                    len(merged_values.columns))).tolist()
        meta_names = []

        for l, lab in enumerate(list(merged_values.columns)):
            print(100 * l / len(merged_values.columns), "%", end="\r")
            infos_label = list(gse.gsms[lab].metadata.values())
            for i, info in enumerate(infos_label):
                try:
                    all_infos[i][l] = ' '.join(info)
                except:
                    all_infos[i][l] = info[0]
                meta_names += [list(gse.gsms[lab].metadata.keys())[i]]
        meta_dict = dict(zip(meta_names, all_infos))
        return meta_dict

    @staticmethod
    def rename_according_to_metadata(gse, meta_dict):
        accept_names = False
        meta_samples = None
        while not accept_names:
            meta_dict = select_meta_number(meta_dict)
            meta_dict = remove_all_unique(meta_dict)
            # meta_dict = remove_all_same(meta_dict)

            number = get_user_int(meta_dict)
            meta_samples = meta_dict[list(meta_dict.keys())[number]]

            try:
                n_samples = int(input("How many samples names you want to see?"))
            except:
                print("Not a valid number, displaying 1...")
                n_samples = 1
            for i in range(n_samples):
                print(meta_samples[i])
            satisfied = input('Are you satisfied with your seletection? (y/n)')
            if len(meta_samples[0]) > 1 and type(meta_samples[0]) != str:
                print("There is a list inside the list! Which one of the following options do you want (the options "
                      "represent the label of one of the samples)?")
                number2 = get_user_int(meta_samples[0])
                meta_samples = []
                for i, sample in enumerate(gse.gsms):
                    elements = list(list(gse.gsms.values())[i].metadata.values())
                    meta_samples.append(elements[number2])

            if satisfied == 'y':
                accept_names = True
        return meta_samples, meta_dict

    def build_dataframe(self, geo_id, labelled, bad_example, automatic_attribute, save_to_disk=True):

        """
        The labels are found in the metadata of merged object

        :param save_to_disk: 
        :param geo_id: ID found on NCBI's database
            EXAMPLE: GSE12417 -> found here -> https://www.ncbi.nlm.nih.gov/Geo/query/acc.cgi?acc=GSE12417
        :param save_to_disk (optional):

        EXAMPLE
        g = get_example_datasets(geo_ids = ["GSE12417","GSE22845"], home_path="/Users/simonpelletier/", load_from_disk=True)
        g.get_geo(geo_ids, load_from_disk=load_from_disk)

        """
        create_missing_folders(self.soft_path)
        gse = Geo.GEOparse.get_GEO(geo=geo_id, destdir=self.soft_path, silent=self.silent)
        gsm_on_choices = list(gse.gsms[list(gse.gsms.keys())[0]].columns.index)
        gpl_on_choices = list(gse.gpls[list(gse.gpls.keys())[0]].columns.index)

        print(str(len(gsm_on_choices)) + " Choices are available for GSM")
        gsm_on_selection = 0
        # gsm_on_selection = get_user_int(gsm_on_choices)
        gsm_on = gsm_on_choices[gsm_on_selection]
        print(str(len(gpl_on_choices)) + " Choices are available for GPL. You must select: ")
        print("1 - An annotation for GPL")
        print("2 - (optional) The annotation you want the row names to take")

        gpl_on_selection = 0
        # gpl_on_selection = get_user_int(gpl_on_choices)
        gpl_on = gpl_on_choices[gpl_on_selection]
        val_selection = None
        if automatic_attribute is False:
            val_selection = get_user_int(gpl_on_choices)
        else:
            self.attribute = automatic_attribute
            for attribute in automatic_attribute:
                try:
                    val_selection = gpl_on_choices.index(attribute)
                except:
                    pass

        if val_selection == None:
            exit("Selection not found " + str(automatic_attribute) + str(gpl_on_choices))
        val = gpl_on_choices[val_selection]

        merged_values = gse.merge_and_average(gse.gpls[next(iter(gse.gpls))], "VALUE", val,
                                              gpl_on_choices, gpl_on=gpl_on, gsm_on=gsm_on)
        merged_values.values[np.isnan(merged_values.values)] = 0

        self.merge_len = merged_values.shape[1]

        if labelled:
            self.df[geo_id] = merged_values
            meta_dict = self.make_metadata_matrix(gse, merged_values)
            labels, meta_dict = self.rename_according_to_metadata(gse, meta_dict)

            labels = ["".join(label) for label in labels]

            labels = rename_labels(labels)
            labels = rename(labels)

            self.df[geo_id].columns = labels

            if len(labels) > merged_values.shape[1]:
                prompt = input(
                    "Duplicates were detected. Do you want to keep only the first labels (only say yes if you are sure, "
                    "or the results could be wrong) [y/n]")
                print("Labels", len(labels))
                print("Labels", merged_values.shape)

                if prompt == "y":
                    labels = labels[:merged_values.shape[1]]
                else:
                    exit()
        else:
            self.unlabelled_df[geo_id] = merged_values
            self.unlabelled_df[geo_id].columns = ["no_label"] * len(self.unlabelled_df[geo_id].columns)
        if save_to_disk:
            create_missing_folders(path=self.dataframes_path)
            if not bad_example:
                self.files_path[geo_id] = self.dataframes_path + '/' + geo_id + "_labelled" + str(labelled) + '_dataframe'
            else:
                self.files_path[geo_id] = self.dataframes_path + '/' + geo_id + "_labelled" + str(labelled) + '_bad_dataframe'

            print("Saving to " + self.files_path[geo_id])
            if labelled:
                self.df[geo_id].to_pickle(self.files_path[geo_id] + '.pickle.npy') # Faster to load pickled files
                #self.df[geo_id].to_csv(self.files_path[geo_id] + '.csv') # For vizualisation
            else:
                self.unlabelled_df[geo_id].to_pickle(self.files_path[geo_id] + '.pickle.npy')
                #self.unlabelled_df[geo_id].to_csv(self.files_path[geo_id] + '.csv')

        return merged_values

    @staticmethod
    def save(filename, merged_values):
        merged_values.to_pickle(filename + '.pickle.npy')
        #merged_values.to_csv(filename + '.csv')

    def remove(self, geo_id='GSE33000'):
        """

        USELESS??
        Removes a dataset
        :param geo_id:
        :return:
        """
        if geo_id not in self.geo_ids:
            print('The object does not contain the GEO ID specified')
        else:
            if len(self.geo_ids) <= 1:
                print('WARNING! The last dataset is being removed (object will be empty) ')

    def merge_datasets(self, labelled, fill_missing=True, load_from_disk=False,
                       meta_destination_folder="meta_pandas_dataframes"):
        """

        :param load_from_disk: 
        :param meta_destination_folder: 
        :param fill_missing: if True, the missing rows will be replaced with 0s for all samples of that dataset.
                The algorythm might be able to do good even without some information. Otherwise, the list might get very small

        from utils import get_example_datasets, create_missing_folders
        fill_missing=True
        geo_ids = ["GSE12417","GSE22845"]
        g = get_example_datasets(geo_ids, home_path="/home/simon/", load_from_disk=True)
        g.get_geo(geo_ids, load_from_disk=True)
        g.merge_datasets(fill_missing=True)

        """

        print("Preparing for merging the selected datasets... labelled:", labelled)
        import os
        import pandas as pd
        import numpy as np

        if labelled:
            dataframe = self.df
            geo_ids = list(self.df.keys())
        else:
            dataframe = self.unlabelled_df
            geo_ids = list(self.unlabelled_df.keys())

        self.meta_destination_folder = meta_destination_folder + "_labelled" + str(labelled)
        self.meta_data_folder_path = "/".join([self.data_folder_path, meta_destination_folder])
        create_missing_folders(self.meta_data_folder_path)
        meta_filename = "_".join(geo_ids) + ".pickle.npy"

        count = 0
        n_samples_list = [len(dataframe[geo_id].columns) for geo_id in geo_ids]
        total_iteration = -n_samples_list[0]
        meta_file = search_meta_file_name(meta_filename, list_meta_files=os.listdir(self.meta_data_folder_path))

        if meta_file is None or load_from_disk is False:
            for i in range(len(n_samples_list)):
                for j in range(i, len(n_samples_list)):
                    total_iteration += n_samples_list[j]
            for g, geo_id in enumerate(geo_ids):
                print("merging file:", g+1, "/", len(geo_ids))
                if g == 0:
                    meta_df = dataframe[geo_id]
                else:
                    if fill_missing:
                        meta_df = pd.concat((meta_df, dataframe[geo_id]), axis=1, sort=True)
                try:
                    assert len(meta_df.index) == len(set(meta_df.index))
                except:
                    print("CONTAINS DUPLICATED ROWNAMES")
                print(meta_df.shape)
            print("Saving files...")
            self.meta_filename = meta_filename
            self.meta_file_path = '/'.join([self.meta_data_folder_path, self.meta_filename])
            self.meta_df[labelled] = meta_df
            self.meta_df[labelled].to_pickle(self.meta_file_path)
            # self.meta_df[labelled].to_csv(self.meta_file_path + ".csv")
        else:
            print("Loading file...")
            self.meta_filename = meta_file
            self.meta_file_path = '/'.join([self.meta_data_folder_path, self.meta_filename])
            self.meta_df[labelled] = pd.read_pickle(self.meta_file_path)
        print("Merged sets loaded.")
        return self.meta_df[labelled]


    def translate(self, geo_id, labelled, old_ids='entrezgene_trans_name', new_ids='uniprot_gn', load=True):
        import subprocess
        translation_destination = "/".join([self.data_folder_path, "translation_results"]) + "/"
        self.translation_destination = translation_destination
        dictionary_path = self.dictionary_path = "/".join([self.data_folder_path, "dictionaries"])
        create_missing_folders(translation_destination)
        create_missing_folders(dictionary_path)
        filename = geo_id + "_" + old_ids + '.txt'
        output_file = geo_id + "_" + old_ids + "2" + new_ids + ".txt"
        output_path = translation_destination + "/" + output_file

        if filename not in os.listdir(translation_destination) or load is False:
            print("new file in" + translation_destination)
            f = open(translation_destination + "/" + filename, "w")
            for id in list(self.meta_df[labelled].index):
                f.write(str(id) + "\n")
            f.close()
        if output_file not in os.listdir(translation_destination):
            print("Translating", geo_id, "from", old_ids, "to", new_ids, "...")
            call = ["./biomart_api.R", translation_destination, geo_id, old_ids, new_ids]
            subprocess.call(call)
        else:
            print("The file", output_file, "was found in", translation_destination)
        file = open(output_path, "r")
        names_translations = np.loadtxt(file, dtype=str, delimiter=";")
        try:
            assert len(names_translations) > 0
        except:
            print("There is not translation to show")
        return names_translations

    def translate_indices_df(self, geo_id, labelled, old_ids='entrezgene_trans_name', new_ids='uniprot_gn', load=True):
        print(geo_id, labelled, old_ids, new_ids, load)
        names_translations = self.translate(geo_id, labelled, old_ids, new_ids, load)

        meta_index = list(self.meta_df[labelled].index)
        new_names = names_translations[:, 1]
        old_names = names_translations[:, 0]

        indices_new_names = [i for i, x in enumerate(np.in1d(meta_index, old_names)) if x]
        indices_to_keep_old_df = np.in1d(meta_index, old_names)

        new_df = self.meta_df[labelled][indices_to_keep_old_df]
        new_df.index = new_names[indices_new_names]
        self.meta_df[labelled] = new_df
