import pandas as pd
import numpy as np
import os

np.random.seed(999)
cleanpath='/content/drive/My Drive/daitan/largescale/MS-SNSD/clean_train'

class MozillaCommonVoiceDataset:

  def __init__(self,k,val_dataset_size):
    #self.basepath = basepath
    self.val_dataset_size = val_dataset_size
    self.k=k
    print(self.k)
    #cleanpath='/content/drive/My Drive/daitan/mozilla/clean/0'
  def _get_common_voice_filenames(self, pathname,dataframe_name): 
    clean_files=[]
    mozilla_metadata = pd.read_csv(os.path.join(pathname, dataframe_name), sep='\t')
    clean_files = mozilla_metadata[mozilla_metadata.columns[0]].values
    #np.random.shuffle(clean_files)
    print("Total number of training examples:", len(clean_files))
    return clean_files

  def get_train_val_filenames(self):
    clean_files = self._get_common_voice_filenames(pathname=cleanpath,dataframe_name='clean_train.csv')
    clean_files3=clean_files[1000*self.k:1000*(self.k+1)]
    # resolve full path
    #clean_files = [os.path.join(cleanpath,filename) for filename in clean_files]
    clean_files2 = [os.path.join(cleanpath,filename) for filename in clean_files3]
    #clean_files2=[file1 for file1 in clean_files if os.path.isfile(file1)]
    clean_train_files = clean_files2[:-self.val_dataset_size]
    clean_val_files = clean_files2[-self.val_dataset_size:]
    print("# of Training clean files:", len(clean_train_files))
    print("# of  Validation clean files:", len(clean_val_files))
    return clean_train_files, clean_val_files

  def get_test_val_filenames(self):
    clean_files = self._get_common_voice_filenames(pathname=cleanpath,dataframe_name='cv-valid-test.csv')
    return clean_files