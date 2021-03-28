import tensorflow
import tensorflow.keras as keras
import numpy as np
import uproot

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_files, features, labels, spectators, batch_size=1024, n_dim=60, 
                 remove_mass_pt_window=False, remove_unlabeled=True, return_spectators=False,
                 max_entry = 20000, scale_mass_pt = [1, 1]):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_files = list_files
        self.features = features
        self.spectators = spectators
        self.return_spectators = return_spectators
        self.scale_mass_pt = scale_mass_pt
        self.n_dim = n_dim
        self.n_channels = len(self.features)
        self.remove_mass_pt_window = remove_mass_pt_window
        self.remove_unlabeled = remove_unlabeled
        self.global_IDs = []
        self.local_IDs = []
        self.file_mapping = []
        self.max_entry = max_entry
        self.open_files = [None]*len(self.list_files)
        running_total = 0
        for i, file_name in enumerate(self.list_files):
            
            root_file = uproot.open(file_name)
            self.open_files.append(root_file)
            try:
                tree = root_file['deepntuplizer/tree']
            except:
                tree = root_file['deepntuplizertree']
            tree_length = min(len(tree),max_entry)
            self.global_IDs.append(np.arange(running_total,running_total+tree_length))
            self.local_IDs.append(np.arange(tree_length))
            self.file_mapping.append(np.repeat(i,tree_length))
            running_total += tree_length
            root_file.close()
        self.global_IDs = np.concatenate(self.global_IDs)
        self.local_IDs = np.concatenate(self.local_IDs)
        self.file_mapping = np.concatenate(self.file_mapping)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.global_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        files = self.file_mapping[index*self.batch_size:(index+1)*self.batch_size]
        
        unique_files = np.unique(files)
        starts = np.array([min(indexes[files==i]) for i in unique_files])
        stops = np.array([max(indexes[files==i]) for i in unique_files])

        # Check if files needed open (if not open them)
        # Also if file is not needed, close it
        for ifile, file_name in enumerate(self.list_files):
            if ifile in unique_files:
                if self.open_files[ifile] is None: 
                    self.open_files[ifile] = uproot.open(file_name)
            else:
                if self.open_files[ifile] is not None: 
                    self.open_files[ifile].close()
                    self.open_files[ifile] = None
            
        # Generate data
        return self.__data_generation(unique_files, starts, stops)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = self.local_IDs

    def __data_generation(self, unique_files, starts, stops):
        'Generates data containing batch_size samples' 
        # X : (n_samples, n_dim, n_channels)
        # y : (n_samples, 2)
        Xs = []
        ys = []
        zs = []
        
        # Generate data
        for ifile, start, stop in zip(unique_files, starts, stops):
            if self.return_spectators:
                X, [y, z] = self.__get_features_labels(ifile, start, stop)
                zs.append(z)
            else:
                X, y = self.__get_features_labels(ifile, start, stop)
            Xs.append(X)
            ys.append(y)
            
        # Stack data if going over multiple files
        if len(unique_files)>1:
            X = np.concatenate(Xs,axis=0)
            y = np.concatenate(ys,axis=0)
            if self.return_spectators:
                z = np.concatenate(zs,axis=0)
            
        if self.return_spectators:
            return X, [y, z]
        
        return X, y
                         
    def __get_features_labels(self, ifile, entrystart, entrystop):
        'Loads data from one file'
        
        # Double check that file is open
        if self.open_files[ifile] is None:
            root_file = uproot.open(self.list_file[ifile])
        else:
            root_file = self.open_files[ifile]
        
        try:
            tree = root_file['deepntuplizer/tree']
        except:
            tree = root_file['deepntuplizertree']
        
        feature_array = tree.arrays(branches=self.features, 
                                    entrystart=entrystart,
                                    entrystop=entrystop,
                                    namedecode='utf-8')

        label_array_all = tree.arrays(branches=self.labels, 
                                      entrystart=entrystart,
                                      entrystop=entrystop,
                                      namedecode='utf-8')
        
        X = np.stack([feature_array[feat].pad(self.n_dim, clip=True).fillna(0).regular() for feat in self.features],axis=2)
        n_samples = X.shape[0]
    
        '''y = np.zeros((n_samples,2))
        y[:,0] = label_array_all['sample_isQCD'] * (label_array_all['label_QCD_b'] + \
                                                    label_array_all['label_QCD_bb'] + \
                                                    label_array_all['label_QCD_c'] + \
                                                    label_array_all['label_QCD_cc'] + \
                                                    label_array_all['label_QCD_others'])
        y[:,1] = label_array_all['label_H_bb']'''
        y = np.zeros((n_samples,6))
        y[:,0] = label_array_all['sample_isQCD'] * label_array_all['label_QCD_b']
        y[:,1] = label_array_all['sample_isQCD'] * label_array_all['label_QCD_bb']
        y[:,2] = label_array_all['sample_isQCD'] * label_array_all['label_QCD_c']
        y[:,3] = label_array_all['sample_isQCD'] * label_array_all['label_QCD_cc']
        y[:,4] = label_array_all['sample_isQCD'] * label_array_all['label_QCD_others']
        y[:,5] = label_array_all['label_H_bb']
        

        if self.remove_mass_pt_window or self.return_spectators:
            spec_array = tree.arrays(branches=self.spectators, 
                                     entrystart=entrystart,
                                     entrystop=entrystop,
                                     namedecode='utf-8')            
            z = np.stack([spec_array[spec] for spec in self.spectators],axis=1)
            
        if self.remove_mass_pt_window:
            # remove data outside of mass/pT range
            X = X[(z[:,0] > 40) & (z[:,0] < 200) & (z[:,1] > 300) & (z[:,1] < 2000)]
            y = y[(z[:,0] > 40) & (z[:,0] < 200) & (z[:,1] > 300) & (z[:,1] < 2000)]
            z = z[(z[:,0] > 40) & (z[:,0] < 200) & (z[:,1] > 300) & (z[:,1] < 2000)]
                        
        if self.remove_unlabeled:
            # remove unlabeled data
            X = X[np.sum(y,axis=1)==1]
            if self.return_spectators:
                z = z[np.sum(y,axis=1)==1]
            y = y[np.sum(y,axis=1)==1]
            
        if self.return_spectators:
            return X, [y, z/self.scale_mass_pt]
        
        return X, y
