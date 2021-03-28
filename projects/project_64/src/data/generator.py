from DataGenerator import DataGenerator

def generator(train_file_name, test_file_name, val_file_name, features, labels, spectators, batch_size,n_dim,remove_mass_pt_window, remove_unlabeled,max_entry):
    
    generators = []
    filenames= [train_file_name, test_file_name, val_file_name]
    
    for name in filenames:
        
        gen = DataGenerator(name, features, labels, spectators, batch_size=batch_size, n_dim=n_dim, 
                                remove_mass_pt_window=remove_mass_pt_window, 
                                remove_unlabeled=remove_unlabeled, max_entry=max_entry)
        generators += [gen]
        
    return generators