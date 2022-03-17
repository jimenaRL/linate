# linate class

#from sklearn.utils.estimator_checks import check_estimator

from linate import CA

def main():
    ca_model = CA(n_components = 5, engine = 'auto', in_degree_threshold = 2, 
                                                        out_degree_threshold = 2, force_bipartite = False)
    #ca_model = CA(n_components = 5, engine = 'linate_ds', in_degree_threshold = 0, out_degree_threshold = 3, standardize_mean=True, standardize_std=False, force_bipartite=False)
    

    # COMMENT: if force_bipartite=True, source nodes that are also target nodes should be eliminated from the list of sources

    # This is the original example
    #network_file_header_names = {'source':'source', 'target':'target'}
    network_file_header_names = None
    X = ca_model.load_input_from_file('/home/foula/FoulaDatasetAttitudinalEmbedding/bipartite_831MPs_4424402followers.csv',
            network_file_header_names = network_file_header_names)
    #X = ca_model.load_input_from_file('data/twitter_networks/test_graph_A.csv',
    #X = ca_model.load_input_from_file('data/twitter_networks/test_graph_B.csv',

    # Example loading the "RawData" filder examples
    #X = ca_model.load_input_from_file('Australia_20201027_MPs_followers_tids.csv',
     #       network_file_header_names = {'source':'followers_id', 'target':'twitter_id'})

    # Fitting
    ca_model.fit(X)

    # Saving to file
    ca_model.save_ca_target_coordinates('ca_results/ca_target_cordinates.csv')
    ca_model.save_ca_source_coordinates('ca_results/ca_source_cordinates.csv')


    # Retrieving the coordinates in program
    target_coords = ca_model.ca_target_coordinates_
    source_coords = ca_model.ca_source_coordinates_

    # Check the number of source and target nodes
    ca_model.target_users_no_
    ca_model.source_users_no_

    #
    print('eigenvalues', ca_model.eigenvalues_)
    print('total inertia', ca_model.total_inertia_)
    print('explained inertia', ca_model.explained_inertia_)

    #check_estimator(ca_model)

if __name__ == "__main__":
    main()
