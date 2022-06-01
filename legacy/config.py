config = {'item_embed': {
    'embedding_dim': 512,
    'sparse': False,
    'padding_idx': -1,
},
    'trans': {
        'input_size': 32,
        'hidden_size': 16,
        'n_layers': 2,
        'n_heads': 4,
        'max_len': 5,
    },
    'context_features': [
            {'num_embeddings': 6, 'embedding_dim': 10, 'sparse': False, 'padding_idx': -1},
            {'num_embeddings': 4, 'embedding_dim': 10, 'sparse': False, 'padding_idx': -1},

        ],

    'cuda': False,
    'init_method': "xavier"
}