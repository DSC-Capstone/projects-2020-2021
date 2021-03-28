from lightfm import LightFM
def collab_model_build(interactions, num_threads, num_components, num_epochs, item_alpha):
    model = LightFM(loss='warp',
                item_alpha=item_alpha,
               no_components=num_components)
    model = model.fit(interactions, epochs=num_epochs, num_threads=num_threads)
    return model
