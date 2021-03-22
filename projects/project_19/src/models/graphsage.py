from spektral.layers import GCNConv
from spektral.layers import GraphSageConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import tensorflow as tf
from tensorflow.keras.regularizers import l2


def GraphSage(A, F, N, X, train_mask, val_mask, labels_encoded, num_classes, aggr, channels, dropout, l2_reg, learning_rate, epochs, es_patience):

    A = GCNConv.preprocess(A).astype('f4')

    X_in = Input(shape=(F, ))
    fltr_in = Input((N, ), sparse=True)

    dropout_1 = Dropout(dropout)(X_in)
    graph_conv_1 = GraphSageConv(channels, aggregate_op=aggr,
                            activation='relu',
                            kernel_regularizer=l2(l2_reg),
                            use_bias=False)([dropout_1, fltr_in])

    dropout_2 = Dropout(dropout)(graph_conv_1)
    graph_conv_2 = GCNConv(num_classes,
                         activation='softmax',
                         use_bias=False)([dropout_2, fltr_in])

    model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_2)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  weighted_metrics=['acc'])

#     tbCallBack_GCN = tf.keras.callbacks.TensorBoard(
#         log_dir='./Tensorboard_GCN_cora',
#     )
#     callback_GCN = [tbCallBack_GCN]

    validation_data = ([X, A], labels_encoded, val_mask)
    model.fit([X, A],
          labels_encoded,
          sample_weight=train_mask,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,
          callbacks=[
              EarlyStopping(patience=es_patience,  restore_best_weights=True)
#               ,tbCallBack_GCN
          ])