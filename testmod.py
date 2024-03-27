import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import nn
from tensorflow.keras import Model
from tensorflow.keras import Sequential
import tensorflow.python
from tensorflow.python import keras


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, \
    Input, Add, PReLU, ReLU


class ResNetBlock(Model):
    def __init__(self, in_channels=(32, 11, 100, 60, 1), out_channels=50, kernel_size=(3,3,3), stride=(1,1,1), padding='same', relu_type='prelu'):
        super(ResNetBlock, self).__init__()

        # Make these hyperparameters optimizable

        self.padding = padding
        self.relu_type = relu_type

        self.conv1 = tf.keras.layers.Conv3D(filters=out_channels, input_shape=in_channels, kernel_size=kernel_size,
                            strides=stride, padding=self.padding)
        self.bn1 = BatchNormalization()
        self.relu1 = PReLU() if self.relu_type == 'prelu' else ReLU()
        self.conv2 = tf.keras.layers.Conv3D(out_channels, kernel_size=kernel_size,
                            strides=stride, padding=self.padding)
        self.bn2 = BatchNormalization()

        self.downsample = None
        self.stride = stride
        if in_channels != out_channels or self.stride != 1:
            self.downsample = Sequential([
                tf.keras.layers.Conv3D(filters=out_channels, kernel_size=(1, 1, 1), strides=stride,
                       padding='valid'),
                BatchNormalization()
            ])

        self.relu2 = PReLU() if self.relu_type == 'prelu' else ReLU()

    def call(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)
        return out

class ResNet(layers.Layer):
    def __init__(self, num_classes, block_depths=[2, 2, 2, 2], num_filters=[64, 128, 256, 512]):
        super(ResNet, self).__init__()

        # Make these hyperparameters optimizable
        self.block_depths = [tf.Variable(depth, trainable=True) for depth in block_depths]
        self.num_filters = [tf.Variable(filters, trainable=True) for filters in num_filters]

        self.conv1 = tf.keras.layers.Conv3D(self.num_filters[0], kernel_size=(5, 7, 7), strides=(1, 2, 2), padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.maxpool=tf.keras.layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')
        self.front = Sequential([
            self.conv1,
            self.bn1,
            PReLU(),
            self.maxpool
        ])



        self.trunk = Sequential()
        in_channels = self.num_filters[0]
        for i in range(len(self.block_depths)):
            self.trunk.add(ResNetBlock(in_channels, self.num_filters[i]))
            in_channels = self.num_filters[i]
            for _ in range(self.block_depths[i] - 1):
                self.trunk.add(ResNetBlock(in_channels, in_channels))

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = Dense(num_classes, activation='softmax')

    def call(self, x):
        print(x.shape)
        x = self.front(x)
        x = tf.squeeze(x, axis=1)  # Squeeze the 1st dimension
        x = self.trunk(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

class TemporalConvolutionalNetwork(Model):
    def __init__(self, num_classes, kernel_size=3, dilation_rates=[1, 2, 4, 8], num_filters=64, dropout_rate=0.2):
        super(TemporalConvolutionalNetwork, self).__init__()

        # Make these hyperparameters optimizable
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate

        self.tcn_layers = []
        for rate in dilation_rates:
            self.tcn_layers.append(
                Sequential([
                    tf.keras.layers.Conv1D(filters=self.num_filters, kernel_size=kernel_size, dilation_rate=rate, padding='causal'),
                    BatchNormalization(),
                    PReLU(),
                    Dropout(self.dropout_rate)
                ])
            )

        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.fc = Dense(num_classes, activation='softmax')

    def call(self, x):
        for layer in self.tcn_layers:
            x = layer(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x


# Combine the ResNet and TCN models



class FrontendBackendModel(Model):
    def __init__(self, num_classes, resnet_block_depths, resnet_num_filters, tcn_kernel_size, tcn_dilation_rates,
                 tcn_num_filters, tcn_dropout_rate):
        super(FrontendBackendModel, self).__init__()
        self.resnet = ResNet(num_classes, resnet_block_depths, resnet_num_filters)
        self.tcn = TemporalConvolutionalNetwork(num_classes, tcn_kernel_size, tcn_dilation_rates, tcn_num_filters,
                                                tcn_dropout_rate)

    def call(self, x):
        print(x)
        x = self.resnet(x)
        x = self.tcn(x)
        return x


# Instantiate the model
"""
model = FrontendBackendModel(
    num_classes=10,
    resnet_block_depths=[2, 2, 2, 2],
    resnet_num_filters=[64, 128, 256, 512],
    tcn_kernel_size=3,
    tcn_dilation_rates=[1, 2, 4, 8],
    tcn_num_filters=64,
    tcn_dropout_rate=0.2
)"""

# Define the model's layers
class FullyDenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_k, d_v, d_model, d_ff, activation='relu', **kwargs):
        super(FullyDenseBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # Define the attention and feed-forward sub-layers
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_k, name='attention')
        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(self.d_ff, activation=self.activation),
            tf.keras.layers.Dense(self.d_model)
        ])

    def call(self, inputs, training=False):
        # Apply the attention and feed-forward sub-layers
        attention_output = self.attention(inputs, inputs)
        feed_forward_output = self.feed_forward(attention_output)
        return feed_forward_output

class PartiallyDenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_k, d_v, d_model, d_ff, activation='relu', **kwargs):
        super(PartiallyDenseBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # Define the attention and feed-forward sub-layers
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_k, name='attention')
        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(self.d_ff, activation=self.activation),
            tf.keras.layers.Dense(self.d_model)
        ])

    def call(self, inputs, training=False):
        # Apply the attention and feed-forward sub-layers
        attention_output = self.attention(inputs, inputs)
        feed_forward_output = self.feed_forward(attention_output)
        return feed_forward_output

# Define the model
class TransformerModel(tf.keras.Model):
    def __init__(self, num_heads, d_k, d_v, d_model, d_ff, num_blocks, num_classes, activation='relu', **kwargs):
        super(TransformerModel, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.activation = tf.keras.activations.get(activation)

        # Define the input and output layers
        self.input_layer = tf.keras.layers.Input(shape=(None, d_model))
        self.output_layer = tf.keras.layers.Dense(num_classes, activation=tf.keras.activations.softmax)

    # Define the blocks
        self.blocks = []
        for i in range(self.num_blocks):
            if i % 2 == 0:
                block = FullyDenseBlock(num_heads, d_k, d_v, d_model, d_ff, activation=self.activation)
            else:
                block = PartiallyDenseBlock(num_heads, d_k, d_v, d_model, d_ff, activation=self.activation)
            self.blocks.append(block)

def call(self, inputs, training=False):
    # Apply the blocks
    for block in self.blocks:
        inputs = block(inputs, training=training)
    return inputs

#Instantiate the model

model = TransformerModel(num_heads=8, d_k=64, d_v=64, d_model=512, d_ff=2048, num_blocks=6, num_classes=10, activation='relu')


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.summary()



"""
This code defines a simple Transformer model with two types of dense blocks, fully dense and partially dense, and a final dense layer for classification. The model is compiled with an optimizer and a loss function.

Please note that this is a very basic example and does not include the full complexity of a Transformer model as seen in the diagram. It also does not include the input and output layers, which would typically be defined in a separate function or class. Additionally, the actual implementation of the attention mechanism and feed-forward sub-layers is not detailed here.

For a complete and functional model, you would need to include the input and output layers, define the attention mechanism and feed-forward sub-layers, and handle the training and evaluation of the model.
"""






# Call the model on a random input
import numpy as np

input_shape = (32,11, 100, 60, 1)

model= ResNet(10, [2, 2, 2, 2], [64, 128, 256, 512])
x = np.ones(input_shape, dtype=np.float32)
output = model(x)
print(output.shape)  # (32, 10)

