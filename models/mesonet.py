"""
MesoNet: Meso4 Architecture for Deepfake Detection

Based on: "MesoNet: a Compact Facial Video Forgery Detection Network"
Original implementation: https://github.com/DariusAf/MesoNet
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam


class Meso4:
    """
    Meso4 - Lightweight CNN for deepfake detection
    
    Architecture:
    - 4 convolutional blocks
    - Batch normalization
    - LeakyReLU activation
    - Dropout for regularization
    - Binary classification output
    """
    
    def __init__(self, learning_rate=0.001, input_shape=(256, 256, 3)):
        self.model = None
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        
    def build(self):
        """Build the Meso4 architecture"""
        
        x = Input(shape=self.input_shape)
        
        # Block 1
        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        # Block 2
        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        # Block 3
        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        # Block 4
        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        # Fully connected layers
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)
        
        self.model = Model(inputs=x, outputs=y)
        
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        
        return self.model
    
    def load_weights(self, weights_path):
        """Load pretrained weights"""
        if self.model is None:
            self.build()
        self.model.load_weights(weights_path)
        
    def predict(self, x, **kwargs):
        """Make prediction on input"""
        if self.model is None:
            raise ValueError("Model not built. Call build() or load_weights() first.")
        return self.model.predict(x, **kwargs)
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            self.build()
        return self.model.summary()


class MesoInception4:
    """
    MesoInception4 - Enhanced version with Inception modules
    
    More complex than Meso4 but still lightweight
    Better feature extraction capabilities
    """
    
    def __init__(self, learning_rate=0.001, input_shape=(256, 256, 3)):
        self.model = None
        self.learning_rate = learning_rate
        self.input_shape = input_shape
    
    def inception_module(self, x, filters):
        """Inception module for multi-scale feature extraction"""
        
        # 1x1 convolution
        branch1 = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
        
        # 1x1 -> 3x3 convolution
        branch2 = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
        branch2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(branch2)
        
        # 1x1 -> 3x3 -> 3x3 convolution
        branch3 = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
        branch3 = Conv2D(filters, (3, 3), padding='same', activation='relu')(branch3)
        branch3 = Conv2D(filters, (3, 3), padding='same', activation='relu')(branch3)
        
        # Max pooling -> 1x1 convolution
        branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch4 = Conv2D(filters, (1, 1), padding='same', activation='relu')(branch4)
        
        # Concatenate all branches
        output = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
        return output
    
    def build(self):
        """Build the MesoInception4 architecture"""
        
        x = Input(shape=self.input_shape)
        
        # Inception block 1
        x1 = self.inception_module(x, 1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        # Inception block 2
        x2 = self.inception_module(x1, 2)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        # Conv block
        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        # Conv block
        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        # Fully connected layers
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)
        
        self.model = Model(inputs=x, outputs=y)
        
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        
        return self.model
    
    def load_weights(self, weights_path):
        """Load pretrained weights"""
        if self.model is None:
            self.build()
        self.model.load_weights(weights_path)
        
    def predict(self, x, **kwargs):
        """Make prediction on input"""
        if self.model is None:
            raise ValueError("Model not built. Call build() or load_weights() first.")
        return self.model.predict(x, **kwargs)
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            self.build()
        return self.model.summary()


if __name__ == "__main__":
    # Test model creation
    print("=" * 60)
    print("Testing Meso4 Architecture")
    print("=" * 60)
    
    meso4 = Meso4(input_shape=(256, 256, 3))
    meso4.build()
    meso4.summary()
    
    print("\n" + "=" * 60)
    print("Testing MesoInception4 Architecture")
    print("=" * 60)
    
    meso_inception = MesoInception4(input_shape=(256, 256, 3))
    meso_inception.build()
    meso_inception.summary()
