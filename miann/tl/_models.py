# TODO go through functions and comment
import logging
import tensorflow as tf
import json
from copy import deepcopy
from functools import partial
from enum import Enum

class ModelEnum(str, Enum):
    BaseAEModel = "BaseAEModel"
    VAEModel = "VAEModel"
    CatVAEModel = "CatVAEModel"
    GMMVAEModel = "GMMVAEModel"
    CondCatVAEModel = "CondCatVAEModel"

    def get_cls(self):
        cls = self.__class__
        if self == cls.BaseAEModel:
            return BaseAEModel
        elif self == cls.VAEModel:
            return VAEModel
        elif self == cls.CatVAEModel:
            return CatVAEModel
        elif self == cls.GMMVAEModel:
            return GMMVAEModel
        elif self == cls.CondCatVAEModel:
            return CondCatVAEModel
        else:
            raise NotImplementedError

# --- tf functions needed for model definition ---
def expand_and_broadcast(x,s=1):
    """expand tensor x with shape (batches,n) to shape (batches,s,s,n)"""
    C = tf.expand_dims(tf.expand_dims(x, 1), 1)
    C = tf.broadcast_to(C, [tf.shape(C)[0], s, s, tf.shape(C)[-1]])
    return C

def reparameterize_gumbel_softmax(latent, temperature=0.1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    def sample_gumbel(shape, eps=1e-20): 
        """Sample from Gumbel(0, 1)"""
        U = tf.random.uniform(shape,minval=0,maxval=1)
        return -tf.math.log(-tf.math.log(U + eps) + eps)
           
    y = latent + sample_gumbel(tf.shape(latent))
    return tf.nn.softmax( y / temperature)
        
def reparameterize_gaussian(latent):
    z_mean, z_log_var = tf.split(latent, 2, axis=-1)
    eps = tf.random.normal(shape=tf.shape(z_mean))
    return eps * tf.exp(z_log_var * .5) + z_mean

# for gradient reversal for adversarial loss
@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)
                                        
# --- Model classes ---   

BASE_MODEL_CONFIG = {
    'name': None,
    # input definition
    'num_neighbors': 1,
    'num_channels': 35, # can be split up in num_input_channels and num_output_channels
    'num_output_channels': None,
    'num_input_channels': None,
    # conditions are appended to the input and the latent representation. They are assumed to be 1d
    'num_conditions': 0, # if > 0, the model is conditional and an additional input w/ conditions is assumed. 
    # if number or list, the condition is encoded using dense layers with this number of nodes
    'encode_condition': None,
    # which layers of encoder and decoder to apply condition to.
    # Give index of layer in encoder and decoder
    'condition_injection_layers': [0],
    # encoder architecture
    'input_noise': None, # 'gaussian', 'dropout', adds noise to encoder input
    'noise_scale': 0, 
    'encoder_conv_layers': [32],
    'encoder_conv_kernel_size': [1],
    'encoder_fc_layers': [32,16],
    # from last encoder layer, a linear fcl to latent_dim is applied
    'latent_dim': 16,  # number of nodes in latent space (for some models == number of classes)
    # decoder architecture
    # from last decoder layer, a linear fcl to num_output_channels is applied
    'decoder_fc_layers': [],
    # decoder regularizer
    'decoder_regularizer': None, # 'l1' or 'l2'
    'decoder_regularizer_weight': 0,
    # for adversarial models, add adversarial layers
    'adversarial_layers': None, # only works with categorical conditions
}
class BaseAEModel:
    """
    Base class for AE and VAE models. Can have neighbors, conditions (concatenated to input + decoder), and
    and adversarial head. 
    
    Defines init functions for setting up AE with encoder and decoder
    Defines encoder and decoder input layers (can be overwritten in subclassed functions)
    Subclassed models can define
        self.default_config (as class variable)
        self.create_encoder() function, returning encoder model and latent (for KL loss)
        self.create_decoder() function, returning decoder model
        self.create_model() function, creating overall model (put encoder and decoder together)
        
    Default architecture:
    Encoder: (noise) - conv layers - fc layers - linear layer to latent_dim
    Decoder: fc_layers - linear (regularized) layer to num_output_channels
    
    Conditional models additionally output "latent", the latent space (for KL loss computation).
    
    Adversarial models additionally output "adv_head", the output of the adversarial head (for adv loss computation)
    (adv_latent - reverse_gradients - adversarial_layers - linear layer to num_conditions)
    """
    default_config = {
        'name': 'BaseAEModel'
    }
    def __init__(self, **kwargs):
        # set up log and config
        self.log = logging.getLogger(self.__class__.__name__)
        self.config = deepcopy(BASE_MODEL_CONFIG)
        self.config.update(self.default_config)
        self.config.update(kwargs)
        if self.config['num_output_channels'] is None:
            self.config['num_output_channels'] = self.config['num_channels']
        if self.config['num_input_channels'] is None:
            self.config['num_input_channels'] = self.config['num_channels']
        if isinstance(self.config['encoder_conv_kernel_size'], int):
            self.config['encoder_conv_kernel_size'] = [self.config['encoder_conv_kernel_size'] for _ in self.config['encoder_conv_layers']]
        self.log.info('Creating model')
        self.log.debug('Creating model with config: {}'.format(json.dumps(self.config, indent=4)))
        
        # set up model
        # input layers for encoder and decoder
        self.encoder_input = tf.keras.layers.Input((self.config['num_neighbors'], 
                                            self.config['num_neighbors'], 
                                            self.config['num_channels']))
        self.decoder_input = tf.keras.layers.Input((self.config['latent_dim'],))
        if self.is_conditional:
            self.encoder_input = [self.encoder_input, 
                                 tf.keras.layers.Input((self.config['num_conditions'],))]
            self.decoder_input = [self.decoder_input,
                                 tf.keras.layers.Input((self.config['num_conditions'],))]
            
        # set self.encoder, self.latent, self.decoder, self.model_output
        self.model = self.create_model()
        
        # expose layers and summary here
        self.layers = self.model.layers
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        self.encoder.summary(print_fn=lambda x: summary.append(x))
        self.decoder.summary(print_fn=lambda x: summary.append(x))
        if self.is_adversarial:
            self.adv_head.summary(print_fn=lambda x: summary.append(x))
        self.summary = "\n".join(summary)
        
    def create_model(self):
        """
        creates keras model using create_encoder and create_decoder functions.
        sets self.encoder, self.latent, self.decoder, self.model_output attributes
        """
        # encoder and decoder
        self.encoder, self.latent = self.create_encoder()
        self.decoder = self.create_decoder()
        if self.is_adversarial:
            self.adv_head = self.create_adversarial_head()
        
        # create model
        if self.is_conditional:
            #self.model_output = self.decoder([self.encoder.output, self.encoder.input[1]])
            self.model_output = self.decoder([self.encoder(self.encoder_input), self.encoder_input[1]])
        else:
            self.model_output = self.decoder([self.encoder(self.encoder_input)])
        if self.latent is not None:
            # model should return both output + latent (for KL loss)
            self.model_output = [self.model_output, self.latent]
        if self.is_adversarial:
            if isinstance(self.model_output, list):
                self.model_output = self.model_output + [self.adv_head(self.encoder(self.encoder_input))]
            else:
                self.model_output = [self.model_output, self.adv_head(self.encoder(self.encoder_input))]
        model = tf.keras.Model(self.encoder_input, self.model_output, name=self.config['name'])
        return model
        
    @property
    def is_conditional(self):
        return self.config['num_conditions'] > 0
    
    @property
    def is_adversarial(self):
        # needs to have adv layers defined, and be conditional
        return self.config['adversarial_layers'] is not None and self.is_conditional
    
    def encode_condition(self, C):
        if self.config['encode_condition'] == None:
            return C
        if not hasattr(self, 'condition_encoder'):
            enc_l = self.config['encode_condition']
            if isinstance(enc_l, int):
                enc_l = [enc_l]
            inpt = tf.keras.layers.Input((self.config['num_conditions'],))
            x = inpt
            for l in enc_l:
                x = tf.keras.layers.Dense(l, activation=tf.nn.relu)(x)
            self.condition_encoder = tf.keras.Model(inpt, x, name='condition_encoder')
        return self.condition_encoder(C)
    
    def _create_base_encoder(self):
        """
        create base encoder structure with conv layers and fcl. 
        does not apply the last (linear) layer to latent_dim - useful for VAE which does this differently
        """
        if self.is_conditional:
            X,C = self.encoder_input
            C = self.encode_condition(C)
            # broadcast C to fit to X
            #fn = partial(expand_and_broadcast, s=self.config['num_neighbors'])
            #C = tf.keras.layers.Lambda(fn)(C)
        else:
            X = self.encoder_input
        if self.config['input_noise'] is not None:
            # add noise
            X = self.add_noise(X)
        #if self.is_conditional:
        #    # concatenate input and conditions
        #    X = tf.keras.layers.concatenate([X, C], axis=-1)
            
        # conv layers
        cond_layers = self.config['condition_injection_layers']
        for i,l in enumerate(self.config['encoder_conv_layers']):
            # check if need to concatenate current X with C
            if self.is_conditional and i in cond_layers:
                # need to broadcast C to fit to X
                fn = partial(expand_and_broadcast, s=self.config['num_neighbors'])
                C_bcast = tf.keras.layers.Lambda(fn)(C)
                X = tf.keras.layers.concatenate([X, C_bcast], axis=-1)
            k = self.config['encoder_conv_kernel_size'][i]
            X = tf.keras.layers.Conv2D(l, kernel_size=(k,k), activation=tf.nn.relu)(X)
        X = tf.keras.layers.Flatten()(X)
        # fully connected layers
        for j,l in enumerate(self.config['encoder_fc_layers']):
            # check if need to concatenate current X with C
            if self.is_conditional and i+1+j in cond_layers:
                X = tf.keras.layers.concatenate([X, C], axis=-1)
            X = tf.keras.layers.Dense(l, activation=tf.nn.relu)(X)
        return X
    
    def create_encoder(self):
        """
        returns encoder and latent (or none) 
        Encoder outputs reparameterized latent
        Latent is potentially returned by overall model for loss calculation, e.g. for VAE
        """
        X = self._create_base_encoder()
        # linear layer to latent
        X = tf.keras.layers.Dense(self.config['latent_dim'], activation=None, name='latent')(X)
        
        # define encoder model
        encoder = tf.keras.Model(self.encoder_input, X, name='encoder')
        return encoder, None
        
    def create_decoder(self):
        """
        returns decoder
        """
        X = self.decoder_input
        if self.is_conditional:
            X, C = self.decoder_input
            C = self.encode_condition(C)
            # concatenate latent + conditions
            #X = tf.keras.layers.concatenate([X, C])
            
        # fully-connected layers
        cond_layers = self.config['condition_injection_layers']
        for i, l in enumerate(self.config['decoder_fc_layers']):
            # check if need to concatenate current X with C
            if self.is_conditional and i in cond_layers:
                X = tf.keras.layers.concatenate([X, C], axis=-1)
            X = tf.keras.layers.Dense(l, activation=tf.nn.relu)(X)
            if i == 0 and self.is_conditional:
                self.entangled_latent = X # might need this later on
        # if no fully-connected layers are build, need to still concatenate current X with C
        if len(self.config['decoder_fc_layers']) == 0 and self.is_conditional:
            X = tf.keras.layers.concatenate([X, C], axis=-1)
        
        # linear layer to num_output_channels (optionally regularized)
        reg = None
        if self.config['decoder_regularizer'] == 'l1':
            reg = tf.keras.regularizers.l1(self.config['decoder_regularizer_weight'])
        else:
            reg = tf.keras.regularizers.l2(self.config['decoder_regularizer_weight'])
        decoder_output = tf.keras.layers.Dense(self.config['num_output_channels'], activation=None)(X)
        
        # define decoder model
        decoder = tf.keras.Model(self.decoder_input, decoder_output, name='decoder')
        return decoder
    
    def create_adversarial_head(self):
        """
        returns adversarial head (reverse_gradient - adversarial_layers - num_conditions)
        """
        assert self.is_conditional
        assert self.is_adversarial

        inpt = tf.keras.layers.Input((self.config['latent_dim'],))
        X = inpt
        X = GradReverse()(X)
        for l in self.config['adversarial_layers']:
            X = tf.keras.layers.Dense(l, activation=tf.nn.relu)(X)
        # linear layer to num_conditions
        adv_head_output = tf.keras.layers.Dense(self.config['num_conditions'], activation=None)(X)
        
        # define adv_head model
        adv_head = tf.keras.Model(inpt, adv_head_output, name='adv_head')
        return adv_head
        
    def add_noise(self, X):
        if self.config['input_noise'] == 'dropout':
            X = tf.keras.layers.Dropout(self.config['noise_scale'])(X)
        elif self.config['input_noise'] == 'gaussian':
            X = tf.keras.layers.GaussianNoise(self.config['noise_scale'])(X)
        else:
            raise NotImplementedError
        return X
    
class VAEModel(BaseAEModel):
    """
    VAE with simple gaussian prior (trainable with KL loss)
    Encoder: (noise) - conv layers - fc layers - linear layer to latent_dim * 2
    Latent: split latent_dim in half, resample using gaussian prior
    Decoder: fc_layers - linear (regularized) layer to num_output_channels
    """
    default_config = {
        'name': 'VAEModel'
    }
    def create_encoder(self):
        """
        returns encoder and latent (or none) 
        Encoder outputs reparameterized latent
        Latent is returned by overall model for loss calculation
        """
        X = self._create_base_encoder()
        # linear layer to latent
        latent = tf.keras.layers.Dense(self.config['latent_dim']*2, activation=None, name='latent')(X)
        # reparameterise
        reparam_latent = reparameterize_gaussian(latent)
        # define encoder
        encoder = tf.keras.Model(self.encoder_input, reparam_latent, name='encoder')
        return encoder, latent

class CatVAEModel(BaseAEModel):
    """
    VAE with categorical prior (softmax gumbel) (trainable with categorical loss)
    Encoder: (noise) - conv layers - fc layers - linear layer to latent_dim * 2
    Latent: split latent_dim in half, resample using gaussian prior
    Decoder: fc_layers - linear (regularized) layer to num_output_channels
    """
    default_config = {
        'name': 'CatVAEModel',
        'temperature': 0.1, # temperature for scaling gumbel_softmax. values close to 0 are close to true categorical distribution
        'initial_temperature': 10,
        'anneal_epochs': 0,
    }
    
    def __init__(self, **kwargs):
        self.temperature = tf.Variable(initial_value=kwargs.get('initial_temperature', self.default_config['initial_temperature']), 
                             trainable=False, dtype=tf.float32)
        super().__init__(**kwargs)
    
    def create_encoder(self):
        """
        returns encoder and latent 
        Encoder outputs reparameterized latent
        Latent is returned by overall model for loss calculation
        """
        X = self._create_base_encoder()
        # linear layer to latent
        latent = tf.keras.layers.Dense(self.config['latent_dim'], activation=None, name='latent')(X)
        # reparameterise
        reparam_latent = reparameterize_gumbel_softmax(latent, self.temperature)
        # define encoder
        encoder = tf.keras.Model(self.encoder_input, reparam_latent, name='encoder')
        return encoder, latent

class CondCatVAEModel(CatVAEModel):
    """
    Conditional Categorical VAE using another concatenation scheme when adding the condition
    to the latent space. This model first calculates a fully connected layer to a vector with length #output_channels x #conditions

    IGNORES decoder_fc_layers - only supports linear decoder!
    """
    
    def create_decoder(self):
        """
        returns decoder
        """

        X, C = self.decoder_input
        
        # dense layer to num_output_channels x num_conditions
        X = tf.keras.layers.Dense(self.config['num_output_channels'] * self.config['encode_condition'], activation=None)(X)
        X = tf.keras.layers.Reshape((self.config['num_output_channels'], self.config['encode_condition']))(X)
        
        C = self.encode_condition(C)
        # multiply X by conditions
        decoder_output = tf.keras.layers.Dot(axes=[2,1])([X,C])
        
        # define decoder model
        decoder = tf.keras.Model(self.decoder_input, decoder_output, name='decoder')
        return decoder
    
class GMMVAEModel(BaseAEModel):
    """
    VAE with gmm prior (trainable with categorical loss for y and weighted kl loss for z)
    Encoder y: (noise) - conv layers y - fc layers y - linear layer to latent_dim
    Encoder: (noise) + y - conv layers - fc layers - linear layer to latent_dim * 2
    Latent: split latent_dim in half, resample using gaussian prior
    Decoder: fc_layers - linear (regularized) layer to num_output_channels
    """
    default_config = {
        'name': 'GMMVAEModel',
        # y encoder architecture
        'y_conv_layers': None,
        'y_conv_kernel_size': None,
        'y_fc_layers': None,
        # pz (gmm prior for zmean and zvar from categorical y)
        'pz_fc_layers': None,
        # number of different gaussians
        'k': 10,
        # temperature for categorical loss on y <- might not need to anneal!
        'temperature': 0.1, # temperature for scaling gumbel_softmax. values close to 0 are close to true categorical distribution
        'initial_temperature': 10,
        'anneal_epochs': 0,
    }
    
    def __init__(self, **kwargs):
        # make sure y_... is defined
        config = deepcopy(BASE_MODEL_CONFIG)
        config.update(self.default_config)
        config.update(kwargs)
        if config['y_conv_layers'] is None:
            config['y_conv_layers'] = config['encoder_conv_layers']
        if config['y_conv_kernel_size'] is None:
            config['y_conv_kernel_size'] = config['encoder_conv_kernel_size']
        if config['y_fc_layers'] is None:
            config['y_fc_layers'] = config['encoder_fc_layers']
        if config['pz_fc_layers'] is None:
            config['pz_fc_layers'] = config['encoder_fc_layers']
        super().__init__(**config)
        
    def qy_graph(self, input_shape):
        """
        returns Y calculated from X (conv layers + fcl layers)
        """
        X_input = tf.keras.layers.Input(input_shape)
        X = X_input
        # conv layers
        for i,l in enumerate(self.config['y_conv_layers']):
            k = self.config['y_conv_kernel_size'][i]
            X = tf.keras.layers.Conv2D(l, kernel_size=(k,k), activation=tf.nn.relu)(X)
        X = tf.keras.layers.Flatten()(X)
        # fully connected layers
        for l in self.config['y_fc_layers']:
            X = tf.keras.layers.Dense(l, activation=tf.nn.relu)(X)
        # linear layer to latent
        Y = tf.keras.layers.Dense(self.config['k'], activation='softmax')(X)
        model = tf.keras.Model(X_input, Y)
        return model
    
    def create_y_encoder(self, X):
        """
        returns Y calculated from X (conv layers + fcl layers)
        """
        # conv layers
        for i,l in enumerate(self.config['y_conv_layers']):
            k = self.config['y_conv_kernel_size'][i]
            X = tf.keras.layers.Conv2D(l, kernel_size=(k,k), activation=tf.nn.relu)(X)
        X = tf.keras.layers.Flatten()(X)
        # fully connected layers
        for l in self.config['y_fc_layers']:
            X = tf.keras.layers.Dense(l, activation=tf.nn.relu)(X)
        # linear layer to latent
        Y = tf.keras.layers.Dense(self.config['k'], activation='softmax', name='latent_y')(X)
        return Y
        
    def qz_graph(self, X_input_shape):
        """
        returns Z calculated from X and Y
        """
        X_input = tf.keras.layers.Input(X_input_shape)
        Y_input = tf.keras.layers.Input((self.config['k'],))
        
        # concatenate Y with X
        fn = partial(expand_and_broadcast, s=self.config['num_neighbors'])
        Y = tf.keras.layers.Lambda(fn)(Y_input)
        X = tf.keras.layers.concatenate([X_input,Y], axis=-1) # this is now the input for the normal encoder
        
        # conv layers
        for i,l in enumerate(self.config['encoder_conv_layers']):
            k = self.config['encoder_conv_kernel_size'][i]
            X = tf.keras.layers.Conv2D(l, kernel_size=(k,k), activation=tf.nn.relu)(X)
        X = tf.keras.layers.Flatten()(X)
        # fully connected layers
        for l in self.config['encoder_fc_layers']:
            X = tf.keras.layers.Dense(l, activation=tf.nn.relu)(X)
        
        # linear layer to latent
        Z = tf.keras.layers.Dense(self.config['latent_dim']*2, activation=None)(X)
        model = tf.keras.Model((X_input, Y_input), Z, name='qz_model')
        return model
    
    def pz_graph(self):
        """prior distibution of Z for different categories (Y should be 1-hot encoded vector)"""
        Y_input = tf.keras.layers.Input((self.config['k'],))
        X = Y_input
        # fully connected layers
        for l in self.config['pz_fc_layers']:
            X = tf.keras.layers.Dense(l, activation=tf.nn.relu)(X)
        Z = tf.keras.layers.Dense(self.config['latent_dim']*2, activation=None)(X)
        model = tf.keras.Model(Y_input, Z, name='pz_model')
        return model
            
    def create_encoder(self):
        """
        returns encoder and latent 
        Encoder outputs reparameterized latent
        Latent is returned by overall model for loss calculation
        """
        if self.is_conditional:
            X,C = self.encoder_input
            # broadcast C to fit to X
            fn = partial(expand_and_broadcast, s=self.config['num_neighbors'])
            C = tf.keras.layers.Lambda(fn)(C)
        else:
            X = self.encoder_input
        if self.config['input_noise'] is not None:
            X = self.add_noise(X)
        if self.is_conditional:
            # concatenate input and conditions
            X = tf.keras.layers.concatenate([X, C], axis=-1)
        
        # define qz and pz subgraphs
        shape_X = X.shape.as_list()[1:]
        qz_model = self.qz_graph(shape_X)
        pz_model = self.pz_graph()
        
        # get qy
        latent_y = self.create_y_encoder(X)
        
        # get qz for qy value (for reconstruction)
        Z = qz_model([X, latent_y])
        # reparameterise
        reparam_Z = reparameterize_gaussian(Z)
        # define encoder
        encoder = tf.keras.Model(self.encoder_input, reparam_Z, name='encoder')
        
        # get pz, qz for different values of y
        pZ = []
        qZ = []
        # functions for expanding Y to have batch_size dim
        def expand_and_broadcastY(Y, s):
            Y = tf.expand_dims(Y, 0)
            Y = tf.broadcast_to(Y, [s, self.config['k']])
            return Y
        exp_fn = partial(expand_and_broadcastY, s=tf.shape(X)[0])
        for i in range(0, self.config['k']):
            Yi = tf.one_hot(i, depth=self.config['k'])
            Yi = tf.keras.layers.Lambda(exp_fn)(Yi)
            pZ.append(pz_model(Yi))
            qZ.append(qz_model([X, Yi]))
        # stack together (shape: None, k, latent_dim)
        pZ = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(pZ)
        qZ = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(qZ)
        # expand latent_y to have shape: None, k, latent_dim
        def expand_and_broadcast_qY(Y, s):
            Y = tf.expand_dims(Y, axis=-1)
            Y = tf.broadcast_to(Y, [tf.shape(Y)[0], tf.shape(Y)[1], s])
            return Y
        fn = partial(expand_and_broadcast_qY, s=self.config['latent_dim']*2)
        qY = tf.keras.layers.Lambda(fn)(latent_y)
        # stack pZ, qZ, qY to one output
        latent_zy = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1), name='latent_zy')([pZ, qZ, qY])
        
        return encoder, latent_y, latent_zy 
    
    
    def create_model(self):
        """
        creates keras model using create_encoder and create_decoder functions.
        sets self.encoder, self.latent, self.decoder, self.model_output attributes
        """
        # encoder and decoder
        self.encoder, self.latent_y, self.latent_zy = self.create_encoder()
        self.decoder = self.create_decoder()
        
        # add encoder_y model
        self.encoder_y = tf.keras.Model(self.encoder_input, self.latent_y, name='encoder_y')
        
        # create model
        if self.is_conditional:
            self.model_output = self.decoder([self.encoder(self.encoder_input), self.encoder_input[1]])
        else:
            self.model_output = self.decoder([self.encoder(self.encoder_input)])
        # model should return both output + latent_y (for cat loss) + latent_zy (for KL loss)
        self.model_output = [self.model_output, self.latent_y, self.latent_zy]
        model = tf.keras.Model(self.encoder_input, self.model_output, name=self.config['name'])
        return model
