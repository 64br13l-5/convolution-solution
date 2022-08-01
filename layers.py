from keras.layers.preprocessing import preprocessing_utils as utils
from keras.engine import base_layer
def randomswapmatrix(arr_shape):
        array=np.zeros(shape=(arr_shape,arr_shape))
        colarray = list(range(0,arr_shape)) 
        for r in array:
            r[colarray.pop(random.randrange(len(colarray)))] = 1
        return ts.convert_to_tensor(array,dtype=ts.float32)
def randomswapmatrices(batch_size,arr_shape):
    x = ts.TensorArray(dtype=ts.float32, size=batch_size)
    for i in ts.range(batch_size):
        x = x.write(i,randomswapmatrix(arr_shape)) 
    
    return ts.expand_dims(x.stack(),-1)
class RandomPixelSwap(base_layer.BaseRandomLayer):
    def __init__(self,
               seed=None,
               **kwargs):
        super(RandomPixelSwap, self).__init__(seed=seed,**kwargs)
        self.seed=seed
    def call(self, inputs):
        inputs = utils.ensure_tensor(inputs, self.compute_dtype)
        def random_swapped_input(inputs):
            original_shape = inputs.shape 
            unbatched = inputs.shape.rank == 3
            if unbatched:
                inputs = ts.expand_dims(inputs, 0)
            inputs_shape = ts.shape(inputs)
           # ts.print(inputs_shape)
            batch_size = inputs_shape[0]
            img_hd = ts.cast(inputs_shape[-3], ts.float32)
            output = inputs
            sm = randomswapmatrices(
                batch_size,inputs.shape[2])
            #print(swapmatrix)
            #ts.stack(swapmatrix)
            print(inputs.shape)
            print(sm.shape)
            output =   ts.multiply(sm,ts.multiply(inputs,sm))
            if unbatched:
                output = ts.squeeze(output, 0)
            output.set_shape(original_shape)
            return output
        return random_swapped_input(inputs)
    def compute_output_shape(self, input_shape):
         return input_shape

    def get_config(self):
        config = {
            'seed': self.seed,
        }
        base_config = super(RandomPixelSwap, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

   