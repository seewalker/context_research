from typing import Dict,Tuple,List,NewType,Union,Callable,TypeVar
import tensorflow as tf
import numpy as np

weight_t = NewType('weight_t',Dict[str,tf.Tensor])
# tuple of weights and biases.
parameters_t = NewType('parameters_t',Tuple[weight_t,weight_t])
debug_t = NewType('debug_t',Union[type(None),Dict])
feed_t = NewType('feed_t',Dict[str,tf.Tensor])

bbox_t = NewType('bbox_t',Tuple[int,int,int,int])
