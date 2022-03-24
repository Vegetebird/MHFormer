import sys
import numpy as np
import torch

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]


def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) 
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) 


def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


def wrap(func, *args, unsqueeze=False):
	args = list(args)
	for i, arg in enumerate(args):
	    if type(arg) == np.ndarray:
	        args[i] = torch.from_numpy(arg)
	        if unsqueeze:
	            args[i] = args[i].unsqueeze(0)

	result = func(*args)

	if isinstance(result, tuple):
	    result = list(result)
	    for i, res in enumerate(result):
	        if type(res) == torch.Tensor:
	            if unsqueeze:
	                res = res.squeeze(0)
	            result[i] = res.numpy()
	    return tuple(result)
	elif type(result) == torch.Tensor:
	    if unsqueeze:
	        result = result.squeeze(0)
	    return result.numpy()
	else:
	    return result

def qrot(q, v):
	assert q.shape[-1] == 4
	assert v.shape[-1] == 3
	assert q.shape[:-1] == v.shape[:-1]

	qvec = q[..., 1:]
	uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
	uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
	return (v + 2 * (q[..., :1] * uv + uuv))


def qinverse(q, inplace=False):
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape) - 1)



        