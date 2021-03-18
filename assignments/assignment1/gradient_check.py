import numpy as np


def check_gradient(f, x, delta=1e-5, tol = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    # WAAAAAAT?
    # This line was in original code
    #analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        """ 
        x[ix] -= delta
        # f_1 = f(x - delta)
        f_1 = f(x)[0] 

        x[ix] += 2 * delta
        # f_2 = f(x + delta)
        f_2 = f(x)[0]
        # Return x to riginal value
        x[ix] -= delta
        numeric_grad_at_ix = (f_2 - f_1) / (2 * delta)
        
        """
        x_at_ix = x[ix]
        
        x[ix] = x_at_ix + delta
        fx2 = f(x)[0]
        
        x[ix] = x_at_ix - delta
        fx1 = f(x)[0]
        
        x[ix] = x_at_ix
        numeric_grad_at_ix = (fx2 - fx1) / (2 * delta)
        
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.7f, Numeric: %2.7f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True

        

        
