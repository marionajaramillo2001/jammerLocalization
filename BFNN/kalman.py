import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as FF
import cubature as cb
from scipy.special import expit


class KalmanFilter:

    def __init__(self, h, f, s0, P0, Q, R, jac):
        """

        :param h:
        :param f:
        :param s0:
        :param P0:
        :param Q:
        :param R:
        :param jac:
        """

        self.h = h
        self.f = f
        self.s = s0
        self.P = P
        self.Q = Q
        self.R = R
        self.jac = jac


    def predict(self):
        s = self.f.dot(self.s)
        # A = self.jac(self.f(self.s))
        # P = A.dot(self.P).dot(A.T)
        P = self.f.dot(self.P).dot(self.f.T)
        return s, P

    def update(self, y):
        s_pred, P_pred = self.predict()
        e = y - self.h.dot(self.s)
        S = self.h.dot(P_pred).dot(self.h.T) + self.R
        K = P_pred.dot(self.h.T).dot(np.linalg.inv(S))
        self.s = s_pred + K.dot(e)
        self.P = (np.eye(len(self.s)) - K.dot(self.h)).dot(P_pred)


class NNBF(nn.Module):
# y1 = sigmoid ( W1*x + Wf*y1 + b)
# yk = w2 * y1k  + b2

    def __init__(self, d_y1, d_x, d_y, y1_0):
        super().__init__()
        self.y1 = y1_0
        self.W1 = torch.nn.Parameter(torch.randn((d_y1, d_x)))
        self.Wf = torch.nn.Parameter(torch.randn((d_y1, d_y1)))
        self.W2 = torch.nn.Parameter(torch.randn((d_y, d_y1)))
        self.b1 = torch.nn.Parameter(torch.randn(()))
        self.b2 = torch.nn.Parameter(torch.randn(()))

        # self.w =

    def forward(self, x):

        # getting value of y1 to avoid backpropagating through time!
        self.y1d = self.y1.detach()
        self.y1d.requires_grad = True

        a = torch.mm(self.W1, x)
        b = torch.mm(self.Wf, self.y1d)
        c = a + b + self.b2
        self.y1 = torch.sigmoid(torch.mm(self.W1, x) + torch.mm(self.Wf, self.y1d) + self.b2)

        y = torch.mm(self.W2, self.y1) + self.b2
        return y

    def update_weights(self):

        # create A using self.W1.grad self.Wf.grad self.b1.grad
        # create C

        pass



class NNEKF(nn.Module):
# y1 = sigmoid ( W1*x + Wf*y1 + b)
# yk = w2 * y1k  + b2

    def __init__(self, observation_model, P0, R, Q):
        super().__init__()

        ##self.state_model = state_model
        self.observation_model = observation_model
        
        # internal state not used if no recursion (it contains initial state of all neurons)
        ##self.s = s0
        
        
        # self.w_dim = len(self.w)
        ##self.w_dim = len(torch.cat((state_model.get_param(), observation_model.get_param())))
        self.w_dim = len(observation_model.get_param())
        
        ##self.s_dim = len(self.s)
        
        
        self.d_dim = self.observation_model.y_dim
        self.P = P0
        self.R = R
        self.Q = Q

    def forward_backward(self, x, d):
        """Compute jacobians from state and observation models. Then perform 
        the Kalman routine"""
        # --- State Model matrix ------------
        # merging inputs
        # xin = x.clone().detatch().requires_grad_(True)
        
        
        ##s_in = self.s.clone().detach().requires_grad_(True)
        ##input = torch.cat((s_in, x))
        

        # passing through the model
        ##s = self.state_model.forward(x)

        # compute gradient of internal outputs
        ##for i, si in enumerate(s):
            ##si.backward(retain_graph=True)
            
            
        """
        # computing Jacobians
        Jss = torch.zeros((self.s_dim, self.s_dim))
        Jsw = torch.zeros((self.s_dim, self.w_dim))

        # Populate Jacobians with gradients
        # for i, si in enumerate(s[0]):
        #     print(si)
        for i, si in enumerate(s):
            si.backward(retain_graph=True)
            Jss[i, :] = s_in.grad.data.T
            # w = self.state_model.get_param()
            # Jsw[i, :] = self.w.grad.data.T
            # Jsw[i, :] = self.get_grad()
            # improve the following line!
            a = self.state_model.get_grad()
            b = torch.zeros(len(self.observation_model.get_param()))
            # b = self.observation_model.get_grad()
            if b.nelement() == 0:
                b = torch.zeros(len(self.observation_model.get_param()))
            Jsw[i, :] = torch.cat((a, b))
        # build F with
        # F = [[I,    0 ]]
        #     [[J_w, J_s]]
        I = torch.eye(self.w_dim)
        Zeros = torch.zeros((self.w_dim, self.s_dim))

        F = torch.cat((torch.cat((I, Jsw)), torch.cat((Zeros, Jss))), 1)
        """
        I = torch.eye(self.w_dim)
        F = I
    
        ##s_mid = s.clone().detach().requires_grad_(True)
        # s_mid = s.requires_grad_(True)

        # --- Observation Model Matrix ------------
        ##y = self.observation_model.forward(s_mid)
        y = self.observation_model.forward(x)

        
        ##Jys = torch.zeros((len(y), self.s_dim))
        
        
        Jyw = torch.zeros((len(y), self.w_dim))
        for i in range(len(y)):
            y[i].backward(retain_graph=True)
            
            ##Jys[i, :] = s_mid.grad.data.T
            
            
            # Jyw[i, :] = self.w.grad.data.T
            ##a = torch.zeros(len(self.state_model.get_param())) #dv of meas equation wrt state weights e.g. wxl (x->l1)
            # a = self.state_model.get_grad()
            b = self.observation_model.get_grad() #dv of meas equation wrt obs weights e.g. wly (l1->y)
            ##Jyw[i, :] = torch.cat((a, b)).T
            Jyw[i, :] = b.T
            

        # H = [[ Jyw , Jys]]
        ##H = torch.cat((Jyw, Jys), 1)
        
        H = Jyw
        
        # --- Extended KF
        e = d - y
        P_ = torch.mm(torch.mm(F, self.P), F.T) + self.R
        K = torch.mm(torch.mm(P_, H.T), torch.inverse(torch.mm(torch.mm(H, P_), H.T) + self.Q))
        # self.P = torch.mm(torch.eye(len(P_)) - torch.mm(K, H), P_) + self.R
        self.P = torch.mm(torch.eye(len(P_)) - torch.mm(K, H), P_)

        ##K1 = K[0:self.w_dim]
        ##K2 = K[self.w_dim:]
        
        K1 = K        
        
        # w = self.w.detach() + torch.mm(K1, e).detach()
        a = self.get_param().reshape(-1, 1) # state predict (i.e. param so far)
        ##b = torch.mm(K1, e).detach()
        w = a + torch.mm(K1, e).detach()
   
        #w.requires_grad = True

        self.set_param(w)

        # Internal state not used if no recursion (internal output)
        ##self.s = s.detach() + torch.mm(K2, e).detach()
        ##self.s.requires_grad = True

        return y, e
    
    def forward(self,x):
        """
        Run the model and return the output
        """
        
        # passing through the model
        # input through the hidden layers
        ##s = self.state_model.forward(x)
        ##s_mid = s.clone().detach().requires_grad_(False)
        # last layer to output
        ##y = self.observation_model.forward(s_mid)
        y = self.observation_model.forward(x)

        return y

    def get_param(self):
        ## return torch.cat((self.state_model.get_param(), self.observation_model.get_param()))
        return self.observation_model.get_param()

    def set_param(self, params):
        ##self.state_model.set_param(params[0:self.state_model.param_dim])
        ##self.observation_model.set_param(params[self.state_model.param_dim:])
        self.observation_model.set_param(params)

    def get_grad(self):
        ##a = self.state_model.get_grad()
        ##b = self.observation_model.get_grad()
        ##return torch.cat(self.state_model.get_grad(), self.observation_model.get_grad())
        return self.observation_model.get_grad()


class StateModel(nn.Module):
    """Build a state model for the bayesian filter so that jacobians can be 
    computed"""
    def __init__(self, x_dim, s_dim, nonlinearity):
        super().__init__()

        # self.wxl = torch.nn.Parameter(torch.randn((s_dim, x_dim)))
        # self.wsl = torch.nn.Parameter(torch.randn((s_dim, s_dim)))
        # self.b = torch.nn.Parameter(torch.randn((1, 1)))
        
        ##self.wxl = torch.nn.Parameter(torch.zeros((s_dim[0], x_dim)))
        
        ##self.wsl = torch.nn.Parameter(torch.zeros((s_dim, s_dim)))
        
        
        ##self.b = torch.nn.Parameter(torch.zeros((1, 1)))
        
        # list of hidden layers' weight matrices
        self.W = nn.ParameterList([torch.nn.Parameter(torch.zeros((s_dim[0], x_dim)))])
        self.B = nn.ParameterList([torch.nn.Parameter(torch.zeros((1, 1)))])
        
        for i in range(len(s_dim)-1):
            w = torch.nn.Parameter(torch.zeros((s_dim[i+1], s_dim[i])))
            self.W.append(w)
            b = torch.nn.Parameter(torch.zeros((1, 1)))
            self.B.append(b)
            
        

        # Define sigmoid activation and softmax output
        self.param_dim = len(self.get_param())


        # Set the nonlinearity
        if nonlinearity == "sigmoid":
            #self.nonlinearity = lambda x: torch.sigmoid(x)
            self.nonlinearity = nn.Sigmoid()
        elif nonlinearity == "softmax":
            self.nonlinearity = nn.Softmax(dim=1)
        elif nonlinearity == "relu":
            #self.nonlinearity = lambda x: FF.relu(x)
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == "softplus":
            #self.nonlinearity = lambda x: FF.softplus(x)
            pass
        elif nonlinearity == 'tanh':
            #self.nonlinearity = lambda x: torch.tanh(x) #FF.tanh(x)
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == 'leaky_relu':
            #self.nonlinearity = lambda x: FF.leaky_relu(x)
            pass
        elif nonlinearity == 'softplus':
            #self.nonlinearity = lambda x: FF.softplus(x)
            pass
        elif nonlinearity == 'leaky_relu':
            #self.nonlinearity = lambda x: FF.leaky_relu(x)
            pass

    def forward(self, xs):
        # Pass the input tensor through each of our operations
        # ox = self.wxl(xs[0:x_dim].T)
        # os = self.wsl(xs[x_dim:].T)
        ox = torch.mm(self.wxl, xs)
        
        
        
        ##os = torch.mm(self.wsl, xs[x_dim:])


        # x = self.output(x)
        # x = self.softmax(x)

        ##s = self.nonlinearity(ox + os + self.b)
        s = self.nonlinearity(ox + self.b)
        
        return s

    def get_param(self):
        """
        Return weights of this net as a vector
        """
        param_vector = torch.tensor([])  # Initial weights
        for param in self.parameters():
            param_vector = torch.cat((param_vector, param.view(-1)))

        return param_vector

    def get_grad(self):
        grad = torch.tensor([])  # Initial weights
        for param in self.parameters():
            # if param.grad is not None:
            grad = torch.cat((grad, param.grad.view(-1)))
        return grad

    def set_param(self, param_vec):
        """
        Given a vector of parameters, sets the parameters of self
        """
        i = 0
        for param in self.parameters():
            j = param.nelement()
            a = param_vec[i:i + j]
            b = param.size()

            param.data = param_vec[i:i + j].view(param.size())
            param.grad.zero_()
            i += j


class ObservationModel(nn.Module):
    """Build a measurement model for the bayesian filter so that jacobians can 
    be computed"""
    def __init__(self, layers, nonlinearity, init_weights = "zeros", bias = "shared"):
        super().__init__()
        
        # Note: substitute s_dim[0] and s_dim[i+1] with 1 in self.B and b to get 1 bias per layer

        # Inputs to hidden layer linear transformation
       
        ##self.ws = nn.Parameter(torch.zeros((y_dim, s_dim)))
        ##self.b = nn.Parameter(torch.zeros((1, 1)))

                
        x_dim = layers[0]
        s_dim = layers[1:]
        
        # initial weights
        if init_weights == "zeros":
            def myzeros(out_features,in_features,bias=0):
                if bias == "independent":
                    return torch.zeros(out_features,1)
                elif bias == "shared":
                    return torch.zeros(1,1)
                else:
                    return torch.zeros(out_features,in_features)
            init_fn = myzeros
        elif init_weights == "rand":
            # uniform random distribution depending on no. of in_features (see torch.nn.Linear)
            def rand_fn(out_features,in_features,bias=0):
                k = np.sqrt(1/in_features)
                if bias == "independent":
                    A = -2*k * torch.rand(out_features, 1) + k
                elif bias == "shared":
                    A = -2*k * torch.rand(1, 1) + k
                else:
                    A = -2*k * torch.rand(out_features, in_features) + k
                return A
            init_fn = rand_fn
            
        
        # list of hidden layers' weight matrices
        self.W = nn.ParameterList([torch.nn.Parameter(init_fn(s_dim[0], x_dim))])
        self.B = nn.ParameterList([torch.nn.Parameter(init_fn(s_dim[0], x_dim, bias))])
        
        for i in range(len(s_dim)-1):
            w = torch.nn.Parameter(init_fn(s_dim[i+1], s_dim[i]))
            self.W.append(w)
            b = torch.nn.Parameter(init_fn(s_dim[i+1], s_dim[i], bias))
            self.B.append(b)
        
        self.y_dim = layers[-1]
        self.param_dim = len(self.get_param())
        
        # Set the nonlinearity
        if nonlinearity == "sigmoid":
            #self.nonlinearity = lambda x: torch.sigmoid(x)
            self.nonlinearity = nn.Sigmoid()
        elif nonlinearity == "softmax":
            self.nonlinearity = nn.Softmax(dim=1)
        elif nonlinearity == "relu":
            #self.nonlinearity = lambda x: FF.relu(x)
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == "softplus":
            #self.nonlinearity = lambda x: FF.softplus(x)
            pass
        elif nonlinearity == 'tanh':
            #self.nonlinearity = lambda x: torch.tanh(x) #FF.tanh(x)
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == 'leaky_relu':
            #self.nonlinearity = lambda x: FF.leaky_relu(x)
            pass
        elif nonlinearity == 'softplus':
            #self.nonlinearity = lambda x: FF.softplus(x)
            pass
        elif nonlinearity == 'leaky_relu':
            #self.nonlinearity = lambda x: FF.leaky_relu(x)
            pass
        

    def forward(self, x):
        # Pass the input tensor through each of our operations
        # y = self.ws(s)
        ##y = torch.mm(self.ws, s) + self.b
        
        
        for i in range(len(self.W)-1):
            zl = torch.mm(self.W[i], x) + self.B[i]
            x = self.nonlinearity(zl)
        y = torch.mm(self.W[i+1], x) + self.B[i+1]
        return y

    def get_param(self):
        """
        Return weights of this net as a vector
        """
        param_vector = torch.tensor([])  # Initial weights
        for param in self.parameters():
            param_vector = torch.cat((param_vector, param.view(-1)))

        return param_vector

    def get_grad(self):
        # return self.ws.grad()
        grad = torch.tensor([])  # Initial weights
        for param in self.parameters():
            if param.grad is not None:
                grad = torch.cat((grad, param.grad.view(-1)))
        return grad

    def set_param(self, param_vec):
        """
        Given a vector of parameters, sets the parameters of self
        """
        i = 0
        for param in self.parameters():
            j = param.nelement()
            param.data = param_vec[i:i + j].view(param.size())
            param.grad.zero_()
            i += j


class CubatureFilter:

    def __init__(self, f, h, x0, P0, Q0, R0, reg_const=0.01):
        """
        class CubatureFilter
        :param f:
        :param h:
        :param x0:
        :param P0:
        :param Q0:
        :param R0:
        :param reg_const:
        """
        self.f = f
        self.fft = lambda x, u: np.outer(self.f(x, u), self.f(x, u))
        self.h = h
        self.hht = lambda x: np.outer(self.h(x), self.h(x))
        self.x = x0
        self.P = P0
        self.Q = Q0
        self.R = R0
        self.dim_x = len(self.x)
        if callable(R0):
            self.dim_y = R0().shape[0]
        else:
            self.dim_y = R0.shape[0]
        self.n_cubature_points = 2*self.dim_x
        self.reg_mat = reg_const*np.eye(len(x0))

    def predict(self, u=None):
        x_pred, cubature_points = cb.cubature_int(self.f, self.x, self.P, return_cubature_points=True, u=u)
        P_pred = cb.cubature_int(self.fft, self.x, self.P, cubature_points=cubature_points, u=u) - \
                 np.outer(x_pred, x_pred) + self.Q

        return x_pred, P_pred

    def update(self, y, u=None):
        x_pred, P_pred = self.predict(u=u)

        # P_pred = P_pred + self.reg_mat
        x_cubature_points = cb.gen_cubature_points(x_pred, P_pred, self.dim_x, self.n_cubature_points)

        # y_pred = cb.cubature_int(self.h, x_pred, P_pred, cubature_points=x_cubature_points)
        y_cubature_points = np.array([self.h(x_cubature_points[i]) for i in range(len(x_cubature_points))])
        y_pred = np.mean(y_cubature_points, axis=0)

        if callable(self.R):
            R = self.R(x_pred)
        else:
            R = self.R

        P_yy = cb.cubature_int(self.hht, x_pred, P_pred, cubature_points=x_cubature_points) - \
               np.outer(y_pred, y_pred) + R

        P_xy = np.mean([np.outer(x_cubature_points[i], y_cubature_points[i]) for i in range(self.n_cubature_points)],
                    axis=0) - np.outer(x_pred, y_pred)

        Kg = np.dot(P_xy, cb.inv_pd_mat(P_yy))

        self.x = x_pred + np.dot(Kg, (y - y_pred))
        self.P = P_pred - np.dot(Kg, P_yy).dot(Kg.T)

        return y_pred, (y - y_pred)

    def get_states(self):
        return self.x


class SquareRootCubatureFilter:

    def __init__(self, f, h, x0, P0, Q0, R0, reg_const=0.01):
        """
        class CubatureFilter
        :param f:
        :param h:
        :param x0:
        :param P0:
        :param Q0:
        :param R0:
        :param reg_const:
        """
        self.f = f
        self.fft = lambda x, u: np.outer(self.f(x, u), self.f(x, u))
        self.h = h
        self.hht = lambda x: np.outer(self.h(x), self.h(x))
        self.x = x0
        # self.P = P0
        self.S = np.linalg.qr(P0)[1].T
        # self.Q = Q0
        self.S_Q = np.linalg.qr(Q0)[1].T
        # self.R = R0
        self.S_R = np.linalg.qr(R0)[1].T
        self.dim_x = len(self.x)
        if callable(R0):
            self.dim_y = R0().shape[0]
        else:
            self.dim_y = R0.shape[0]
        self.n_cubature_points = 2*self.dim_x
        self.reg_mat = reg_const*np.eye(len(x0))

    def predict(self, u=None):

        cubature_points = cb.gen_cubature_points(self.x, self.S, self.dim_x, self.n_cubature_points, sqr=True)
        if u is not None:
            X_ = np.concatenate([self.f(xi, u).reshape(1, -1) for xi in cubature_points], axis=0)
        else:
            X_ = np.concatenate([self.f(xi).reshape(1, -1) for xi in cubature_points], axis=0)

        x_pred = np.mean(X_, axis=0)

        XX = ((X_ - x_pred)/np.sqrt(self.n_cubature_points)).T
        # getting R^T (lower triangular matrix)
        S_pred = np.linalg.qr(np.concatenate((XX, self.S_Q), axis=1).T)[1].T

        return x_pred, S_pred

    def update(self, y, u=None):

        x_pred, S_pred = self.predict(u=u)

        # P_pred = P_pred + self.reg_mat
        x_cubature_points = cb.gen_cubature_points(x_pred, S_pred, self.dim_x, self.n_cubature_points, sqr=True)

        # y_pred = cb.cubature_int(self.h, x_pred, P_pred, cubature_points=x_cubature_points)
        # y_cubature_points = np.array([self.h(x_cubature_points[i]) for i in range(len(x_cubature_points))])
        # y_pred = np.mean(y_cubature_points, axis=0)
        Y_cubature_points = np.concatenate([self.h(xcb).reshape(1, -1) for xcb in x_cubature_points], axis=0)
        y_pred = np.mean(Y_cubature_points, axis=0)

        YY = ((Y_cubature_points - y_pred)/np.sqrt(self.n_cubature_points)).T
        S_yy = np.linalg.qr(np.concatenate((YY, self.S_R), axis=1).T)[1].T
        XX = ((x_cubature_points - x_pred)/np.sqrt(self.n_cubature_points)).T

        P_xy = XX.dot(YY.T)

        # Kalman Gain:
        S_yy_inv = np.linalg.inv(S_yy.T)
        # Kg = P_xy.dot(np.linalg.inv(S_yy.T)).dot(np.linalg.inv(S_yy))
        Kg = P_xy.dot(S_yy_inv.T).dot(S_yy_inv)

        self.x = x_pred + np.dot(Kg, (y - y_pred))
        self.S = np.linalg.qr(np.concatenate((XX - Kg.dot(YY), Kg.dot(self.S_R)), axis=1).T)[1].T

        return y_pred, (y - y_pred)

    def get_states(self):
        return self.x


if __name__ == '__main__':
    plt.close('all')

    
    # --- Generates an input tensor x
    T = 50
    N = T*100
    x = torch.sin(10*torch.linspace(0, T, steps=N)) + torch.sin(5*torch.linspace(0, T, steps=N))
    x += 0.01* torch.randn_like(x)      
    
    x.requires_grad_()

    x = torch.stack((x, torch.ones_like(x)), -1)
    # x = torch.stack((x, torch.randn_like(x)), 0)
    # x = torch.stack((x, torch.randn_like(x)), 0)
    # # x = x.view(N, 2)

    # --- No. of neurons (input, hidden layer1, output)
    # d_y1, d_x, d_y, y1_0):
    d_x = 2
    d_y1 = 4
    d_y = 4

    # xx = torch.randn((N, d_x))
    # xx[:, 1] = x
    # x = xx

    # --- Generate target output y through a RNN
    nnbf_gen = NNBF(d_y1, d_x, d_y, torch.zeros((d_y1, 1)))
    nnbf_gen.parameters()
    # y = torch.zeros(N, 1)
    y = torch.zeros(d_y1, 1)
    count = 0
    save_y = torch.zeros(N, d_y)
    for xi in x:
        # xi = x[1]
        # y = nnekf.forward(xi.reshape(-1, 1), torch.tensor([[y.item()]]))
        y = nnbf_gen.forward(xi.reshape(-1, 1))
        for i in range(d_y):
            save_y[count, i] = y[i].item()
        count += 1
        # y.backward()
    # print(xi.grad)


    # print(x.grad)
    # print(y.grad)

    noise_pw = 1e-3
    ydata = save_y + np.sqrt(noise_pw)*torch.randn_like(save_y)
    xdata = x.detach().numpy()

    plt.figure(0)
    plt.plot(ydata)
    plt.plot(save_y)
    plt.show()
    plt.plot(x.detach().numpy())
    plt.show()

    # useless?
    y0 = torch.zeros(1, 1)
    save_y_est = torch.zeros(N, 1)

    # Hyperparameters for our network
    # input_size = dim_x
    # hidden_sizes = [128, 64]
    # output_size = 10
    # # Build a feed-forward network
    # model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
    #                       nn.ReLU(),
    #                       nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    #                       nn.ReLU(),
    #                       nn.Linear(hidden_sizes[1], output_size),
    #                       nn.Softmax(dim=1))
    #
    # state_model =

    # --- Init EKF (with state and meas dimensions, initial state, and cov matrices)
    x_dim = d_x
    s_dim = d_y1
    input_size = x_dim + s_dim
    y_dim = d_y

    state_model = StateModel(x_dim, s_dim)
    observation_model = ObservationModel(s_dim, y_dim)
    s0 = torch.zeros((s_dim, 1))

    w_dim = state_model.get_param().data.shape[0] + observation_model.get_param().data.shape[0]

    P = np.sqrt(noise_pw) * torch.eye(w_dim + s_dim)
    Q = np.sqrt(noise_pw) * torch.eye(y_dim)
    R = 0.1 * np.sqrt(noise_pw) * torch.eye(len(P))

    print('I1')
    nnekf = NNEKF(state_model, observation_model, s0, P, R, Q)

    # --- Init Cubature filter
    # z = [b0, b1, wx, ws, wo, s]
    # f = lambda z, input: np.concatenate(z[0:-s_dim], 1.0/np.exp(np.dot(z[0] + z[2:2 + s_dim*x_dim].reshape(s_dim, x_dim), input) + np.dot(z[2 + s_dim*x_dim:2 + s_dim*x_dim + s_dim**2], z[-s_dim:])))
    def f(z, input):
        f1 = z[0:-s_dim]
        b = z[0]
        W1 = z[2:2 + s_dim*x_dim].reshape(s_dim, x_dim)
        Wf = z[2 + s_dim*x_dim:2 + s_dim*x_dim + s_dim**2].reshape(s_dim, s_dim)
        s = z[-s_dim:]
        # f2 = 1.0/np.exp(np.dot(z[0] + z[2:2 + s_dim*x_dim].reshape(s_dim, x_dim), input) + np.dot(z[2 + s_dim*x_dim:2 + s_dim*x_dim + s_dim**2], z[-s_dim:]))
        a1 = np.dot(W1, input).ravel()
        a2 = np.dot(Wf, s)#.reshape(-1, 1)
        # f2 = 1.0 / np.exp(a1 + a2 + b)
        f2 = expit(a1 + a2 + b)
        return np.concatenate((f1, f2), axis=0)

    # h = lambda z: np.dot(z[-2*s_dim:-s_dim], z[-s_dim:])
    def h(z):
        return np.dot(z[-d_y*s_dim - s_dim:-s_dim].reshape(d_y, s_dim), z[-s_dim:]) + z[1]

    z_dim = len(P)
    x0 = np.zeros((z_dim,)) + 1
    P0 = P.numpy()
    # Q0 = 1e-4 * np.eye(len(P))
    Q0 = 0.1 * np.sqrt(noise_pw) * np.eye(len(P))
    R0 = np.sqrt(noise_pw) * np.eye(y_dim)
    nncbf = CubatureFilter(f, h, x0, P0, Q0, R0)
    # nncbf = SquareRootCubatureFilter(f, h, x0, P0, Q0, R0)
    
    # --- Train EKF and Cubature
    save_e = torch.zeros((N, 1))
    save_w = torch.zeros((len(x), w_dim))
    save_d = torch.zeros_like(save_y)
    save_e_cbf = np.zeros((N, 1))
    save_d_cbf = np.zeros_like(save_y)
    for i in range(len(x)-1):
        if not i%100:
            print(i)
        xi = x[i].reshape(-1, 1)

        yi, ei = nnekf.forward_backward(xi, ydata[i].reshape(-1, 1))
        save_d[i+1] = yi.reshape(d_y,)
        save_e[i+1] = torch.norm(ei)**2

        yi_cbf, ei_cbf = nncbf.update(ydata[i], u=xi.detach().numpy())
        save_d_cbf[i+1] = yi_cbf
        save_e_cbf[i+1] = np.linalg.norm(ei_cbf) ** 2

    # --- Plot prediction error
    plt.figure(1)
    # plt.plot(np.log10(save_e.detach().numpy()), label='EKF', alpha=0.5)
    # plt.plot(np.log10(save_e_cbf), label='CKF', alpha=0.5)
    
    plt.plot(save_e.detach().numpy(), label='EKF', alpha=0.5)
    plt.plot(save_e_cbf, label='CKF', alpha=0.5)
    
    
    # plt.plot(save_e_cbf, label='CKF')
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Squared Error')
    plt.grid()
    plt.show()

    # --- Plot output
    plt.figure(2)
    # plt.plot(ydata)
    plt.plot(save_y.detach().numpy(), '--', label='true')
    plt.plot(save_d.detach().numpy(), label='EKF')
    plt.plot(save_d_cbf, ':', label='CKF')
    plt.xlabel('Sample')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

    # # # W1 = torch.nn.Parameter(torch.randn((1, 1)))
    # # # Wf = torch.nn.Parameter(torch.randn((1, 1)))
    # # # W2 = torch.nn.Parameter(torch.randn((1, 1)))
    # # # b1 = torch.nn.Parameter(torch.randn(()))
    # # # b2 = torch.nn.Parameter(torch.randn(()))
    # dw = d_x*d_y1 + d_y1**2 + d_y1*d_y + 2
    # # w = torch.nn.Parameter(0.01*torch.randn((dw, 1)))
    # w = torch.nn.Parameter(torch.zeros((dw, 1)))
    #
    # y1_old = torch.zeros(d_y1, 1, requires_grad=True)
    #
    # P = np.sqrt(noise_pw) * torch.eye(len(w) + len(y1_old))
    # Q = np.sqrt(noise_pw) * torch.eye(1)
    # R = np.sqrt(noise_pw) * torch.eye(len(P))
    # save_e = torch.zeros_like(save_y)
    # save_w = torch.zeros((len(x), len(w)))
    # save_d = torch.zeros_like(save_y)
    # print('I0')
    # for i in range(len(x)):
    #     xi = x[i].reshape(-1, 1)
    #     # y = nnekf.forward(xi.reshape(-1, 1), torch.tensor([[y.item()]]))
    #     y1 = torch.sigmoid(torch.mm(w[0:d_y1], xi) + torch.mm(w[d_y1:d_y1 + d_y1**2].reshape(d_y1, d_y1), y1_old) +
    #                        w[d_y1 + d_y1**2:d_y1 + d_y1**2+1])
    #     y1.backward()
    #     # print(y_old.grad)
    #     # print(w.grad)
    #
    #     # A = [[I_(d_w x d_w) , 0_(d_w, d_y1)]
    #     #      [Js_w.T,              Js_y1.T]]
    #     A = torch.cat((torch.cat((torch.eye((len(w))), w.grad.T), 0),
    #                    torch.cat((torch.zeros((len(w), len(y1))), y1_old.grad.T), 0)), 1)
    #     # print(A[-1, :])
    #     # print(w.grad.T)
    #     # detaching layers
    #     # y1_old = torch.tensor([[y1.item()]], requires_grad=True)
    #     # y11 = torch.tensor([[y1.item()]], requires_grad=True)
    #     y11 = y1.detach()
    #     y11.requires_grad = True
    #
    #     d = torch.mm(w[3:4], y11) + w[4:5]
    #     d.backward()
    #     # print(w.grad)
    #     # print(y11.grad)
    #
    #     # C = [[ Jh_w.T , Js_y1.T]]
    #     C = torch.cat((w.grad.T, y11.grad.T), 1)
    #     # Extended KF
    #
    #     # e = save_y[i] - d
    #     e = ydata[i] - d
    #     P = torch.mm(torch.mm(A, P), A.T)
    #     K = torch.mm(torch.mm(P, C.T), torch.inverse(torch.mm(torch.mm(C, P), C.T) + Q))
    #     P = torch.mm(torch.eye(len(P)) - torch.mm(K, C), P) + R
    #     print(A)
    #     print(C)
    #     print(K)
    #     # print(R)
    #     print(P)
    #     K1 = K[0:len(w)]
    #     K2 = K[len(w):]
    #     w = w.detach() + torch.mm(K1, e).detach()
    #     # print(w)
    #     w.requires_grad = True
    #     # y1_old = torch.tensor([[y1.item() + torch.mm(K2, e).item()]], requires_grad=True)
    #     y1_old = y1.detach() + torch.mm(K2, e).detach()
    #     y1_old.requires_grad = True
    #     # save_y_est[i] = y.item()
    #     # y.backward()
    #     # # print(xi.grad)
    #     # print(nnekf.W1.grad)
    #     # print(x.grad[i])
    #     save_e[i] = e
    #     save_w[i] = w.T
    #     save_d[i] = d
    # plt.plot(save_e[0:20].detach().numpy() ** 2)
    # plt.show()
    # plt.plot(save_y.detach().numpy())
    # plt.plot(save_d.detach().numpy())
    # plt.show()

    #
    #
    # Nt = 100
    # Np = 500
    # var_prob = 0.0
    # Ts = 1
    # # s_t = F s_t - 1 + Gamma * w_t
    #
    # F = np.array([[1, Ts, 0, 0],
    #               [0, 1, 0, 0],
    #               [0, 0, 1, Ts],
    #               [0, 0, 0, 1]])
    #
    # Gamma = np.array([[(Ts **2) / 2,  0],
    #                 [Ts, 0],
    #                 [0, (Ts**2) / 2],
    #                 [0, Ts]])
    #
    # # #hfun = @(s)  [30 - 10 * log10(norm(-s(1: 2:3)) ^ 2.2); atan2(s(3), s(1))];
    # # hfun = lambda s: np.array([[30 - 10 * np.log10(np.linalg.norm(-s[0:2:2]) ** 2.2)], [math.atan2(s[3], s[1])]])
    # H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    # # w_t ~ N(0, Q)
    # q = 2e-3
    # Q = q**2 * np.eye(2)
    # # Q4 = np.diag([sqrt(0.5) * q ^ 2, q ^ 2, sqrt(0.5) * q ^ 2, q ^ 2]);
    # Q4 = Gamma.dot(Q).dot(Gamma.T)
    # r = 5e-2
    # # r = 1;
    # R = (r**2) * np.eye(2)
    #
    # s0 = np.array([[5.3, 0.43, 4.5, -0.52]]).T
    # P0 = np.diag([1, 0.1, 1, 0.1])
    #
    # s = s0
    # P = P0
    # save_s = np.zeros((Nt, 2))
    # save_y = np.zeros((Nt, 2))
    #
    # save_kf_s = np.zeros((Nt, 2))
    # save_kf_y = np.zeros((Nt, 2))
    #
    #                   # h, f, s0, P0, Q, R, jac):
    # kf = KalmanFilter(H, F, s0, P0, Q4, R, None)
    #
    # for n in range(Nt):
    #     s = F.dot(s) + Gamma.dot(np.random.multivariate_normal([0, 0], Q)).reshape(1, -1).T
    #     y = H.dot(s) + np.random.multivariate_normal([0, 0], R).reshape(1, -1).T
    #
    #     kf.update(y)
    #     save_s[n, :] = np.array([s[0], s[2]]).T
    #     save_y[n, :] = y.T
    #
    #     save_kf_s[n, :] = np.array([kf.s[0], kf.s[2]]).T
    #     save_kf_y[n, :] = y.T
    #
    #
    # plt.figure()
    # plt.plot(save_s[:, 0], save_s[:, 1], label='True')
    # plt.plot(save_kf_s[:, 0], save_kf_s[:, 1], label='KF')
    # # RMSE_KF = sqrt((norm(save_kf_x - save_x). ^ 2) / length(save_x))
    # # RMSE_GP_MAP = sqrt((norm(save_gppf_x_map - save_x). ^ 2) / length(save_x))
    # # RMSE_GP_MMSE = sqrt((norm(save_gppf_x_mmse - save_x). ^ 2) / length(save_x))
    # # RMSE_PF_MAP = sqrt((norm(save_pf_x_map - save_x). ^ 2) / length(save_x))
    # # RMSE_PF_MMSE = sqrt((norm(save_pf_x_mmse - save_x). ^ 2) / length(save_x))
    # # legend('True', 'KF', 'GPPF (MAP)', 'GPPF (MMSE)', 'PF (MAP)', 'PF (MMSE)')
    # #
    # plt.legend()
    # plt.show()