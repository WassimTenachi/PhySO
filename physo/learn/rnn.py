import torch
import numpy as np

class Cell(torch.nn.Module):
    """
        A custom RNN cell for symbolic expression generation.
        Attributes
        ----------
        input_size  : int
            Size of observation vector.
        n_layers    : int
            Number of stacked RNNs.
        hidden_size : int
            Number of hidden features in the RNN cells states.
        output_size : int
            Number of features ie number of choices possible in the library of tokens.

        input_dense       : torch.nn
            Input dense layer.
        stacked_cells     : torch.nn.ModuleList of torch.nn
            Stacked RNN cells.
        output_dense      : torch.nn
            Output dense layer.
        output_activation : function
            Output activation function.

        is_lobotomized : bool
            Should the probs output of the neural net be replaced by random numbers.

        logTemperature : torch.tensor
            Annealing parameter

        Methods
        -------
        forward (input_tensor, states)
            RNN cell call returning categorical logits.
        get_zeros_initial_state (batch_size)
            Returns a cell state containing zeros.
        count_parameters()
            Returns number of trainable parameters.

        Examples
        -------
        #  RNN initialization ---------
        input_size  = 3*7 + 33
        output_size = 7
        hidden_size = 32
        n_layers    = 1
        batch_size  = 1024
        time_steps  = 30

        initial_states = torch.zeros(n_layers, 2, batch_size, hidden_size)
        initial_obs    = torch.zeros(batch_size, input_size)

        RNN_CELL = Cell(input_size  = input_size,
                        output_size = output_size,
                        hidden_size = hidden_size,
                        n_layers    = n_layers)

        print(RNN_CELL)
        print("n_params = %i"%(RNN_CELL.count_parameters()))
        #  RNN run --------------
        observations = initial_obs
        states       = initial_states
        outputs      = []
        for _ in range (time_steps):
            output, states = RNN_CELL(input_tensor = observations,
                                            states = states      )
            observations   = observations
            outputs.append(output)
        outputs = torch.stack(outputs)
        print("outputs shape = ", outputs.shape)

    """
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 n_layers      = 1,
                 input_dense   = None,
                 stacked_cells = None,
                 output_dense  = None,
                 is_lobotomized = False,
                 ):
        super().__init__()
        # --------- Input dense layer ---------
        self.input_size  = input_size
        self.hidden_size = hidden_size
        if input_dense is None:
            input_dense = torch.nn.Linear(self.input_size, self.hidden_size)
        self.input_dense = input_dense
        # --------- Stacked RNN cells ---------
        self.n_layers      = n_layers
        if stacked_cells is None:
            stacked_cells = torch.nn.ModuleList([torch.nn.LSTMCell(input_size  = self.hidden_size,
                                                                    hidden_size = self.hidden_size)
                                for _ in range(self.n_layers) ])
        self.stacked_cells = stacked_cells
        # --------- Output dense layer ---------
        self.output_size       = output_size
        if output_dense is None:
            output_dense      = torch.nn.Linear(self.hidden_size, self.output_size)
        self.output_dense = output_dense
        self.output_activation = lambda x: -torch.nn.functional.relu(x) # Mapping output to log(p)
                                 #lambda x: torch.nn.functional.softmax(x, dim=1)
                                 #torch.sigmoid
        # --------- Annealing param ---------
        self.logTemperature = torch.nn.Parameter(1.54*torch.ones(1), requires_grad=True)
        # --------- Lobotomization ---------
        self.is_lobotomized = is_lobotomized

    def get_zeros_initial_state(self, batch_size):
        zeros_initial_state = torch.zeros(self.n_layers, 2, batch_size, self.hidden_size, requires_grad=False,)
        return zeros_initial_state

    def forward(self,
                input_tensor,                                         # (batch_size, input_size)
                states,                                               # (n_layers, 2, batch_size, hidden_size)
               ):
        # --------- Input dense layer ---------
        hx = self.input_dense(input_tensor)                           # (batch_size, hidden_size)
        # layer norm + activation
        # --------- Stacked RNN cells ---------
        new_states = [] # new states of stacked RNNs
        for i in range(self.n_layers):
            hx, cx = self.stacked_cells[i](hx,                        # (batch_size, hidden_size)
                                           (states[i,0,:,:],          # (batch_size, hidden_size)
                                            states[i,1,:,:]           # (batch_size, hidden_size)
                                           ))
            new_states.append(torch.stack([hx, cx]))
        # --------- Output dense layer ---------
        # Probabilities from neural net
        res = self.output_dense(hx) + self.logTemperature             # (batch_size, output_size)
        # Applying activation function
        res = self.output_activation(res)                             # (batch_size, output_size)
        # Probabilities from random number generator
        if self.is_lobotomized:
            res = torch.log(torch.rand(res.shape))
        out_states = torch.stack(new_states)                          # (n_layers, 2, batch_size, hidden_size)
        # --------------- Return ---------------
        return res, out_states                                        # (batch_size, output_size), (n_layers, 2, batch_size, hidden_size)

    def count_parameters (self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        return n_params
