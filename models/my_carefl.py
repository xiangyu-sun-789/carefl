import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Laplace, Uniform, TransformedDistribution, SigmoidTransform
from torch.utils.data import DataLoader, Dataset

from data.generate_synth_data import CustomSyntheticDatasetDensity
from nflib import AffineCL, NormalizingFlowModel, MLP1layer, MAF, NSF_AR, ARMLP, MLP4


class MY_CAREFL:
    """
    The CAREFL model.

    This class defines the CAREFL model developed in Causal Autoregressive Flows, by Khemakhem et al. (2021)
    manuscript available at: https://arxiv.org/abs/2011.02268

    CAREFL can be used to find causal direction between paris of (multivariate) random variables, or
    to perform interventions and answer counterfactual queries.

    Parameters:
    ----------
    config: dict
        A configuration dict that defines all necessary parameters.
        Refer to one of the provided config files for more info/

    Methods:
    ----------
    flow_lr: Init and train two normalizing flow models for each direction.
        Return their likelihood ratio evaluated on held out test data.
        The causal direction is x->y if the likelihood is positive, and y->x instead.
    predict_proba: A wrapper around flow_lr which also returns the direction.
    fit_to_sem: fits an autoregressive flow model to an SEM.
    predict_intervention: Perform an intervention on a given variable in a fitted DAG.
    predict_counterfactual: Answer counterfactual queries on a fitted DAG.

    """

    def __init__(self, config):
        self.config = config
        self.n_layers = config.flow.nl
        self.n_hidden = config.flow.nh
        self.epochs = config.training.epochs
        self.device = config.device
        self.verbose = config.training.verbose

        # initial guess on correct model to be updated after each fit
        self.dim = None
        self.direction = 'none'

        if type(self.n_layers) is list:
            raise Exception("self.n_layers cannot be a list.")

        if type(self.n_hidden) is list:
            raise Exception("self.n_hidden cannot be a list.")

    def flow_lr(self, data, return_scores=False):
        """
        For each direction, fit a flow model, then compute the log-likelihood ratio to determine causal direction.

        Parameters:
        ----------
        data: numpy.ndarray
            A dataset where the first half of the columns are observations of a (multivariate)
            random variable X, and the second half are those of a (multivariate) r.v. Y.
        return_scores: bool
            If True, return the lists of the test likelihoods of the multiple flows trained for each direction.

        Returns:
        ----------
        p: float
            The test likelihood which indicates direction.
        score_xy: list
            If `return_scores` is True, return all test likelihoods of the different flows trained for 'x->y'
        score_yx: list
            If `return_scores` is True, return all test likelihoods of the different flows trained for 'y->x'
        """
        training_dataset, testing_dataset, dim = self._get_datasets(data)
        self.dim = dim

        # Conditional Flow Model: X->Y
        torch.manual_seed(self.config.training.seed)
        flow_xy, _ = self._train(training_dataset)
        score_xy = self._evaluate(flow_xy, testing_dataset)

        # Conditional Flow Model: Y->X
        torch.manual_seed(self.config.training.seed)
        flow_yx, _ = self._train(training_dataset, parity=True)
        score_yx = self._evaluate(flow_yx, testing_dataset, parity=True)

        # compute LR
        p = score_xy - score_yx
        self._update_direction(p)
        if return_scores:
            return p, score_xy, score_yx
        else:
            return p

    def _get_optimizer(self, parameters):
        """
        Returns an optimizer according to the config file
        """
        optimizer = optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                               betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        if self.config.optim.scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=self.verbose)
        else:
            scheduler = None
        return optimizer, scheduler

    def _get_flow_arch(self, parity=False):
        """
        Returns a normalizing flow according to the config file.

        Parameters:
        ----------
        parity: bool
            If True, the flow follows the (1, 2) permutations, otherwise it follows the (2, 1) permutation.
        """
        # this method only gets called by _train, which in turn is only called after self.dim has been initialized
        dim = self.dim
        # prior
        if self.config.flow.prior_dist == 'laplace':
            prior = Laplace(torch.zeros(dim).to(self.device), torch.ones(dim).to(self.device))
        else:
            prior = TransformedDistribution(Uniform(torch.zeros(dim).to(self.device), torch.ones(dim).to(self.device)),
                                            SigmoidTransform().inv)
        # net type for flow parameters
        if self.config.flow.net_class.lower() == 'mlp':
            net_class = MLP1layer
        elif self.config.flow.net_class.lower() == 'mlp4':
            net_class = MLP4
        elif self.config.flow.net_class.lower() == 'armlp':
            net_class = ARMLP
        else:
            raise NotImplementedError('net_class {} not understood.'.format(self.config.flow.net_class))

        # flow type
        def ar_flow(hidden_dim):
            if self.config.flow.architecture.lower() in ['cl', 'realnvp']:
                return AffineCL(dim=dim, nh=hidden_dim, scale_base=self.config.flow.scale_base,
                                shift_base=self.config.flow.shift_base, net_class=net_class, parity=parity,
                                scale=self.config.flow.scale)
            elif self.config.flow.architecture.lower() == 'maf':
                return MAF(dim=dim, nh=hidden_dim, net_class=net_class, parity=parity)
            elif self.config.flow.architecture.lower() == 'spline':
                return NSF_AR(dim=dim, hidden_dim=hidden_dim, base_network=net_class)
            else:
                raise NotImplementedError('Architecture {} not understood.'.format(self.config.flow.architecture))

        # construct normalizing flows
        flow_list = [ar_flow(self.n_hidden) for _ in range(self.n_layers)]
        normalizing_flow = NormalizingFlowModel(prior, flow_list).to(self.device)

        return normalizing_flow

    def _train(self, dset, parity=False):
        """
        Train one flow for a single direction, specified by `parity`.
        """
        train_loader = DataLoader(dset, shuffle=True, batch_size=self.config.training.batch_size)
        flow = self._get_flow_arch(parity)

        optimizer, scheduler = self._get_optimizer(flow.parameters())

        flow.train()
        loss_vals = []
        for e in range(self.epochs):
            loss_val = 0
            for _, x in enumerate(train_loader):
                x = x.to(self.device)
                if parity and self.config.flow.architecture == 'spline':
                    # spline flows don't have parity option and should only be used with 2D numpy data:
                    x = x[:, [1, 0]]
                # compute loss
                _, prior_logprob, log_det = flow(x)
                loss = - torch.sum(prior_logprob + log_det)
                loss_val += loss.item()
                # optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.config.optim.scheduler:
                scheduler.step(loss_val / len(train_loader))
            if self.verbose:
                print('epoch {}/{} \tloss: {}'.format(e, self.epochs, loss_val))
            loss_vals.append(loss_val)

        return flow, loss_vals

    def _evaluate(self, flow, test_dset, parity=False):
        """
        Evaluate the trained flow on test dataset.
        """
        loader = DataLoader(test_dset, batch_size=128)

        if parity and self.config.flow.architecture == 'spline':
            # spline flows don't have parity option and should only be used with 2D numpy data:
            score = np.nanmean(np.concatenate([flow.log_likelihood(x.to(self.device)[:, [1, 0]]) for x in loader]))
        else:
            score = np.nanmean(np.concatenate([flow.log_likelihood(x.to(self.device)) for x in loader]))

        return score

    def _get_datasets(self, input):
        """
        Check data type, which can be:
            - an np.ndarray, in which case split it and wrap it into a train Dataset and and a test Dataset
            - a Dataset, in which case duplicate it (test dataset is the same as train dataset)
            - a tuple of Datasets, in which case just return.
        return a train Dataset, and a test Dataset
        """
        assert isinstance(input, (np.ndarray, Dataset, tuple, list))
        if isinstance(input, np.ndarray):
            dim = input.shape[-1]
            if self.config.training.split == 1.:
                data_test = np.copy(input)
            else:
                data_test = np.copy(input[int(self.config.training.split * input.shape[0]):])
                input = input[:int(self.config.training.split * input.shape[0])]
            dset = CustomSyntheticDatasetDensity(input.astype(np.float32))
            test_dset = CustomSyntheticDatasetDensity(data_test.astype(np.float32))
            return dset, test_dset, dim
        if isinstance(input, Dataset):
            dim = input[0].shape[-1]
            return input, input, dim
        if isinstance(input, (tuple, list)):
            dim = input[0][0].shape[-1]
            return input[0], input[1], dim

    def _update_direction(self, p):
        self.direction = 'x->y' if p >= 0 else 'y->x'
