import numpy as np
import scipy.stats as stats
import pdb


class DAG:
    def __init__(self):
        self.node_list = {}
        self.node_order = []
        
    def add_node(self, node_id, parents, parent_mean_function, distribution='Gaussian', std=1., 
                 node_name=None, use_link=False, parent_transformation=None, parent_posthoc=None,
                 is_latent=False):
        '''
        node_id: integer label for node. Must add nodes in number order. 
        parents: list of parent ids (as ints). Nodes can only have parents with lower labels
        parent_mean_function: function of the parents, that gives the expected value of the node's output
            e.g. if the mean is X_3^2 + X_4, then we would have 
            parents = [3, 4]
            parent_mean_function = lambda x: x[0]**2 + x[1]
            if parents is None, parent_mean_function should be a constant
        distribution: 'Gaussian' or 'Categorical' implemented so far. 
        std: standard deviation, if using Gaussian
        node_name: string that describes the node. If none, will use node_id
        use_link: If False, use parent_mean_function directly as the mean of the distribution. 
                    (Note, requires parent_mean_function to lie in the parameter space,eg be positive for Poisson)
                  If True, apply appropriate link function: softmax for categorical, exp for Poisson
        parent_transformation: If not None, a function that is applied to the random variable. 
            E.g. if we want log-normal distributions, we would set: 
                distribution = Gaussian
                parent_transformation = lambda x: np.exp(x)
        parent_posthoc: if not None, a function of the parents that we multiply the random variable by
            eg if we want to set the output to zero if the first parent is less than 3, we would set:
                parent_posthoc = lambda x: int(x[0] < 3)
        is_latent: If True, do not include in final sample
        '''
        assert node_id not in self.node_list.keys() # can't add an existing node   
        if parents is not None:
            assert parents == list(np.sort(parents)) # parents must be in number order
            for parent_id in parents:
                assert parent_id < node_id # must be a DAG
                assert parent_id in self.node_list.keys() # must have seen parent before
        if node_name is None:
            node_name = '{}'.format(node_id)
        self.node_list[node_id] = Node(node_id, parents, parent_mean_function, distribution, std=std,
                                       node_name=node_name, use_link=use_link, 
                                       parent_transformation=parent_transformation, parent_posthoc=parent_posthoc,
                                       is_latent=is_latent)
        self.node_order.append(node_id)
        
        if parents is not None:
            for parent_id in parents:
                self.node_list[parent_id].add_child(node_id)
            
    def gen_sample(self, size=1, verbose=False):
        samples = {}
        for node_id in self.node_order:
            node_name = self.node_list[node_id].node_name
            # generate node's random variable
            raw_samples = self.node_list[node_id].generate_sample(size)
            if self.node_list[node_id].is_latent == False:
                if self.node_list[node_id].distribution == 'Categorical':
                    samples[node_name] = np.array([sample.argmax() for sample in raw_samples])
                else:
                    samples[node_name] = raw_samples + 0
                if verbose:
                    print('node {}: {}'.format(node_id, samples[node_id]))
            # propogate to children
            for child_id in self.node_list[node_id].children:
                self.node_list[child_id].get_message_from_parent(node_id, raw_samples)
            
        return samples

class Node:
    def __init__(self, node_id, parents, parent_mean_function, distribution, std=None, node_name=None,
                 use_link=False, parent_transformation=None, parent_posthoc=None, is_latent=False
                ):
        '''
        node_id: integer label for node. Must add nodes in number order. 
        parents: list of parent ids (as ints). Nodes can only have parents with lower labels
        parent_mean_function: function of the parents, that gives the expected value of the node's output
            e.g. if the mean is X_3^2 + X_4, then we would have 
            parents = [3, 4]
            parent_mean_function = lambda x: x[0]**2 + x[1]
            if parents is None, parent_mean_function should be a constant
        distribution: 'Gaussian' or 'Categorical' implemented so far. 
        std: standard deviation, if using Gaussian
        node_name: string that describes the node. 
        use_link: If False, use parent_mean_function directly as the mean of the distribution. 
                    (Note, requires parent_mean_function to lie in the parameter space,eg be positive for Poisson)
                  If True, apply appropriate link function: softmax for categorical, exp for Poisson
        parent_transformation: If not None, a function that is applied to the random variable 
            E.g. if we want log-normal distributions, we would set: 
                distribution = Gaussian
                parent_transformation = lambda x: np.exp(x)
        parent_posthoc: if not None, a function of the parents that we multiply the random variable by
            eg if we want to set the output to zero if the first parent is less than 3, we would set:
                parent_posthoc = lambda x: int(x[0] < 3)
        is_latent: If True, do not include in final sample
        '''
        self.node_id = node_id
        if node_name is None:
            node_name = '{}'.format(node_id)
        self.node_name=node_name
        self.parents = parents
        self.parent_mean_function = parent_mean_function
        self.parent_transformation = parent_transformation
        self.distribution = distribution
        self.std = std 
        self.children = []
        self.sample = None
        self.parent_messages = []
        self.use_link = use_link
        self.is_latent = is_latent
        self.parent_posthoc = parent_posthoc

            
    def add_child(self, child_id):
        '''
        Adds child_id to self.children
        '''
        if child_id not in self.children:
            self.children.append(child_id)
            
    def generate_sample(self, size):
        '''
        generates a sample, based on samples from parents
        size: number of samples (if self.parents is not None, must match number of samples from all parents)
        '''
        
        if self.parents == None:
            combined_mean = self.parent_mean_function
        else:
            assert len(self.parent_messages) == len(self.parents)
            try:
                combined_mean = self.parent_mean_function(self.parent_messages) # apply parent_mean_function to samples from parents
            except ValueError:
                pdb.set_trace()
            if self.distribution == 'Categorical':
                if len(combined_mean) != size:
                    assert len(combined_mean[0]) == size
                    combined_mean = np.array(combined_mean).T
            else:
                assert len(combined_mean) == size

            
        if self.distribution == 'Gaussian':
            
            self.sample = np.random.normal(loc=combined_mean, scale=self.std, size=size)
        elif self.distribution == 'Bernoulli':
            if self.parents == None:
                assert 0 <= combined_mean <= 1
            else:
                assert np.all([0 <= x <= 1 for x in combined_mean])
            self.sample = np.random.binomial(1, combined_mean, size=size)
        elif self.distribution == 'Categorical':
            if self.use_link:
                combined_mean = np.exp(combined_mean)
                row_sums = np.sum(combined_mean, axis=1)
                combined_mean = combined_mean / row_sums[:, np.newaxis]
            if self.parents == None:
                self.sample = np.array([np.random.multinomial(1, combined_mean) for i in range(size)])
            else:
                self.sample = np.array([np.random.multinomial(1, score) for score in combined_mean])
        elif self.distribution == 'Poisson':
            if self.use_link:
                combined_mean = np.exp(combined_mean)
            else:
                if self.parents == None:
                    assert combined_mean >= 0
                else:
                    assert np.all([x >= 0 for x in combined_mean])
            self.sample = np.random.poisson(combined_mean, size=size)
        else:
            print('{} is not a recognized distribution'.format(self.distribution) )
            return None
            
        if self.parent_transformation is not None:
            self.sample = self.parent_transformation(self.sample)
        if self.parent_posthoc is not None:
            self.sample = self.parent_posthoc(self.parent_messages) * self.sample
        #reset parent messages
        self.parent_messages = []
            
        return self.sample
            
 
    def get_message_from_parent(self, parent_id, raw_message):
        assert parent_id in self.parents
        self.parent_messages.append(raw_message)

        
