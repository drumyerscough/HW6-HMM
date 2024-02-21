import numpy as np

class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, 
                 prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """
        
        assert len(observation_states) == emission_p.shape[1], "number of observation states doesn't match shape of emission_p"
        assert len(hidden_states) == transition_p.shape[1], "number of hidden states doesn't match shape of transition_p"
        assert len(hidden_states) == len(prior_p), "number of hidden states doesn't match shape of prior_p"
        assert transition_p.shape[0] == transition_p.shape[1], "transition matrix should be square"

        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        # do linear scaling to ensure probabilities sum to 1
        self.prior_p = prior_p / np.sum(prior_p)
        self.transition_p = transition_p / np.sum(transition_p, axis=1, keepdims=True)
        self.emission_p = emission_p / np.sum(emission_p, axis=1, keepdims=True)


    def forward(self, input_observation_states: np.ndarray) -> float:
        """

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """

        if len(input_observation_states) == 0:
            raise Exception('The input input_observation_states is empty')
        
        if not set(input_observation_states) <= set(self.observation_states):
            raise ValueError('states in input_observation_states are not present in the HMM')

        # Step 1. Initialize variables
        fwd_mat = np.zeros((self.hidden_states.shape[0], input_observation_states.shape[0]))
        fwd_mat[:, 0] = self.prior_p * self.emission_p[:, self.observation_states_dict[input_observation_states[0]]]

        # Step 2. Calculate probabilities
        for t in range(1, input_observation_states.shape[0]):
            state_idx = self.observation_states_dict[input_observation_states[t]] # index of observed state in emission_p
            for s in range(self.hidden_states.shape[0]):
                fwd_mat[s, t] = np.sum(fwd_mat[:, t-1] * self.transition_p[:,s] * self.emission_p[s, state_idx])

        # Step 3. Return final probability
        return np.sum(fwd_mat[:, -1])

    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """

        if len(decode_observation_states) == 0:
            raise Exception('The input decode_observation_states is empty')
        
        if not set(decode_observation_states) <= set(self.observation_states):
            raise ValueError('states in decode_observation_states are not present in the HMM')
        
        # Step 1. Initialize variables
        
        #store probabilities of hidden state at each step
        num_hidden_states = self.hidden_states.shape[0]
        viterbi_table = np.zeros((num_hidden_states, decode_observation_states.shape[0]))
        viterbi_back = np.zeros((num_hidden_states, decode_observation_states.shape[0]))

        viterbi_table[:, 0] = np.log(self.prior_p * self.emission_p[:, self.observation_states_dict[decode_observation_states[0]]])
        viterbi_back[:, 0] = 0

        # Step 2. Calculate Probabilities
        for t in range(1, decode_observation_states.shape[0]):
            state_idx = self.observation_states_dict[decode_observation_states[t]]
            for s in range(num_hidden_states):
                #viterbi_table[s, t] = np.max(viterbi_table[:, t-1] * self.transition_p[:,s] * self.emission_p[:, state_idx])
                viterbi_table[s, t] = np.max(viterbi_table[:, t-1] + np.log(self.transition_p[:,s]) + np.log(self.emission_p[s, state_idx]))
                viterbi_back[s, t] = np.argmax(viterbi_table[:, t-1] + np.log(self.transition_p[:,s]) + np.log(self.emission_p[s, state_idx]))

        # Step 3. Traceback
        print(viterbi_table)
        best_path = []
        s = int(np.argmax(viterbi_table[:, -1]))
        for t in range(decode_observation_states.shape[0]):
            best_path.append(s)
            s = int(viterbi_back[s, -(t+1)])
        best_path.reverse()

        # Step 4. Return best hidden state sequence 
        return [self.hidden_states_dict[hidden_state_idx] for hidden_state_idx in best_path]