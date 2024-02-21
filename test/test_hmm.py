import pytest
from hmm import HiddenMarkovModel
import numpy as np


def test_mini_weather():
    """
    
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')
    my_hmm = HiddenMarkovModel(mini_input['observation_states'], mini_hmm['hidden_states'], mini_hmm['prior_p'], mini_hmm['transition_p'], mini_hmm['emission_p'])
    
    # check that forward algorithm returns correct likelihood
    assert my_hmm.forward(mini_input['observation_state_sequence']) == 0.0350
    
    # check that viterbi algorithm returns correct sequence of hidden states
    assert my_hmm.viterbi(mini_input['observation_state_sequence']) == mini_input['best_hidden_state_sequence'].tolist()

    # check edge cases of bad inputs
    with pytest.raises(Exception):
        my_hmm.forward([])

    with pytest.raises(ValueError):
        my_hmm.forward([char for char in 'abcdef'])

    with pytest.raises(Exception):
        my_hmm.viterbi([])

    with pytest.raises(ValueError):
        my_hmm.viterbi([char for char in 'abcdef'])


def test_full_weather():

    """

    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')
    my_hmm = HiddenMarkovModel(full_hmm['observation_states'], full_hmm['hidden_states'], full_hmm['prior_p'], full_hmm['transition_p'], full_hmm['emission_p'])
    
    # check that forward algorithm returns correct likelihood
    #assert my_hmm.forward(full_input['observation_state_sequence'])
    
    # check that viterbi algorithm returns correct sequence of hidden states
    assert my_hmm.viterbi(full_input['observation_state_sequence']) == full_input['best_hidden_state_sequence'].tolist()












