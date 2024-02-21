[![Project Tests](https://github.com/drumyerscough/HW6-HMM/actions/workflows/ci.yml/badge.svg)](https://github.com/drumyerscough/HW6-HMM/actions/workflows/ci.yml)

# HW6-HMM

In this assignment, you'll implement the Forward and Viterbi Algorithms (dynamic programming). 


# Assignment

## Overview 

The goal of this assignment is to implement the Forward and Viterbi Algorithms for Hidden Markov Models (HMMs).

For a helpful refresher on HMMs and the Forward and Viterbi Algorithms you can check out the resources [here](https://web.stanford.edu/~jurafsky/slp3/A.pdf), 
[here](https://towardsdatascience.com/markov-and-hidden-markov-model-3eec42298d75), and [here](https://pieriantraining.com/viterbi-algorithm-implementation-in-python-a-practical-guide/). 





## Tasks and Data 
Please complete the `forward` and `viterbi` functions in the HiddenMarkovModel class. 

We have provided two HMM models (mini_weather_hmm.npz and full_weather_hmm.npz) which explore the relationships between observable weather phenomenon and the temperature outside. Start with the mini_weather_hmm model for testing and debugging. Both include the following arrays:
* `hidden_states`: list of possible hidden states 
* `observation_states`: list of possible observation states 
* `prior_p`: prior probabilities of hidden states (in order given in `hidden_states`) 
* `transition_p`: transition probabilities of hidden states (in order given in `hidden_states`)
* `emission_p`: emission probabilities (`hidden_states` --> `observation_states`)



For both datasets, we also provide input observation sequences and the solution for their best hidden state sequences. 
 * `observation_state_sequence`: observation sequence to test 
* `best_hidden_state_sequence`: correct viterbi hidden state sequence 


Create an HMM class instance for both models and test that your Forward and Viterbi implementation returns the correct probabilities and hidden state sequence for each of the observation sequences.

Within your code, consider the scope of the inputs and how the different parameters of the input data could break the bounds of your implementation.
  * Do your model probabilites add up to the correct values? Is scaling required?
  * How will your model handle zero-probability transitions? 
  * Are the inputs in compatible shapes/sizes which each other? 
  * Any other edge cases you can think of?
  * Ensure that your code accomodates at least 2 possible edge cases. 

Finally, please update your README with a brief description of your methods. 

## Methods

This repo contains a module called hmm, which includes a lightweight class for hidden Markov models that can be initialized from arrays containing hidden and observed states, prior probabilities of hidden states, transition probabilities between hidden states, and emission probabilities for each observed state from each hidden state. The hmm class has two methods, forward() and viterbi(), which respectively implement the forward and viterbi algorithms using dynamic programming. The forward algorithm calculates the likelihood of a sequence of observed states given an HMM instance, and the Viterbi algorithm calculates the most likely sequence of hidden states that would produce a given a sequence of observed states.

In more detail:
The forward algorithm recursively calculates likelihood of a sequence of observed states given all possible sequences of hidden states that could have produced each observed state. Because of the Markov property, each hidden state depends only on the previous hidden state, and each observed state depends only on the current hidden state, so likelihoods can be efficiently calculated for each i-th observed state based on the likelihoods of the (i-1)-th states. Starting from a prior distribution, the likelihood of the observed states for each possible hidden state are accumulated in a table across iterations through the observed state sequence, and the sum over the final column of the table yields the likelihood of the observed sequence. The Viterbi algorithm is similar in concept and implementation to the forward algorithm, but instead of summing over all possible sequences of hidden states that could have produced an observed state, the most likely sequence is taken instead. As in the forward algorithm, the likelihoods for the hidden states are accumulated in a table, and additionally the most likely hidden state transitions are stored in a backtrace table. When the algorithm terminates, the transitions from the most likely ending state to the most likely initial state can be traced to produce the most likely sequence of hidden states to produce the observed states.


## Task List

[TODO] Complete the HiddenMarkovModel Class methods  <br>
  [ ] complete the `forward` function in the HiddenMarkovModelClass <br>
  [ ] complete the `viterbi` function in the HiddenMarkovModelClass <br>

[TODO] Unit Testing  <br>
  [ ] Ensure functionality on mini and full weather dataset <br>
  [ ] Account for edge cases 

[TODO] Packaging <br>
  [ ] Update README with description of your methods <br>
  [ ] pip installable module (optional)<br>
  [ ] github actions (install + pytest) (optional)


## Completing the Assignment 
Push your code to GitHub with passing unit tests, and submit a link to your repository [here](https://forms.gle/xw98ZVQjaJvZaAzSA)

### Grading 

* Algorithm implementation (6 points)
    * Forward algorithm is correct (2)
    * Viterbi is correct (2)
    * Output is correct on small weather dataset (1)
    * Output is correct on full weather dataset (1)

* Unit Tests (3 points)
    * Mini model unit test (1)
    * Full model unit test (1)
    * Edge cases (1)

* Style (1 point)
    * Readable code and updated README with a description of your methods 

* Extra credit (0.5 points)
    * Pip installable and Github actions (0.5)