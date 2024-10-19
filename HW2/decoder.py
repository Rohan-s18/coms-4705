import sys
import copy

import numpy as np
import torch

from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from extract_training_data import FeatureExtractor, State
from train_model import DependencyModel
import torch.nn.functional as F

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.extractor = extractor

        # Create a new model and load the parameters
        self.model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))
        self.model.load_state_dict(torch.load(modelfile))
        sys.stderr.write("Done loading model")

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        # Initialize the parser state
        state = State(range(1, len(words)))
        state.stack.append(0)  # Start with the first word (index 0) on the stack

        while state.buffer:
            # Get the input representation based on the current state
            input_representation = self.extractor.get_input_representation(words, pos, state)

            # Create a LongTensor for the input representation
            features = torch.LongTensor(input_representation).unsqueeze(0)  # Add batch dimension
            
            # Get the model predictions by calling the forward method
            output_logits = self.model(features)  # Forward pass
            
            # Apply softmax to get probabilities
            action_probs = F.softmax(output_logits, dim=1).detach().numpy()  # Convert to probabilities

            # Create a list of possible actions based on output labels
            possible_actions = []
            for idx, prob in enumerate(action_probs[0]):  # Access the first (and only) batch
                action = self.output_labels[idx]
                possible_actions.append((action, prob))  # Store action and its probability
            
            # Sort possible actions by their probabilities in descending order
            possible_actions.sort(key=lambda x: x[1], reverse=True)

            # Find the highest scoring legal action
            action_taken = None
            for action, _ in possible_actions:
                if self.is_legal_action(action, state):
                    action_taken = action
                    break

            if action_taken is None:
                break  # No legal actions available, break the loop

            # Update the state based on the action taken
            self.update_state(state, action)

        result = DependencyStructure()
        for p, c, r in state.deps:
            result.add_deprel(DependencyEdge(c, words[c], pos[c], p, r))

        return result


    def is_legal_action(self, action, state):
        """Check if the action is legal given the current state."""
        if action == 'shift':
            return len(state.buffer) > 0
        elif action == 'arc-left':
            return len(state.stack) > 1  # At least two words on stack to make a left arc
        elif action == 'arc-right':
            return len(state.stack) > 0 and len(state.buffer) > 0  # Need a word on stack and buffer
        return False

    def update_state(self, state, action):
        """Update the parser state based on the action taken."""
        if action == 'shift':
            state.stack.append(state.buffer.pop(0))  # Shift the first word from buffer to stack
        elif action == 'arc-left':
            child = state.stack.pop()  # Child is the top word on the stack
            parent = state.stack[-1]  # Parent is the next word on the stack
            state.deps.append((parent, child, 'left_arc'))  # Add dependency (parent, child, relation)
        elif action == 'arc-right':
            child = state.stack.pop()  # Child is the top word on the stack
            parent = state.stack[-1]  # Parent is the next word on the stack
            state.deps.append((parent, child, 'right_arc'))  # Add dependency (parent, child, relation)



if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
