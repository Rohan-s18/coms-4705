import sys

import torch

from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from extract_training_data import FeatureExtractor, State
from train_model import DependencyModel
import torch.nn.functional as F


class Parser(object):
    def __init__(self, extractor, modelfile):
        self.extractor = extractor
        # Create a new model and load the parameters
        self.model = DependencyModel(
            len(extractor.word_vocab), len(extractor.output_labels)
        )
        self.model.load_state_dict(torch.load(modelfile))
        sys.stderr.write("Done loading model")
        self.output_labels = dict(
            [(index, action) for (action, index) in extractor.output_labels.items()]
        )

    def parse_sentence(self, words, pos):
        # initializing parser state
        state = State(range(1, len(words)))
        state.stack.append(0)

        while state.buffer:
            # getting the input representation based on the current state
            input_representation = self.extractor.get_input_representation(
                words, pos, state
            )

            # creating a tensot for the input representation
            features = torch.LongTensor(input_representation).unsqueeze(0)

            # getting the model predictions
            output_logits = self.model(features)
            # applying softmax to obtain probs
            action_probs = F.softmax(output_logits, dim=1).detach().numpy()

            # creating a list of possible actions based on output labels
            possible_actions = []
            for idx, prob in enumerate(action_probs[0]):
                action = self.output_labels[idx]
                possible_actions.append((action, prob))

            # sorting possible actions by their probabilities in descending order
            possible_actions.sort(key=lambda x: x[1], reverse=True)

            # finding the highest scoring legal action
            action_taken = None
            for action, _ in possible_actions:
                if self.is_legal_action(action, state):
                    action_taken = action
                    break

            if action_taken is None:
                break

            # updating the state based on the action taken
            self.update_state(state, action_taken)

        # building the dependency structure
        result = DependencyStructure()
        for p, c, r in state.deps:
            result.add_deprel(DependencyEdge(c, words[c], pos[c], p, r))
            # print(f"pcr :{p}-{c}-{r}")

        return result

    """
    helper function to check if the action is legal based on the current state
    """

    def is_legal_action(self, action, state):
        transition, relation = action
        if transition == "shift":
            return len(state.buffer) > 0
        elif transition == "left_arc":
            return len(state.stack) > 1 and state.stack[-1] != 0
        elif transition == "right_arc":
            return len(state.stack) > 1
        return False

    """
    Helper function to update the state based on the given action.
    """

    def update_state(self, state, action):
        transition, relation = action
        if transition == "shift":
            state.shift()
        elif transition == "left_arc":
            state.left_arc(relation)
        elif transition == "right_arc":
            state.right_arc(relation)


if __name__ == "__main__":
    WORD_VOCAB_FILE = "data/words.vocab"
    POS_VOCAB_FILE = "data/pos.vocab"

    try:
        word_vocab_f = open(WORD_VOCAB_FILE, "r")
        pos_vocab_f = open(POS_VOCAB_FILE, "r")
    except FileNotFoundError:
        print(
            "Could not find vocabulary files {} and {}".format(
                WORD_VOCAB_FILE, POS_VOCAB_FILE
            )
        )
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2], "r") as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
