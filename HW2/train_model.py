import sys
import numpy as np
import torch

from torch.nn import Module, Linear, Embedding, NLLLoss
from torch.nn.functional import relu, log_softmax
from torch.utils.data import Dataset, DataLoader 

from extract_training_data import FeatureExtractor

class DependencyDataset(Dataset):

  def __init__(self, inputs_filename, output_filename):
    self.inputs = np.load(inputs_filename)
    self.outputs = np.load(output_filename)

  def __len__(self): 
    return self.inputs.shape[0]

  def __getitem__(self, k): 
    return (self.inputs[k], self.outputs[k])


class DependencyModel(Module): 

    def __init__(self, word_types, outputs):
        super(DependencyModel, self).__init__()
        # Embedding layer: maps word types to 128-dimensional embeddings
        self.embedding = Embedding(num_embeddings=word_types, embedding_dim=128)
        # Hidden layer: input size is 768 (6 tokens * 128 embedding size), output size is 128
        self.hidden = Linear(in_features=768, out_features=128)
        # Output layer: input size is 128, output size is the number of possible transitions (91)
        self.output = Linear(in_features=128, out_features=outputs)

    def forward(self, inputs):
        # Pass the inputs through the embedding layer
        embedded = self.embedding(inputs)  # (batch_size, 6, 128)
        # Flatten the embeddings to (batch_size, 768)
        embedded_flattened = embedded.view(embedded.size(0), -1)  # (batch_size, 768)
        # Pass through the hidden layer and apply ReLU
        hidden_output = relu(self.hidden(embedded_flattened))  # (batch_size, 128)
        # Pass through the output layer to get logits
        output_logits = self.output(hidden_output)  # (batch_size, 91)
        return output_logits



def train(model, loader): 

    loss_function = torch.nn.CrossEntropyLoss(reduction='mean')  # Use CrossEntropyLoss

    LEARNING_RATE = 0.01 
    optimizer = torch.optim.Adagrad(params=model.parameters(), lr=LEARNING_RATE)

    tr_loss = 0 
    tr_steps = 0

    # put model in training mode
    model.train()

    correct = 0 
    total =  0 
    for idx, batch in enumerate(loader):
    
        inputs, targets = batch

        predictions = model(torch.LongTensor(inputs))

        # Ensure targets are in the correct data type (long)
        targets = targets.long()

        # Ensure that targets is a 1D tensor with class indices
        if targets.ndim == 2 and targets.shape[1] == 91:
            targets = torch.argmax(targets, dim=1)

        loss = loss_function(predictions, targets)
        tr_loss += loss.item()

        tr_steps += 1
        
        if idx % 1000 == 0:
            curr_avg_loss = tr_loss / tr_steps
            print(f"Current average loss: {curr_avg_loss}")

        # To compute training accuracy for this epoch 
        correct += sum(torch.argmax(predictions, dim=1) == targets)
        total += len(inputs)
          
        # Run the backward pass to update parameters 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / tr_steps
    acc = correct / total
    print(f"Training loss epoch: {epoch_loss},   Accuracy: {acc}")




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


    model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))

    dataset = DependencyDataset(sys.argv[1], sys.argv[2])
    loader = DataLoader(dataset, batch_size = 16, shuffle = True)

    print("Done loading data")

    # Now train the model
    for i in range(5): 
      train(model, loader)


    torch.save(model.state_dict(), sys.argv[3]) 
