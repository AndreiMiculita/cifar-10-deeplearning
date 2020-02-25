from typing import Callable, List, Tuple

import torch
from torch.utils.data import DataLoader

# define features and labels tensor dtypes
feats_dtype = torch.FloatTensor
label_dtype = torch.LongTensor


def train(network: torch.nn.Module,
          trainloader: DataLoader,
          testloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: Callable[[feats_dtype, label_dtype], feats_dtype],
          NUM_EPOCHS: int,
          l_rate: float,
          device: str) -> Tuple[List[float], List[float]]:
    # define training of a single data batch -> return error of the batch
    def train_batch(network: torch.nn.Module,  # the network
                    X_batch: feats_dtype,  # the features batch
                    Y_batch: label_dtype,  # the labels batch
                    # a function from a FloatTensor (prediction) and a LongTensor (Y) to a FloatTensor (the loss)
                    loss_fn: Callable[[feats_dtype, label_dtype], feats_dtype],
                    # the optimizer
                    optimizer: torch.optim.Optimizer) -> float:

        network.train()

        prediction_batch = network(X_batch)
        batch_loss = loss_fn(prediction_batch, Y_batch)
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return batch_loss.item()

    # define training of the whole dataset for one training step (epoch) -> return error of the entire set
    def train_epoch(network: torch.nn.Module,
                    # a list of data points x
                    dataloader: DataLoader,
                    loss_fn: Callable[[feats_dtype, label_dtype], feats_dtype],
                    optimizer: torch.optim.Optimizer,
                    device: str) -> float:

        loss = 0.
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            loss += train_batch(network=network, X_batch=x_batch, Y_batch=y_batch, loss_fn=loss_fn, optimizer=optimizer)
        loss /= batch_idx  # divide loss by number of batches for consistency

        return loss

    def test_batch(network: torch.nn.Module,
                   X_batch: feats_dtype,
                   Y_batch: label_dtype,
                   loss_fn: Callable[[feats_dtype, label_dtype], feats_dtype]) -> float:

        # signal testing time
        network.eval()

        with torch.no_grad():  # dont do differentation when not needed
            p_batch = network(X_batch)
            batch_loss = loss_fn(p_batch, Y_batch)

        return batch_loss.item()

    def test_epoch(network: torch.nn.Module,
                   dataloader: DataLoader,
                   loss_fn: Callable[[feats_dtype, label_dtype], feats_dtype],
                   device: str) -> float:

        loss = 0.
        for i, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)  # give back to GPU if available
            y_batch = y_batch.to(device)

            loss += test_batch(network=network, X_batch=x_batch, Y_batch=y_batch, loss_fn=loss_fn)
        loss /= i  # just for consistency

        return loss

    train_losses = []
    eval_losses = []
    for t in range(NUM_EPOCHS):
        train_loss = train_epoch(network=network, dataloader=trainloader, optimizer=optimizer(network.parameters(), lr=l_rate),
                                 loss_fn=loss_fn,
                                 device=device)
        test_loss = test_epoch(network=network, dataloader=testloader, loss_fn=loss_fn, device=device)

        print('\nEpoch {}'.format(t))
        print('Training loss %.4f' % train_loss)
        print('Validation loss %.4f' % test_loss)

        train_losses.append(train_loss)
        eval_losses.append(test_loss)

    return (train_losses, eval_losses)


def test(network: torch.nn.Module,
         dataloader: DataLoader,
         loss_fn: Callable[[feats_dtype, label_dtype], feats_dtype],
         device: str) -> float:
    network.eval()
    correct = 0.
    total = 0.
    for data, labels in dataloader:
        # data = Variable(data)
        # labels = Variable(labels)

        data = data.to(device)
        labels = labels.to(device)

        predictions = network(data)
        _, predicted = torch.max(predictions.data, 1)
        total += labels.size(0)
        temp = (predicted == labels.data).sum()
        correct += temp

    accuracy = 100 * correct / total

    return accuracy
