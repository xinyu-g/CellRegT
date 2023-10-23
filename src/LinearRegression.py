import torch.nn as nn
import torch.optim as optim


class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.output = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.output(x)
        return x
    

def train_linear_model(train_data, valid_data, model, batch_size=32, num_epochs=500, lr=0.01):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    train_num_batches = 0
    valid_num_batches = 0
    train_loss = []
    valid_loss = []

    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        model.train()
        train_total_loss = 0.0

        for i, (src, tgt) in enumerate(train_data):
            # Get the current batch
            batch_fixed_inputs = src
            batch_y_true = tgt

            # Forward pass
            optimizer.zero_grad()
            y_pred = model(batch_fixed_inputs)

            # Compute loss
            loss = loss_func(y_pred, batch_y_true)
            train_total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            train_num_batches += 1

        if epoch % 50 == 0:
            avg_loss = train_total_loss / train_num_batches
            train_loss.append(avg_loss)
            print(f"Epoch {epoch}, Train average Loss: {avg_loss}" )



        model.eval()
        valid_total_loss = 0.0

        for i, (src, tgt) in enumerate(valid_data):
            # Get the current batch
            batch_fixed_inputs = src
            batch_y_true = tgt

            # Forward pass
            optimizer.zero_grad()
            y_pred = model(batch_fixed_inputs)

            # Compute loss
            loss = loss_func(y_pred, batch_y_true)
            valid_total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            valid_num_batches += 1

        if epoch % 50 == 0:
            avg_loss = valid_total_loss / valid_num_batches
            valid_loss.append(avg_loss)
            print(f"Epoch {epoch}, Valid average Loss: {avg_loss}" )
           

    print("Training completed.")
    return train_loss, valid_loss
    