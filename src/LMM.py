import torch.nn as nn
import torch
import torch.optim as optim



class LinearMixedModel(nn.Module):
    def __init__(self, num_fixed_effects, num_random_effects):
        super(LinearMixedModel, self).__init__()
        self.fixed_effects = nn.Linear(num_fixed_effects, 1)
        self.random_effects = nn.Linear(num_random_effects, 1)

    def forward(self, fixed_inputs, random_inputs):
        fixed_output = self.fixed_effects(fixed_inputs)
        random_output = self.random_effects(random_inputs)
        output = fixed_output + random_output
        return output
    

def log_likelihood(y_true, y_pred):
    residual = y_true - y_pred
    noise_variance = torch.var(residual)
    log_likelihood = -0.5 * (torch.log(2 * torch.pi * noise_variance) + (residual.pow(2) / noise_variance))
    return log_likelihood.mean()


def train_linear_mixed_model(train_data, valid_data, model, random_inputs, batch_size=32, num_epochs=500, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_num_batches = 0
    valid_num_batches = 0
    train_loss = []
    valid_loss = []

    for epoch in range(num_epochs):
        model.train()
        train_total_loss = 0.0

        for i, (src, tgt) in enumerate(train_data):
            # Get the current batch
            batch_fixed_inputs = src
            batch_y_true = tgt

            # Forward pass
            optimizer.zero_grad()
            y_pred = model(batch_fixed_inputs, random_inputs)

            # Compute loss
            loss = -log_likelihood(batch_y_true, y_pred)
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
            y_pred = model(batch_fixed_inputs, random_inputs)

            # Compute loss
            loss = -log_likelihood(batch_y_true, y_pred)
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
