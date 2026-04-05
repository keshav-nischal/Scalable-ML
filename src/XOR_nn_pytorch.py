import torch.nn as nn
import torch

torch.manual_seed(23)
neuralNetwork = nn.Sequential(
    nn.Linear(2, 2),
    nn.Sigmoid(),
    nn.Linear(2, 1)
)


x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

loss_fn = nn.BCEWithLogitsLoss() 
optimizer = torch.optim.Adam(neuralNetwork.parameters(), lr=0.1)

x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

for epoch in range(5000):
    optimizer.zero_grad()
    output = neuralNetwork(x_tensor)
    loss = loss_fn(output, y_tensor)
    loss.backward()
    optimizer.step()

# print(nn.parameters[0])
# check prediction for all the x possible
with torch.no_grad():
    predictions = neuralNetwork(x_tensor)
    predictions = torch.sigmoid(predictions)
    for i, (input_val, pred) in enumerate(zip(x, predictions)):
        print(f"Input: {input_val}, Prediction: {pred.item():.4f}, Expected: {y[i][0]}")