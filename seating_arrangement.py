import random
# Function to generate random unfriendly relationship matrix
def generate_unfriendly_matrix(num_people, num_unfriendly):
    matrix = [[0 for _ in range(num_people)] for _ in range(num_people)]
    while num_unfriendly > 0:
        i = random.randint(0, num_people - 1)
        j = random.randint(0, num_people - 1)
        # Ensure i != j and avoid duplicate unfriendly relationships
        if i != j and matrix[i][j] == 0:
            matrix[i][j] = 1
            matrix[j][i] = 1 # Make it symmetric
            num_unfriendly -= 1
    return matrix
def calculate_conflict_cost(seating_arrangement, unfriendly_matrix, num_seats):
    total_cost = 0
    for i in range(len(seating_arrangement)):
        for j in range(i + 1, len(seating_arrangement)): # Avoid double counting
            distance = abs(i - j)
            # Check for critical unfriendly positions (beside, front, or rear)
            if distance == 1 or distance == num_seats - 1 or distance == num_seats:
                cost = unfriendly_matrix[seating_arrangement[i]][seating_arrangement[j]]
            else:
                # Reduce penalty for other distances with a distance-based weight
                weight = 1.0 / (distance**2 + 1) # Avoid division by zero and give higher weight to closer positions
                cost = unfriendly_matrix[seating_arrangement[i]][seating_arrangement[j]] * weight
    total_cost += cost
    return total_cost

# generate training data
def generate_training_data(num_people, num_unfriendly, num_samples):
    training_data = []
    for _ in range(num_samples):
        _unfriendly_matrix = generate_unfriendly_matrix(num_people, num_unfriendly)
        training_data.append(_unfriendly_matrix)
    return training_data

def convert_to_permutation(output):
    _, indices = torch.sort(output, descending=True)
    return indices + 1  # تبدیل به دامنه 1 تا 24
def convert_index_to_names(predicted_seating_arrangement):
    first_names = ["Ali", "Zahra", "Reza", "Sara", "Mohammad", "Fatemeh", "Hossein", "Maryam", "Mehdi", "Narges", "Hamed", "Roya"]
    last_names = ["Ahmadi", "Hosseini", "Karimi", "Rahimi", "Hashemi", "Ebrahimi", "Moradi", "Mohammadi", "Rostami", "Fazeli", "Hosseinzadeh", "Niknam"]
    random.seed(40)
    random_names = set()
    while len(random_names) < 24:
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        random_names.add(f"{first_name} {last_name}")

    random_names = list(random_names) # primary list of names
    seating_arrangement = [[0 for _ in range(6)] for _ in range(4)]
    t = 0
    for i in range(4):
        for j in range(6):
            seating_arrangement[i][j] = random_names[predicted_seating_arrangement[t]]
            t += 1
    return seating_arrangement

import torch
import torch.nn as nn
import torch.optim as optim
# Neural network architecture
class SeatingArrangementNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SeatingArrangementNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.flat = nn.Flatten(0, -1)
    def forward(self, x):
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
# Hyperparameters
num_people = 24
num_unfriendly_pairs = 40 
learning_rate = 0.001
num_epochs = 10
num_samples = 1000

# Model definition
input_size = num_people * num_people 
hidden_size = 24
output_size = num_people # Output represents predicted seating order
model = SeatingArrangementNet(input_size, hidden_size, output_size)
# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = calculate_conflict_cost
# Load training data
urm_data = generate_training_data(num_people, num_unfriendly_pairs, num_samples)
urm_data = torch.tensor(urm_data).float()

# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(urm_data)):
       
        urm_tensor = torch.tensor(urm_data[i])
        predictions = model(urm_tensor.type(torch.FloatTensor))
        predictions = predictions.argsort(dim=0)
        permuted_outputs = torch.stack([convert_to_permutation(output) for output in predictions])
        loss = loss_fn(permuted_outputs, urm_tensor, num_people) # Update loss function arguments
        optimizer.zero_grad()
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
       
    print(f"Epoch: {epoch+1}/{num_epochs}")
print("Training complete!", )

# test
unfreindly_matrix = generate_unfriendly_matrix(num_people, num_unfriendly_pairs)
test_urm_tensor = torch.tensor(unfreindly_matrix)
predicted_seating_arrangement = model(test_urm_tensor.flatten().float())

predicted_seating_arrangement = convert_to_permutation(predicted_seating_arrangement)
predicted_seating_arrangement = [element - 1 for element in predicted_seating_arrangement]
names = convert_index_to_names(predicted_seating_arrangement)
conflict = loss_fn(predicted_seating_arrangement, unfreindly_matrix, num_people)
print("names: ", names, "conflict:", conflict)