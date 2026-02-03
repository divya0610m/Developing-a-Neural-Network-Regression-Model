# EXPERIMENT 1 - Developing a Neural Network Regression Model
## NAME : DIVYA LAKSHMI M
## REGISTRATION NUMBER : 212224040082

## AIM :
To develop a neural network regression model for the given dataset.

## THEORY :
The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features.
The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

## Neural Network Model :
<img width="1820" height="1017" alt="Screenshot 2026-02-02 094607EXP1" src="https://github.com/user-attachments/assets/91177b10-6ef2-428c-b60f-6fbbf3c926d2" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM :

### Name: DIVYA LAKSHMI M

### Register Number: 212224040082

```
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)

        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss=criterion(ai_brain(X_train),y_train)
        loss.backward()
        optimizer.step()


        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```

### Dataset Information

<img width="198" height="422" alt="Screenshot 2026-02-03 085556" src="https://github.com/user-attachments/assets/0d37fcc8-51d5-46ff-8ba6-5a1fa57e7653" />

### OUTPUT

<img width="395" height="239" alt="Screenshot 2026-02-03 085254" src="https://github.com/user-attachments/assets/5e550758-d4ff-4940-ab71-a9388ddd39fb" />

<img width="257" height="46" alt="Screenshot 2026-02-03 085301" src="https://github.com/user-attachments/assets/398d4fd3-126f-4a47-aef4-6f47188b8177" />

### Training Loss Vs Iteration Plot

<img width="748" height="590" alt="Screenshot 2026-02-02 094612" src="https://github.com/user-attachments/assets/678af924-379f-494b-9911-54d34d39f9cf" />

### New Sample Data Prediction

<img width="340" height="52" alt="Screenshot 2026-02-03 085307" src="https://github.com/user-attachments/assets/d00366db-6973-4c8b-b1b0-2a29f18ffe4e" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
