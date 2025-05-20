import tensorflow as tf
from keras import layers, Sequential
import load_data

def train_data():
    model = Sequential([
        layers.Input(shape = (48,48,1)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        
        layers.Dropout(0.5),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    train_dataset, validation_dataset = load_data.load_data()
    
    model.fit(train_dataset, validation_data=validation_dataset, epochs=100)
    
    _, final_accuracy = model.evaluate(validation_dataset)
    print("Final validation accuracy:", final_accuracy)
    model.save("aa.keras")



train_data()