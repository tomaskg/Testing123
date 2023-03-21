from textgenrnn import textgenrnn

textgen = textgenrnn()

num_epochs = int(input('Enter the number of epochs to fine-tune for: '))
data_path = input('Enter the path to your training data: ')
learning_rate = float(input('Enter the learning rate: '))

textgen.train_from_file(data_path, num_epochs=num_epochs, gen_optimizer=tf.keras.optimizers.Adam(learning_rate))
