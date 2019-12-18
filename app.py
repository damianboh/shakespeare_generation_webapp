import flask
import pickle
import tensorflow as tf
import numpy as np


app = flask.Flask(__name__, template_folder='templates')

with open(f'model/char2idx.pkl', 'rb') as f:
   char2idx = pickle.load(f)

with open(f'model/idx2char.pkl', 'rb') as f:
   idx2char = pickle.load(f)

model = tf.keras.models.load_model('model/shakespeare_generation_model.h5')

def generate_text(model, start_string, num_generate):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = num_generate

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


    
@app.route('/', methods=['GET', 'POST'])
def main():

	
	if flask.request.method == 'GET':
		return(flask.render_template('main.html'))
		
	if flask.request.method == 'POST':
		start_text = ' '
		start_text = flask.request.form['start_text']
		num_char = flask.request.form['num_char']
		prediction = generate_text(model, start_string=start_text, num_generate=int(num_char))
		return flask.render_template('main.html',
                                     original_input={'Starting Text':start_text},
                                     result=prediction,
                                     )
        
if __name__ == '__main__':
    
	app.config['TEMPLATES_AUTO_RELOAD'] = True
	app.run()