# TensorFlowWorkshop

[Presentation](https://docs.google.com/presentation/d/1PnwbcGh6zjPV5Jx1kYZiL83k1sizCzuYhg1SM5ET9jk/edit?usp=sharing)

[Beginner's Tutorial](https://www.tensorflow.org/versions/r1.1/get_started/mnist/beginners)

[Advanced Tutorial](https://www.tensorflow.org/tutorials/layers)

## Usage

`
python3 main.py -h

usage: main.py [-h] [--nn NN] [--n_itr N_ITR] [--log_freq LOG_FREQ]
               [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help              show this help message and exit
  --nn NN                 which neural network to use: 'simple' 'intermediate' 'advanced'
  --n_itr N_ITR           number of iterations
  --log_freq LOG_FREQ     number of iterations to skip after logging
  --batch_size BATCH_SIZE number of tuples per iteration to feed
`