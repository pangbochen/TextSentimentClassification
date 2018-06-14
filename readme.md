# Text Sentiment Classification

loss and accuaracy

## BiLSTM model

## LSTM Model

## CNN Model

## Mixed Model

take lstm nueral network as the text feature extractor, then use 

ectract the middle layer output of the pytorch nerual network?

1) return the middle output in the forward funtion 
2) define a new modle
3) hook

I accomplish way 1) and way 3, way 1 is easy and safe compared with way 3, for hook may cause:

when you run model(input) the forward prediciton way, pytorch will check the hook function.

I also use the fc to evaluate, so it may accumulate the cum_tensor also in nn ealuate, so it cause 
the difference between sizes of train_X and train_y

safe is 