# Probabilistic neural networks for dialogues systems

This is the source code for my fourth year project.

In this project we aim to explore methods for emotion-based text generation in the context of Twitter using neural neworks, more precisely LSTM and GRU cells. 
The hypothesis is that as long as the agent is trained on data, highly expressive on a certain emotion, the agent will generate text, expressing that same emotion.
To achieve this, a naural network classifier is implemented, for analysing text. With it, a dataset of labelled tweets is created.
Four GRU models are trained on that dataset - one for each of the following emotions: joy, sadness, anger, fear. Chapter 1 is Introduction - 
it describes motion, possible applications and scope of the project. Chapter 2 is Background and Literature Review - it explains 
concepts, required for the reader, and discusses other research in the topic. Chapter 3 is Methodology - it illustrates the high-level
approach to solving the problem at hand. Chapter 4 is Data - it outlines how data has been pre-processed and processed for training.
Chapter 5 is Evaluation - describes the methods of evaluation, used in this project, as well as the results. Chapter 6 is Discussion 
and speculation for the evaluation results. It also discusses ideas for future improvement. Chapter 7 is Conclusion.
