# Project

Losses in the Electric Energy Distribution System can be defined as the difference between
the electric energy purchased by the distributors and that billed to their consumers. These
losses can be technical or non-technical. Combating Non-Technical Losses is extremely
important, both for distributors and for society in general, making it interesting to study
and improve techniques that aim to find and stop the sources of these losses. This work
aims to develop a tool to combat Non-Technical Losses, focusing on solutions without the
use of hardware, making use of machine learning techniques. In this sense, the data to be
used were selected and adapted from the consumer database of a Brazilian company. With
such data, an exploratory analysis was carried out and, with the information obtained,
the variables to be used in the training of the models were chosen, after going through
the pre-processing stage. For the construction of such models, the techniques of Support
Vector Machines, Artificial Neural Networks (more specifically Multilayer Perceptron),
Optimum-Path Forests, Random Forests and Gradient Boosting were used and evaluated.
The last one presented better results in the theoretical tests. This model was used to
select installations to be inspected in the field, with some divergence in relation to the
theoretical tests, which can be explained by factors such as the quality of the database and
the distributor’s service, disregard of the temporal evolution of the data and seasonality,
as well as the use of a reduced sample. Finally, other methodologies were proposed to
boost the results of the models, which can be put into practice in future works.

See "Utilização de Técnicas de Aprendizado de Máquina para Identificação de Fraudes na Distribuição de Energia Elétrica.pdf" for more information.

# Files

**training.py:** used to create and test the models.

**selection.py:** used to select installations for inspection.
