# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 21:19:23 2024

@author: andre
"""

import numpy as np
import pandas as pd

from scipy.stats import ttest_1samp
import tensorflow as tf

from scipy import stats 

# Load gene expression data
# Load the data
data = pd.read_csv(r'~\Desktop\BRCA_project\Processed\BRCA_mRNA_annotated.csv', header=0, index_col=0)
data = data.astype(float)
data_n = stats.zscore(data)
data2 = pd.read_csv(r'~\Desktop\BRCA_project\Processed\BRCA_mirna_processed.csv', header=0, index_col=0)
data2 = data2.astype(float)
data2_n = stats.zscore(data2)

data3 = pd.read_csv(r'~\Desktop\BRCA_project\Processed\BRCA_meth_processed.csv', header=0, index_col=0)
data3 = data3.astype(float)
data3_n = stats.zscore(data3)

data4 = pd.read_csv(r'~\Desktop\BRCA_project\Processed\BRCA_proteome_processed.csv', header=0, index_col=0)
data4 = data4.astype(float)
data4_n = stats.zscore(data4)


# Concatenate the dataframes
df = pd.concat([data, data2, data3, data4], axis=1)
df_normalized = pd.concat([data_n, data2_n, data3_n, data4_n], axis=1)


# Check for NaN values in the normalized dataframe
if df_normalized.isna().any().any():
    print("There are NaN values in the normalized DataFrame.")
else:
    print("No NaN values found in the normalized DataFrame.")

# Display the normalized dataframe
print(df_normalized)

# Convert the normalized data array back to a DataFrame
tcga_input = df_normalized.astype(np.float32)  # Ensure data is float32
length1 = tcga_input.shape[1]

# Parameters
learning_rate = 0.0001
training_epochs = 100
batch_size = 128
display_step = 2
examples_to_show = 10
n_input = tcga_input.shape[1]

# Define the encoder and decoder models using tf.keras
class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(200, activation='relu'),
            
            
        ])
        self.decoder = tf.keras.Sequential([
            
           
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(n_input, activation='sigmoid')
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Prepare the data
dataset = tf.data.Dataset.from_tensor_slices(tcga_input)
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

# Instantiate the autoencoder
autoencoder = Autoencoder()

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_fn = tf.keras.losses.MeanSquaredError()

# Training loop
@tf.function
def train_step(batch_xs):
    with tf.GradientTape() as tape:
        noise = 0.2 * tf.random.normal(shape=batch_xs.shape, dtype=tf.float32)  # Ensure noise is float32
        batch_xs_noisy = batch_xs + noise
        reconstructed = autoencoder(batch_xs_noisy)
        loss = loss_fn(batch_xs, reconstructed)
    gradients = tape.gradient(loss, autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
    return loss

for epoch in range(training_epochs):
    for batch_xs in dataset:
        loss = train_step(batch_xs)
    if epoch % display_step == 0:
        print(f"Epoch: {epoch + 1}, loss={loss.numpy()}")

# Get the features from the encoder
encoded_data = autoencoder.encoder(tcga_input).numpy()
np.savetxt(r'~/Desktop/BRCA_project/results/encoded_data.csv', encoded_data, delimiter=',')

print("Optimization Finished!")
print(encoded_data.shape)

df2 = df_normalized.transpose()
df3 = np.matmul(df2,encoded_data)
t_stat, p_values = ttest_1samp(df3, popmean=0, axis=1)

# Bonferroni correction
num_tests = len(p_values)
bonferroni_corrected_p_values = p_values * num_tests
bonferroni_corrected_p_values = np.minimum(bonferroni_corrected_p_values, 1.0)  # Cap at 1

# Create a DataFrame to organize the results
results_df = pd.DataFrame({
    't_stat': t_stat,
    'p_value': p_values,
    'bonferroni_corrected_p_value': bonferroni_corrected_p_values
})

# Sort by corrected p-values and select the top 100 features
top_500_features = results_df.nsmallest(500, 'bonferroni_corrected_p_value')
index_obj = top_500_features.index
index_list = index_obj.tolist()
print(index_list)
column_names_list_comprehension = [df.columns[i] for i in index_list]
print("Using list comprehension:", column_names_list_comprehension)
genes = [col for col in column_names_list_comprehension if col in data.columns]
genes_df = pd.DataFrame(genes, columns=['Gene'])

mirna = [col for col in column_names_list_comprehension if col.startswith('hsa')]
mirna_df = pd.DataFrame(mirna, columns=['miRNA'])

methylation = [col for col in column_names_list_comprehension if col.startswith('cg')]
methylation_df = pd.DataFrame(methylation, columns=['Metyhlation site'])

protein = [col for col in column_names_list_comprehension if col in data4.columns]
protein_df = pd.DataFrame(protein, columns=['proteins'])
# Save the DataFrame to a CSV file
genes_df.to_csv('~/Desktop/BRCA_project/results/autoencoder_gene_results1.csv', index=False)
mirna_df.to_csv('~/Desktop/BRCA_project/results/autoencoder_mirna_results1.csv', index=False)
methylation_df.to_csv('~/Desktop/BRCA_project/results/autoencoder_methylation_results1.csv', index=False)
protein_df.to_csv('~/Desktop/BRCA_project/results/autoencoder_protein_results1.csv', index=False)
top_500_features["names"] = column_names_list_comprehension
top_500_features.to_csv('~/Desktop/BRCA_project/results/complete_df1.csv', index=False)

print(top_500_features)