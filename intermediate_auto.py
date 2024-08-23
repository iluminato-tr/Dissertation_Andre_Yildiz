# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 00:56:47 2024

@author: andre
"""
import numpy as np
import pandas as pd
from scipy.stats import zscore
import tensorflow as tf
from scipy.stats import ttest_1samp
from scipy import stats 

def load_and_preprocess_data(file_paths):
    """
    Load and preprocess multiple datasets.
    
    Parameters:
    - file_paths: List of file paths to the datasets.
    
    Returns:
    - df_normalized: Concatenated and normalized DataFrame of all datasets.
    """
    data_list = []
    data_normalized_list = []
    
    for file_path in file_paths:
        data = pd.read_csv(file_path, header=0, index_col=0)
        data = data.astype(float)
        data_normalized = zscore(data)
        data_list.append(data)
        data_normalized_list.append(pd.DataFrame(data_normalized, index=data.index, columns=data.columns))
    
    df_normalized = pd.concat(data_normalized_list, axis=1)
    
    # Check for NaN values
    if df_normalized.isna().any().any():
        print("There are NaN values in the normalized DataFrame.")
    else:
        print("No NaN values found in the normalized DataFrame.")
    
    return df_normalized.astype(np.float32) 

def build_autoencoder(n_input,n_layer):
    """
    Build an autoencoder model.
    
    Parameters:
    - n_input: Number of input features.
    
    Returns:
    - autoencoder: Compiled autoencoder model.
    """
    class Autoencoder(tf.keras.Model):
        def __init__(self):
            super(Autoencoder, self).__init__()
            self.encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(n_layer, activation='relu')
            ])
            self.decoder = tf.keras.Sequential([
                tf.keras.layers.Dense(n_layer, activation='relu'),
                tf.keras.layers.Dense(n_input, activation='sigmoid')
            ])
        
        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    
    autoencoder = Autoencoder()
    return autoencoder

def train_autoencoder(tcga_input, autoencoder, learning_rate=0.0001, training_epochs=100, batch_size=128, display_step=2):
    """
    Train the autoencoder model.
    
    Parameters:
    - tcga_input: Input data for training.
    - autoencoder: Autoencoder model.
    - learning_rate: Learning rate for the optimizer.
    - training_epochs: Number of epochs for training.
    - batch_size: Batch size for training.
    - display_step: Frequency of displaying the training loss.
    """
    dataset = tf.data.Dataset.from_tensor_slices(tcga_input)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    @tf.function
    def train_step(batch_xs):
        with tf.GradientTape() as tape:
            noise = 0.2 * tf.random.normal(shape=batch_xs.shape, dtype=tf.float32)
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
    return encoded_data

def main(file_paths,n_layer):
    df_normalized = load_and_preprocess_data(file_paths)
    n_input = df_normalized.shape[1]
    
    autoencoder = build_autoencoder(n_input,n_layer)
    
    encoded_data = train_autoencoder(df_normalized, autoencoder)
    
    # Display the encoded data
    return(encoded_data)
    
# Example usage:

mrna = r'~\Desktop\BRCA_project\Processed\BRCA_mRNA_annotated.csv',
mirna = r'~\Desktop\BRCA_project\Processed\BRCA_mirna_processed.csv',
meth = r'C~\Desktop\BRCA_project\Processed\BRCA_meth_processed.csv',
proteome = r'~\Desktop\BRCA_project\Processed\BRCA_proteome_processed.csv',


encoded_data = main(mrna,100)
encoded_data2 =  main(mirna,100)
encoded_data3 = main(meth,100)
encoded_data4 = main(proteome,100)


overall = np.concatenate((encoded_data,encoded_data2 ,encoded_data3,encoded_data4),axis =1)

np.savetxt(r'~/Desktop/BRCA_project/results/intermediate.csv', overall, delimiter=',')

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
df_normalized = pd.concat([data_n, data2_n, data3_n, data4_n], axis=1)



df2 = df_normalized.transpose()
df3 = np.matmul(df2,overall)
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
column_names_list_comprehension = [df_normalized.columns[i] for i in index_list]
print("Using list comprehension:", column_names_list_comprehension)
genes = [col for col in column_names_list_comprehension if col in data.columns]
genes_df = pd.DataFrame(genes, columns=['Gene'])

mirnaa = [col for col in column_names_list_comprehension if col.startswith('hsa')]
mirna_df = pd.DataFrame(mirnaa, columns=['miRNA'])

methylation = [col for col in column_names_list_comprehension if col.startswith('cg')]
methylation_df = pd.DataFrame(methylation, columns=['Metyhlation site'])

protein = [col for col in column_names_list_comprehension if col in data4.columns]
protein_df = pd.DataFrame(protein, columns=['proteins'])
# Save the DataFrame to a CSV file
genes_df.to_csv('~/Desktop/BRCA_project/results/autoencoder_gene_results.csv', index=False)
mirna_df.to_csv('~/Desktop/BRCA_project/results/autoencoder_mirna_results.csv', index=False)
methylation_df.to_csv('~/Desktop/BRCA_project/results/autoencoder_methylation_results.csv', index=False)
protein_df.to_csv('~/Desktop/BRCA_project/results/autoencoder_protein_results.csv', index=False)
top_500_features["names"] = column_names_list_comprehension
top_500_features.to_csv('~/Desktop/BRCA_project/results/complete_df.csv', index=False)

print(top_500_features)








