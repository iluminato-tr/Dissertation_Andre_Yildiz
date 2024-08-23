library(dplyr)
library(tibble)
library(stringr)
library(matrixStats)

###Loading Data files
load(file="C:/Users/andre/Desktop/BRCA_project/Data/BRCA_mrna.RData") #BRCA mRNA
load(file="C:/Users/andre/Desktop/BRCA_project/Data/BRCA_mirna.RData") #BRCA miRNA
load(file="C:/Users/andre/Desktop/BRCA_project/Data/BRCA_proteome.RData") #BRCA proteome
load(file="C:/Users/andre/Desktop/BRCA_project/Data/BRCA_meth.RData") #BRCA methylation beta value
load(file="C:/Users/andre/Desktop/BRCA_project/Data/df_vecs.RData") #BRCA Tumor or not 
load(file="C:/Users/andre/Desktop/BRCA_project/Data/subtype.RData") #BRCA subtype file


### Step 1: Processing mRNA data
mrna_multimodal <- as.data.frame(mrna_multimodal)
mrna_multimodal <- mrna_multimodal %>% arrange(barcode)

barcodes_multimodal <- mrna_multimodal$barcode
mrna_multimodal_processing <- subset(mrna_multimodal, select = -barcode)
limit <- nrow(mrna_multimodal_processing)*.2
mrna_multimodal_processing <- as.matrix(mrna_multimodal_processing)
class(mrna_multimodal_processing) <- "numeric"
mrna_multimodal_processing <- mrna_multimodal_processing[, which(as.numeric(colSums(mrna_multimodal_processing == 0)) < limit)]
mrna_multimodal_processing <- as.data.frame(mrna_multimodal_processing)
write.csv(mrna_multimodal_processing,file = "C:/Users/andre/Desktop/BRCA_project/Processed/BRCA_mRNA_processed.csv")

### Step 5: Preprocessing Methylation Data 
#Getting rid-off unwanted methlation sites 
meth_multimodal_processing <- as.data.frame(meth_multimodal)
meth_multimodal_processing <- meth_multimodal_processing %>% arrange(barcode)

meth_multimodal_processing <- subset(meth_multimodal_processing, select = -barcode)
meth_multimodal_processing <- meth_multimodal_processing %>% dplyr::select(!starts_with("rs"))
#Removing genes with more than 20% zeros 
limit_probes <- nrow(meth_multimodal_processing)*.2
meth_nas <- meth_multimodal_processing[ , which(colSums(is.na(meth_multimodal_processing)) < limit_probes)]


# Mapping probes to cpg islands
# CpG Data Taken Downloaded Directly from the Illumina Website, file named: "infinium-methylationepic-v-1-0-b5-manifest-file.csv"
cpg_map <- read.csv("C:/Users/andre/Desktop/Multimodal/Methylation/CPG_ProbeData/infinium-methylationepic-v-1-0-b5-manifest-file.csv",header=TRUE, skip=3)
cpg_map <- cpg_map %>% dplyr::select(IlmnID, UCSC_CpG_Islands_Name, Relation_to_UCSC_CpG_Island)
probe_index2 <- which(cpg_map$Relation_to_UCSC_CpG_Island=="Island")
probe_id2 <- cpg_map$IlmnID[probe_index2]
meth_cpg <- meth_nas[ ,colnames(meth_nas) %in% probe_id2]
cpg_barcodes <- rownames(meth_cpg)

cpg <- as.matrix(meth_cpg)
class(cpg) <- "numeric"
ranks <- colVars(cpg, na.rm=TRUE)
ranks_sort <- sort(ranks, index.return=TRUE, decreasing=TRUE)
top_indexes <- ranks_sort$ix[1:18000] #top 20,000 methylation probes by variance
cpg_variance <- cpg[,top_indexes]
cpg_variance <- as.data.frame(cpg_variance)
meth_filtered <- as.data.frame(cbind(barcodes_multimodal,cpg_variance))
colnames(meth_filtered)[1] <- 'barcode'
#Replacing NA values with median
meth_filtered <- meth_filtered %>%
  mutate(across(everything(), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

meth_filtered <- subset(meth_filtered, select = -barcode)

#Saving

### Step 6: miRNA Processing
#Removing genes more than 20% zeros
mirna_multimodal <- mirna_multimodal %>% arrange(barcode)
miRNA_processing <- subset(mirna_multimodal, select = -barcode)
limit <- nrow(miRNA_processing)*.2
miRNA_processing <- as.matrix(miRNA_processing)
class(miRNA_processing) <- "numeric"
miRNA_filtered <- miRNA_processing[, which(as.numeric(colSums(miRNA_processing == 0)) < limit)]
miRNA_filtered <- as.data.frame(miRNA_filtered)


### Step 7: proteome Processing
proteome_processing <- as.data.frame(proteome_multimodal)
proteome_processing <- proteome_processing %>% arrange(barcode)

proteome_processing <- subset(proteome_processing, select = -barcode)
limit <- nrow(proteome_processing)*.2
proteome_processing <- as.matrix(proteome_processing)
class(proteome_processing) <- "numeric"
proteome_processing <- proteome_processing[, which(as.numeric(colSums(proteome_processing == 0)) < limit)]
proteome_processing <- as.data.frame(proteome_processing)



df <- as.data.frame(filtered_df_vecs4)
df <- df %>% arrange(barcode)
write.csv(df, file = "C:/Users/andre/Desktop/BRCA_project/Processed/BRCA_df_processed.csv")


df <- read.csv(file = "C:/Users/andre/Desktop/BRCA_project/Processed/BRCA_mRNA_processed.csv")
df <- df[,-1]

RNA_transposed <- t(mrna_multimodal_processing)

# Remove the first row after using it as column names
mymat <- as.matrix(as.data.frame(RNA_transposed, stringsAsFactors = FALSE))
mymat<- as.numeric(mymat)
mymat <- matrix(mymat, nrow = nrow(RNA_transposed), byrow = FALSE,
                dimnames = list(rownames(RNA_transposed), colnames(RNA_transposed)))
ens <- rownames(mymat)

symbols <- mapIds(org.Hs.eg.db, keys = ens,
                  column = c('SYMBOL'), keytype = 'ENSEMBL')
symbols <- symbols[!is.na(symbols)]
symbols <- symbols[match(rownames(mymat), names(symbols))]
rownames(mymat) <- symbols
keep <- !is.na(rownames(mymat))
mymat <- mymat[keep,]
mymat <- t(mymat)






load(file = "C:/Users/andre/Desktop/BRCA_project/Data/subtype.RData")
df =read.csv(file= "C:/Users/andre/Desktop/BRCA_project/Processed/BRCA_df_processed.csv" )
df_filtered <- subset(df, Value != 0)
df_filtered2<- subset(df, Value != 0)
colnames(subtype_multimodal)[1] <- "barcode"
df_filtered$barcode <- substring(df_filtered$barcode,1,12)
df_merged <- df_filtered %>% left_join(subtype_multimodal %>% select(barcode,BRCA_Subtype_PAM50 ), by = "barcode")
df_merged$barcode2 <- df_filtered2$barcode
df_merged$barcode <-df_merged$barcode2
df_merged <- subset(df_merged, select = -barcode2)
rows_to_add <- df[df$Value == 0, ]
rows_to_add$BRCA_Subtype_PAM50 <- "Control"
df_merged <- subset(df_merged, select = -X)
rows_to_add <- subset(rows_to_add, select = -X)

# Append these rows to df2
df_merged <- rbind(df_merged, rows_to_add)
df_merged <- df_merged %>% arrange(barcode)
row.names(df_merged) <- NULL
na_indices <- which(is.na(df_merged$BRCA_Subtype_PAM50))
df_cleaned <- df_merged[-na_indices, ]

df_cleaned <- df_cleaned %>%
  mutate(Multi_code = case_when(
    BRCA_Subtype_PAM50 == "Basal" ~ 1,
    BRCA_Subtype_PAM50 == "LumA" ~ 2,
    BRCA_Subtype_PAM50 == "LumB" ~ 3,
    BRCA_Subtype_PAM50 == "Normal" ~ 4,
    BRCA_Subtype_PAM50 == "Her2" ~ 5,
    BRCA_Subtype_PAM50 == "Control" ~ 0,
    TRUE ~ NA_real_  # Ensure NA remains as NA
  ))


write.csv(df_merged, file= "C:/Users/andre/Desktop/BRCA_project/Processed/Multi_code.csv")

mymat <- as.data.frame(mymat)
mymat <- mymat[-na_indices, ]
proteome_processing <- proteome_processing[-na_indices, ]
miRNA_filtered <- miRNA_filtered[-na_indices, ]
meth_filtered <- meth_filtered[-na_indices, ]


write.csv(meth_filtered,file = "C:/Users/andre/Desktop/BRCA_project/Processed/BRCA_meth_processed.csv")
write.csv(mymat, file = "C:/Users/andre/Desktop/BRCA_project/Processed/BRCA_mRNA_annotated.csv")
write.csv(miRNA_filtered, file = "C:/Users/andre/Desktop/BRCA_project/Processed/BRCA_mirna_processed.csv")
write.csv(proteome_processing, file = "C:/Users/andre/Desktop/BRCA_project/Processed/BRCA_proteome_processed.csv")
write.csv(df_cleaned, file= "C:/Users/andre/Desktop/BRCA_project/Processed/Multi_code.csv")


