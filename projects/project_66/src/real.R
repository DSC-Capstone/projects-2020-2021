myArgs <- commandArgs(trailingOnly = TRUE)

library('magrittr')
library('dplyr')
library('compcodeR')

in_dir <- myArgs[1]

gene_matrix <- read.csv(file.path(in_dir, "gene_matrix.csv"))
annotations <- read.csv(file.path(in_dir, "SraRunTable.csv"))

valids <- colnames(gene_matrix)
valids <- valids[-1]

annotations <- annotations[annotations$Run %in% valids, ]

# Real dataset divided by different clinical diagnosis
schizo <- annotations[(annotations[, "clinical_diagnosis"] == 'Control' | annotations[, "clinical_diagnosis"] == 'Schizophrenia'),]
mdd <- annotations[(annotations[, "clinical_diagnosis"] == 'Control' | annotations[, "clinical_diagnosis"] == 'Major Depression'),]
bipolar <- annotations[(annotations[, "clinical_diagnosis"] == 'Control' | annotations[, "clinical_diagnosis"] == 'Bipolar Disorder'),]

# Create gene matrixes for each clinical diagnosis
schizo_gene <- gene_matrix %>% select(schizo$Run)
schizo_gene <- sapply(schizo_gene, round)

mdd_gene <- gene_matrix %>% select(mdd$Run)
bipolar_gene <- gene_matrix %>% select(bipolar$Run)

# For DESeq2 and NOISeq
schizo_gene_1 <- gene_matrix %>% select(schizo$Run)
schizo_gene_1 <- sapply(schizo_gene_1, round)

mdd_gene_1 <- gene_matrix %>% select(mdd$Run)
mdd_gene_1 <- sapply(mdd_gene_1, round)

bipolar_gene_1 <- gene_matrix %>% select(bipolar$Run)
bipolar_gene_1 <- sapply(bipolar_gene_1, round)

# First, try to run each with only condition and gene name
schizo_annotation_1 <- data.frame(schizo$clinical_diagnosis)
row.names(schizo_annotation_1) <- schizo$Run
colnames(schizo_annotation_1) <- "condition"

mdd_annotation_1 <- data.frame(mdd$clinical_diagnosis)
row.names(mdd_annotation_1) <- mdd$Run
colnames(mdd_annotation_1) <- "condition"

bipolar_annotation_1 <- data.frame(bipolar$clinical_diagnosis)
row.names(bipolar_annotation_1) <- bipolar$Run
colnames(bipolar_annotation_1) <- "condition"

# Try more stuff
schizo_annotation <- data.frame(schizo$clinical_diagnosis, schizo$age_at_death, schizo$Brain_pH)
row.names(schizo_annotation) <- schizo$Run
schizo_annotation <- schizo_annotation %>% rename(condition = schizo.clinical_diagnosis, age_at_death = schizo.age_at_death, Brain_pH = schizo.Brain_pH)

mdd_annotation <- data.frame(mdd$clinical_diagnosis, mdd$age_at_death, mdd$Brain_pH)
row.names(mdd_annotation) <- mdd$Run
mdd_annotation <- mdd_annotation %>% rename(condition = mdd.clinical_diagnosis, age_at_death = mdd.age_at_death, Brain_pH = mdd.Brain_pH)

bipolar_annotation <- data.frame(bipolar$clinical_diagnosis, bipolar$age_at_death, bipolar$Brain_pH)
row.names(bipolar_annotation) <- bipolar$Run
bipolar_annotation <- bipolar_annotation %>% rename(condition = bipolar.clinical_diagnosis, age_at_death = bipolar.age_at_death, Brain_pH = bipolar.Brain_pH)

# Create an info parameters for each condition
schizo_info <- list(dataset = "schizo", uID = "1000000000")
mdd_info <- list(dataset = "mdd", uID = "1000000001")
bipolar_info <- list(dataset = "bipolar", uID = "1000000002")

schizo_info_1 <- list(dataset = "schizo", uID = "1000000003")
mdd_info_1 <- list(dataset = "mdd", uID = "1000000004")
bipolar_info_1 <- list(dataset = "bipolar", uID = "1000000005")

# Create the compCodeR objects
schizo_object_1 <- compData(schizo_gene_1, schizo_annotation, schizo_info_1)
mdd_object_1 <- compData(mdd_gene_1, mdd_annotation, mdd_info_1)
bipolar_object_1 <- compData(bipolar_gene_1, bipolar_annotation, bipolar_info_1)

schizo_object <- compData(schizo_gene, schizo_annotation, schizo_info)
mdd_object <- compData(mdd_gene, mdd_annotation, mdd_info)
bipolar_object <- compData(bipolar_gene, bipolar_annotation, bipolar_info)

# Save the objects to a .rds file
saveRDS(schizo_object_1, file = file.path(in_dir, "schizo_1.rds"))
saveRDS(mdd_object_1, file = file.path(in_dir, "mdd_1.rds"))
saveRDS(bipolar_object_1, file = file.path(in_dir, "bipolar_1.rds"))

saveRDS(schizo_object, file = file.path(in_dir, "schizo.rds"))
saveRDS(mdd_object, file = file.path(in_dir, "mdd.rds"))
saveRDS(bipolar_object, file = file.path(in_dir, "bipolar.rds"))

print("Successfully created the rds file for the real dataset. Ready to implement tools on real datasets")
print("-----------------------------------")