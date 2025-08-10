# Gene Expression Data Processing and Analysis
# Author: [Your Name]
# Description: This script processes RNA-seq count data, maps gene IDs to gene names,
# calculates FPKM and TPM, performs normalization and basic statistical analysis,
# and visualizes gene expression for publication-quality figures.

# ========================= [ Environment Setup ] =========================
rm(list = ls())                          # Clear environment
options(stringsAsFactors = FALSE)       # Prevent automatic factor conversion
gc()                                     # Garbage collection

# ========================= [ Load Required Packages ] =========================
library(rtracklayer)        # Reading GFF/GTF files
library(dplyr)              # Data manipulation
library(ggpubr)             # Publication-quality plots
library(corrplot)           # Correlation plots (not used here)
library(pheatmap)           # Heatmaps (not used here)
library(patchwork)          # Plot composition (not used here)
library(edgeR)              # RNA-seq normalization
library(ggplot2)            # Data visualization
library(tidyverse)          # Data wrangling
library(DGEobj.utils)       # FPKM and TPM calculation
library(compareGroups)      # Descriptive statistics
library(purrr)              # Functional programming
library(tidyr)              # Data tidying
library(flextable)          # Table formatting
library(glue)               # String formatting

# ========================= [ Set Working Directory ] =========================
setwd("C:/Users/xinyu/Desktop/code")
# ========================= [ Load and Filter Raw Count Data ] =========================
rawcount <- read.table("featureCounts_clean.txt", row.names = 1, sep = "\t", header = TRUE)
test <- rowSums(rawcount > 0) >= floor(0.5 * ncol(rawcount))
filter_count <- rawcount[test, ]

# Replace invalid values (NA, negative, infinite) with 0
invalid_values <- sapply(filter_count, function(x) !is.finite(x) | x < 0)
filter_count[invalid_values] <- 0
filter_count <- data.frame(gene_id = row.names(filter_count), filter_count)
rownames(filter_count) <- NULL
# ========================= [ Load GTF and Map Gene IDs to Gene Names ] =========================
gff <- readGFF("Homo_sapiens.GRCh38.110.gtf")
mapid <- gff[gff$type == "gene", c("gene_id", "gene_name", "gene_biotype")]
write.csv(mapid, file = "gtfgene.csv", row.names = FALSE)
mapid <- read.csv("gtfgene.csv")

df <- merge(filter_count, mapid, by = "gene_id")
df_pro <- subset(df, gene_biotype == "protein_coding")
df_no <- df_pro[!is.na(df_pro$gene_name), ]
df_dup <- df_no[!duplicated(df_no$gene_name), ]

# Create named and ID-based matrices

# Gene names matrix
df_clean <- subset(df_dup, select = -c(gene_id, gene_biotype))
df_name <- df_clean[, c(ncol(df_clean), 1:(ncol(df_clean) - 1))]

# Gene IDs matrix
df_id <- subset(df_dup, select = -c(gene_name, gene_biotype))

# Save cleaned data
write.csv(df_name, file = "pro_name.csv", row.names = FALSE)
write.csv(df_id, file = "pro_id.csv", row.names = FALSE)

# ========================= [ FPKM and TPM Calculation ] =========================
genelength <- read.table("genelength.txt", sep = "\t", header = TRUE)
pro_name <- read.csv("pro_name.csv", row.names = 1)
pro_id <- read.csv("pro_id.csv")

df <- merge(pro_id, genelength, by.x = "gene_id", by.y = "Geneid")
countsMatrix <- apply(pro_name, 2, as.numeric)
rownames(countsMatrix) <- rownames(pro_name)

fpkm <- convertCounts(countsMatrix, "fpkm", df$Length, log = FALSE)
tpm  <- convertCounts(countsMatrix, "tpm", df$Length, log = FALSE)

write.csv(fpkm, file = "fpkm.csv")
write.csv(tpm, file = "tpm.csv")

# ========================= [ Normalization and Boxplot ] =========================
filter_count <- read.csv("pro_name.csv", row.names = 1)
filter_count_cpm <- log2(cpm(filter_count) + 1)

plot_data <- data.frame(expression = c(filter_count_cpm),
                        sample = rep(colnames(filter_count_cpm), each = nrow(filter_count_cpm)))

ggboxplot(plot_data, x = "sample", y = "expression") + 
  theme_bw() + 
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 90, face = "bold"),
        axis.text.y = element_text(face = "bold"))


# ========================= [ Descriptive Statistics ] =========================
mymeta <- read.csv("mymeta.csv", row.names = 1)
mymeta$group <- factor(mymeta$group, levels = c("LS", "HS"))
mymeta$sex <- factor(mymeta$sex, levels = c("female", "male"))
mymeta$Boosterinjection <- factor(mymeta$Boosterinjection, levels = c(0, 1, 2))

# Shapiro-Wilk Normality Test
quant_vars <- c("age", "bmi", "lastdose", "IgG", "IgM")
normality_test <- function(data, group_var, variables) {
  map_dfr(variables, ~{
    group_data <- split(data[[.x]], data[[group_var]])
    map_dfr(names(group_data), ~{
      test_result <- shapiro.test(group_data[[.x]])
      tibble(
        Variable = .x,
        Group = .y,
        W = test_result$statistic,
        p_value = test_result$p.value,
        Normality = ifelse(p_value > 0.05, "Normal", "Non-normal")
      )
    }, .y = .x)
  })
}
normality_results <- normality_test(mymeta, "group", quant_vars)

# Descriptive Table
quant_vars <- c("age", "bmi", "Cough", "Fatigue", "Sorethroat", 
                "Decreasedsense", "Nasalobstruction", "Runnynose",
                "conjunctivitis", "Muscle.soreness", "Diarrhea",
                "scoresum", "symptom", "lastdose", "IgG", "IgM")
desc_stats <- mymeta %>%
  group_by(group) %>%
  summarise(across(all_of(quant_vars),
                   list(M = ~median(., na.rm = TRUE),
                        Q1 = ~quantile(., 0.25, na.rm = TRUE),
                        Q3 = ~quantile(., 0.75, na.rm = TRUE)),
                   .names = "{.col}_{.fn}")) %>%
  pivot_longer(-group, names_to = c("Variable", ".value"), names_sep = "_") %>%
  mutate(Value = glue::glue("{M} ({Q1}-{Q3})")) %>%
  select(-M, -Q1, -Q3) %>%
  pivot_wider(names_from = group, values_from = Value)

write.csv(desc_stats, file = "desc_stats.csv", row.names = FALSE)

# Group Comparisons
descrTable(group ~ ., data = mymeta)
descrTable(~ ., data = mymeta)
descrTable(group ~ sex + age + bmi + Boosterinjection + group, data = mymeta, method = c(pathsize = NA))

# ========================= [ Barplot for PPI Results ] =========================
colors <- c("#339999", '#F0E445', "#cc3333", '#0072B2')
description <- factor(c("Immune system process",
                        "Cell migration",
                        "Blood vessel development",
                        "Positive regulation of secretion by cell"),
                      levels = c("Immune system process",
                                 "Cell migration",
                                 "Blood vessel development",
                                 "Positive regulation of secretion by cell"))
num <- c(16, 12, 9, 6)
data <- data.frame(description, num)

p <- ggplot(data) +
  geom_bar(aes(description, num, fill = description), stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = colors) +
  labs(x = "Description", y = "Count", fill = "Description") +
  theme_bw() +
  theme(legend.position = "none",
        axis.title.x = element_text(size = 18, face = "bold"),
        axis.title.y = element_text(size = 18, face = "bold"),
        axis.text.x = element_text(size = 16, face = "bold"),
        axis.text.y = element_text(size = 16, face = "bold"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())

print(p)
ggsave("F2C.pdf", p, width = 20, height = 12, units = "cm")