# RNA-seq Differential Expression Analysis Pipeline in R

##  Load Required Packages
rm(list = ls())
gc()

# Custom ggplot2 theme
mytheme <- theme(
  axis.text.x = element_text(hjust = 0.5, size = 20), 
  axis.text.y = element_text(size = 20), 
  axis.title.x = element_text(size = 20), 
  axis.title.y = element_text(size = 20), 
  plot.margin = unit(c(1, 1, 1, 1), "cm"), 
  plot.title = element_text(hjust = 0.5, size = 22),
  legend.title = element_text(size = 22), 
  legend.text = element_text(size = 22), 
  legend.position = "right",
  panel.grid.major = element_blank(), 
  panel.grid.minor = element_blank(), 
  legend.background = element_rect(fill = 'transparent')
)

# Load libraries
library(DESeq2)
library(dplyr)
library(biomaRt)
library(clusterProfiler)
library(enrichplot)
library(tidyverse)
library(ggstatsplot)
library(ggnewscale)
library(org.Hs.eg.db)
library(DOSE)
library(ggplot2)
library(ggrepel)
library(ggprism)
library(gridExtra)
library(cowplot)
library(aPEAR)

# Set working directory
setwd("C:/Users/xinyu/Desktop/code")
# RNA-seq Differential Expression Analysis Pipeline in R

##  Load Required Packages
rm(list = ls())
gc()

# Custom ggplot2 theme
mytheme <- theme(
  axis.text.x = element_text(hjust = 0.5, size = 20), 
  axis.text.y = element_text(size = 20), 
  axis.title.x = element_text(size = 20), 
  axis.title.y = element_text(size = 20), 
  plot.margin = unit(c(1, 1, 1, 1), "cm"), 
  plot.title = element_text(hjust = 0.5, size = 22),
  legend.title = element_text(size = 22), 
  legend.text = element_text(size = 22), 
  legend.position = "right",
  panel.grid.major = element_blank(), 
  panel.grid.minor = element_blank(), 
  legend.background = element_rect(fill = 'transparent')
)

# Load libraries
library(DESeq2)
library(dplyr)
library(biomaRt)
library(clusterProfiler)
library(enrichplot)
library(tidyverse)
library(ggstatsplot)
library(ggnewscale)
library(org.Hs.eg.db)
library(DOSE)
library(ggplot2)
library(ggrepel)
library(ggprism)
library(gridExtra)
library(cowplot)
library(aPEAR)

# Set working directory
setwd("C:/Users/xinyu/Desktop/code")

##  Load Data
count <- read.csv("pro_name.csv", row.names = 1, header = TRUE)
count <- apply(count, 2, as.integer)  # Convert to integer
rownames(count) <- rownames(read.csv("pro_name.csv", row.names = 1))

metadata <- read.csv("mymeta.csv", stringsAsFactors = TRUE, row.names = 1)
metadata <- dplyr::select(metadata, c("group", "sex"))
metadata$group <- factor(metadata$group, levels = c("LS", "HS"))

stopifnot(identical(rownames(metadata), colnames(count)))

##  Differential Expression Analysis
dds <- DESeqDataSetFromMatrix(countData = count, colData = metadata, design = ~ group + sex)
dds <- DESeq(dds)
res <- results(dds, contrast = c("group", "HS", "LS"))
res <- data.frame(res)

write.csv(res, row.names = TRUE, file = "deseq2.csv")

##  Annotate DEG Results
deg_data <- read.csv("deseq2.csv", row.names = 1)
deg_data <- deg_data %>%
  mutate(group = case_when(
    log2FoldChange >= 0.25 & pvalue < 0.05 ~ "UP",
    log2FoldChange <= -0.25 & pvalue < 0.05 ~ "DOWN",
    TRUE ~ "NOT_CHANGE"
  ))

write.csv(deg_data, "deseq2group.csv", row.names = TRUE)

# Filter DEGs
deg_filtered <- subset(deg_data, group != "NOT_CHANGE")
write.csv(deg_filtered, "deg.csv", row.names = TRUE)

deg_genes <- rownames(deg_filtered)
save(deg_genes, file = "deg.Rdata")

##  DEG Machine Learning Table
load("deg.Rdata")
tpm <- read.csv("tpm.csv", row.names = 1)
deg_expr <- t(tpm[deg_genes, ])
group_info <- read.csv("mymeta.csv", row.names = 1)[, "group", drop = FALSE]
group_info$group <- ifelse(group_info$group == "LS", 0, 1)
ml_table <- cbind(group_info, deg_expr)
write.csv(ml_table, "deg571.csv", row.names = FALSE)

##  PPI Top30 ML Table
top30 <- read.csv("top30.csv")
ppi_expr <- t(tpm[top30$name, ])
ppi_table <- cbind(group_info, ppi_expr)
ppi_table$group <- ifelse(ppi_table$group == "LS", 0, 1)
write.csv(ppi_table, "ppi30.csv", row.names = FALSE)

## 🔗 Export Node Attributes for PPI Network
deg_for_node <- read.csv("deg.csv", row.names = 1)
node_data <- data.frame(
  node1 = rownames(deg_for_node),
  log2FC = deg_for_node$log2FoldChange,
  pvalue_log10 = -log10(deg_for_node$pvalue)
)
write.csv(node_data, "node_data.csv", row.names = FALSE, quote = FALSE)

##  Volcano Plot
volcano_df <- read.csv("deseq2group.csv")
volcano <- ggplot(volcano_df, aes(x = log2FoldChange, y = -log10(pvalue), colour = group)) +
  geom_point(alpha = 0.85, size = 1.5) +
  scale_color_manual(values = c("#0072B2", 'gray', "#cc3333")) +
  xlim(c(-1.2, 1.2)) +
  ylim(c(0, 10)) +
  geom_vline(xintercept = c(-0.25, 0.25), lty = 4, col = "black", lwd = 0.8) +
  geom_hline(yintercept = -log10(0.05), lty = 4, col = "black", lwd = 0.8) +
  labs(x = "log2FoldChange", y = "-log10(p-value)") +
  theme_bw() + mytheme + theme_prism(border = TRUE)

ggsave("F1A.pdf", plot = volcano, width = 6, height = 4, dpi = 300)

##  GO Enrichment Analysis
gene_up <- rownames(deg_data[deg_data$log2FoldChange >= 0.25 & deg_data$pvalue <= 0.05, ])
gene_down <- rownames(deg_data[deg_data$log2FoldChange <= -0.25 & deg_data$pvalue <= 0.05, ])
gene_diff <- unique(c(gene_up, gene_down))

entrez_diff <- bitr(gene_diff, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)[, 2]
entrez_diff <- as.character(na.omit(entrez_diff))

go_enrich <- enrichGO(gene = entrez_diff, OrgDb = org.Hs.eg.db, ont = "BP", 
                      pvalueCutoff = 0.05, qvalueCutoff = 0.05, readable = TRUE)
go_result <- go_enrich@result
go_top <- go_result[1:15, ]

## GO Network Plot
go_net <- enrichmentNetwork(go_top, fontSize = 8, drawEllipses = TRUE,
                            colorBy = "pvalue", colorType = "pval") +
  scale_color_gradientn(colours = c("#cc3333", "white", "#0072B2"), name = "p-value")
ggsave("F1C.pdf", plot = go_net, width = 12, height = 8, dpi = 300)

## GO Bubble Plot
go_top$GeneRatio <- sapply(strsplit(as.character(go_top$GeneRatio), "/"), function(x) as.numeric(x[1]) / as.numeric(x[2]))
go_top <- go_top[order(go_top$GeneRatio), ]
go_top$Description <- factor(go_top$Description, levels = go_top$Description)

go_bubble <- ggplot(go_top, aes(x = Description, y = GeneRatio, fill = -log10(pvalue))) +
  geom_point(aes(size = Count), shape = 21, color = "black") +
  labs(x = NULL, y = "GeneRatio", fill = bquote("-"~Log[10]~"(P value)")) +
  scale_fill_gradientn(colours = c("#0072B2", "#cc3333")) +
  scale_size_continuous(range = c(4, 10)) +
  coord_flip() + theme_bw() + mytheme +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 30))

ggsave("F1B.pdf", plot = go_bubble, width = 11, height = 8, dpi = 300)

##  BTM Enrichment (custom term2gene)
btm <- read.csv("btmenrich.csv", row.names = 1)
btm_enrich <- enricher(gene_diff, TERM2GENE = btm, pvalueCutoff = 0.05, qvalueCutoff = 0.2)
btm_top <- btm_enrich@result[1:15, ]

btm_net <- enrichmentNetwork(btm_top, fontSize = 8, drawEllipses = TRUE,
                             colorBy = "pvalue", colorType = "pval") +
  scale_color_gradientn(colours = c("#cc3333", "white", "#0072B2"), name = "p-value")
ggsave("F1C.pdf", plot = btm_net, width = 11, height = 8, dpi = 300)

## BTM Bubble Plot
btm_top$GeneRatio <- sapply(strsplit(as.character(btm_top$GeneRatio), "/"), function(x) as.numeric(x[1]) / as.numeric(x[2]))
btm_top <- btm_top[order(btm_top$GeneRatio), ]
btm_top$Description <- factor(btm_top$Description, levels = btm_top$Description)

btm_bubble <- ggplot(btm_top, aes(x = Description, y = GeneRatio, fill = -log10(pvalue))) +
  geom_point(aes(size = Count), shape = 21, color = "black") +
  labs(x = NULL, y = "GeneRatio", fill = bquote("-"~Log[10]~"(P value)")) +
  scale_fill_gradientn(colours = c("#0072B2", "#cc3333")) +
  scale_size_continuous(range = c(4, 10)) +
  coord_flip() + theme_bw() + mytheme +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 30))

ggsave("F1C_bubble.pdf", plot = btm_bubble, width = 11, height = 8, dpi = 300)

## Combine Plots
combined_plot <- plot_grid(go_bubble, btm_bubble, nrow = 1)
ggsave("F1BC_combined.pdf", plot = combined_plot, width = 22, height = 8, dpi = 300)

##  KEGG Enrichment
kegg_enrich <- enrichKEGG(gene = entrez_diff, organism = "hsa", keyType = "kegg",
                          pvalueCutoff = 0.05, qvalueCutoff = 0.05)
kegg_enrich <- setReadable(kegg_enrich, OrgDb = org.Hs.eg.db, keyType = "ENTREZID")
kegg_top <- kegg_enrich@result[1:10, ]

kegg_top$GeneRatio <- sapply(strsplit(as.character(kegg_top$GeneRatio), "/"), function(x) as.numeric(x[1]) / as.numeric(x[2]))
kegg_top <- kegg_top[order(kegg_top$GeneRatio), ]
kegg_top$Description <- factor(kegg_top$Description, levels = kegg_top$Description)

kegg_bubble <- ggplot(kegg_top, aes(x = Description, y = GeneRatio, fill = -log10(pvalue))) +
  geom_point(aes(size = Count), shape = 21, color = "black") +
  labs(x = NULL, y = "GeneRatio", title = "KEGG Pathway Enrichment", 
       fill = bquote("-"~Log[10]~"(P value)")) +
  scale_fill_gradientn(colours = c("#0072B2", "#cc3333")) +
  scale_size_continuous(range = c(4, 10)) +
  coord_flip() + theme_bw() + mytheme +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 30))

ggsave("F1D_KEGG.pdf", plot = kegg_bubble, width = 11, height = 8, dpi = 300)













