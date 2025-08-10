rm(list = ls())
gc()

# Load required libraries
library(ReactomePA) 
library(tidyverse)
library(data.table)
library(org.Hs.eg.db)
library(clusterProfiler)
library(biomaRt)
library(enrichplot)
library(cols4all)
library(DOSE)
library(msigdr)
library(GseaVis)
library(gridExtra)

setwd("C:/Users/xinyu/Desktop/code")  # Modify as needed

# ---- Prepare custom gene sets (BTM) ----
btm_data <- readRDS("btm_list_2020_12_23.rds")
btm_df <- data.frame(matrix(ncol = 2, nrow = sum(lengths(btm_data))))
row_index <- 1
for (term in names(btm_data)) {
  for (gene in btm_data[[term]]) {
    btm_df[row_index, ] <- c(term, gene)
    row_index <- row_index + 1
  }
}
colnames(btm_df) <- c("term", "gene")
write.csv(btm_df, "btmenrich.csv", row.names = FALSE)

# ---- Prepare gene list for GSEA ----
deg <- read.csv("deseq2.csv", header = TRUE)
gene_list <- deg$log2FoldChange
names(gene_list) <- deg[, 1]
gene_list <- sort(gene_list[gene_list != 0], decreasing = TRUE)

# Convert SYMBOL to ENTREZID
gene_symbols <- as.character(deg[, 1])
gene_map <- select(org.Hs.eg.db, keys = gene_symbols, keytype = "SYMBOL", columns = c("ENTREZID"))
gene_map <- gene_map[!duplicated(gene_map$SYMBOL), ]
colnames(gene_map)[1] <- "Gene"
names(deg)[1] <- "Gene"

# Join expression and ID mapping
merged <- inner_join(gene_map, deg, by = "Gene") %>%
  select(ENTREZID, log2FoldChange) %>%
  na.omit() %>%
  arrange(desc(log2FoldChange))
geneList <- merged$log2FoldChange
names(geneList) <- as.character(merged$ENTREZID)

# ---- GSEA using BTM gene sets ----
btm_terms <- read.csv("btmenrich.csv")
set.seed(1234)
btm_gsea <- GSEA(
  geneList,
  TERM2GENE = btm_terms,
  minGSSize = 10,
  maxGSSize = 10000,
  pvalueCutoff = 1,
  pAdjustMethod = 'BH'
)

btm_results <- as.data.frame(btm_gsea)
btm_results <- btm_results[btm_results$p.adjust < 0.05, ]
write.csv(btm_results, "deseq2btm.csv", row.names = FALSE)

# ---- Create lollipop plot of top pathways ----
btm_plot_data <- read.csv("deseq2btm.csv")
btm_plot_data$log10Padj <- ifelse(
  btm_plot_data$NES > 0,
  -log10(btm_plot_data$p.adjust),
  log10(btm_plot_data$p.adjust)
)
btm_plot_data$group <- ifelse(btm_plot_data$NES > 0, "UP", "DOWN")
setnames(btm_plot_data, "setSize", "count")

# Sort and subset top terms
btm_plot_data$Description <- factor(
  btm_plot_data$Description,
  levels = rev(btm_plot_data$Description)
)
btm_plot_data <- btm_plot_data %>%
  arrange(desc(abs(log10Padj))) %>%
  group_by(group) %>%
  slice_head(n = 10) %>%
  ungroup()

# Plot theme
mytheme <- theme(
  axis.text.x = element_text(hjust = 0.5, size = 20),
  axis.ticks.y = element_blank(),
  axis.text.y = element_text(size = 20),
  axis.title.x = element_text(size = 20),
  axis.title.y = element_text(size = 20),
  axis.line = element_line(size = 1),
  plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
  plot.margin = unit(c(1, 1, 1, 1), "cm"),
  legend.title = element_text(size = 22),
  legend.text = element_text(size = 20),
  legend.position = "right",
  legend.background = element_rect(fill = 'transparent'),
  axis.line.y = element_blank(),
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),
  panel.border = element_blank()
)

# Plot lollipop chart
p <- ggplot(btm_plot_data, aes(x = Description, y = log10Padj)) +
  coord_flip() +
  geom_segment(aes(xend = Description, yend = log10Padj, color = group), size = 2) +
  geom_point(aes(size = count, color = group)) +
  scale_size_continuous(range = c(4, 10)) +
  scale_color_manual(values = c('#0072B2', "#cc3333")) +
  labs(
    x = NULL, 
    y = bquote("-" ~ Log[10] ~ "(P value)"),
    title = "BTM Pathway Enrichment",
    color = "Group"
  ) +
  scale_y_continuous(expand = expansion(add = c(0.1, 0.1))) +
  theme_bw() + mytheme +
  scale_x_discrete(labels = NULL)

ggsave("gsea_lollipop.pdf", p, width = 33, height = 16, units = "cm")

# ---- Single-term GSEA plot (example) ----
p1 <- gseaNb(
  object = btm_gsea,
  geneSetID = "enriched in T cells (I) (M7.0)",
  addPval = TRUE,
  pvalX = 0.55, pvalY = 0.8,
  pCol = 'black', pvalSize = 5,
  pHjust = 0, subPlot = 2,
  curveCol = c('#0072B2', "#cc3333"),
  htCol = c('#0072B2', "#cc3333"),
  rankCol = c('#0072B2', "white", "#cc3333")
)

# ---- Plot multiple terms ----
terms <- c(
  "enriched in neutrophils (I) (M37.1)",
  "enriched in T cells (I) (M7.0)",
  "enriched in monocytes (II) (M11.0)"
)

plots <- lapply(terms, function(term) {
  gseaNb(
    object = btm_gsea,
    geneSetID = term,
    addPval = TRUE,
    pvalX = 0.55, pvalY = 0.8,
    pCol = 'black', pvalSize = 5,
    pHjust = 0, subPlot = 2,
    curveCol = c('#0072B2', "#cc3333"),
    htCol = c('#0072B2', "#cc3333"),
    rankCol = c('#0072B2', "white", "#cc3333")
  )
})

# Combine plots vertically and horizontally
p_vert <- cowplot::plot_grid(plotlist = plots, nrow = 3)
ggsave("btm_gsea_vertical.pdf", p_vert, width = 15, height = 40, units = "cm")

p_horiz <- cowplot::plot_grid(plotlist = plots, ncol = 3)
ggsave("btm_gsea_horizontal.pdf", p_horiz, width = 45, height = 12, units = "cm")
