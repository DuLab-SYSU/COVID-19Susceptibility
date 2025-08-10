# Clear environment and set options
rm(list = ls())
options(stringsAsFactors = FALSE)
gc()

# =========================[ Load Required Packages ]=========================
library(ggplot2)          # For data visualization
library(tinyarray)        # For array data manipulation
library(GSVA)             # For ssGSEA analysis
library(dplyr)            # For data manipulation
library(Hmisc)            # For statistical functions (not used in this script)
library(pheatmap)         # For heatmap generation (not used in this script)
library(ggpubr)           # For publication-quality plots
library(patchwork)        # For combining plots
library(ggsci)            # For scientific color palettes
library(tidyverse)        # For data wrangling
library(corrplot)         # For correlation plots (not used in this script)
library(rio)              # For importing various file formats
library(readxl)           # For reading Excel files
library(writexl)          # For writing Excel files
library(rstatix)          # For statistical testing

# =========================[ Set Working Directory ]=========================
setwd("C:/Users/xinyu/Desktop/code")

# =========================[ Load and Prepare Metadata ]=========================
# Load metadata and format group variable
metadata <- read.csv("mymeta.csv", header = TRUE, row.names = 1)
metadata$group <- factor(metadata$group, levels = c("LS", "HS"))
metadata <- dplyr::select(metadata, "group")

# =========================[ Load Gene Expression Data ]=========================
# Load TPM-normalized gene expression matrix
tpm_matrix <- read.csv("tpm.csv", header = TRUE, row.names = 1)
tpm_matrix <- as.matrix(tpm_matrix)


# =========================[ Load Gene Sets ]=========================
# Load cell marker gene sets
cell_markers <- rio::import("cellMarker.xlsx")
geneset_cell <- split(cell_markers$Metagene, cell_markers$`Cell type`)

# Load pathway gene sets
pathway_data <- read_excel("pathway1.xlsx")
pathway <- pivot_longer(data = pathway_data,
                        cols = 1:ncol(pathway_data),
                        names_to = "pathway",
                        values_to = "Metagene") %>%
  na.omit()
geneset_pathway <- split(pathway$Metagene, pathway$pathway)

# Combine gene sets (use pathway gene sets only in this analysis)
geneset <- geneset_pathway

# =========================[ Perform ssGSEA Analysis ]=========================
# Create ssGSEA parameter object
ssgsea_params <- ssgseaParam(
  exprData = tpm_matrix,
  geneSets = geneset,
  minSize = 1,
  maxSize = Inf,
  alpha = 0.25,
  normalize = TRUE
)

# Run ssGSEA to obtain enrichment score matrix
gsva_scores <- gsva(ssgsea_params)

# =========================[ Boxplot Visualization with Significance ]=========================
# Prepare data for plotting
plot_data <- reshape2::melt(gsva_scores) %>%
  setNames(c("Pathway", "Sample", "EnrichmentScore")) %>%
  mutate(group = metadata[Sample, "group"])

# Perform Mann-Whitney U test for each pathway
stat_test <- plot_data %>%
  group_by(Pathway) %>%
  wilcox_test(EnrichmentScore ~ group, alternative = "two.sided") %>%
  add_xy_position(x = "Pathway", dodge = 0.8)

# Add significance labels
stat_test$label <- case_when(
  stat_test$p < 0.001 ~ "***",
  stat_test$p < 0.01 ~ "**",
  stat_test$p < 0.05 ~ "*",
  TRUE ~ "ns"
)

# Filter significant pathways (p < 0.05)
significant_pathways <- stat_test %>%
  filter(p < 0.05) %>%
  pull(Pathway) %>%
  as.character()

stat.test_sig <- stat.test %>% filter(p < 0.05)

# Prepare data for significant pathways
plot_data_sig <- plot_data %>%
  filter(Pathway %in% significant_pathways) %>%
  mutate(Pathway = factor(Pathway, levels = significant_pathways))

# Recalculate statistics for significant pathways with significance symbols
stat_test_sig <- plot_data_sig %>%
  group_by(Pathway) %>%
  wilcox_test(EnrichmentScore ~ group) %>%
  add_significance("p") %>%
  add_xy_position(x = "Pathway", dodge = 0.8)

# Create boxplot
p<-ggplot(plot_data_sig, aes(x = Pathway, y = EnrichmentScore)) +
  geom_boxplot(aes(fill = group), width = 0.7, outlier.shape = NA) +
  geom_jitter(aes(color = group), 
              position = position_jitterdodge(jitter.width = 0.2),
              alpha = 0.6, size = 2) +
  stat_pvalue_manual(
    stat.test_sig,
    label = "p.signif",  
    tip.length = 0.01,
    size = 5,
    bracket.nudge.y = 0.1
  ) +
  scale_fill_manual(values = c(LS = "#0072B2", HS = "#cc3333")) +
  scale_color_manual(values = c(LS = "#0072B2", HS = "#cc3333")) +
  labs(x = NULL, y = "Enrichment Score") +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
    legend.position = "top"
  )
p
p <- ggplot(plot_data_sig, aes(x = Pathway, y = EnrichmentScore)) +
  geom_boxplot(aes(fill = group), width = 0.7, outlier.shape = NA) +
  geom_jitter(aes(color = group), 
              position = position_jitterdodge(jitter.width = 0.2),
              alpha = 0.6, size = 2) +
  stat_pvalue_manual(
    stat.test_sig,
    label = "p.signif",
    tip.length = 0.01,
    size = 5,
    bracket.nudge.y = 0.1
  ) +
  scale_fill_manual(values = c(LS = "#0072B2", HS = "#cc3333")) +
  scale_color_manual(values = c(LS = "#0072B2", HS = "#cc3333")) +
  labs(x = NULL, y = "Enrichment Score") +
  theme_bw() +
  theme(
     plot.margin = unit(c(2, 2, 2, 2), "cm"),  # 单位为cm或inches
    axis.text.x = element_text(angle = 45, hjust = 1, size = 12, margin = margin(t = 15)),
    axis.title.y = element_text(margin = margin(r = 15)),  
    legend.position = "top"
  )
p

p + scale_x_discrete(expand = expansion(add = 0.5))
ggsave("figure3.pdf",width = 10,height = 8)
gsva_data <- gsva(ssgseaP)
ssgsea1<-gsva_data[c("Central memory CD4 T cell","Activated CD8 T cell"),]
ssgsea2<-gsva_data[c("HALLMARK_COMPLEMENT","Interferons","Interferon Receptor","HALLMARK_COAGULATION","BCR Signaling Pathway"),]
gsva_data<-rbind(ssgsea1,ssgsea2)
save(gsva_data,file = "ssgsea_final.RData")
load("ssgsea_final.RData")
draw_boxplot(gsva_data,mymeta$group,color = c( "#0072B2","#cc3333"))

ggsave("ssgsea_boxplot.pdf",width = 12,height = 8)
ggsave("ssgsea_boxplot.png",width = 12,height = 8)




# =========================[ Prepare Machine Learning Feature Table ]=========================
pathway_file <- "pathway1.xlsx"
pathway_data <- read_excel(pathway_file)
pathway<-pivot_longer(data = pathway_data,
                      cols = 1:ncol(pathway_data),
                      names_to = "pathway",
                      values_to = "Metagene")
pathway<-na.omit(pathway)
pathwaygene <- subset(pathway, pathway %in% c("Interferons","Interferon Receptor","HALLMARK_COMPLEMENT","HALLMARK_COAGULATION","BCR Signaling Pathway"))
pathwaygene <- subset(pathway, pathway %in% c("Interferons","Interferon Receptor","HALLMARK_COMPLEMENT"))
gene<-pathwaygene$Metagene
gene<-unique(gene)
#geneset = rio::import("cellMarker.xlsx")
#pathway <- subset(geneset, Celltype %in% c("Activated CD8 T cell","Central memory CD4 T cell"))
#gene2<-pathway$Metagene
#gene<-c(gene1,gene2)
#btm = read.csv("btmenrich.csv")
#pathway <- subset(btm, term %in% c("T cell activation (I) (M7.1)"))
#gene<-pathway$gene
tpm<-read.csv("tpm.csv",row.names=1,header = T)
count1<-t(tpm[gene,])
count1<-data.frame(count1)

library(dplyr)
count1<- count1[,!apply(is.na(count1), 2, any)]
mymeta<-read.csv("mymeta.csv",stringsAsFactors = T,row.names=1,header = T)
mymeta1<-mymeta["group"]
rt<-cbind(mymeta1,count1)
rt$group<-ifelse(rt$group=="LS",0,1)
write.csv(rt, file = "ssGSEAfeature.csv",row.names = FALSE)
btmgene = read.csv("btmenrich.csv")
geneset= split(btmgene$gene,btmgene$term)

gsva_data<-read.csv("ssgsea_BTM.csv",row.names = 1)
data1<-cbind(mymeta,t(gsva_data))
rownames(data1)<-NULL
data2<-pivot_longer(data = data1,
                    cols = 2:ncol(data1),
                    names_to = "celltype",
                    values_to = "proportion")
table(data2$celltype)
table(data2$group)

p_values <- data2 %>%
  group_by(celltype) %>%
  do({
    data.frame(
      celltype = unique(.$celltype),
      p_value = t.test(proportion ~ group, data = .)$p.value
    )
  })
df_sorted <- p_values [order(p_values$p_value, decreasing = TRUE), ]
df_filtered <- filter(df_sorted, p_value < 0.05)
df_filtered <- filter(df_filtered, !grepl("TBA", celltype))
name<-df_filtered$celltype
m<-gsva_data[name,]
draw_boxplot(m,mymeta$group,color = c( "#ED5462","#0072B2"))
data3 <-subset(data2,celltype%in% c("T cell activation (II) (M7.3)", "signaling in T cells (I) (M35.0)"))
#data3 <-subset(data2,celltype%in%c("Activated.CD8.T.cell","Central.memory.CD4.T.cell"))
#data3 <-subset(data2,celltype%in%c("HALLMARK_COMPLEMENT","Interferons","Interferon.Receptor","HALLMARK_COAGULATION","BCR.Signaling.Pathway"))









