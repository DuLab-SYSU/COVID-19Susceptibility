# Clear environment and set options
rm(list = ls())
options(stringsAsFactors = FALSE)
gc()
# =========================[ Load Required Packages ]=========================
library(WGCNA)            # For weighted gene co-expression network analysis
library(tidyverse)        # For data wrangling
library(dplyr)            # For data manipulation
library(clusterProfiler)   # For enrichment analysis
library(org.Hs.eg.db)     # For human gene annotations
library(ggplot2)          # For data visualization
library(ggrepel)          # For adding labels to scatter plots
# =========================[ Set Working Directory ]=========================
setwd("C:/Users/xinyu/Desktop/code")
# =========================[ Load Gene Expression Data ]=========================
count<-read.csv("tpm.csv",row.names=1,header = T)
count<-as.matrix(count)
WGCNA_matrix = t(count[order(apply(count,1,mad), decreasing = T)[1:3000],])
datExpr <- WGCNA_matrix
sampleNames = rownames(datExpr)

# =========================[ Load and Prepare Metadata ]=========================
# Load metadata and format group variable
mymeta<-read.csv("mymeta.csv",row.names=1,header = T)
m<-factor(mymeta$group,levels = c("LS","HS"))
df<-binarizeCategoricalVariable(m,includePairwise = F,includeLevelVsAll = T)
df<-data.frame(df)
rownames(df)<-rownames(mymeta)
colnames(df)[colnames(df)%in%c("LS.vs.all","HS.vs.all")]<-c("LS","HS")

# Sample clustering to detect outliers
pdf("sample_clustering.pdf", width = 12, height = 9)
sampleTree = hclust(dist(datExpr), method = "average")
sizeGrWindow(12,9)
par(cex = 0.6)
par(mar = c(0,6,2,0))
plot(sampleTree, main = "Sample clustering to detect outliers", sub="", 
     xlab="", cex.lab = 1.5, cex.axis = 1.5, cex.main = 2)
dev.off()
nGenes   = ncol(datExpr)
nSamples = nrow(datExpr)
powers = c(c(1:10), seq(from = 12, to=20, by=2))
sft = pickSoftThreshold(datExpr, powerVector = powers, verbose = 5)
par(mfrow = c(1,2));
cex1 = 0.9;
pdf(file = "SoftThreshold.pdf",width = 8, height = 4)
par(mfrow = c(1,2))
plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     xlab="Soft Threshold (power)",ylab="Scale Free Topology Model Fit,signed R^2"
     ,type="n",main = paste("Scale independence"));
text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     labels=powers,cex=cex1,col="red");
abline(h=0.85,col="red")
plot(sft$fitIndices[,1], sft$fitIndices[,5],
     xlab="Soft Threshold (power)",ylab="Mean Connectivity", type="n",
     main = paste("Mean connectivity"))
text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers, cex=cex1,col="red")
dev.off()
power = sft$powerEstimate
power #8

# Network construction
cor <- WGCNA::cor
net = blockwiseModules(
  datExpr,
  power = power,
  maxBlockSize = 6000,
  TOMType = "unsigned", minModuleSize = 50,
  reassignThreshold = 0, mergeCutHeight = 0.25,
  numericLabels = TRUE, pamRespectsDendro = FALSE,
  saveTOMs = TRUE,
  saveTOMFileBase = "TOM",
  verbose = 3)
table(net$colors)
cor<-stats::cor
pdf(file = "moduleCluster.pdf", width = 5, height = 4)  
mergedColors = labels2colors(net$colors)
plotDendroAndColors(dendro = net$dendrograms[[1]], 
                    colors = mergedColors[net$blockGenes[[1]]],
                    groupLabels = "Module colors",
                    dendroLabels = FALSE, 
                    hang = 0.03,
                    addGuide = TRUE, 
                    guideHang = 0.05,
                    main = "Gene dendrogram and module colors")
dev.off()
MEs = moduleEigengenes(datExpr, mergedColors)$eigengenes
MEs = orderMEs(MEs)
moduleColors=mergedColors
# Save network data
save(MEs,nGenes,sft,net, nSamples, mergedColors,sampleNames, datExpr,moduleColors,file = "Module.RData")
load("Module.RData")
# Module-trait correlation
datTraits <- df
moduleTraitCor=cor(MEs, datTraits, use="p")
moduleTraitCor<-moduleTraitCor[order(moduleTraitCor[,2],decreasing=T),]
moduleTraitPvalue=corPvalueStudent(moduleTraitCor, nSamples)
pdf(file="Module_trait_relationships.pdf", width = 8, height = 6)
textMatrix=paste(signif(moduleTraitCor,2),"\n(",signif(moduleTraitPvalue,1),")",sep="")
par(mar=c(2,15,1,10))
dim(textMatrix)=dim(moduleTraitCor)
custom_colors <- c("#0072B2",'white', "#cc3333")
color_func <- colorRampPalette(custom_colors)
gradient_colors <- color_func(100)
labeledHeatmap(Matrix = moduleTraitCor,
               xLabels = colnames(datTraits),
               yLabels = rownames(moduleTraitCor),
               xLabelsAngle = 0,
               ySymbols = rownames(moduleTraitCor),
               colorLabels = FALSE,
               colors = gradient_colors,  
               textMatrix = textMatrix,
               setStdMargins = FALSE,
               cex.text = 1.2,
               zlim = c(-0.2, 0.2))
              # ,main = "module-trait relationships")
dev.off()

# Gene significance and module membership
SY_stage = as.data.frame(datTraits$HS)
names(SY_stage) = "symptomatic"
modNames = substring(names(MEs), 3)
geneModuleMembership = as.data.frame(cor(datExpr, MEs, use = "p"));
MMPvalue = as.data.frame(corPvalueStudent(as.matrix(geneModuleMembership), nSamples));
names(geneModuleMembership) = paste("SY", modNames, sep="");
names(MMPvalue) = paste("p.SY", modNames, sep="");
geneTraitSignificance = as.data.frame(cor(datExpr, SY_stage, use = "p"));
GSPvalue = as.data.frame(corPvalueStudent(as.matrix(geneTraitSignificance), nSamples));
names(geneTraitSignificance) = paste("GS.", names(SY_stage), sep="")
names(GSPvalue) = paste("p.GS.", names(SY_stage), sep="")
module = "turquoise"#1315/50hub
column = match(module, modNames)
moduleGenes = moduleColors==module
color_module<-as.data.frame(dimnames(data.frame(datExpr))[[2]][moduleGenes]) 
names(color_module)="genename"
turquoise<-color_module$genename
save(turquoise, file = "turquoise.RData")
write.table(turquoise,"turquoise.txt",quote = F,row.names = F,col.names = F)
# Hub gene selection
MM<-abs(geneModuleMembership[moduleGenes,column])
GS<-abs(geneTraitSignificance[moduleGenes, 1])
c<-as.data.frame(cbind(MM,GS))
c %>% mutate(group = case_when(MM >= 0.8 & GS >= 0.2 ~ TRUE,TRUE ~ FALSE)) -> c
rownames(c)=color_module$genename
table(c$group)
hub<-rownames(c[c$group==TRUE,])
save(hub, file = "turquoisehub.RData")
hub
# Machine learning table
load("turquoisehub.RData")
gene<-hub
tpm<-read.csv("tpm.csv",row.names=1,header = T)
count1<-t(tpm[gene,])
mymeta<-read.csv("mymeta.csv",stringsAsFactors = T,row.names=1,header = T)
mymeta1<-mymeta["group"]
rt<-cbind(mymeta1,count1)
rt$group<-ifelse(rt$group=="LS",0,1)
write.csv(rt, file = "turquoisehub.csv",row.names = FALSE)

# Scatter plot with ggplot2
c$genename <- rownames(c)
hub_data <- subset(c, group == TRUE)
target_genes <- hub_data %>% 
  filter(grepl("FURIN|IFNGR1|KLF7|PABPC4", genename, ignore.case = TRUE))


library(ggplot2)
library(ggrepel)

p1<-ggplot(data=c, aes(x=MM, y=GS, color=group))+
  geom_point(size=1.5)+scale_colour_manual(values=c("grey60", module))+ 
  theme_bw()+  theme(panel.grid.major = element_blank(),
                     panel.grid.minor = element_blank())+  
  labs(x="Module Membership in turquoise module", y="Gene significance for HS",
       title = "Module membership vs. gene significance \n cor=0.49,p=2.3e-80 ")+
  theme(axis.title.x =element_text(size=14), 
        axis.title.y=element_text(size=14),axis.text = element_text(size = 12),
        axis.text.x = element_text(colour = "black"),axis.text.y = element_text(colour = "black"),
        plot.title = element_text(hjust = 0.5,size = 16,face = "bold"),plot.margin = unit(rep(2,4),'lines')) +
  theme(legend.position = 'none')+geom_hline(aes(yintercept=0.2),colour="#5B9BD5",lwd=1,linetype=5)+
  geom_vline(aes(xintercept=0.8),colour="#5B9BD5",lwd=1,linetype=5)
p1
if(nrow(target_genes) > 0) {
  p1 <- p1 + 
    geom_point(
      data = target_genes,
      aes(x = MM, y = GS),
      color = "red",
      size = 3
    ) +
    geom_text_repel(
      data = target_genes,
      aes(label = genename),
      color = "red",
      size = 4,
      box.padding = 0.8,
      segment.color = "grey50",
      max.overlaps = 50,  
      min.segment.length = 0.2  
    )
}
p1
ggsave("turquoise.pdf",plot=p1,width=7,height=7)
ggsave("turquoise.png",plot=p1,width=7,height=7)

png(file="Module_membership_vs_gene_significance_2.png")
par(mfrow = c(1,1));
verboseScatterplot(abs(geneModuleMembership[moduleGenes, column]),
                   abs(geneTraitSignificance[moduleGenes, 1]),
                   xlab = paste("Module Membership in", module, "module"),
                   ylab = "Gene significance for symptomatic",
                   main = paste("Module membership vs. gene significance\n"),
                   cex.main = 1.2, cex.lab = 1.2, cex.axis = 1.2, col = module)
dev.off()


module = "blue"#335/18hub
column = match(module, modNames)
moduleGenes = moduleColors==module
color_module<-as.data.frame(dimnames(data.frame(datExpr))[[2]][moduleGenes]) 
names(color_module)="genename"
blue<-color_module$genename
save(blue, file = "blue.RData")
write.table(blue,"blue.txt",quote = F,row.names = F,col.names = F)
 
MM<-abs(geneModuleMembership[moduleGenes,column])
GS<-abs(geneTraitSignificance[moduleGenes, 1])
c<-as.data.frame(cbind(MM,GS))
c %>% mutate(group = case_when(MM >= 0.8 & GS >= 0.2 ~ TRUE,TRUE ~ FALSE)) -> c
rownames(c)=color_module$genename
table(c$group)
hub<-rownames(c[c$group==TRUE,])
save(hub, file = "bluehub.RData")
hub
load("bluehub.RData")
gene<-hub
tpm<-read.csv("tpm.csv",row.names=1,header = T)
count1<-t(tpm[gene,])
mymeta<-read.csv("mymeta.csv",stringsAsFactors = T,row.names=1,header = T)
mymeta1<-mymeta["group"]
rt<-cbind(mymeta1,count1)
rt$group<-ifelse(rt$group=="LS",0,1)
write.csv(rt, file = "bluehub.csv",row.names = FALSE)


library(ggplot2)
c$genename <- rownames(c)
hub_data <- subset(c, group == TRUE)
target_genes <- hub_data %>% 
  filter(grepl("PSAP|CD93|BST1", genename, ignore.case = TRUE))

library(ggplot2)
library(ggrepel)
p1<-ggplot(data=c, aes(x=MM, y=GS, color=group))+
  geom_point(size=1.5)+scale_colour_manual(values=c("grey60", module))+ 
  theme_bw()+  theme(panel.grid.major = element_blank(),
                     panel.grid.minor = element_blank())+  
  labs(x="Module Membership in blue module", y="Gene significance for HS",
       title = "Module membership vs. gene significance \n cor=0.40,p=2.7e-14 ")+
  theme(axis.title.x =element_text(size=14), 
        axis.title.y=element_text(size=14),axis.text = element_text(size = 12),
        axis.text.x = element_text(colour = "black"),axis.text.y = element_text(colour = "black"),
        plot.title = element_text(hjust = 0.5,size = 16,face = "bold"),plot.margin = unit(rep(2,4),'lines')) +
  theme(legend.position = 'none')+geom_hline(aes(yintercept=0.2),colour="#5B9BD5",lwd=1,linetype=5)+
  geom_vline(aes(xintercept=0.8),colour="#5B9BD5",lwd=1,linetype=5)
p1
library(ggplot2)
library(ggrepel)

if(nrow(target_genes) > 0) {
  p1 <- p1 + 
    geom_point(
      data = target_genes,
      aes(x = MM, y = GS),
      color = "red",
      size = 3
    ) +
    geom_text_repel(
      data = target_genes,
      aes(label = genename),
      color = "red",
      size = 4,
      box.padding = 0.8,
      segment.color = "grey50",
      max.overlaps = 50, 
      min.segment.length = 0.2  
    )
}
p1
dev.off()
ggsave("blue.pdf",plot=p1,width=7,height=7)
ggsave("blue.png",plot=p1,width=7,height=7)


png(file="Module_membership_vs_gene_significance_2.png")
par(mfrow = c(1,1));
verboseScatterplot(abs(geneModuleMembership[moduleGenes, column]),
                   abs(geneTraitSignificance[moduleGenes, 1]),
                   xlab = paste("Module Membership in", module, "module"),
                   ylab = "Gene significance for symptomatic",
                   main = paste("Module membership vs. gene significance\n"),
                   cex.main = 1.2, cex.lab = 1.2, cex.axis = 1.2, col = module)
dev.off()

 
library(clusterProfiler)
library(org.Hs.eg.db)
library(tidyverse)
load("turquoise.RData")
load("blue.RData")
genelist <- bitr(turquoise, fromType="SYMBOL",
                 toType="ENTREZID", OrgDb='org.Hs.eg.db')

enrichGO <- enrichGO(gene = genelist$ENTREZID,
                 OrgDb = org.Hs.eg.db, 
                 ont = "all",
                 pAdjustMethod = "BH",
                 minGSSize = 1,
                 pvalueCutoff =0.1, 
                 qvalueCutoff =0.2,
                 readable = TRUE)
enrichKEGG <- enrichKEGG(gene  = genelist$ENTREZID,
           organism  = "hsa", 
           pvalueCutoff = 0.05,
           qvalueCutoff = 0.2)
enrichgo<-data.frame(enrichGO@result)
enrichkegg<-data.frame(enrichKEGG@result)
# write.csv(enrichGO@result, 'GO_gene_up_BP_enrichresults.csv') 
# write.csv(enrichKEGG@result, 'KEGG_gene_up_BP_enrichresults.csv') 

#GO
barplot(enrichGO, drop = TRUE, showCategory =5,split="ONTOLOGY") + 
  facet_grid(ONTOLOGY~., scale='free')
#KEGG
barplot(enrichKEGG, font.size=14, showCategory=5)+
  theme(plot.margin=unit(c(1,1,1,1),'lines'))  +
  ggtitle("UP-regulated genes symptomatic vs uninfected")


library(clusterProfiler)
btm<-read.csv("btmenrich.csv",row.names = 1)
load("green.RData")
load("pink.RData")
BTM<-enricher(pinkhub, pvalueCutoff = 0.05, pAdjustMethod = "BH",  minGSSize = 10, maxGSSize = 500, qvalueCutoff = 0.2, TERM2GENE=btm)
BTM_RESULT<-BTM@result

btmenrich<-read.csv("btmenrich.csv",row.names = 1)
table(moduleColors)

group_g=data.frame(gene=names(net$colors),
                   group=moduleColors)
head(group_g)
table(group_g$group)
write.csv(group_g,"group_g.csv")
#save(group_g,file='wgcna_group_g.Rdata') 
group_g<-read.csv("group_g.csv",row.names=1,header = T)
btmenrich<-read.csv("btmenrich.csv",row.names = 1)
library(clusterProfiler)
# Convert gene ID into entrez genes
group_g <-subset(group_g,group%in%c("blue","turquoise"))
tmp <- bitr(group_g$gene, fromType="SYMBOL", 
            toType="ENTREZID", 
            OrgDb="org.Hs.eg.db")
de_gene_clusters=merge(tmp,group_g,by.x='SYMBOL',by.y='gene')
table(de_gene_clusters$group)
head(de_gene_clusters)
gcSample <- split(de_gene_clusters$ENTREZID, 
                  de_gene_clusters$group) 
gcSample1<- split(de_gene_clusters$SYMBOL, 
                 de_gene_clusters$group) 

Allkegg <- compareCluster(gcSample, fun="enrichKEGG",
                     organism="hsa", pvalueCutoff=0.05)
Allgo <- compareCluster(gcSample, fun="enrichGO",
                     OrgDb="org.Hs.eg.db", pvalueCutoff=0.05)
Allbtm <- compareCluster(gcSample1, fun="enricher",
                     TERM2GENE=btmenrich)
FF<-data.frame(Allgo@compareClusterResult)
table(FF@compareClusterResult$Cluster)
dotplot(Allkegg)  
dotplot(Allgo)
dotplot(Allbtm)
 
as_go  <-data.frame(Allgo@compareClusterResult)
B<-as_go%>%group_by(Cluster)%>%arrange(Cluster,p.adjust)%>%slice(1:5)%>%ungroup()
B$Cluster<- factor(B$Cluster,levels = c("blue","turquoise"))
 
ratios <- B$GeneRatio
numeric_ratios <- rep(NA, length(ratios))
B$Description <- fct_inorder(B$Description)
for (i in 1:length(ratios)) {
  parts <- strsplit(ratios[i], "/")[[1]]
  if (length(parts) == 2) {
    numerator <- as.numeric(parts[1])
    denominator <- as.numeric(parts[2])
    numeric_ratios[i] <- numerator / denominator
  } else {
    numeric_ratios[i] <- NA
  }
}
B$GeneRatio <- numeric_ratios
 

p1<-ggplot(B, aes(Cluster, Description)) +
  geom_point(aes(color=Cluster, size=GeneRatio))+theme_bw()+
  theme(panel.grid = element_blank(),
        axis.text.x=element_text(angle=90,hjust = 1,vjust=0.5,size=12,face="bold"),
        axis.text.y=element_text(size=12,face="bold"),
        plot.margin =unit(c(1,6,1,6),"cm"))+
  scale_color_manual(values=c( "blue","turquoise"))+
  labs(x=NULL,y=NULL)+guides(size=guide_legend(order=1))
#+ggtitle("GO")
 
ggsave("WGCNA_GO.pdf", plot = p1, width = 9.6, height = 4, units = "in")

 
as_btm  <-data.frame(Allbtm@compareClusterResult)
A<-as_btm%>%group_by(Cluster)%>%arrange(Cluster,p.adjust)%>%slice(1:5)%>%ungroup()
A$Cluster<- factor(A$Cluster,levels = c("blue","turquoise"))
 
ratios <- A$GeneRatio
numeric_ratios <- rep(NA, length(ratios))
A$Description <- fct_inorder(A$Description)
for (i in 1:length(ratios)) {
  parts <- strsplit(ratios[i], "/")[[1]]
  if (length(parts) == 2) {
    numerator <- as.numeric(parts[1])
    denominator <- as.numeric(parts[2])
    numeric_ratios[i] <- numerator / denominator
  } else {
    numeric_ratios[i] <- NA
  }
}
A$GeneRatio <- numeric_ratios
 
p2<-ggplot(A, aes(Cluster, Description)) +
  geom_point(aes(color=Cluster, size=GeneRatio))+theme_bw()+
  theme(panel.grid = element_blank(),
        axis.text.x=element_text(angle=90,hjust = 1,vjust=0.5,size=12,face="bold"),
        axis.text.y=element_text(size=12,face="bold"),
        plot.margin =unit(c(1,3,1,6),"cm"))+
  scale_color_manual(values=c( "blue","turquoise"))+
  labs(x=NULL,y=NULL)+guides(size=guide_legend(order=1))
  #+ggtitle("BTM")
p2
 
ggsave("WGCNA_BTM.pdf", plot = p2, width = 9.5, height = 4, units = "in")

colors<-c("#339999",'#0072B2','#F0E445',"#cc3333")
description<- c("Platelet activation, signaling and aggregation",
                "Nucleosome",
                "Regulation of cytokine production",
                "Immune system process")
description <- factor(description, levels = c("Platelet activation, signaling and aggregation",
                                              "Nucleosome",
                                              "Regulation of cytokine production",
                                              "Immune system process"))
num<-c(15,17,39,53)
data<-data.frame(description,num)
ggplot(data) +
  geom_bar(aes(description,num,fill=description),stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = colors) +
  labs(x="Description",y="Count",fill="Description") +
  theme_bw() +
  theme(legend.position = "none",
        axis.title.x = element_text(size = 18, face = "bold"), 
        axis.title.y = element_text(size = 18, face = "bold") ,  
        axis.text.x = element_text(size = 16, face = "bold"),
        axis.text.y = element_text(size = 16, face = "bold")
        )
load("bluehub.Rdata")
