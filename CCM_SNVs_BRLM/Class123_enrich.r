### conda activate R4.2
library(enrichplot)
library(tidyverse)
library(edgeR)
library(dplyr)
library(clusterProfiler)
library(org.Hs.eg.db)
library(RColorBrewer)
library(scales)

##################### 1_1 KEGG dotplot ###########################
anno_filter <- read.csv("res_CCM_Classify/anno_filter.csv",header=T,row.names=1)
anno_filter_Class123 <- subset(anno_filter,Class %in% c("Class1","Class2","Class3"))
genelist <- names(table(anno_filter_Class123$Gene.refGene))

setwd("7.Class123_enrich")
gene.df<-bitr(genelist, fromType = "SYMBOL",  toType = c("ENTREZID"),OrgDb = org.Hs.eg.db)
colnames(gene.df)[1] <- "gene"
ekegg <- enrichKEGG(
  gene  = gene.df$ENTREZID,
  keyType  = 'kegg',
  organism = 'human',
  pvalueCutoff = 1,
  pAdjustMethod     = "BH",
  qvalueCutoff  = 1
)
egokegg=DOSE::setReadable(ekegg, OrgDb='org.Hs.eg.db',keyType='ENTREZID')
#write.csv(egokegg,file="kegg1.csv")
#egokegg2=read.csv("kegg1.csv",header=T,row.names = 1)
#pfs <- read.csv("pfs_treeplot.csv",header=T,row.names=1)
#num=which(egokegg2$Description %in% c(pfs$Row.names[2:11]))
#egokegg@result<-egokegg[num,]
#write.csv(egokegg,file="kegg.csv")
#set.seed(111)
#x2 <- pairwise_termsim(egokegg)
#pdf("emapplot_cluster230718.pdf",height=10,width=10)
#        emapplot_cluster(x2, color = "pvalue",showCategory = 100,cex_line=.3,nCluster=5,cex_label_group = 2.5,
#			alpha_hilight=0.01,alpha_no_hilight=0.01)+
#                scale_fill_continuous(low = "#e06663", high = "#327eba",
#                        name = "pvalue",
#                        guide = guide_colorbar(reverse = TRUE, order=1),
#                        trans='log10')
# dev.off()

####### 0731 selected cnet
num=which(egokegg$ID %in% c("hsa04010","hsa04014","hsa04270","hsa04015", "hsa04727"))
egokegg@result<-egokegg[num,]
pdf(paste("kegg_upsetplot_F50801.pdf", sep=""))
upsetplot(egokegg)
dev.off()
terms <- egokegg$Description
p1 <- pmcplot(terms, 2010:2023)+theme_test()+
	theme(legend.position = c(0.35, 0.85),
	legend.background = element_blank())
#pdf(paste("kegg_pmcplot_F50801.pdf", sep=""))
#print(p1)
#dev.off()
terms <- c("FGF1","FGF6")
p2 <- pmcplot(terms, 2010:2023)+theme_test()+
	theme(legend.position = c(0.1, 0.9),
	legend.background = element_blank())
#pdf(paste("genes2_pmcplot_F50801.pdf", sep=""))
#print(p2)
#dev.off()
terms <- c("rs17217240","rs17223975","rs60436519","rs61324948","rs73795020","rs78769659","rs2241286")
p3 <- pmcplot(terms, 2010:2023)+theme_test()+
	theme(legend.position = c(0.8, 0.8),
        legend.background = element_blank())
#pdf(paste("SNV6_pmcplot_F50801.pdf", sep=""))
#print(p3)
#dev.off()
library(cowplot)
pdf(paste("pmcplot_inter_F50802.pdf", sep=""),width=15,height=7)
plot_grid(p3, p2, p1, labels = LETTERS,ncol = 3, scale = 0.9, hjust = -1.5)
dev.off()

#pdf(paste("kegg_cnet.pdf", sep=""),height=10,width=15)
#cnetplot(egokegg,showCategory = 5,colorEdge = TRUE,ann=True)
#dev.off()
#pdf(paste("enrichment_kegg_dotplot.pdf", sep=""),height=7,width=7.5)
#dotplot(egokegg,showCategory = 10, color = "pvalue",title="The KEGG enrichment analysis of all genes in Class1,2,3")+
#scale_size(range=c(2, 12))+
#scale_y_discrete(labels=function(egokegg) str_wrap(egokegg,width = 25))+
#scale_color_continuous(low='red', high='#FFD2D2')
#dev.off()

#ego <- enrichGO(gene  = gene.df$ENTREZID,
#                OrgDb         = org.Hs.eg.db,
#                keyType       = 'ENTREZID',## gene id type
#                ont           = "BP",## GO
#                pAdjustMethod = "BH",
#                pvalueCutoff  = 0.05,## threshold
#                qvalueCutoff  = 0.05,
#                readable=TRUE)
#write.csv(ego,file="ego_bp.csv")
#library(simplifyEnrichment)
#gobp <- read.csv("ego_bp.csv",header=T,row.names=1)
#GOIDs <- gobp$ID
#mat = GO_similarity(GOIDs)
#pdf("Heatmap_GO.pdf")
#df = simplifyGO(mat,method = "kmeans",min_term=50)
#print(df)
#dev.off()
#
################### 2 compare
##try to combine
#anno_filter <- read.csv("res_CCM_Classify/anno_filter.csv",header=T,row.names=1)
#anno_filter_Class12 <- subset(anno_filter,Class %in% c("Class1","Class2","Class3"))
#genelist12 <- names(table(anno_filter_Class12$Gene.refGene))
#anno_filter_Class3 <- subset(anno_filter,Class %in% c("Class3"))
#genelist3 <- names(table(anno_filter_Class3$Gene.refGene))
#setwd("7.Class123_enrich")
#gene.df12<-bitr(genelist12, fromType = "SYMBOL",  toType = c("ENTREZID"),OrgDb = org.Hs.eg.db)
#colnames(gene.df12)[1] <- "gene"
#gene.df3<-bitr(genelist3, fromType = "SYMBOL",  toType = c("ENTREZID"),OrgDb = org.Hs.eg.db)
#colnames(gene.df3)[1] <- "gene"
#
#        com<-list()
#        com$Class12 <- gene.df12$ENTREZID
#        com$Class3 <- gene.df3$ENTREZID
#        xx <- compareCluster(com,fun="enrichKEGG",organism="hsa",pvalueCutoff=1,qvalueCutoff=1)
#        x2 <- pairwise_termsim(xx)
#        class(x2)
#        #看对象有哪些属性
#        str(x2)
#        write.csv(x2@compareClusterResult,file="comkegg1.csv")
#        com2 <- read.csv("comkegg1.csv",header=T,row.names = 1)
#	pfs <- read.csv("pfs_treeplot.csv",header=T,row.names=1)
#	#num=which(x2@compareClusterResult$ID %in% c(pfs$Row.names))
#	num=which(x2@compareClusterResult$Description %in% c(pfs$Row.names))
#        x2@compareClusterResult <- x2@compareClusterResult[num,]
#	print(x2@compareClusterResult[num,])
#	col1 <- brewer.pal(n=5, name = "Pastel2")
#        pdf(paste("KEGG_com_treeplot_selected.pdf", sep=""),height=8,width=10)
#        treeplot(x2,pie="count", color = "pvalue",geneClusterPanel = "pie",showCategory=10,cex_category=1.5, layout="kk",group_color=col1)+
#	scale_fill_manual(values=c("#E41A1C","#377EB8"))
#        dev.off()
#

###################### 2_2 pfs treeplot ###########################
#anno_filter <- read.csv("res_CCM_Classify/anno_filter.csv",header=T,row.names=1)
#anno_filter_Class12 <- subset(anno_filter,Class %in% c("Class1","Class2"))
#genelist12 <- names(table(anno_filter_Class12$Gene.refGene))
#more_genelist <- read.csv("7.Class123_enrich/more_pfs_genes.csv",header=F)
#genelist <- c(genelist12,more_genelist$V1)
#genelist <- genelist[!duplicated(genelist)]
#
#setwd("7.Class123_enrich")
#gene.df<-bitr(genelist, fromType = "SYMBOL",  toType = c("ENTREZID"),OrgDb = org.Hs.eg.db)
#colnames(gene.df)[1] <- "gene"
#ekegg <- enrichKEGG(
#  gene  = gene.df$ENTREZID,
#  keyType  = 'kegg',
#  organism = 'human',
#  pvalueCutoff = 1,
#  pAdjustMethod     = "BH",
#  qvalueCutoff  = 1
#)
#egokegg=DOSE::setReadable(ekegg, OrgDb='org.Hs.eg.db',keyType='ENTREZID')
#write.csv(egokegg,file="kegg1.csv")
#egokegg2=read.csv("kegg1.csv",header=T,row.names = 1)
#pfs <- read.csv("pfs_treeplot.csv",header=T,row.names=1)
#num=which(egokegg2$Description %in% c(pfs$Row.names[2:11]))
#egokegg@result<-egokegg[num,]
#pdf(paste("kegg_pfs_cnet.pdf", sep=""),height=10,width=15)
#cnetplot(egokegg,showCategory = 10,colorEdge = TRUE,ann=True,layout = "mds",cex_label_category=1.5)
#dev.off()
#
