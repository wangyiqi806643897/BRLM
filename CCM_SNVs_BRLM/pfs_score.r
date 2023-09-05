library(PMAPscore)
library(dplyr)
library(stringr)
library(org.Hs.eg.db)
library(clusterProfiler)

## Class status 0-1 matrix construction
data <- read.table("res_CCM_Classify/Prediction_Class_maf.txt",header = F,sep = "\t",stringsAsFactors = FALSE,quote = "")
data <- subset(data, V2%in%c("Class1","Class2","Class3"))
anno <- read.table("/home/wangyiqi/wyq/WGS/lixuewei12065/case_control230616/maftools/case_control_more.hg19_multianno_maf_noBenign_info.txt",header=F,sep="\t")
anno_gene <- anno[,c(1,8)]
data_gene <- dplyr::left_join(anno_gene,data,by=c('V1'))
data_gene <- na.omit(data_gene)
data_gene_sub <- data_gene[grep(pattern=";",data_gene$V8,invert=TRUE),]
gene_GPT = read.csv("/home/wangyiqi/wyq/GNN/test230315_chatGPT_CCM/res/genelist_tissue_node.csv",header=F)
data_gene_sub[which(data_gene_sub$V8%in%gene_GPT$V1),]$V2="Class1"
#write.csv(data_gene_sub,file="res_CCM_Classify/Prediction_Class_sub.csv")
colnames(data_gene_sub) <- c("ID","gene","class")
data_gene_sub <- data_gene_sub[!duplicated(data_gene_sub$gene),]
t1<-unique(data_gene_sub$gene) #构建行名
t2<-unique(data_gene_sub$class) #构建列名
datamatrix<-matrix(0,nrow(data_gene_sub),length(t2))
rownames(datamatrix)<-t1
colnames(datamatrix)<-t2
for (i in 1:nrow(data_gene_sub)) {
  datamatrix[data_gene_sub[i,2],data_gene_sub[i,3]]<-1 #对应位置打上1
}
write.table(datamatrix,"res_CCM_Classify/mut_class_matrix.txt",sep = "\t",quote = FALSE)

kegg_SYMBOL_hsa <- function(genes){
  gene.df <- bitr(genes, fromType = "SYMBOL",
                  toType = c("SYMBOL", "ENTREZID"),
                  OrgDb = org.Hs.eg.db)
  head(gene.df)
  diff.kk <- enrichKEGG(gene   = gene.df$ENTREZID,
                        organism     = 'hsa',
                        pvalueCutoff = 0.05,
                        qvalueCutoff = 1
  )
  return( setReadable(diff.kk, OrgDb = org.Hs.eg.db,keyType = 'ENTREZID'))
}
predict_sub <- read.csv("res_CCM_Classify/Prediction_Class_sub.csv",header=T,row.names=1)
genes_Class3 <- subset(predict_sub,V2=="Class3")
genes_Class12 <- subset(predict_sub,V2%in%c("Class1","Class2"))
genes_Class3_res=kegg_SYMBOL_hsa(genes_Class3$V8)@result
write.csv(genes_Class3_res,file="res_CCM_Classify/genes_Class3_enrich_res.csv")
genes_Class12_res=kegg_SYMBOL_hsa(genes_Class12$V8)@result
write.csv(genes_Class12_res,file="res_CCM_Classify/genes_Class12_enrich_res.csv")


### pfs_score 计算
get_pfs_score<-function(mut_status,gene_Ucox_res,gene_symbol_Entrez,data.dir=NULL,organism="hsa",
                        verbose=TRUE,Entrez_ID=TRUE,gene_set=NULL){
  rownames(mut_status)<-gsub(pattern = "-",replacement = ".",rownames(mut_status))
  if(is.null(gene_set)){
    a<-apply(mut_status,1,function(x){(length(which(x!=0))/length(x))})
    mut_status<-cbind(mut_status,a)
    mut_status<-mut_status[,-dim(mut_status)[2]]
    mut_status<-mut_status[match(rownames(gene_Ucox_res)[which(gene_Ucox_res$HR<1)],rownames(mut_status)),]
  }else{
    mut_status<-mut_status[match(gene_set,rownames(mut_status)),]
  }
  mut_status<-get_Entrez_ID(mut_status,gene_symbol_Entrez,Entrez_ID=Entrez_ID)
  .myDataEnv <- new.env(parent = emptyenv())
  datload <- paste(organism, "SPIA", sep = "")
  if (is.null(data.dir)) {
    if (!paste(datload, ".RData", sep = "") %in%
        dir(system.file("extdata", package = "PMAPscore"))) {
      cat("The KEGG pathway data for your organism is not present in the extdata folder of the SPIA package!!!")
      cat("\n")
      cat("Please generate one first using makeSPIAdata and specify its location using data.dir argument or copy it in the extdata folder of the SPIA package!")
    }
    else {
      load(file = paste(system.file("extdata", package = "PMAPscore"),
                        paste("/", organism, "SPIA", sep = ""),
                        ".RData", sep = ""), envir = .myDataEnv)
    }
  }
  if (!is.null(data.dir)) {
    if (!paste(datload, ".RData", sep = "") %in%
        dir(data.dir)) {
      cat(paste(data.dir, " does not contin a file called ",
                paste(datload, ".RData", sep = "")))
    }
    else {
      load(file = paste(data.dir, paste(datload, ".RData",
                                        sep = ""), sep = ""), envir = .myDataEnv)
    }
  }
  path.info = .myDataEnv[["path.info"]]
  newlist<-list()
  for(i in 1:dim(mut_status)[2])
  {
    path_matrix<-as.numeric(mut_status[,i])
    names(path_matrix)<-rownames(mut_status)
    path_matrix<-path_matrix[which(path_matrix!=0)]
    if(length(path_matrix[which(path_matrix!=0)])>0){
      res<-newspia(de=path_matrix[which(path_matrix!=0)],all=rownames(mut_status),organism="hsa",
                   beta=NULL,verbose=verbose,data.dir=data.dir,pathids = NULL)
    }
    newlist[[i]]<-res
    names(newlist)[[i]]<-colnames(mut_status)[i]
  }
  pathname<-c()
  for (j in 1:length(path.info)) {
    pathname<-c(pathname,path.info[[j]]$title)
  }
  pfs_score<-matrix(data=0,nrow = length(pathname),ncol = dim(mut_status)[2])
  rownames(pfs_score)<-pathname
  colnames(pfs_score)<-colnames(mut_status)
  for(a in 1:length(newlist))
  {
    loc<-match(names(newlist)[[a]],colnames(pfs_score))
    loc2<-match(newlist[[a]][,1],rownames(pfs_score))
    pfs_score[loc2,loc]<-newlist[[a]][,3]
  }
  return(pfs_score)
}
mut_status1 <- read.table("res_CCM_Classify/mut_class_matrix.txt",header=T,row.names=1)
#pfs_score<-get_pfs_score(mut_status[1:30,1:2],gene_Ucox_res,gene_symbol_Entrez)
head(mut_status1)
pfs_score<-get_pfs_score(mut_status1,gene_Ucox_res,gene_symbol_Entrez)
write.csv(pfs_score,file="res_CCM_Classify/pfs_score.csv")

### Cross
setwd("res_CCM_Classify/")
pfs_score <- read.csv("pfs_score.csv",header=T,row.names=1)
Class3_enrich_res <- read.csv("genes_Class3_enrich_res.csv",header=T,row.names=1)
rownames(Class3_enrich_res) <- Class3_enrich_res$Description
Class12_enrich_res <- read.csv("genes_Class12_enrich_res.csv",header=T,row.names=1)
rownames(Class12_enrich_res) <- Class12_enrich_res$Description

Class3_enrich_pfs <- merge(pfs_score,Class3_enrich_res,by=0)
write.csv(Class3_enrich_pfs,file="Class3_enrich_pfs.csv")
Class12_enrich_pfs <- merge(pfs_score,Class12_enrich_res,by=0)
write.csv(Class12_enrich_pfs,file="Class12_enrich_pfs.csv")
