## Influence Diagram
library(idr)
source("idr-prj.r")

## Influence Diagram 03-09-08

bypass2 = list(

  PAIN = node( Type = "CHANCE", Name = "PAIN", Values = c("ABSENT","PRESENT"), Preds = c("HEARTDISEASE"), 
Pots = matrix( data = c(
0.80, 0.20, ## AB
0.20, 0.80),## PR
  nrow = 2, ncol = 2, byrow = TRUE, dimnames = NULL)), 

  ANGIOGRAM = node( Type = "CHANCE", Name = "ANGIOGRAM", Values = c("NEGATIVE","POSITIVE"), Preds = c("HEARTDISEASE"), 
Pots = matrix( data = c(
0.95, 0.05, ## AB
0.14, 0.86),## PR
  nrow = 2, ncol = 2, byrow = TRUE, dimnames = NULL)), 

  HEARTDISEASE = node(Type="CHANCE", Name = "HEARTDISEASE", Values=c("ABSENT","PRESENT"), Preds = c(),
Pots = matrix( data = c(
0.86, 0.14),
  nrow = 1,ncol = 2,byrow = TRUE,dimnames = NULL)),

  HEARTSURGERY = node( Type = "DECISION", Name = "HEARTSURGERY", Values = c("NO","YES"), Preds=c("PAIN","ANGIOGRAM"), 
Pots = matrix( data = c(1.0), dimnames = list("phase","HEARTSURGERY"))), 

  EARLYRESULTS = node( Type = "CHANCE", Name = "EARLYRESULTS", Values = c("CoRem", "PaRem", "NoChng", "PrgrsvDisease"), Preds = c("ANGIOGRAM","HEARTSURGERY"),
Pots = matrix( data = c(
0.97, 0.01, 0.01, 0.01, ##  NE NO
0.95, 0.03, 0.01, 0.01, ##  NE YS
0.01, 0.04, 0.10, 0.80, ##  PO NO
0.55, 0.20, 0.15, 0.10),##  PO YS
  nrow = 4, ncol = 4, byrow = TRUE, dimnames = NULL)),

  HEARTPHARMA = node( Type = "DECISION", Name = "HEARTPHARMA", Values = c("NO","YES"), Preds=c("PAIN","ANGIOGRAM","EARLYRESULTS"),
Pots = matrix( data = c(2.0), dimnames = list("phase","HEARTPHARMA"))),

  LIFEQ = node(Type="CHANCE", Name = "LIFEQ", Values = c("DEAD","LIVE2ALQ","LIVE2AHQ"), Preds = c("HEARTDISEASE","HEARTSURGERY"),
Pots = matrix( data = c(
0.01, 0.08, 0.91, ## AB NO
0.05, 0.65, 0.30, ## AB YS
0.45, 0.40, 0.15, ## PR NO
0.20, 0.20, 0.60),## PR YS
  nrow = 4, ncol = 3, byrow = TRUE, dimnames = NULL)), 

  ECONOMICALC = node(Type="CHANCE", Name = "ECONOMICALC", Values = c("LOW","MEDIUM","HIGH"), Preds = c("EARLYRESULTS","HEARTSURGERY","HEARTPHARMA"),
Pots = matrix( data = c(
0.90, 0.05, 0.05, ## CR, NO, NO
0.80, 0.10, 0.10, ## CR, NO, YS
0.50, 0.25, 0.25, ## CR, YS, NO
0.45, 0.25, 0.30, ## CR, YS, YS
0.85, 0.10, 0.05, ## PR, NO, NO
0.95, 0.03, 0.02, ## PR, NO, YS
0.45, 0.30, 0.25, ## PR, YS, NO
0.40, 0.30, 0.30, ## PR, YS, YS
0.80, 0.10, 0.10, ## NC, NO, NO
0.90, 0.05, 0.05, ## NC, NO, YS
0.40, 0.35, 0.25, ## NC, YS, NO
0.35, 0.30, 0.35, ## NC, YS, YS
0.75, 0.10, 0.15, ## PD, NO, NO
0.85, 0.10, 0.05, ## PD, NO, YS
0.35, 0.35, 0.30, ## PD, YS, NO
0.30, 0.35, 0.35),## PD, YS, YS
  nrow = 16,ncol = 3,byrow=TRUE,dimnames=NULL)),

  UTILITY = node(Type="UTILITY", Name = "UTILITY", Values = c(0.0,100.0), Preds =c("LIFEQ","ECONOMICALC"), 
Pots = matrix( data = c(
2.0,  ## DEAD, LOW
0.70, ## DEAD, MED
0.05, ## DEAD, HIG
3.10, ## LQL, LOW
2.90, ## LQL, MED
2.80, ## LQL, HIG
7.80, ## HQL, LOW
4.80, ## HQL, MED
2.80),## HQL, HIG
  nrow = 9,ncol = 1,byrow = TRUE,dimnames = list( NULL, c("UTILITY"))))

)

cat( "Influence Diagram -- bypass2: ", names(bypass2),"\n")

influence.diagram( bypass2)
dump.netG( bypass2, filename="bypass2")
arcrevalg.eval( bypass2, DUMP=TRUE)


