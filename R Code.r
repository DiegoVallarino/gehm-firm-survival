# ============================
# Survival Analysis with Full Dataset (CSV from Python)
# ============================

# --- Libraries ---
library(survival)
library(survminer)
library(dplyr)
library(FactoMineR)
library(factoextra)
library(corrplot)

# ----------------------------
# Load dataset
# ----------------------------
data.full <- read.csv("firms_survival_full.csv")

# ----------------------------
# Train/Test Split
# ----------------------------
set.seed(123)
train_index <- sample(1:nrow(data.full), size = 0.7*nrow(data.full))
data.train <- data.full[train_index, ]
data.test <- data.full[-train_index, ]
surv_obj_test <- Surv(data.test$Event_Time, data.test$Status)

# ----------------------------
# Kaplan–Meier Overall
# ----------------------------
fit_km <- survfit(Surv(Event_Time, Status) ~ 1, data = data.train)
ggsurvplot(fit_km, data = data.train, conf.int = TRUE,
           title = "Kaplan–Meier Survival Curve",
           xlab = "Time (Days)", ylab = "Survival Probability")

summary(fit_km, times=c(20, 50, 100, 350, 1000, 2000, 5000, 10000, 20000))

# ----------------------------
# Kaplan–Meier by Sector
# ----------------------------
fit_km_sector <- survfit(Surv(Event_Time, Status) ~ Sector, data = data.train)
ggsurvplot(fit_km_sector, data = data.train, pval = TRUE,
           title = "Survival Curves by Sector",
           xlab = "Time (Days)", ylab = "Survival Probability")

# ----------------------------
# Kaplan–Meier by Region
# ----------------------------
fit_km_region <- survfit(Surv(Event_Time, Status) ~ Region, data = data.train)
ggsurvplot(fit_km_region, data = data.train, pval = TRUE,
           title = "Survival Curves by Region",
           xlab = "Time (Days)", ylab = "Survival Probability")

# ----------------------------
# Cox PH Model with Observables + Embeddings
# ----------------------------
fit_cox <- coxph(Surv(Event_Time, Status) ~ Size + Leverage + Profit_Margin +
                   Org_Complexity + RD_Intensity + Sector + Region +
                   Embed_1 + Embed_2 + Embed_3 + Embed_4,
                 data = data.train, x = TRUE)
summary(fit_cox)

# ----------------------------
# Cox PH with Embeddings Only
# ----------------------------
fit_cox_embed <- coxph(Surv(Event_Time, Status) ~ Embed_1 + Embed_2 + Embed_3 + Embed_4,
                       data = data.train, x = TRUE)
summary(fit_cox_embed)

# ----------------------------
# Correlation Analysis: Covariates vs Embeddings
# ----------------------------
vars_numeric <- data.train %>%
  select(Size, Leverage, Profit_Margin, Org_Complexity, RD_Intensity)

embeds <- data.train %>%
  select(Embed_1, Embed_2, Embed_3, Embed_4)

cor_mat <- cor(cbind(vars_numeric, embeds), use = "pairwise.complete.obs")
corrplot(cor_mat, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45,
         title = "Correlations between Embeddings and Firm Variables")

# ----------------------------
# PCA Visualization of Embeddings
# ----------------------------
pca_res <- PCA(embeds, graph = FALSE)
fviz_pca_biplot(pca_res,
                repel = TRUE,
                col.ind = data.train$Region,
                palette = "Set1",
                geom.ind = "point",
                title = "PCA Projection of Firm Embeddings (colored by Region)")