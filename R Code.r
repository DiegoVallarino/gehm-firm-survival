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
data.full <- firms_survival_full_clean
data.full$Event_Time <- as.numeric(gsub(",", ".", data.full$Event_Time))
data.full$Status     <- as.numeric(gsub(",", ".", data.full$Status))

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
library(corrplot)
library(FactoMineR)
library(factoextra)

# ----------------------------
# Ensure numeric format & clean NAs
# ----------------------------
vars_numeric <- data.train[, c("Size", "Leverage", "Profit_Margin", "Org_Complexity", "RD_Intensity")]
vars_numeric <- data.frame(lapply(vars_numeric, function(x) as.numeric(gsub(",", ".", as.character(x)))))

embeds <- data.train[, c("Embed_1", "Embed_2", "Embed_3", "Embed_4")]
embeds <- data.frame(lapply(embeds, function(x) as.numeric(gsub(",", ".", as.character(x)))))

# Replace NA by column means (optional, safer for cor())
for(i in 1:ncol(vars_numeric)) {
  vars_numeric[is.na(vars_numeric[,i]), i] <- mean(vars_numeric[,i], na.rm = TRUE)
}
for(i in 1:ncol(embeds)) {
  embeds[is.na(embeds[,i]), i] <- mean(embeds[,i], na.rm = TRUE)
}

# ----------------------------
# Correlation matrix
# ----------------------------
cor_mat <- cor(cbind(vars_numeric, embeds), use = "pairwise.complete.obs")

library(corrplot)
corrplot(cor_mat,
         method = "color",
         type = "upper",
         tl.col = "black",
         tl.srt = 45,
         title = "Correlations between Embeddings and Firm Variables",
         mar = c(0,0,2,0))


# ----------------------------
# PCA Visualization of Embeddings (cleaned)
# ----------------------------
library(FactoMineR)
library(factoextra)

# Ensure numeric again (just in case)
embeds <- data.train[, c("Embed_1", "Embed_2", "Embed_3", "Embed_4")]
embeds <- data.frame(lapply(embeds, function(x) as.numeric(as.character(x))))

# Replace NA by column means
for(i in 1:ncol(embeds)) {
  embeds[is.na(embeds[,i]), i] <- mean(embeds[,i], na.rm = TRUE)
}

# PCA
pca_res <- PCA(embeds, graph = FALSE)

# Plot with colors by Region
fviz_pca_biplot(pca_res,
                repel = TRUE,
                col.ind = data.train$Region,
                palette = "Set1",
                geom.ind = "point",
                title = "PCA Projection of Firm Embeddings (colored by Region)")


# -----------------------
# Subsample of 500 firms for clearer dentate KM curves
# -----------------------
set.seed(123)
plot_sample <- data.train %>% dplyr::sample_n(500)

# Round event times to mimic yearly observations (makes curves more stepped/dentate)
plot_sample$Event_Time <- round(as.numeric(plot_sample$Event_Time))

# -----------------------
# Kaplan-Meier by Sector
# -----------------------
km_sector <- survfit(Surv(Event_Time, Status) ~ Sector, data = plot_sample)
ggsurvplot(km_sector, data = plot_sample, pval = TRUE,
           risk.table = TRUE, palette = "Dark2",
           title = "Survival Curves by Sector",
           xlab = "Time (Days)", ylab = "Survival Probability")

# -----------------------
# Kaplan-Meier by Region
# -----------------------
km_region <- survfit(Surv(Event_Time, Status) ~ Region, data = plot_sample)
ggsurvplot(km_region, data = plot_sample, pval = TRUE,
           risk.table = TRUE, palette = "Set1",
           title = "Survival Curves by Region",
           xlab = "Time (Days)", ylab = "Survival Probability")

# -----------------------
# Kaplan-Meier by Firm Size
# -----------------------
plot_sample <- plot_sample %>%
  dplyr::mutate(SizeQuartile = ntile(Size, 4))

km_size <- survfit(Surv(Event_Time, Status) ~ SizeQuartile, data = plot_sample)
ggsurvplot(km_size, data = plot_sample, pval = TRUE, risk.table = TRUE,
           palette = "Accent",
           title = "Survival by Firm Size Quartiles",
           xlab = "Time (Days)", ylab = "Survival Probability")

# -----------------------
# Kaplan-Meier by Leverage
# -----------------------
plot_sample <- plot_sample %>%
  dplyr::mutate(LevQuartile = ntile(Leverage, 4))

km_lev <- survfit(Surv(Event_Time, Status) ~ LevQuartile, data = plot_sample)
ggsurvplot(km_lev, data = plot_sample, pval = TRUE, risk.table = TRUE,
           palette = "Paired",
           title = "Survival by Leverage Quartiles",
           xlab = "Time (Days)", ylab = "Survival Probability")

# -----------------------
# Kaplan-Meier by Profit Margin
# -----------------------
plot_sample <- plot_sample %>%
  dplyr::mutate(PMQuartile = ntile(Profit_Margin, 4))

km_pm <- survfit(Surv(Event_Time, Status) ~ PMQuartile, data = plot_sample)
ggsurvplot(km_pm, data = plot_sample, pval = TRUE, risk.table = TRUE,
           palette = "Set3",
           title = "Survival by Profit Margin Quartiles",
           xlab = "Time (Days)", ylab = "Survival Probability")

# -----------------------
# Kaplan-Meier combining Sector × Region
# -----------------------
plot_sample$SectorRegion <- interaction(plot_sample$Sector, plot_sample$Region, sep = " - ")

km_sector_region <- survfit(Surv(Event_Time, Status) ~ SectorRegion, data = plot_sample)
ggsurvplot(km_sector_region, data = plot_sample, pval = TRUE,
           risk.table = FALSE, palette = "Spectral",
           title = "Survival Curves by Sector × Region",
           xlab = "Time (Days)", ylab = "Survival Probability")
