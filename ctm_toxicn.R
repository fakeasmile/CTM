# ============================================================
# CTM 建模脚本：TOXICN 训练集 + 形容词伪文档
# ============================================================
#
# 输入：output/preprocessing/ 下的 DTM 文件（由 prepare_dtm.py 生成）
# 输出：output/ctm_model/ 下的模型结果矩阵和日志
#
# 输出文件：
#   - beta_matrix.csv        : β 矩阵 (V × K, 行=词, 列=主题)
#   - theta_sample.csv       : 样本 θ 矩阵 (n_sample × K)
#   - theta_adj.csv          : 形容词 θ 矩阵 (n_adj × K)
#   - theta_all.csv          : 全部 θ 矩阵 (n_docs × K)
#   - sigma_matrix.csv       : Σ 协方差矩阵
#   - concept_matrix_ctm.csv : 文本-形容词余弦相似度矩阵
#   - adj_similarity.csv     : 形容词间余弦相似度矩阵
#   - topic_top_terms.csv    : 各主题 Top-N 高频词
#   - topic_top_terms_full.csv : 各主题所有词的概率分布
#   - model_config.csv       : 模型配置参数记录
#   - ctm_training.log       : 完整训练日志
#
# 用法：在 R 控制台中运行
#   source("d:/CTM/ctm_toxicn.R")
# ============================================================

library(tm)
library(topicmodels)
library(Matrix)

# ---- 配置 ----
BASE_DIR <- "d:/CTM"
PREP_DIR <- file.path(BASE_DIR, "output", "preprocessing")  # Python 输出目录
OUTPUT_DIR <- file.path(BASE_DIR, "output", "ctm_model")     # R 输出目录
K <- 30            # 主题数（可调整）
SEED <- 42         # 随机种子
TOP_N <- 15        # 每个主题展示的 Top-N 词数

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ---- 日志函数：同时输出到控制台和文件 ----
log_file <- file(file.path(OUTPUT_DIR, "ctm_training.log"), encoding = "UTF-8", open = "wt")

log_msg <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  cat(msg, "\n", file = log_file)
}

log_sep <- function() log_msg(paste(rep("=", 60), collapse = ""))

# ---- 保存模型配置 ----
log_sep()
log_msg("CTM 模型配置")
log_sep()
log_msg(sprintf("K (主题数)    = %d", K))
log_msg(sprintf("SEED (随机种子)= %d", SEED))
log_msg(sprintf("TOP_N         = %d", TOP_N))
log_msg(sprintf("输入目录      = %s", PREP_DIR))
log_msg(sprintf("输出目录      = %s", OUTPUT_DIR))
log_msg(sprintf("开始时间      = %s", Sys.time()))

# 保存配置到 CSV
config_df <- data.frame(
  parameter = c("K", "SEED", "TOP_N", "input_dir", "output_dir", "start_time"),
  value = as.character(c(K, SEED, TOP_N, PREP_DIR, OUTPUT_DIR, as.character(Sys.time())))
)
write.csv(config_df, file.path(OUTPUT_DIR, "model_config.csv"), row.names = FALSE)

# ---- 1. 读取数据 ----
log_sep()
log_msg("Step 1: 读取数据")

triplet_path <- file.path(PREP_DIR, "dtm_triplet.csv")
vocab_path <- file.path(PREP_DIR, "vocab.txt")
meta_path <- file.path(PREP_DIR, "dtm_metadata.csv")

log_msg("读取 triplet 文件...")
triplet <- read.csv(triplet_path, stringsAsFactors = FALSE)

log_msg("读取词表...")
vocab <- readLines(vocab_path, encoding = "UTF-8")

log_msg("读取元数据...")
meta <- read.csv(meta_path, stringsAsFactors = FALSE)

n_docs <- nrow(meta)
n_terms <- length(vocab)
log_msg(sprintf("文档数: %d, 词数: %d", n_docs, n_terms))

# 从元数据动态推断样本数和形容词数
N_SAMPLE <- sum(meta$doc_type == "sample")
N_ADJ <- sum(meta$doc_type == "adjective")
log_msg(sprintf("样本文档: %d, 形容词文档: %d", N_SAMPLE, N_ADJ))

# ---- 2. 构建 DocumentTermMatrix ----
log_sep()
log_msg("Step 2: 构建 DocumentTermMatrix")

sparse_mat <- sparseMatrix(
  i = triplet$doc_idx,
  j = triplet$term_idx,
  x = triplet$count,
  dims = c(n_docs, n_terms),
  dimnames = list(meta$doc_id, vocab)
)

dtm <- as.DocumentTermMatrix(sparse_mat, weighting = weightTf)

log_msg(sprintf("DTM 维度: %d × %d", nrow(dtm), ncol(dtm)))
log_msg(sprintf("稀疏度: %.2f%%", 100 * (1 - nnzero(sparse_mat) / (n_docs * n_terms))))

# ---- 3. 拟合 CTM 模型 ----
log_sep()
log_msg(sprintf("Step 3: 拟合 CTM 模型 (K=%d)", K))
log_msg("这可能需要几分钟...")

start_time <- Sys.time()
ctm_model <- CTM(dtm, k = K, control = list(seed = SEED, verbose = 1))
end_time <- Sys.time()
elapsed <- as.numeric(end_time - start_time, units = "secs")

log_msg(sprintf("CTM 拟合完成！耗时: %.1f 秒 (%.1f 分钟)", elapsed, elapsed / 60))
log_msg(sprintf("完成时间: %s", end_time))

# ---- 4. 提取矩阵 ----
log_sep()
log_msg("Step 4: 提取矩阵")

# β 矩阵: 自动检测维度并转置
beta_raw <- posterior(ctm_model)$terms
log_msg(sprintf("posterior()$terms 原始维度: %d × %d", nrow(beta_raw), ncol(beta_raw)))

if (nrow(beta_raw) == K && ncol(beta_raw) == n_terms) {
  beta_matrix <- t(beta_raw)
  log_msg("检测到 K × V 格式，已转置为 V × K")
} else if (nrow(beta_raw) == n_terms && ncol(beta_raw) == K) {
  beta_matrix <- beta_raw
  log_msg("检测到 V × K 格式，无需转置")
} else {
  beta_matrix <- beta_raw
  log_msg(sprintf("[WARNING] β 矩阵维度异常: %d × %d", nrow(beta_raw), ncol(beta_raw)))
}
rownames(beta_matrix) <- colnames(dtm)
colnames(beta_matrix) <- paste0("Topic", 1:ncol(beta_matrix))
log_msg(sprintf("β 矩阵: %d × %d", nrow(beta_matrix), ncol(beta_matrix)))

# θ 矩阵
theta_all <- posterior(ctm_model)$topics
if (ncol(theta_all) != K) {
  log_msg(sprintf("[WARNING] θ 矩阵列数=%d, 期望 K=%d", ncol(theta_all), K))
}
colnames(theta_all) <- paste0("Topic", 1:ncol(theta_all))
log_msg(sprintf("θ 矩阵 (全部): %d × %d", nrow(theta_all), ncol(theta_all)))

# 根据元数据中的 doc_type 分离
sample_idx <- which(meta$doc_type == "sample")
adj_idx <- which(meta$doc_type == "adjective")
theta_sample <- theta_all[sample_idx, , drop = FALSE]
theta_adj <- theta_all[adj_idx, , drop = FALSE]

doc_ids_sample <- meta$doc_id[sample_idx]
doc_ids_adj <- meta$doc_id[adj_idx]
rownames(theta_sample) <- doc_ids_sample
rownames(theta_adj) <- doc_ids_adj

log_msg(sprintf("θ_sample: %d × %d", nrow(theta_sample), ncol(theta_sample)))
log_msg(sprintf("θ_adj: %d × %d", nrow(theta_adj), ncol(theta_adj)))

# Σ 协方差矩阵
sigma_matrix <- ctm_model@Sigma
log_msg(sprintf("@Sigma 原始维度: %d × %d", nrow(sigma_matrix), ncol(sigma_matrix)))

sigma_dim <- nrow(sigma_matrix)
if (sigma_dim == K) {
  log_msg("检测到 K × K 格式的 Σ")
} else if (sigma_dim == K - 1) {
  log_msg("检测到 (K-1) × (K-1) 格式的 Σ (logistic-normal 参数化)")
} else {
  log_msg(sprintf("[WARNING] Σ 维度异常: %d × %d", sigma_dim, sigma_dim))
}
rownames(sigma_matrix) <- paste0("Topic", 1:sigma_dim)
colnames(sigma_matrix) <- paste0("Topic", 1:sigma_dim)
log_msg(sprintf("Σ 协方差矩阵: %d × %d", nrow(sigma_matrix), ncol(sigma_matrix)))

# ---- 5. 保存矩阵 ----
log_sep()
log_msg("Step 5: 保存矩阵")

write.csv(beta_matrix, file.path(OUTPUT_DIR, "beta_matrix.csv"), row.names = TRUE)
write.csv(theta_sample, file.path(OUTPUT_DIR, "theta_sample.csv"), row.names = TRUE)
write.csv(theta_adj, file.path(OUTPUT_DIR, "theta_adj.csv"), row.names = TRUE)
write.csv(sigma_matrix, file.path(OUTPUT_DIR, "sigma_matrix.csv"), row.names = TRUE)
write.csv(theta_all, file.path(OUTPUT_DIR, "theta_all.csv"), row.names = TRUE)

log_msg("已保存: beta_matrix.csv, theta_sample.csv, theta_adj.csv, sigma_matrix.csv, theta_all.csv")

# ---- 6. 主题 Top 词展示与保存 ----
log_sep()
log_msg(sprintf("Step 6: 各主题 Top-%d 高频词", TOP_N))

top_terms_mat <- terms(ctm_model, TOP_N)

# 保存 Top 词到 CSV
top_terms_df <- data.frame(Topic = paste0("Topic", 1:K))
for (rank in 1:TOP_N) {
  top_terms_df[[paste0("Rank", rank)]] <- as.character(top_terms_mat[rank, ])
}
write.csv(top_terms_df, file.path(OUTPUT_DIR, "topic_top_terms.csv"), row.names = FALSE)

# 输出到日志
for (k in 1:K) {
  log_msg(sprintf("主题 %2d: %s", k, paste(top_terms_mat[, k], collapse = ", ")))
}
log_msg(sprintf("主题 Top 词已保存: topic_top_terms.csv"))

# 保存完整 β 矩阵中每个主题的 Top-30 词及其概率
log_msg("保存每个主题的 Top-30 词概率分布...")
top30_per_topic <- list()
for (k in 1:ncol(beta_matrix)) {
  probs <- beta_matrix[, k]
  names(probs) <- rownames(beta_matrix)
  sorted_idx <- order(probs, decreasing = TRUE)[1:min(30, length(probs))]
  top30_per_topic[[k]] <- data.frame(
    rank = 1:length(sorted_idx),
    word = names(probs)[sorted_idx],
    probability = probs[sorted_idx],
    stringsAsFactors = FALSE
  )
}
# 合并保存
top30_combined <- do.call(rbind, lapply(1:K, function(k) {
  df <- top30_per_topic[[k]]
  df$topic <- k
  df
}))
top30_combined <- top30_combined[, c("topic", "rank", "word", "probability")]
write.csv(top30_combined, file.path(OUTPUT_DIR, "topic_top_terms_full.csv"), row.names = FALSE)
log_msg("已保存: topic_top_terms_full.csv")

# ---- 7. 计算文本-形容词余弦相似度（概念向量矩阵）----
log_sep()
log_msg("Step 7: 计算文本-形容词余弦相似度")

cosine_sim <- function(A, B) {
  A_norm <- A / sqrt(rowSums(A^2) + 1e-10)
  B_norm <- B / sqrt(rowSums(B^2) + 1e-10)
  sim <- A_norm %*% t(B_norm)
  return(sim)
}

concept_matrix <- cosine_sim(theta_sample, theta_adj)
colnames(concept_matrix) <- doc_ids_adj
rownames(concept_matrix) <- doc_ids_sample

log_msg(sprintf("概念向量矩阵: %d × %d", nrow(concept_matrix), ncol(concept_matrix)))
log_msg(sprintf("值范围: [%.4f, %.4f]", min(concept_matrix), max(concept_matrix)))

# 统计信息
log_msg(sprintf("均值: %.4f, 中位数: %.4f", mean(concept_matrix), median(concept_matrix)))
log_msg(sprintf(">0.5 的比例: %.2f%%", 100 * mean(concept_matrix > 0.5)))
log_msg(sprintf(">0.8 的比例: %.2f%%", 100 * mean(concept_matrix > 0.8)))

write.csv(concept_matrix, file.path(OUTPUT_DIR, "concept_matrix_ctm.csv"), row.names = TRUE)
log_msg("已保存: concept_matrix_ctm.csv")

# ---- 8. 形容词间的主题分布相似度 ----
log_sep()
log_msg("Step 8: 形容词间主题分布相似度")

adj_sim <- cosine_sim(theta_adj, theta_adj)
adj_names <- meta$source[adj_idx]
rownames(adj_sim) <- doc_ids_adj
colnames(adj_sim) <- doc_ids_adj

# 保存完整的形容词间相似度矩阵
adj_sim_df <- as.data.frame(as.matrix(adj_sim))
adj_sim_df <- cbind(adj_name = adj_names, adj_sim_df)
write.csv(adj_sim_df, file.path(OUTPUT_DIR, "adj_similarity.csv"), row.names = TRUE)
log_msg("已保存: adj_similarity.csv")

# 取 top-5 最相似对（排除自身）
log_msg("最相似的形容词对 (Top-5):")
sim_list <- list()
for (i in 1:(N_ADJ - 1)) {
  for (j in (i + 1):N_ADJ) {
    sim_list <- c(sim_list, list(c(i, j, adj_sim[i, j])))
  }
}
sim_vals <- do.call(rbind, sim_list)
sim_vals <- sim_vals[order(-sim_vals[, 3]), ]
for (r in 1:min(5, nrow(sim_vals))) {
  idx_i <- sim_vals[r, 1]
  idx_j <- sim_vals[r, 2]
  log_msg(sprintf("  %s ↔ %s : %.4f", adj_names[idx_i], adj_names[idx_j], sim_vals[r, 3]))
}

# 保存 Top-20 最相似形容词对
top_pairs <- data.frame(
  adj_i = character(),
  adj_j = character(),
  cosine_sim = numeric(),
  stringsAsFactors = FALSE
)
for (r in 1:min(20, nrow(sim_vals))) {
  idx_i <- sim_vals[r, 1]
  idx_j <- sim_vals[r, 2]
  top_pairs <- rbind(top_pairs, data.frame(
    adj_i = adj_names[idx_i],
    adj_j = adj_names[idx_j],
    cosine_sim = sim_vals[r, 3],
    stringsAsFactors = FALSE
  ))
}
write.csv(top_pairs, file.path(OUTPUT_DIR, "adj_top_similar_pairs.csv"), row.names = FALSE)
log_msg("已保存: adj_top_similar_pairs.csv")

# ---- 完成 ----
log_sep()
log_msg("完成！输出文件总结")
log_sep()
log_msg(sprintf("  %s", file.path(OUTPUT_DIR, "beta_matrix.csv")))
log_msg(sprintf("  %s", file.path(OUTPUT_DIR, "theta_sample.csv")))
log_msg(sprintf("  %s", file.path(OUTPUT_DIR, "theta_adj.csv")))
log_msg(sprintf("  %s", file.path(OUTPUT_DIR, "theta_all.csv")))
log_msg(sprintf("  %s", file.path(OUTPUT_DIR, "sigma_matrix.csv")))
log_msg(sprintf("  %s", file.path(OUTPUT_DIR, "concept_matrix_ctm.csv")))
log_msg(sprintf("  %s", file.path(OUTPUT_DIR, "adj_similarity.csv")))
log_msg(sprintf("  %s", file.path(OUTPUT_DIR, "adj_top_similar_pairs.csv")))
log_msg(sprintf("  %s", file.path(OUTPUT_DIR, "topic_top_terms.csv")))
log_msg(sprintf("  %s", file.path(OUTPUT_DIR, "topic_top_terms_full.csv")))
log_msg(sprintf("  %s", file.path(OUTPUT_DIR, "model_config.csv")))
log_msg(sprintf("  %s", file.path(OUTPUT_DIR, "ctm_training.log")))
log_msg(sprintf("\n结束时间: %s", Sys.time()))
log_msg(sprintf("总耗时: %.1f 秒 (%.1f 分钟)", elapsed, elapsed / 60))

# 关闭日志文件
close(log_file)
