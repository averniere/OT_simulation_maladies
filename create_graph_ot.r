#import de librairies
library(dplyr)
library(tidyr)
library(ggplot2)
library(glue)

#chargements des résultats de simulation
data <- read.csv("results/results_mean.csv") # nolint

#réorganisation
data_long <- data |>
  pivot_longer(
    cols = starts_with("Precision_quantile_"),
    names_to = c("Metric", "Quantile"),
    names_pattern = "(.*)_quantile_(.*)",
    values_to = "Precision"
  ) |>
  mutate(Quantile = round(as.numeric(Quantile), 4))

#labeller
custom_labeller <- labeller(
  n_match = function(x) paste("Monogenic matches =", x),
  noise_level = function(x) {
    x_num <- as.numeric(x)
    paste("Noise level =", x_num * 100, "%")
  }
)

#plot
theme_set(theme_bw())

p1 <- ggplot(data_long, aes(x = Quantile, y = Precision, color = as.factor(OT_type))) + # nolint
  geom_line(aes(linetype = factor(overlap_rate))) +
  facet_grid(n_match ~ noise_level, labeller = custom_labeller) +
  labs(x = "Monogenic to complex association threshold",
       color = NULL,
       linetype = "Overlap Rate") +
  ylim(0, 1) +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom",
    legend.title = element_text(size = 8),
    legend.text = element_text(size = 7)
  )
ggsave("results/plot_simu_ot.png", plot = p1, dpi = 300, width = 12, height = 7, units ="in") # nolint