library(ggplot2)
library(forecast)
library(tidyverse)

setwd("~/Documents/CANADA/STUDY/2020Winter/CPSC340/Midterm/Kaggle")
dat0 = read.csv("data/phase2_training_data.csv")
str(dat)
dat = dat0 %>%
  group_by(country_id) %>%
  mutate(date = as.Date(date, format = "%m/%d/%Y"),
         days = as.numeric(date) - as.numeric(date[1]) + 1,
         deaths_lag1 = lag(deaths),
         daily_deaths = c(NA,diff(deaths)),
         daily_deaths_lag1 = lag(daily_deaths),
         cases_lag1 = lag(cases),
         cases_lag2 = lag(lag(cases)),
         daily_cases = c(NA,diff(cases)),
         daily_cases_lag1 = lag(daily_cases),
         daily_cases_lag2 = lag(daily_cases,2),
         cases_14_100k_lag1 = lag(cases_14_100k), 
         cases_100k_lag1 = lag(cases_100k))


dat %>%
  as.data.frame() %>%
  filter(country_id %in% unique(country_id)[1:8]) %>%
  ggplot(aes(date, cases, color=country_id)) +
  geom_line() 
dat %>%
  as.data.frame() %>%
  filter(country_id %in% unique(country_id)[1:8]) %>%
  ggplot(aes(date, daily_cases, color=country_id)) +
  geom_line() 
dat %>%
  as.data.frame() %>%
  filter(country_id %in% unique(country_id)[1:8]) %>%
  ggplot(aes(date, cases_100k, color=country_id)) +
  geom_line() 
dat %>%
  as.data.frame() %>%
  filter(country_id %in% unique(country_id)[1:8]) %>%
  ggplot(aes(date, cases_14_100k, color=country_id)) +
  geom_line() 
dat %>%
  as.data.frame() %>%
  filter(country_id %in% unique(country_id)[1:8]) %>%
  ggplot(aes(date, deaths, color=country_id)) +
  geom_line() 


dat %>%
  as.data.frame() %>%
  filter(country_id %in% unique(country_id)[1:8]) %>%
  ggplot(aes(cases, cases_lag1, color=country_id)) +
  geom_line() +
  geom_point()
dat %>%
  as.data.frame() %>%
  filter(country_id %in% unique(country_id)[1:8]) %>%
  ggplot(aes(daily_cases, daily_cases_lag1, color=country_id)) +
  geom_point()


cor(dat$cases, dat$cases_lag1, use="complete.obs")

dat %>%
  as.data.frame() %>%
  filter(country_id %in% unique(country_id)[11:20]) %>%
  ggplot(aes(daily_cases, daily_cases_lag1, color=country_id)) +
  geom_line() +
  geom_point()


dat_wd = dat %>%
  pivot_wider(names_from = country_id, values_from = -c(country_id, date))

deaths = dat_wd %>%
  select(starts_with("deaths")) 

corr = apply(deaths, 2, function(x){cor(x, deaths$deaths_CA)})
sort(corr)


### Canada
dat.cty = dat %>%
  as.data.frame() %>%
  filter(country_id %in% unique(country_id)[33])

pairs(dat.cty[,c("days", "deaths", "cases", "cases_14_100k","cases_100k")])
pairs(dat.cty[(nrow(dat.cty)-100):(nrow(dat.cty)),c("days", "daily_deaths", "daily_deaths_lag1", "deaths_lag1", "cases_lag1", "daily_cases_lag1")])

summary(lm1 <- lm(deaths ~ deaths_lag1, data = dat.cty))
plot(lm1, which=1)
summary(lm2 <- lm(daily_deaths ~ daily_deaths_lag1 + poly(deaths_lag1,2), data = dat.cty %>% filter(deaths>0)))
plot(lm2, which=1)
summary(lm2 <- lm(daily_deaths ~ daily_deaths_lag1, data = dat.cty %>% na.omit))
plot(lm2, which=1)

dat.cty %>%
  as.data.frame() %>%
  ggplot(aes(date, deaths)) +
  geom_line() 

acf(na.omit(dat.cty$daily_deaths-u_deaths_lag1))

dat.cty %>%
  as.data.frame() %>%
  ggplot(aes(date, daily_deaths)) +
  geom_line() 

dat.cty %>%
  as.data.frame() %>%
  ggplot(aes(date, daily_deaths-daily_deaths_lag1)) +
  geom_line() 



# ==== predict with only deaths ====
### Canada
dat.cty = dat %>%
  as.data.frame() %>%
  filter(country_id =="CA")

lm2 <- lm(daily_deaths ~ daily_deaths_lag1, data = dat.cty %>% na.omit)
dat.pred = dat.cty[,c("deaths","deaths_lag1","daily_deaths","daily_deaths_lag1")]
for(i in 1:11){
  dd_pred <- predict(lm2, newdata = dat.pred[nrow(dat.pred),])
  d_pred <- dat.pred$deaths[nrow(dat.pred)] + dd_pred
  d1_pred <- dat.pred$deaths[nrow(dat.pred)]
  dd1_pred <- d1_pred - dat.pred$deaths_lag1[nrow(dat.pred)]
  dat.pred <- rbind(dat.pred, c(d_pred, d1_pred, dd_pred, dd1_pred))
}
tail(dat.pred,20)

dat.pred %>%
  as.data.frame() %>%
  ggplot(aes(1:nrow(dat.pred), daily_deaths)) +
  geom_line() 


library(forecast)
fit = auto.arima(dat.cty$deaths[(nrow(dat.cty)-100):(nrow(dat.cty))])
fit
pred = forecast(fit, 11)
plot(pred)
diff(pred$mean)
plot(diff(c(dat.cty$deaths[(nrow(dat.cty)-100):(nrow(dat.cty))], pred$mean)))

fit = arima(dat.cty$deaths[(nrow(dat.cty)-103):(nrow(dat.cty))], c(3,1,0))
pred = forecast(fit, 11)
plot(pred)
pred$mean
diff(pred$mean)

df = data.frame(deaths = pred$mean, Id = 0:10)
write.csv(df, "data/predictions.csv", row.names = F)

summary(lm <- lm(daily_deaths ~ daily_deaths_lag1, data = dat.cty[(nrow(dat.cty)-100):(nrow(dat.cty)),]))

dat.cty[(nrow(dat.cty)-100):(nrow(dat.cty)),] %>%
  ggplot(aes(daily_deaths_lag1, daily_deaths)) + 
  geom_point() +
  geom_smooth(method="lm")

dat.std = apply(dat.cty, 2, function(x){(x-mean(x, na.rm=T))/sd(x, na.rm=T)})



summary(lm2 <- step(lm(daily_deaths ~ ., data = na.omit(dat.cty[,-c(1:2,7:8)] %>% filter(deaths>0)))))
plot(lm2,which=1)
summary(lm3 <- lm(daily_deaths ~ daily_deaths_lag1, data = na.omit(dat.cty)))
plot(lm3, which=1)


# ==== using similar countries ====
dat %>%
  group_by(country_id) %>%
  slice_tail(n=100) %>%
  filter(country_id %in% c('CA', 'DE', 'PL', 'DZ')) %>%
  ggplot(aes(date, daily_deaths, color = country_id)) +
  geom_line() +
  ggsave("plots/death_similar_countries_100days.png")

dat %>%
  filter(country_id %in% c('CA', 'DE', 'PL', 'DZ')) %>%
  ggplot(aes(date, deaths, color = country_id)) +
  geom_line() 

dat %>%
  filter(country_id %in% c('CA', 'DE', 'PL', 'DZ')) %>%
  ggplot(aes(date, daily_deaths, color = country_id)) +
  geom_line()

dat_4ct = dat %>%
  filter(country_id %in% c('CA', 'DE', 'PL', 'DZ', 'BE', 'AM', 'MD')) %>%
  group_by(country_id) %>%
  mutate(daily_deaths_lag2 = lag(daily_deaths,2),
         daily_deaths_lag3 = lag(daily_deaths,3),
         deaths_lag2 = lag(deaths,2),
         deaths_lag3 = lag(deaths,3)) %>%
  slice_tail(n=100) %>%
  pivot_wider(names_from = country_id, values_from = -c(country_id, date, days))

dat_4ct_sub = dat_4ct %>%
  select(daily_deaths_CA,daily_deaths_lag1_CA,daily_deaths_lag2_CA,daily_deaths_lag3_CA,
         daily_deaths_lag1_DE,daily_deaths_lag1_DZ,daily_deaths_lag1_PL,
         daily_deaths_lag1_BE,daily_deaths_lag1_AM,daily_deaths_lag1_MD,
         daily_cases_lag1_CA, cases_lag1_CA, cases_14_100k_lag1_CA, cases_100k_lag1_CA) 
rmat=cor(dat_4ct_sub)

summary(step(lm(daily_deaths_CA~., data = dat_4ct_sub)))


ggplot(data=reshape2::melt(rmat), aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile(color="white")  +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed() +
  ggsave("plots/heatmap_daily_deaths.png")


dat_4ct_sub0 = dat_4ct %>%
  select(deaths_CA,deaths_lag1_CA,deaths_lag2_CA,deaths_lag3_CA,
         deaths_lag1_DE,deaths_lag1_DZ,deaths_lag1_PL,
         cases_lag1_CA, cases_lag1_CA, cases_14_100k_lag1_CA, cases_100k_lag1_CA) 

summary(step(lm(deaths_CA~., data = dat_4ct_sub0)))
rmat2=cor(dat_4ct_sub0)


ggplot(data=reshape2::melt(rmat2), aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile(color="white")  +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed() +
  ggsave("plots/heatmap_total_deaths.png")

library(GGally)
dat_4ct_sub2 = dat_4ct_sub[, rownames(rmat)[rmat[,1]>0.15]] 
png("plots/pairs.png", width=700, height=700)
ggpairs(dat_4ct_sub2)
dev.off()


(fit <- auto.arima(dat_4ct$daily_deaths_BE))
pred = forecast(fit, 11)
plot(pred)
diff(pred$mean)


lm_step <- step(lm0 <- lm(daily_deaths_CA~., data=dat_4ct_sub2[,-9]))
summary(lm0)
summary(lm_step)

dat_pred = as.data.frame(dat_4ct_sub2)[,1:3]
for(i in 1:11){
  dat_pred = rbind(dat_pred, c(NA, tail(dat_pred$daily_deaths_CA,1), tail(dat_pred$daily_deaths_CA,3)[1]))
  dat_pred[nrow(dat_pred),1] = predict(lm_step, newdata = tail(dat_pred,n=1))
}
tail(dat_pred,20)

deaths_CA = c(dat$daily_deaths[dat$country_id=="CA"][1:280], dat_pred[101:111,1])
deaths_CA[is.na(deaths_CA)] = 0
total_deaths_CA = cumsum(deaths_CA)

df = data.frame(deaths = total_deaths_CA[281:291], Id = 0:10)
write.csv(df, "data/predictions.csv", row.names = F)

# real data
deaths_CA_true = c(dat$daily_deaths[dat$country_id=="CA"][1:280], c(26,11,16,28,23,5,14,27,10,35,23))
deaths_CA_true[1] = 0
total_deaths_CA_true = cumsum(deaths_CA_true)

sqrt(mean((total_deaths_CA[281:291]-total_deaths_CA_true[281:291])^2))
sqrt(mean((deaths_CA[281:291]-deaths_CA_true[281:291])^2))

