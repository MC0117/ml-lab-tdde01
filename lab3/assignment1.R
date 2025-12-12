set.seed(1234567890)
library(geosphere)
library("dplyr")



stations <- read.csv("stations.csv", fileEncoding = "latin1")
temperatures <- read.csv("temps50k.csv")
st <- merge(stations, temperatures, by="station_number")


#date and time pre-processing
st$date <- as.Date(st$date)
st$time <- substr(st$time, 1, 5)
st$hour <- as.numeric(substr(st$time, 1, 2))

#position pre-processing, calculating haversine distance
positions <- stations%>%select(longitude, latitude)
space_distances <- distHaversine(positions, c(b,a))


h_space <- 80000
h_date <- 10
h_time <- 2

a <- 58.4274 #point to predict
b <-  14.826

date <- as.Date("2013-12-12") #date to predict
times <- c(
  "04:00:00",
  "06:00:00",
  "08:00:00",
  "10:00:00",
  "12:00:00",
  "14:00:00",
  "16:00:00",
  "18:00:00",
  "20:00:00",
  "22:00:00",
  "24:00:00"
)

times <- as.numeric(substr(times, 1, 2))

temp <- vector(length = length(times))

#scaled_distance <- function(data, prediction, smoothing) (data-prediction)/h
gaussian_kernel <- function(u) exp(-u^2)


for(i in seq_along(times)){
  pred_time <- times[i]
  
  st_filtered <- st %>%
    filter(date < date | (date == date & hour <= pred_time))
  
  #calculate the distances in each domain
  d_space <- distHaversine(cbind(st_filtered$longitude, st_filtered$latitude), c(b,a))
  d_date <- abs(as.numeric(st_filtered$date - date))
  d_time <- abs(st_filtered$hour - pred_time)

  #scale the distances
  u_space <- d_space/h_space
  u_date <- d_date/h_date
  u_time <- d_time/h_time
  
  #calculate the kernel values
  k_space <- gaussian_kernel(u_space)
  k_date <- gaussian_kernel(u_date)
  k_time <- gaussian_kernel(u_time)
  
  w <- k_space + k_date + k_time
  #w <- k_space * k_date * k_time #product of kernels
  
  temp[i] <- sum(w * st_filtered$air_temperature) / sum(w)
  
}

plot(temp, type="o")

