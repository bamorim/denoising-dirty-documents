# libraries

require("png")
require("data.table")

dirtyFolder = "./data/train"
cleanFolder = "./data/train_cleaned"
outFolder = "./out"

outPath = file.path(outFolder, "trainingdata.csv")
filenames = list.files(dirtyFolder)

for (f in filenames){
  print(f)
  imgX = readPNG(file.path(dirtyFolder, f))
  imgY = readPNG(file.path(cleanFolder, f))

  # turn the images into vectors
  x = matrix(imgX, nrow(imgX) * ncol(imgX), 1)
  y = matrix(imgY, nrow(imgY) * ncol(imgY), 1)

  dat = data.table(cbind(y, x))
  setnames(dat,c("y", "x"))
  write.table(dat, file=outPath, append=(f != filenames[1]), sep=",", row.names=FALSE, col.names=(f == filenames[1]), quote=FALSE)
}

# view the data
dat = read.csv(outPath)
head(dat)
rows = sample(nrow(dat), 10000)
plot(dat$x[rows], dat$y[rows])
