#!/usr/bin/env ruby

def train data, model
  log = "logs/train_#{data}_#{model}"
  puts "Training #{model} with #{data}"
  system("python train.py data/train_#{data}.npz data/#{data}_#{model}.p #{model} 2>#{log}.err > #{log}.out")
end

train(:r0, :Linear)
train(:r1, :Linear)

train(:k3_r0, :Linear)
train(:k3_r1, :Linear)
train(:c_k3_r0, :Linear)
train(:c_k3_r1, :Linear)

# ANNs
["3", "5", "5_3", "9", "5_3_5"].each do |topology|
  train(:r1, "ANN_#{topology}")
  train(:c_k3_r0, "ANN_#{topology}")
  train(:c_k3_r1, "ANN_#{topology}")
end
