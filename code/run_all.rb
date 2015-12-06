#!/usr/bin/env ruby

def validate data, model
  log = "out/#{data}_#{model}"
  puts "Validating #{model} with #{data}"
  system("python validate.py data/train_#{data}.npz #{model} 2>#{log}.err > #{log}.out")
end

validate(:r0, :Linear)
validate(:r0, :ANN_3)
validate(:r0, :KMeansThresholding)
validate(:r1, :Linear)

validate(:k3_r0, :Linear)
validate(:k3_r1, :Linear)
validate(:c_k3_r0, :Linear)
validate(:c_k3_r1, :Linear)

# ANNs
["3", "5", "5_3", "9", "5_3_5"].each do |topology|
  validate(:r1, "ANN_#{topology}")
  validate(:c_k3_r0, "ANN_#{topology}")
  validate(:c_k3_r1, "ANN_#{topology}")
end
