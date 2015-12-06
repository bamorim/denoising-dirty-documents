#!/usr/bin/env ruby

def validate data, model
  log = "logs/validate_#{data}_#{model}"
  puts "Validating #{model} with #{data}"
  system("python validate.py data/train_#{data}.npz #{model} 2>#{log}.err > #{log}.out")
end

def validate_basic
  validate(:r0, :Linear)
  validate(:r0, :KMeansThresholding)
  validate(:r1, :Linear)
end

def validate_prev_kmeans
  validate(:k3_r0, :Linear)
  validate(:k3_r1, :Linear)
  validate(:c_k3_r0, :Linear)
  validate(:c_k3_r1, :Linear)
end

def validate_anns
  ["3", "5", "3_3", "5_3", "9", "5_3_5"].each do |topology|
    #validate(:r1, "ANN_#{topology}")
    #validate(:c_k3_r0, "ANN_#{topology}")
    validate(:c_k3_r1, "ANN_#{topology}")
  end
end

validate_anns
