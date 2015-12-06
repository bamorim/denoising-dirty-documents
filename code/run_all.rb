def validate data, model
  log = "out/#{data}_#{model}"
  puts "Validating #{model} with #{data}"
  system("python validate.py data/train_#{data}.npz #{model} 2>#{log}.err > #{log}.out")
end

validate(:r0, :Linear)
validate(:r0, :ANN_3)
validate(:r0, :KMeansThresholding)
validate(:r1, :Linear)

validate(:k3, :Linear)
validate(:r0_k3, :Linear)
validate(:r1_k3, :Linear)

# ANNs
["3", "5", "5_3", "9", "5_3_5"].each do |topology|
  validate(:r1, "ANN_#{topology}")
  validate(:r1_k3, "ANN_#{topology}")
  validate(:r1_k3, "ANN_#{topology}")
end
