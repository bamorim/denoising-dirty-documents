#!/usr/bin/env ruby

def test data, model
  executable = data.to_s[0] == "c" ? "test_combined.py" : "test.py"
  test_folder = data.to_s[0] == "k" ? "out/r0_KMeansThresholding" : "data/test"
  r = data.to_s[-1]
  log = "logs/test_#{data}_#{model}"
  puts "Testing #{model} with #{data}"
  Dir.glob("#{test_folder}/*").each do |f|
    puts "|-> #{f}"
    system("python test.py #{f} #{r} data/#{data}_#{model}.p 2>#{log}.err >> #{log}.out")
  end
end

test(:r0, :Linear)
test(:r0, :KMeansThresholding)
test(:r1, :Linear)

test(:k3_r0, :Linear)
test(:k3_r1, :Linear)
test(:c_k3_r0, :Linear)
test(:c_k3_r1, :Linear)

# ANNs
["3", "5", "5_3", "9", "5_3_5"].each do |topology|
  test(:r1, "ANN_#{topology}")
  test(:c_k3_r0, "ANN_#{topology}")
  test(:c_k3_r1, "ANN_#{topology}")
end
