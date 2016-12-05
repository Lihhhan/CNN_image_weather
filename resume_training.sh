#!/usr/bin/env sh  
  
~/workspace/caffe/build/tools/caffe train --solver=/home/han/workspace/CNN_image_weather/models/demo/solver.prototxt --snapshot=/home/han/workspace/CNN_image_weather/models/demo_iter_30000.solverstate 2>&1 | tee ./demo.log 





