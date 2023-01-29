#To be ran in Colab notebook, I don't own a GPU

#from CudaNet import Net

def testEpochAccuracy(size,accuracy,lr):
  testNet.setSize(size)

  weightsFile = "sigmoid-untrained-weights"
  testNet.loadWeights(weightsFile)
  testNet.learningRate = numpy.float32(lr)
  testNet.copyToDevice()

  for epoch in range(1):
    print("\nEPOCH",epoch,"\n")
    start_time = time.time()
    for i in range(len(img_train)):
      trainImg32 = img_train[i].astype(numpy.float32)
      cuda.memcpy_htod(img_gpu,trainImg32)

      testNet.forward(img_gpu)
      testNet.getLoss(label_train[i])
      checkTrainingAnswer(testNet,label_train[i])
      testNet.backward()
      
      if i % batchSize == 0 or i == (len(img_train) - 1):
        testNet.optimize()
        testNet.zero_grad()

    print("--- %s seconds ---" % (time.time() - start_time))
    cuda.memcpy_dtoh(training_correct,training_correct_gpu)
    reset_values(training_correct_gpu,numpy.int32(1),block=(1,1,1))
    print("train dataset: correct = ",(training_correct[0]/len(img_train)))
    test(testNet)
  assert test_correct[0]/len(img_test) == accuracy,"test accuracy has changed."
  testNet.free()


testNet = Net()

batchSize = 1
testEpochAccuracy([784,16,10],0.8948,0.1)
testEpochAccuracy([784,4,10],0.7046,0.1)
testEpochAccuracy([784,16,10,10],0.8181,1)
testEpochAccuracy([784,1200,10],0.8572,0.1)
batchSize = 10
testEpochAccuracy([784,16,10],0.8614,0.1)
batchSize = 1