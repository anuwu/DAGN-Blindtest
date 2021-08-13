import sys
import batch
Batch = batch.Batch

if __name__ == "__main__" :
    batch_name = sys.argv[1]
    batchObj = Batch.getBatch(batch_name)

    if batchObj is not None :
       batchObj.classifySerial()
       batchObj.genResults()
    else :
        print("Please enter the correct batch name")
        sys.exit(1)

