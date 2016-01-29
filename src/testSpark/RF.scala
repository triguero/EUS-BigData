package testSpark

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Gini

import org.apache.log4j.Logger
/**
 * @author david
 */
object RF {
  def main(arg: Array[String]) {

    var logger = Logger.getLogger(this.getClass())

    if (arg.length < 4) {
      logger.error("=> wrong parameters number")
      System.err.println("Parameters \n\t<path-to-header>\n\t<path-to-train>\n\t<path-to-test>\n\t<number-of-partition>")
      System.exit(1)
    }

    //Reading parameters
    val pathHeader = arg(0)
    val pathTrain = arg(1)
    val pathTest = arg(2)
    val numPartition = arg(3).toInt
    val pathOutput = arg(4)

    //Basic setup
    val jobName = "RFMR"

    //Linea para ejecutar desde SCALA IDE
    val conf = new SparkConf().setAppName(jobName).setMaster("local[*]")
    //Linea para ejecutar desde local / cluster
    //val conf = new SparkConf().setAppName(jobName)
    val sc = new SparkContext(conf)

    logger.info("=> jobName \"" + jobName + "\"")
    logger.info("=> pathToHeader \"" + pathHeader + "\"")
    logger.info("=> pathToTrain \"" + pathTrain + "\"")
    logger.info("=> pathToTest \"" + pathTest + "\"")
    logger.info("=> NumberPartition \"" + numPartition + "\"")
    logger.info("=> pathToOuput \"" + pathOutput + "\"")

    //Reading dataset
    val trainRaw = sc.textFile(pathTrain: String, numPartition)
    val testRaw = sc.textFile(pathTest: String)
    
     trainRaw.take(10).foreach(println)

    //Count the samples of each dataset and the number of classes
    val numSamplesTrain = trainRaw.count().toInt
    val numSamplesTest = testRaw.count()
    val numClasses = keelParser.getNumClassFromHeader(sc, pathHeader)

    //Reading dataset header and normalize(JM)
        //val converter = keelParser.getParserFromHeader(sc, pathHeader)
        //val train = trainRaw.map(line => keelParser.parserToDouble(converter, line))
        //val timestamp: Long = System.currentTimeMillis / 1000
        //train.saveAsTextFile(pathOutput+"-TR-"+timestamp)

    // Parse data and creates LabeledPoint
    val trainLabeled = trainRaw.map { line =>
      val parts = line.split(',').map(_.toDouble)
      LabeledPoint(parts.last, Vectors.dense(parts.init))
    }
    val cachedTrainLabeled = trainLabeled.cache()

    val testLabeled = testRaw.map { line =>
      val parts = line.split(',').map(_.toDouble)
      LabeledPoint(parts.last, Vectors.dense(parts.init))
    }
    val cachedTestLabeled = testLabeled.cache()

    // RUS 
    //##########################################################################################
        val TrainRus = cachedTrainLabeled.mapPartitions(usMapPartition.rus)
        val timestamp: Long = System.currentTimeMillis / 1000
        TrainRus.saveAsTextFile(pathOutput+"-USMP-"+timestamp)

    // RANDOM FOREST
    //##########################################################################################
/*
    val maxDepth = 5
    val maxBins = 32

    //val model = DecisionTree.trainClassifier(cachedTrainLabeled, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    val model = DecisionTree.train(cachedTrainLabeled, Classification, Gini, maxDepth)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = cachedTestLabeled.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / cachedTestLabeled.count()
    println("Test Error = " + testErr)
    println("Learned classification tree model:\n" + model.toDebugString)

    print("\n\n++++++++++++++++++++++++++++++++++++++++++++++++END")
    /*
    
    //Reading header of the dataset and dataset
    val typeConversion = KeelParser.parseHeaderFile(sc, pathHeader)
    val bcTypeConv = sc.broadcast(typeConversion)
    val trainRaw = sc.textFile(pathTrain: String)
    val testRaw = sc.textFile(pathTest: String)

    //Count the samples of each data set
    val numSamplesTrain = trainRaw.count().toInt
    val numSamplesTest = testRaw.count()

    //Parsing categorical attribute and class, splitting all data sets 
    //val train = trainRaw.map(line => Utilities.toArrayDouble(KeelParser.parseLabeledPoint(bcTypeConv.value, line).toString()))
    val test = testRaw.map(line => Utilities.toArrayDouble(KeelParser.parseLabeledPoint(bcTypeConv.value, line).toString()))
    
    // and save the result
    test.saveAsTextFile(pathOutput)
    
    //Get class of training set.
    val trainClassAux = train.map(line => line)
    val trainClass = trainClassAux.toArray()

    Aqui tengo que hacer algo similar pero con el resto de elementos del array
    val trainDataAux = train.map()

    print("\n\n****************************************************************************"+trainClass.getClass)

    for (i <- 0 to numSamplesTrain-1)
      print(prueba(i)+" ")

    
    val test1 = test.map({  print("hola")
                            line => var tra = line(0).toDouble
                            print("holaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\n\n\n\n"+tra+"\n\n\n\n\n")}
    )
    

    
    var trozos = new StringTokenizer (test.toString(),",");

    Tengo que pasar del string porcachón que devuelve el lector de Keel a dos vectores de doubles de java. Con eso puedo hacer crear el prototype.
    var z = new Array[Double](1)
    var hola = "1.0"
    z(0) = hola.toDouble
    val pro = new Prototype(z, z)
    print(pro.toString())
    
    */
*/
  }
}