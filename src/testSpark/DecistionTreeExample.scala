package testSpark

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j.Logger

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Gini

import org.apache.spark.mllib.evaluation.MulticlassMetrics

/**
 * @author david
 */
object DecisionTreeExample {
  def main(args: Array[String]) {

    val jobName = "DecisionTreeExample"

    var logger = Logger.getLogger(this.getClass())

    if (args.length < 2) {
      logger.error("=> wrong parameters number")
      System.err.println("Usage: DecisionTreeExample <path-to-files> <output-path>")
      System.exit(1)
    }

    val pathToFile = args(0)
    val outputPath = args(1)

    val conf = new SparkConf().setAppName(jobName).setMaster("local[*]")
    val sc = new SparkContext(conf)

    logger.info("=> jobName \"" + jobName + "\"")
    logger.info("=> pathToFiles \"" + pathToFile + "\"")

    // Load and parse the data file
    val data = sc.textFile(pathToFile)
    val parsedData = data.map { line =>
      val parts = line.split(',').map(_.toDouble)
      LabeledPoint(parts(0), Vectors.dense(parts.tail))
    }

    // Run training algorithm to build the model
    val maxDepth = 5
    val model = DecisionTree.train(parsedData, Classification, Gini, maxDepth)

    // Evaluate model on training examples and compute training error
    val labelAndPreds = parsedData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val trainScores = parsedData.map(p => (model.predict(p.features), p.label))
    val trainMetrics = new MulticlassMetrics(trainScores)

    println(trainMetrics.confusionMatrix)
  }
}