package testSpark

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j.Logger

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

/**
 * @author david
 */
object undersampling {
  def main(args: Array[String]) {

    val jobName = "undersampling---"

    var logger = Logger.getLogger(this.getClass())

    if (args.length < 2) {
      logger.error("=> wrong parameters number")
      System.err.println("Usage: undersampling <path-to-files> <output-path>")
      System.exit(1)
    }

    val pathToFile = args(0)
    val outputPath = args(1)

    val conf = new SparkConf().setAppName(jobName).setMaster("local[*]")
    val sc = new SparkContext(conf)

    logger.info("=> jobName \"" + jobName + "\"")
    logger.info("=> pathToFiles \"" + pathToFile + "\"--")

    // Load and parse the data file
    val data = sc.textFile(pathToFile, 2)
    val parsedData = data.map { line =>
      val parts = line.split(',').map(_.toDouble)
      LabeledPoint(parts(0), Vectors.dense(parts.tail))
    }
    parsedData.cache()

    val numPos = parsedData.filter { p => p.label == 1 }.count().toDouble
    val numNeg = parsedData.filter { p => p.label == 0 }.count().toDouble

    val porRed = numPos / numNeg
    val timestamp: Long = System.currentTimeMillis / 1000

    parsedData.filter { p => p.label == 0 }.sample("FALSE".toBoolean, porRed, 0).union(parsedData.filter { p => p.label == 1 }).saveAsTextFile(outputPath + "-US-" + timestamp)
  }
}