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
object usMapPartition {
  def main(args: Array[String]) {

    val jobName = "usMapPartition"

    var logger = Logger.getLogger(this.getClass())

    if (args.length < 2) {
      logger.error("=> wrong parameters number")
      System.err.println("Usage: usMapPartition <path-to-files> <output-path>")
      System.exit(1)
    }

    val pathToFile = args(0)
    val outputPath = args(1)

    val conf = new SparkConf().setAppName(jobName).setMaster("local[*]")
    val sc = new SparkContext(conf)

    val timestamp: Long = System.currentTimeMillis / 1000

    logger.info("=> jobName \"" + jobName + "\"")
    logger.info("=> pathToFiles \"" + pathToFile + "\"")

    // Load and parse the data file
    val data = sc.textFile(pathToFile, 2)
    val parsedData = data.map { line =>
      val parts = line.split(',').map(_.toDouble)
      LabeledPoint(parts(0), Vectors.dense(parts.tail))
    }
    val cachedParsedData = parsedData.cache()
    cachedParsedData.saveAsTextFile(outputPath + "-data-" + timestamp)
    cachedParsedData.mapPartitions(rus).saveAsTextFile(outputPath + "-USMP-" + timestamp)
  }

  def rus[T](iter: Iterator[T]): Iterator[T] = {

    var pos = new scala.collection.mutable.ListBuffer[T]()
    var neg = new scala.collection.mutable.ListBuffer[T]()

    while (iter.hasNext) {
      val cur = iter.next
      if (cur.asInstanceOf[LabeledPoint].label == 3) {
        pos += cur
      } else {
        neg += cur
      }
    }

    if (pos.size < neg.size) {

      val diff = neg.size - pos.size
      val usCandidates = (0 to neg.size).toList
      val usSelected = scala.util.Random.shuffle(usCandidates).take(diff)

      for (i <- (neg.size - 1) to 0 by -1) if (usSelected contains i) neg remove i
    }
    val output = pos.++(neg)
    output.iterator
  }

}

