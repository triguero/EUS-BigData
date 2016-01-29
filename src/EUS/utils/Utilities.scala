package EUS.utils

object Utilities {

  //Parser Keel to array[double] where 1º elements is the class
  def toArrayDouble(str: String): Array[Double] = {
    //Replace ()[]to facilities parse to array double
    val clear: Array[String] = str.replaceAllLiterally("(", "").replaceAllLiterally(")", "").replaceAllLiterally("[", "").replaceAllLiterally("]", "").split(",")

    //print("\n\n"+str+"\n")
    // Parse Array[String] to Array[Double]
    val size = clear.length
    var sample: Array[Double] = new Array[Double](size)
    for (i <- 0 to size - 1) {
      sample(i) = clear(i).toDouble
      //print(clear(i)+" ")
    }
    //print("\n\n")
    sample
  }

  // Replace ()[]to facilities parse to array double
  def cleanString(str: String): String = {
    val clear: String = str.replaceAllLiterally("(", "").replaceAllLiterally(")", "").replaceAllLiterally("[", "").replaceAllLiterally("]", "") //replaceAll(")", "")replaceAll("[", "")replaceAll("]", "")
    //print("\n\n" + clear + "\n\n")
    clear
  }

  // Parse Array[String] to Array[Double]
  def parseDouble(listSTR: Array[String]): Array[Double] = {
    val size = listSTR.length
    var sample: Array[Double] = new Array[Double](size)

    for (i <- 1 to size)
      sample(i) = listSTR(i).toDouble

    sample
  }
  
  def calculateConfusionMatrix(rightPredictedClas: Array[Array[Int]], numClass: Int): Array[Array[Int]] = {
    //Create and initializate the confusion matrix
    var confusionMatrix = new Array[Array[Int]](numClass)
    for (i <- 0 to numClass - 1) {
      confusionMatrix(i) = new Array[Int](numClass)
      for (j <- 0 to numClass - 1) {
        confusionMatrix(i)(j) = 0
      }
    }

    val size = rightPredictedClas.length
    for (i <- 0 to size - 1) {
      confusionMatrix(rightPredictedClas(i)(0))(rightPredictedClas(i)(1)) = confusionMatrix(rightPredictedClas(i)(0))(rightPredictedClas(i)(1)) + 1
    }

    confusionMatrix
  }

}