package EUS.utils

import org.apache.mahout.keel.Dataset._
import java.util.ArrayList
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import collection.JavaConverters._
import org.apache.spark.SparkContext

object KeelParser {

  /**
   * @brief Get the labels of a feature or the main class as Array[String]
   *
   * @param str string to parser
   * @author jesus
   */
  def getLabels(str: String): Array[String] = {
    var result = str.substring(str.indexOf("{") + 1, str.indexOf("}")).replaceAll(" ", "").split(",")
    result
  }

  /**
   * @brief Get the min and max of a feature as a Array[Double]
   *
   * @param str string to parser
   * @author jesus
   */
  def getRange(str: String): Array[Double] = {
    var aux = str.substring(str.indexOf("[") + 1, str.indexOf("]")).replaceAll(" ", "").split(",")
    var result = new Array[Double](2)
    result(0) = aux(0).toDouble
    result(1) = aux(1).toDouble
    result
  }

  /**
   * @brief Get the information necesary for parser with function parserToDouble
   *
   * @param sc The SparkContext
   * @param file path of the header
   * @author jesus
   */
  def getParserFromHeader(sc: SparkContext, file: String): Array[Map[String, Double]] = {
    //Reading header. Each element is a line
    val header = sc.textFile(file)
    var linesHeader = header.toArray()

    //Calculate number of featires + 1 for the class
    var numFeatures = 0
    for (i <- 0 to (linesHeader.length - 1)) {
      if (linesHeader(i).toUpperCase().contains("@INPUTS")) {
        numFeatures = linesHeader(i).length - linesHeader(i).replaceAllLiterally(",", "").length + 2
      } //end if
    } //end for

    //Calculate transformation to normalize and erase categorical features
    val conv = new Array[Map[String, Double]](numFeatures)
    for (i <- 0 to numFeatures - 1) {
      conv(i) = Map()
    }

    var auxParserClasses = 0.0
    var auxNumFeature = 0
    for (i <- 0 to (linesHeader.length - 1)) {
      if (linesHeader(i).toUpperCase().contains("@ATTRIBUTE CLASS")) {
        val labelsClasses = getLabels(linesHeader(i)) //Array of String with the labels of the objetive variable
        for (key <- labelsClasses) { //Calculate map for parser label classes
          conv(numFeatures - 1) += (key -> auxParserClasses)
          auxParserClasses = auxParserClasses + 1
        }
      } else if (linesHeader(i).toUpperCase().contains("[")) { //Real or integer feature
        val range = getRange(linesHeader(i)) //Min and max of the feature
        conv(auxNumFeature) += ("min" -> range(0), "max" -> range(1)) //Do the parser for this feature
        auxNumFeature = auxNumFeature + 1 //Increase for the next feature
      } else if (linesHeader(i).toUpperCase().contains("{") && !(linesHeader(i).toUpperCase().contains("@ATTRIBUTE CLASS"))) {
        //print("\n\n\n que te pillo: " + linesHeader(i))
        var auxParserCategories = 0.0
        val labelsClasses = getLabels(linesHeader(i)) //Array of String with the labels of the objetive variable
        for (key <- labelsClasses) { //Calculate map for parser label categorical attributes
          conv(auxNumFeature) += (key -> auxParserCategories)
          auxParserCategories = auxParserCategories + 1
        }
        
//        val labelsClasses = getLabels(linesHeader(i)) //Array String with the labels of the feature
//        val size = labelsClasses.length
//
//        //Calculate the increase. If categorical variable only have a value (WTF) it must do 0 and the increase 1. Dont /0
//        var inc: Double = 0.0
//        if (size == 1) {
//          inc = 1.0
//        } else {
//          inc = 1 / (size - 1)
//        }
//
//        for (i <- 0 to labelsClasses.length - 1) { //Map to parser the label class
//          conv(auxNumFeature) += (labelsClasses(i) -> i * inc)
//        }
        
        auxNumFeature = auxNumFeature + 1
      }
    } //end for

    /*
    //Muestro todos los map que se forman
    print("\n\n\nMap creado para el parser. Lenght: " + conv.length + "\n\n")

    for (x <- conv){
      println("Keys: " + x.keys)
      println("\nValues: " + x.values + "\n\n\n")
    }
    */

    conv
  }

  /**
   * @brief Parser a line to a Array[Double]
   *
   * @param conv Array[Map] with the information to parser
   * @param line The string to be parsed
   * @author jesus
   */
  def parserToDouble(conv: Array[Map[String, Double]], line: String): Array[Double] = {
    val size = conv.length
    var result: Array[Double] = new Array[Double](size)

    //Change the line to Array[String]
    val auxArray = line.split(",")

    //Iterate over the array parsing to double each element with the knwolegde of the header
    for (i <- 0 to size - 1) {
      if (conv(i).contains("min") && conv(i).contains("max") && (conv(i).size == 2)) { //If dictionary have like key (only) min and max is real or integer, else, categorical
        result(i) = (auxArray(i).toDouble - conv(i).get("min").get) / (conv(i).get("max").get - conv(i).get("min").get)
      } else {
        result(i) = conv(i).get(auxArray(i)).get
      }
    }

    /*
    //Muestro el Array[Double] resultante
    print("\n\n***MUESTRO EL PARSER DOUBLE***\n\n")  

    for (x <- result){
      print(x+",")  
    } 
    */

    result
  }

  
    def parserCategoricalToDouble(conv: Array[Map[String, Double]], line: String): Array[Double] = {
      val size = conv.length
      var result: Array[Double] = new Array[Double](size)
  
      //Change the line to Array[String]
      val auxArray = line.split(",")
  
      //Iterate over the array parsing to double each element with the knwolegde of the header
      for (i <- 0 to size - 1) {
        if (conv(i).contains("min") && conv(i).contains("max") && (conv(i).size == 2)) { //If dictionary have like key (only) min and max is real or integer, else, categorical
          result(i) = auxArray(i).toDouble //- conv(i).get("min").get) / (conv(i).get("max").get - conv(i).get("min").get)
        } else {
          result(i) = conv(i).get(auxArray(i).trim()).get
       }
      }
      
    result
  }
  
  
  def getNumClassFromHeader(sc: SparkContext, file: String): Int = {
    var numClass = 0
    val header = sc.textFile(file)
    var linesHeader = header.toArray()

    //val size = linesHeader.length

    for (i <- 0 to (linesHeader.length - 1)) {
      if (linesHeader(i).toUpperCase().contains("@ATTRIBUTE CLASS")) {
        //print("\n\nLa linea es: " + linesHeader(i) + "\n\n")

        numClass = linesHeader(i).length - linesHeader(i).replaceAllLiterally(",", "").length + 1

        //Muestro el numero de clases que calcula
        //print("Numero de clases: "+numClass+"\n\n\n")

      } //end if
    } //end for

    numClass

  }

}