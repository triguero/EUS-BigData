package testSpark

import org.apache.spark.SparkContext

/**
 * @author david
 */
object keelParser {

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
    //Leemos la cabecera. Cada elemento es una linea
    val header = sc.textFile(file)
    var linesHeader = header.toArray()

    //Calculamos el numero de caracteristicas +1 por la clase
    var numFeatures = 0
    for (i <- 0 to (linesHeader.length - 1)) {
      if (linesHeader(i).toUpperCase().contains("@INPUTS")) {
        //print("\n\nLa linea es: " + linesHeader(i) + "\n\n")
        numFeatures = linesHeader(i).length - linesHeader(i).replaceAllLiterally(",", "").length + 2
        //Muestro el numero de caracteristicas
        //print("Numero de clases: "+numFeatures+"\n\n\n")
      } //end if
    } //end for

    //Calculamos las transformaciones para normalizar y quitar categoricas
    val conv = new Array[Map[String, Double]](numFeatures)
    //Inicializamos el array de map.
    for (i <- 0 to numFeatures - 1) {
      conv(i) = Map()
    }

    var auxParserClasses = 0.0
    var auxNumFeature = 0
    for (i <- 0 to (linesHeader.length - 1)) {
      if (linesHeader(i).toUpperCase().contains("@ATTRIBUTE CLASS")) { //Clase objetivo
        //print("\n\nLa linea es: " + linesHeader(i) + "\n\n")
        val labelsClasses = getLabels(linesHeader(i)) //Obtengo un array de string con los label de las clases
        for (key <- labelsClasses) { //Calculo el map que parseara las etiquetas de clase
          conv(numFeatures - 1) += (key -> auxParserClasses)
          auxParserClasses = auxParserClasses + 1
        }
      } else if (linesHeader(i).toUpperCase().contains("[")) { //Caracteristica real o entera
        val range = getRange(linesHeader(i)) //Minimo y maximo de la caracteristica
        conv(auxNumFeature) += ("min" -> range(0), "max" -> range(1)) //Creo el parser para esa caracteristica
        auxNumFeature = auxNumFeature + 1 //Aumento para completar la siguiente caracteristica
      } else if (linesHeader(i).toUpperCase().contains("{") && !(linesHeader(i).toUpperCase().contains("@ATTRIBUTE CLASS"))) {
        print("\n\n\n que te pillo: " + linesHeader(i))
        val labelsClasses = getLabels(linesHeader(i)) //Obtengo un array de string con los label de la caracteristica
        val size = labelsClasses.length

        //Calculamos el incremento. Si la variable categorica solo tiene un valor (WTF) que valga 0 y el incremento 1. Evito division por cero.
        var inc: Double = 0.0
        if (size == 1) {
          inc = 1.0
        } else {
          inc = 1 / (size - 1)
        }

        for (i <- 0 to labelsClasses.length - 1) { //Calculo el map que parseara las etiquetas de clase
          conv(auxNumFeature) += (labelsClasses(i) -> i * inc)
        }
        auxNumFeature = auxNumFeature + 1 //Aumento para completar la siguiente caracteristica
      }
    } //end for

    //Muestro todos los map que se forman
    print("\n\n\nMap creado para el parser. Lenght: " + conv.length + "\n\n")

    for (x <- conv) {
      println("Keys: " + x.keys)
      println("\nValues: " + x.values + "\n\n\n")
    }

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

    //Tomamos la linea y la pasamos a Array[String]
    val auxArray = line.split(",")

    //Iteramos sobre el array parseando a double cada elemento aprovechando la información de la cabecera
    for (i <- 0 to size - 1) {
      print(i + "\n")
      if (conv(i).contains("min") && conv(i).contains("max") && (conv(i).size == 2)) { //Si tiene la clave min y max y solo esas es real o entera, else, es categorica
        result(i) = (auxArray(i).toDouble - conv(i).get("min").get) / (conv(i).get("max").get - conv(i).get("min").get)
      } else {
        result(i) = conv(i).get(auxArray(i)).get
      }
    }

    //Muestro el Array[Double] resultante
    print("\n***PARSER DOUBLE***\n")
    print("\n" + line + "\n")
    for (x <- result) {
      print(x + ",")
    }

    result
  }

  def getNumClassFromHeader(sc: SparkContext, file: String): Int = {
    var numClass = 0
    val header = sc.textFile(file)
    var linesHeader = header.toArray()

    //val size = linesHeader.length
    print("\n\nlinesHeader.length es: " + linesHeader.length + "\n\n")

    for (i <- 0 to (linesHeader.length - 1)) {
      if (linesHeader(i).toUpperCase().contains("@ATTRIBUTE CLASS")) {
        print("\n\nLa linea es: " + linesHeader(i) + "\n\n")

        numClass = linesHeader(i).length - linesHeader(i).replaceAllLiterally(",", "").length + 1

        //Muestro el numero de clases que calcula
        //print("Numero de clases: "+numClass+"\n\n\n")

      } //end if
    } //end for

    numClass

  }
}