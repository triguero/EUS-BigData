/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010
	
	F. Herrera (herrera@decsai.ugr.es)
    L. Sánchez (luciano@uniovi.es)
    J. Alcalá-Fdez (jalcala@decsai.ugr.es)
    S. García (sglopez@ujaen.es)
    A. Fernández (alberto.fernandez@ujaen.es)
    J. Luengo (julianlm@decsai.ugr.es)

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see http://www.gnu.org/licenses/
  
**********************************************************************/

package org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles;


/**
 * <p>Title: Selector</p>
 * <p>Description: This class implements an attribute condition of a rule
 * <p>Company: KEEL </p>
 * @author Alberto Fernandez (University of Jaen) 11/10/2012
 * @version 1.1
 * @since JDK1.6
 */
import org.core.Randomize;

public class Selector  implements java.io.Serializable {

  int attribute; //position of the attribute/variable in the dataset
  int operator; // =, <= or >
  public static int EQUAL = 0;
  public static int LESS_EQUAL = 1;
  public static int GREATER = 2;
  double value; //if attribute is real
  transient String nominalValues[]; //if attribute is nominal (for printing)
  double values[]; //if attribute is nominal (checking)
  transient String attNames[];
  transient myDataset train;

  /**
   * Defalt constructor
   */
  public Selector() {
  }

  /**
   * Parameter constructor
   * @param attribute name of the variable
   * @param operator operator type
   * @param value value of the condition
   */
  public Selector(String attribute, String operator, String value) {
    //attribute is type AttX with X == attribute's position
    String numero = new String("" + attribute.charAt(3));
    if (attribute.length() > 4) {
      numero += attribute.charAt(4);
    }
    this.attribute = Integer.parseInt(numero);
    nominalValues = new String[1];
    values = new double[1];
    if (operator.equalsIgnoreCase("=")) {
      this.operator = EQUAL;
      nominalValues[0] = value;
      values[0] = myDataset.valorReal(this.attribute,value);
    }
    else if (operator.equalsIgnoreCase("<=")) {
      this.operator = LESS_EQUAL;
      this.value = Double.parseDouble(value);
    }
    else if (operator.equalsIgnoreCase(">")) {
      this.operator = GREATER;
      this.value = Double.parseDouble(value);
    }
    else {
      System.err.println("There was an error in the parsing of the tree");
      System.exit(0);
    }
  }

  /**
   * Parameter constructor
   * @param attribute variable position
   * @param train full training set
   */
  public Selector(int attribute, myDataset train){
    this.train = train;
    this.attribute = attribute;
    adjuntaNombres(train.nombres());
    if (train.getTipo(attribute) == train.NOMINAL){
      this.operator = EQUAL;
      int totalNominales = train.totalNominales(attribute);
      int nominalesEscogidos = Randomize.RandintClosed(1,totalNominales);
      nominalValues = new String[nominalesEscogidos];
      values = new double[nominalesEscogidos];
      int [] noSeleccionados = new int[totalNominales];
      for (int i = 0; i < totalNominales; i++){
        noSeleccionados[i] = i;
      }
      for (int i = 0; i < nominalValues.length; i++){
        int seleccion = Randomize.RandintClosed(0,totalNominales-1);
        values[i] = 1.0*noSeleccionados[seleccion];
        nominalValues[i] = train.valorNominal(attribute,values[i]);
        noSeleccionados[seleccion] = noSeleccionados[totalNominales-1];
        totalNominales--;
      }
    }else{
      nominalValues = new String[1];
      values = new double[1];
      this.operator = Randomize.RandintClosed(this.LESS_EQUAL, this.GREATER);
      int ejemplo = Randomize.RandintClosed(0, train.size()-1);
      this.value = train.getExample(ejemplo)[attribute];
    }
  }

  public void adjuntaNombres(String[] attributes) {
    attNames = new String[attributes.length];
    attNames = attributes.clone();
  }

  public String printString() {
    String cadena = new String("");
    cadena += " " + attNames[attribute];
    if (operator == EQUAL) {
      cadena += " = {";
      int i;
      for (i = 0; i < values.length - 1; i++) {
        cadena += nominalValues[i] + ", ";
      }
      cadena += nominalValues[i] + "} ";
    }
    else if (operator == LESS_EQUAL) {
      cadena += " <= " + value + " ";
    }
    else {
      cadena += " > " + value + " ";
    }
    return cadena;
  }

  /**
   * Creates a copy of the Selector
   * @return new copy of Selector
   */
  public Selector copy(){
    Selector s = new Selector();
    s.attribute = attribute;
    s.operator =  operator; // =, <= � >
    s.value = value;
    s.nominalValues = new String[nominalValues.length];
    s.nominalValues = nominalValues.clone();
    s.values = new double[values.length];
    s.values = values.clone();
    s.attNames = new String[attNames.length];
    s.attNames = attNames.clone();
    s.train = this.train;
    return s;
  }

  /**
   * Checks if the examples is covered by the selector
   * @param example
   * @return true if the selector covers the example. False otherwise.
   */
  public boolean covers(double[] example) {
    boolean cubierto = false;
    if (this.operator == EQUAL) {
      for (int i = 0; i < values.length; i++) { //Si es EQUAL a alguno
        cubierto = cubierto || (example[attribute] == values[i]);
      }
    }
    else if (this.operator == LESS_EQUAL){
      cubierto = example[attribute] <= value;
    }else{
      cubierto = example[attribute] > value;
    }
    return cubierto;
  }

  public int getattribute(){
    return attribute;
  }

}
