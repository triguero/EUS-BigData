/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010

	F. Herrera (herrera@decsai.ugr.es)
    L. S�nchez (luciano@uniovi.es)
    J. Alcal�-Fdez (jalcala@decsai.ugr.es)
    S. Garc�a (sglopez@ujaen.es)
    A. Fern�ndez (alberto.fernandez@ujaen.es)
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


import java.util.ArrayList;
import java.util.StringTokenizer;

/**
 * <p>Title: Rule</p>
 * <p>Description: Rule representation
 * <p>Company: KEEL </p>
 * @author Alberto Fernandez (University of Jaen) 09/10/2012
 * @version 1.1
 * @since JDK1.6
 */
public class Rule  implements java.io.Serializable {

	ArrayList<Selector> antecedent;
	String clase;
	transient myDataset train;
	transient int coveredExamples[];
	transient int positiveCoveredEx[];
	int nCubiertos,nCubiertosOK; //number of covered examples
	float fCubiertos, fCubiertosOK;
	double fitness; //si es una Rule producida por el GA
	int codigoRule; //si es una Rule producida por el GA

	/**
	 * Default constructor
	 */
	public Rule() {
		antecedent = new ArrayList<Selector> ();
		coveredExamples = new int[1];
	}

	/**
	 * Constructor with parameters. It initializes a rule for a given class
	 * @param clase class value
	 * @param train training set
	 */
	public Rule(String clase, myDataset train) {
		antecedent = new ArrayList<Selector> ();
		this.train = train;
		this.clase = clase;
		coveredExamples = new int[train.size()];
		positiveCoveredEx = new int[train.size()];
	}

	/**
	 * Constructor with parameters. It initializes a rule from a C4.5 file of rules (just a single line) 
	 * @param train training set
	 * @param linea Line of the file which stores the C4.5 rule set
	 */
	public Rule(myDataset train, String linea) {
		antecedent = new ArrayList<Selector> ();
		this.train = train;
		coveredExamples = new int[train.size()];
		positiveCoveredEx = new int[train.size()];
		String [] nombres = train.nombres();
		StringTokenizer campo = new StringTokenizer(linea, " ");
		campo.nextToken(); //RULE-X:
		String aux = campo.nextToken(); //IF
		while(!aux.equalsIgnoreCase("THEN")){
			String atributo = campo.nextToken();
			String operador = campo.nextToken();
			String valor = campo.nextToken();
			Selector s = new Selector(atributo,operador,valor);
			s.adjuntaNombres(nombres);
			antecedent.add(s);
			aux = campo.nextToken();
		}
		campo.nextToken(); //class
		campo.nextToken(); //=
		clase = campo.nextToken();
	}

	public void incluyeSelector(Selector s) {
		antecedent.add(s);
	}

	public String printString() {
		String cadena = new String("");
		cadena += "IF ";
		for (int i = 0; i < antecedent.size()-1; i++) {
			cadena += antecedent.get(i).printString()+ "AND ";
		}
		cadena += antecedent.get(antecedent.size()-1).printString();
		cadena += " THEN Class = " + clase + " ("+nCubiertosOK+"/"+nCubiertos+")\n";
		return cadena;
	}

	public String printStringF() {
		String cadena = new String("");
		cadena += "IF ";
		for (int i = 0; i < antecedent.size()-1; i++) {
			cadena += antecedent.get(i).printString()+ "AND ";
		}
		if (antecedent.size() > 0)
			cadena += antecedent.get(antecedent.size()-1).printString();
		cadena += " THEN Class = " + clase + " ("+fCubiertosOK+"/"+fCubiertos+")\n";
		return cadena;
	}

	/**
	 * Creates a copy of a rule
	 * @return a new copy of the rule
	 */
	public Rule copy(){
		Rule r = new Rule(clase, train);
		r.antecedent = new ArrayList<Selector>();
		for (int i = 0; i < antecedent.size(); i++){
			r.antecedent.add(antecedent.get(i).copy());
		}
		r.nCubiertos = nCubiertos;
		r.nCubiertosOK = nCubiertosOK;
		r.coveredExamples = new int[coveredExamples.length];
		r.coveredExamples = coveredExamples.clone();
		r.positiveCoveredEx = new int[positiveCoveredEx.length];
		r.positiveCoveredEx = positiveCoveredEx.clone();
		r.fitness = fitness;
		r.codigoRule = codigoRule;
		return r;
	}

	public int covered(){
		return nCubiertos;
	}

	public int positiveCovered(){
		return nCubiertosOK;
	}

	/**
	 * It counts how many examples are covered and also how many of them are positive
	 */
	public void coverExamples(){
		nCubiertos = nCubiertosOK = 0;
		for (int i = 0; i < train.size(); i++){
			double [] ejemplo = train.getExample(i);
			if (this.covers(ejemplo)){
				coveredExamples[nCubiertos] = i;
				nCubiertos++;
				if (train.getOutputAsString(i).compareToIgnoreCase(this.clase) == 0){
					positiveCoveredEx[nCubiertosOK] = i;
					nCubiertosOK++;
				}
			}
		}
	}


	/**
	 * It checks the weights of the covered examples
	 * @param weights the weights of all examples
	 */
	public void coverExamples(double[] weights){
		fCubiertos = fCubiertosOK = 0;
		for (int i = 0; i < train.size(); i++){
			double [] ejemplo = train.getExample(i);
			if (this.covers(ejemplo)){
				fCubiertos += weights[i];
				if (train.getOutputAsString(i).compareToIgnoreCase(this.clase) == 0){
					fCubiertosOK += weights[i];
				}
			}
		}
		train = null;
		coveredExamples = null;
		positiveCoveredEx = null;
	}

	/**
	 * To compute whether the rule covers an example
	 * @param example
	 * @return True if the rule covers the example. False otherwise.
	 */
	public boolean covers(double [] example){
		boolean cubierto = true;
		for (int i = 0; (i < antecedent.size())&&(cubierto); i++){
			cubierto = cubierto && (antecedent.get(i).covers(example));
		}
		return cubierto;
	}

	public int size(){
		return antecedent.size();
	}

	public boolean contieneAtributo(int att){
		boolean contiene = false;
		for (int i = 0; i < antecedent.size() && !contiene; i++){
			contiene = (antecedent.get(i).attribute == att);
		}
		return contiene;

	}

}
