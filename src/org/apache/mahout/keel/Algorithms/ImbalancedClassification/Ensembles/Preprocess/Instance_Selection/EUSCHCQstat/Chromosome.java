package org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Instance_Selection.EUSCHCQstat;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Instance_Selection.EUSCHCQstat.Chromosome;

import org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Basic.KNN;
import org.core.*;

public class Chromosome implements Comparable {

	/*Cromosome data structure*/
	boolean cuerpo[];

	/*Useful data for cromosomes*/
	double calidad;
	boolean cruzado;
	boolean valido;

	boolean prediction[];

	/** 
	 * Construct a random chromosome of specified size
	 * @param size 
	 */
	public Chromosome (int size) {

		double u;
		int i;

		cuerpo = new boolean[size];
		for (i=0; i<size; i++) {
			u = Randomize.Rand();
			if (u < 0.5) {
				cuerpo[i] = false;
			} else {
				cuerpo[i] = true;
			}
		}
		cruzado = true;
		valido = true;
	}

	/** 
	 * It creates a copied chromosome
	 * @param size
	 * @param a Chromosome to copy
	 */
	public Chromosome (int size, Chromosome a) {

		int i;

		cuerpo = a.cuerpo.clone(); 
//                cuerpo new boolean[size];
//		for (i=0; i<cuerpo.length; i++)
//			cuerpo[i] = a.getGen(i);
		calidad = a.getCalidad();
		cruzado = false;
		valido = true;
		prediction = a.prediction.clone();
	}

	/**
	 * It returns a given gen of the chromsome
	 * @param indice
	 * @return
	 */
	public boolean getGen (int indice) {
		return cuerpo[indice];
	}

	/**
	 * Ite returns the fitness of the chrom.
	 * @return
	 */
	public double getCalidad () {
		return calidad;
	}

	/**
	 * It sets a value for a given chrom.
	 * @param indice
	 * @param valor
	 */
	public void setGen (int indice, boolean valor) {
		cuerpo[indice] = valor;
	} 

	/**
	 * Function that evaluates a cromosome
	 */
	public void evalua (double datos[][], double real[][], int nominal[][], boolean nulos[][], int clases[],
                double train[][], double trainR[][], int trainN[][], boolean trainM[][], int clasesT[], 
                String wrapper, int K, String evMeas, boolean MS, boolean pFactor, double P, int posID, int nPos, boolean distanceEu, 
                org.apache.mahout.keel.Dataset.Attribute entradas[], boolean[][] anteriores, boolean[][] salidasAnteriores, ArrayList<Integer> evalStrata, KNN knn) {

		int i, j, m, h;
		int aciertosP = 0, aciertosN = 0;
		int totalP = 0, totalN = 0;
		double beta;
		int claseObt;

		prediction = new boolean[train.length];

		if (MS) {
			h=0;
                        HashMap<Integer, Boolean> mascHM = new HashMap<Integer, Boolean>();
			for (m=0; m<cuerpo.length; m++, h++) {
				for (;h<clasesT.length && clasesT[h]==posID;h++);
				if (cuerpo[m]) { //the instance must be copied to the solution
                                        mascHM.put(h, true);
				}
			}
			for (m=0; m<train.length; m++) {
				if (clasesT[m] == posID) {
                                        mascHM.put(m, true);
				}
			}

//			if (wrapper.equalsIgnoreCase("k-NN")) {
                            int auxK = mascHM.size() < K ? mascHM.size() : K;
                            
//				for (i=0; i<datos.length; i++) {
                                Iterator<Integer> iter = evalStrata.iterator();
                                while(iter.hasNext()){
                                    i = iter.next();
                                    claseObt = knn.evaluacionKNN2(auxK, mascHM, i, clasesT);
					
					if (claseObt >= 0)
						if (clases[i] == claseObt && clases[i] != posID) {
							aciertosN++;
							totalN++;
							prediction[i] = true;
						} else if (clases[i] != claseObt && clases[i] != posID) {
							totalN++;
							prediction[i] = false;
						} else if (clases[i] == claseObt && clases[i] == posID) {
							aciertosP++;
							totalP++;
							prediction[i] = true;
						} else if (clases[i] != claseObt && clases[i] == posID) {
							totalP++;
							prediction[i] = false;
						} 
				}	    		
//			}		    
		} else {
                        HashMap<Integer, Boolean> mascHM = new HashMap<Integer, Boolean>();
			for (j=0; j<train.length; j++) {
				if (cuerpo[j]) { //the instance must be copied to the solution
                                    mascHM.put(j, true);
				}
			}    	
                            int auxK = mascHM.size() < K ? mascHM.size() : K;
//			for (i=0; i<datos.length; i++) {
                        Iterator<Integer> iter = evalStrata.iterator();
                        while(iter.hasNext()){
                                i = iter.next();
                                claseObt = knn.evaluacionKNN2(auxK, mascHM, i, clasesT);
				
				if (claseObt >= 0)
					if (clases[i] == claseObt && clases[i] != posID) {
                                                aciertosN++;
                                                totalN++;
                                                prediction[i] = true;
                                        } else if (clases[i] != claseObt && clases[i] != posID) {
                                                totalN++;
                                                prediction[i] = false;
                                        } else if (clases[i] == claseObt && clases[i] == posID) {
                                                aciertosP++;
                                                totalP++;
                                                prediction[i] = true;
                                        } else if (clases[i] != claseObt && clases[i] == posID) {
                                                totalP++;
                                                prediction[i] = false;
                                        } 
			}				
		}

//		if (evMeas.equalsIgnoreCase("geometric-mean")) {
			calidad = Math.sqrt(((double)aciertosP/(double)totalP)*((double)aciertosN/(double)totalN));			
//		} else if (evMeas.equalsIgnoreCase("auc")) {
//			if (totalP < totalN)
//				calidad = (((double)aciertosP / ((double)totalP)) * ((double)aciertosN / ((double)totalN))) + ((1.0 - ((double)aciertosN / ((double)totalN)))*((double)aciertosP / ((double)totalP)))/2.0 + ((1.0 - ((double)aciertosP / ((double)totalP)))*((double)aciertosN / ((double)totalN)))/2.0;
//			else
//				calidad = (((double)aciertosN / ((double)totalN)) * ((double)aciertosP / ((double)totalP))) + ((1.0 - ((double)aciertosP / ((double)totalP)))*((double)aciertosN / ((double)totalN)))/2.0 + ((1.0 - ((double)aciertosN / ((double)totalN)))*((double)aciertosP / ((double)totalP)))/2.0;			
//		} 
//                else if (evMeas.equalsIgnoreCase(("cost-sensitive"))) {
//			calidad = ((double)totalN - aciertosN) + ((double)totalP - aciertosP) * (double)totalN/(double)totalP;
//			calidad /= (2*(double)totalN);
//			calidad = 1 - calidad;
//		} else if (evMeas.equalsIgnoreCase(("kappa"))) {
//			double sumDiagonales = 0.0, sumTrTc = 0.0;
//			sumDiagonales = aciertosP + aciertosN;
//			sumTrTc = totalP * (totalN - aciertosN) + totalN * (totalP - aciertosP);
//			calidad = (((double)datos.length * sumDiagonales - sumTrTc) / ((double)datos.length * (double)datos.length - sumTrTc));
//		}
//
//		else {
//			precision = (((double)aciertosP / ((double)totalP))) / (((double)aciertosP / ((double)totalP)) + (1.0 - ((double)aciertosN / ((double)totalN))));
//			recall = (((double)aciertosP / ((double)totalP))) / (((double)aciertosP / ((double)totalP)) + (1.0 - ((double)aciertosP / ((double)totalP))));
//			calidad = (2 * precision * recall)/(recall + precision);
//		}

		if (pFactor) {
			if (MS) {
				beta = (double)genesActivos()/(double)nPos;				
			} else {
				beta = (double)genes0Activos(clasesT)/(double)genes1Activos(clasesT);				
			}
			calidad -= Math.abs(1.0-beta)*P;
		}

		if (anteriores[0] != null) {
			/* Calcular la distancia de Hamming mínima entre el cromosoma y anteriores[][] */
			double q = -Double.MAX_VALUE;
			for (i = 0; i < anteriores.length && anteriores[i] != null; i++) {
				double qaux = Qstatistic(anteriores[i], cuerpo, clases.length);
				if (q < qaux)
					q = qaux;
			}
			double peso = (double)(anteriores.length - i) / (double) (anteriores.length);
			double IR = 0;
                        if (MS)
                            IR = (double)cuerpo.length / (double)nPos * 0.1;
                        else
                            IR = (double)(cuerpo.length - nPos) / (double)nPos * 0.1;
			calidad = calidad * (1.0 / peso) * (1.0 / IR) - q * peso;
		}
		cruzado = false;
	}


	private double Qstatistic(boolean[] v1, boolean[] v2, int n) {
		double[][] t = new double[2][2];
		double ceros = 0;
		if (v1.length < n)
			n = v1.length;
		for (int i = 0; i < n; i++) {
			if (v1[i] == v2[i] && v1[i] == true)
				t[0][0]++;
			else if (v1[i] == v2[i] && v1[i] == false)
				t[1][1]++;
			else if (v1[i] != v2[i] && v1[i] == true)
				t[1][0]++;
			else
				t[0][1]++;
			if (!v2[i])
				ceros++;
		}
		if (ceros == n)
			return 2.0;
		return (t[1][1] * t[0][0] - t[0][1] * t[1][0]) / (t[1][1] * t[0][0] + t[0][1] * t[1][0]);
	}

	/**
	 * Function that does the CHC diverge
	*/
	public void divergeCHC (double r, Chromosome mejor, double prob) {

		int i;

		for (i=0; i<cuerpo.length; i++) {
			if (Randomize.Rand() < r) {
				if (Randomize.Rand() < prob) {
					cuerpo[i] = true;
				} else {
					cuerpo[i] = false;
				}
			} else {
				cuerpo[i] = mejor.getGen(i);
			}
		}
		cruzado = true;
	}

	public boolean estaEvaluado () {
		return !cruzado;
	}

	public int genesActivos () {

		int i, suma = 0;

		for (i=0; i<cuerpo.length; i++) {
			if (cuerpo[i]) suma++;
		}

		return suma;
	}

	public int genes0Activos (int clases[]) {

		int i, suma = 0;

		for (i=0; i<cuerpo.length; i++) {
			if (cuerpo[i] && clases[i] == 0) suma++;
		}

		return suma;
	}

	public int genes1Activos (int clases[]) {

		int i, suma = 0;

		for (i=0; i<cuerpo.length; i++) {
			if (cuerpo[i] && clases[i] == 1) suma++;
		}

		return suma;
	}

	public boolean esValido () {
		return valido;
	}

	public void borrar () {
		valido = false;
	}

	/**
	 * Function that lets compare cromosomes for an easilier sort
	*/
	public int compareTo (Object o1) {

		if (this.calidad > ((Chromosome)o1).calidad)
			return -1;
		else if (this.calidad < ((Chromosome)o1).calidad)
			return 1;
		else return 0;
	}

	/**
	 * Prints the chrosome into a string value
	 */
	public String toString() {

		int i;
		String temp = "[";

		for (i=0; i<cuerpo.length; i++)
			if (cuerpo[i])
				temp += "1";
			else
				temp += "0";
		temp += ", " + String.valueOf(calidad) + ", " + String.valueOf(genesActivos()) + "]";

		return temp;
	}


	@Override
        public boolean equals(Object o1)
        {
            if (this.calidad == ((Chromosome)o1).calidad)
                return true;
            else
                return false;
        }

}

