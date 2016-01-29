//
//  SMOTE.java
//
//  Salvador Garc�a L�pez
//
//  Created by Salvador Garc�a L�pez 30-3-2006.
//  Copyright (c) 2004 __MyCompanyName__. All rights reserved.
//

package org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.SMOTE;

import org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.Basic.*;
import org.apache.mahout.keel.Dataset.*;

import org.core.*;

import java.util.StringTokenizer;
import org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.multi_C45;

public class SMOTE extends Metodo {

  /*Own parameters of the algorithm*/
  private long semilla;
  private int kSMOTE;
  private int ASMO;
  private boolean balance;
  private double smoting;

  public SMOTE (String ficheroScript) {
    super (ficheroScript);
  }

  public SMOTE (InstanceSet IS, long seed, int k, int ASMO, boolean bal, double smoting, String distance) {
     int nClases, i, j, l, m, n;
    double VDM;
    int Naxc, Nax, Nayc, Nay;
    double media, SD;

     this.semilla = seed;
     this.training = IS;
     this.test = IS;
     this.kSMOTE = k;
     this.balance = bal;
     this.smoting = smoting;
     this.ASMO = ASMO;
     distanceEu = distance.equalsIgnoreCase("Euclidean")?true:false;
     ficheroSalida = new String[2];
     ficheroSalida[0] = multi_C45.outputTr + "train.tra";
     ficheroSalida[1] = multi_C45.outputTr + "train.tst";

       try {
         /*Normalize and check the data*/
         normalizar();
       }
       catch (Exception e) {
         System.err.println(e);
         System.exit(1);
       }

     /*Previous computation for HVDM distance*/
    if (distanceEu == false) {
      stdDev = new double[Attributes.getInputNumAttributes()];
      nominalDistance = new double[Attributes.getInputNumAttributes()][][];
      nClases = Attributes.getOutputAttribute(0).getNumNominalValues();
      for (i = 0; i < nominalDistance.length; i++) {
        if (Attributes.getInputAttribute(i).getType() == Attribute.NOMINAL) {
          nominalDistance[i] = new double[Attributes.getInputAttribute(i).
              getNumNominalValues()][Attributes.getInputAttribute(i).
              getNumNominalValues()];
          for (j = 0; j < Attributes.getInputAttribute(i).getNumNominalValues();
               j++) {
            nominalDistance[i][j][j] = 0.0;
          }
          for (j = 0; j < Attributes.getInputAttribute(i).getNumNominalValues();
               j++) {
            for (l = j + 1;
                 l < Attributes.getInputAttribute(i).getNumNominalValues(); l++) {
              VDM = 0.0;
              Nax = Nay = 0;
              for (m = 0; m < training.getNumInstances(); m++) {
                if (nominalTrain[m][i] == j) {
                  Nax++;
                }
                if (nominalTrain[m][i] == l) {
                  Nay++;
                }
              }
              for (m = 0; m < nClases; m++) {
                Naxc = Nayc = 0;
                for (n = 0; n < training.getNumInstances(); n++) {
                  if (nominalTrain[n][i] == j && clasesTrain[n] == m) {
                    Naxc++;
                  }
                  if (nominalTrain[n][i] == l && clasesTrain[n] == m) {
                    Nayc++;
                  }
                }
                VDM +=
                    ( ( (double) Naxc / (double) Nax) - ( (double) Nayc / (double) Nay)) *
                    ( ( (double) Naxc / (double) Nax) -
                     ( (double) Nayc / (double) Nay));
              }
              nominalDistance[i][j][l] = Math.sqrt(VDM);
              nominalDistance[i][l][j] = Math.sqrt(VDM);
            }
          }
        }
        else {
          media = 0;
          SD = 0;
          for (j = 0; j < training.getNumInstances(); j++) {
            media += realTrain[j][i];
            SD += realTrain[j][i] * realTrain[j][i];
          }
          media /= (double) realTrain.length;
          stdDev[i] = Math.sqrt( Math.abs((SD / ( (double) realTrain.length)) -
                                (media * media)));
        }
      }
    }

  }

  public void ejecutar () {

    int nPos = 0;
    int nNeg = 0;
    int i, j, l, m;
    int tmp;
    int posID, negID;
    int positives[];
    double conjS[][];
    double conjR[][];
    int conjN[][];
    boolean conjM[][];
    int clasesS[];
    double genS[][];
	double genR[][];
	int genN[][];
	boolean genM[][];
    int clasesGen[];
    int tamS;
    int pos;
    int neighbors[][];
    int nn;

    long tiempo = System.currentTimeMillis();


    /*Count of number of positive and negative examples*/
    for (i=0; i<clasesTrain.length; i++) {
      if (clasesTrain[i] == 0)
        nPos++;
      else
        nNeg++;
    }
    if (nPos > nNeg) {
      tmp = nPos;
      nPos = nNeg;
      nNeg = tmp;
      posID = 1;
      negID = 0;
    } else {
      posID = 0;
      negID = 1;
    }

    /*Localize the positive instances*/
    positives = new int[nPos];
    for (i=0, j=0; i<clasesTrain.length; i++) {
      if (clasesTrain[i] == posID) {
        positives[j] = i;
        j++;
      }
    }

    /*Randomize the instance presentation*/
  //  Randomize.setSeed (semilla);
    for (i=0; i<positives.length; i++) {
      tmp = positives[i];
      pos = Randomize.Randint(0,positives.length-1);
      positives[i] = positives[pos];
      positives[pos] = tmp;
    }

    /*Obtain k-nearest neighbors of each positive instance*/
    neighbors = new int[positives.length][kSMOTE];
    for (i=0; i<positives.length; i++) {
    	switch (ASMO) {
        	case 0:
        		KNN.evaluacionKNN2 (kSMOTE, datosTrain, realTrain, nominalTrain, nulosTrain, clasesTrain, datosTrain[positives[i]], realTrain[positives[i]], nominalTrain[positives[i]], nulosTrain[positives[i]], 2, distanceEu, neighbors[i]);
        		break;
        	case 1:
        		evaluacionKNNClass (kSMOTE, datosTrain, realTrain, nominalTrain, nulosTrain, clasesTrain, datosTrain[positives[i]], realTrain[positives[i]], nominalTrain[positives[i]], nulosTrain[positives[i]], 2, distanceEu, neighbors[i],posID);
        		break;
        	case 2:
        		evaluacionKNNClass (kSMOTE, datosTrain, realTrain, nominalTrain, nulosTrain, clasesTrain, datosTrain[positives[i]], realTrain[positives[i]], nominalTrain[positives[i]], nulosTrain[positives[i]], 2, distanceEu, neighbors[i],negID);
        		break;
    	}
    }

    /*Interpolation of the minority instances*/
    if (balance) {
    	genS = new double[nNeg-nPos][datosTrain[0].length];
    	genR = new double[nNeg-nPos][datosTrain[0].length];
    	genN = new int[nNeg-nPos][datosTrain[0].length];
    	genM = new boolean[nNeg-nPos][datosTrain[0].length];
    	clasesGen = new int[nNeg-nPos];
    } else {
    	genS = new double[(int)(nPos*smoting)][datosTrain[0].length];
    	genR = new double[(int)(nPos*smoting)][datosTrain[0].length];
    	genN = new int[(int)(nPos*smoting)][datosTrain[0].length];
    	genM = new boolean[(int)(nPos*smoting)][datosTrain[0].length];
    	clasesGen = new int[(int)(nPos*smoting)];
    }
    for (i=0; i<genS.length; i++) {
    	clasesGen[i] = posID;
    	nn = Randomize.Randint(0,kSMOTE-1);
    	interpola (realTrain[positives[i%positives.length]],realTrain[neighbors[i%positives.length][nn]],nominalTrain[positives[i%positives.length]],nominalTrain[neighbors[i%positives.length][nn]],nulosTrain[positives[i%positives.length]],nulosTrain[neighbors[i%positives.length][nn]],genS[i],genR[i],genN[i],genM[i]);
    }

	if (balance) {
		tamS = 2*nNeg;
	} else {
		tamS = nNeg + nPos + (int)(nPos*smoting);
	}
   /*Construction of the S set from the previous vector S*/
    conjS = new double[tamS][datosTrain[0].length];
    conjR = new double[tamS][datosTrain[0].length];
    conjN = new int[tamS][datosTrain[0].length];
    conjM = new boolean[tamS][datosTrain[0].length];
    clasesS = new int[tamS];
    for (j=0; j<datosTrain.length; j++) {
      for (l=0; l<datosTrain[0].length; l++) {
        conjS[j][l] = datosTrain[j][l];
        conjR[j][l] = realTrain[j][l];
        conjN[j][l] = nominalTrain[j][l];
        conjM[j][l] = nulosTrain[j][l];
      }
      clasesS[j] = clasesTrain[j];
    }
    for (m=0;j<tamS; j++, m++) {
      for (l=0; l<datosTrain[0].length; l++) {
        conjS[j][l] = genS[m][l];
        conjR[j][l] = genR[m][l];
        conjN[j][l] = genN[m][l];
        conjM[j][l] = genM[m][l];
      }
      clasesS[j] = clasesGen[m];
    }

    System.out.println("SMOTE "+ relation + " " + (double)(System.currentTimeMillis()-tiempo)/1000.0 + "s");

    OutputIS.escribeSalida(ficheroSalida[0], conjR, conjN, conjM, clasesS, entradas, salida, nEntradas, relation);
  //  OutputIS.escribeSalida(ficheroSalida[1], test, entradas, salida, nEntradas, relation);
  }

	public static int evaluacionKNNClass (int nvec, double conj[][], double real[][], int nominal[][], boolean nulos[][], int clases[], double ejemplo[], double ejReal[], int ejNominal[], boolean ejNulos[], int nClases, boolean distance, int vecinos[], int clase) {

		int i, j, l;
		boolean parar = false;
		int vecinosCercanos[];
		double minDistancias[];
		int votos[];
		double dist;
		int votada, votaciones;

		if (nvec > conj.length)
			nvec = conj.length;

		votos = new int[nClases];
		vecinosCercanos = new int[nvec];
		minDistancias = new double[nvec];
		for (i=0; i<nvec; i++) {
			vecinosCercanos[i] = -1;
			minDistancias[i] = Double.POSITIVE_INFINITY;
		}

		for (i=0; i<conj.length; i++) {
			dist = KNN.distancia(conj[i], real[i], nominal[i], nulos[i], ejemplo, ejReal, ejNominal, ejNulos, distance);
			if (dist > 0 && clases[i] == clase) {
				parar = false;
				for (j = 0; j < nvec && !parar; j++) {
					if (dist < minDistancias[j]) {
						parar = true;
						for (l = nvec - 1; l >= j+1; l--) {
							minDistancias[l] = minDistancias[l - 1];
							vecinosCercanos[l] = vecinosCercanos[l - 1];
						}
						minDistancias[j] = dist;
						vecinosCercanos[j] = i;
					}
				}
			}
		}

		for (j=0; j<nClases; j++) {
			votos[j] = 0;
		}

		for (j=0; j<nvec; j++) {
			if (vecinosCercanos[j] >= 0)
				votos[clases[vecinosCercanos[j]]] ++;
		}

		votada = 0;
		votaciones = votos[0];
		for (j=1; j<nClases; j++) {
			if (votaciones < votos[j]) {
				votaciones = votos[j];
				votada = j;
			}
		}

		for (i=0; i<vecinosCercanos.length; i++)
			vecinos[i] = vecinosCercanos[i];

		return votada;
	}

	void interpola (double ra[], double rb[], int na[], int nb[], boolean ma[], boolean mb[], double resS[], double resR[], int resN[], boolean resM[]) {

		int i;
		double diff;
		double gap;
		int suerte;

		for (i=0; i<ra.length; i++) {
			if (ma[i] == true && mb[i] == true) {
				resM[i] = true;
				resS[i] = 0;
			} else if (ma[i] == true){
				if (entradas[i].getType() == Attribute.REAL) {
					resR[i] = rb[i];
					resS[i] = (resR[i] + entradas[i].getMinAttribute()) / (entradas[i].getMaxAttribute() - entradas[i].getMinAttribute());
				} else if (entradas[i].getType() == Attribute.INTEGER) {
					resR[i] = rb[i];
					resS[i] = (resR[i] + entradas[i].getMinAttribute()) / (entradas[i].getMaxAttribute() - entradas[i].getMinAttribute());
				} else {
					resN[i] = nb[i];
					resS[i] = (double)resN[i] / (double)(entradas[i].getNominalValuesList().size() - 1);
				}
			} else if (mb[i] == true) {
				if (entradas[i].getType() == Attribute.REAL) {
					resR[i] = ra[i];
					resS[i] = (resR[i] + entradas[i].getMinAttribute()) / (entradas[i].getMaxAttribute() - entradas[i].getMinAttribute());
				} else if (entradas[i].getType() == Attribute.INTEGER) {
					resR[i] = ra[i];
					resS[i] = (resR[i] + entradas[i].getMinAttribute()) / (entradas[i].getMaxAttribute() - entradas[i].getMinAttribute());
				} else {
					resN[i] = na[i];
					resS[i] = (double)resN[i] / (double)(entradas[i].getNominalValuesList().size() - 1);
				}
			} else {
				resM[i] = false;
				if (entradas[i].getType() == Attribute.REAL) {
					diff = rb[i] - ra[i];
					gap = Randomize.Rand();
					resR[i] = ra[i] + gap*diff;
					resS[i] = (resR[i] + entradas[i].getMinAttribute()) / (entradas[i].getMaxAttribute() - entradas[i].getMinAttribute());
				} else if (entradas[i].getType() == Attribute.INTEGER) {
					diff = rb[i] - ra[i];
					gap = Randomize.Rand();
					resR[i] = Math.round(ra[i] + gap*diff);
					resS[i] = (resR[i] + entradas[i].getMinAttribute()) / (entradas[i].getMaxAttribute() - entradas[i].getMinAttribute());
				} else {
					suerte = Randomize.Randint(0, 2);
					if (suerte == 0) {
						resN[i] = na[i];
					} else {
						resN[i] = nb[i];
					}
					resS[i] = (double)resN[i] / (double)(entradas[i].getNominalValuesList().size() - 1);
				}
			}
		}
	}

  public void leerConfiguracion (String ficheroScript) {

    String fichero, linea, token;
    StringTokenizer lineasFichero, tokens;
    byte line[];
    int i, j;

    ficheroSalida = new String[2];

    fichero = Fichero.leeFichero (ficheroScript);
    lineasFichero = new StringTokenizer (fichero,"\n\r");

    lineasFichero.nextToken();
    linea = lineasFichero.nextToken();

    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    token = tokens.nextToken();

    /*Getting the names of the training and test files*/
    line = token.getBytes();
    for (i=0; line[i]!='\"'; i++);
    i++;
    for (j=i; line[j]!='\"'; j++);
    ficheroTraining = new String (line,i,j-i);
    for (i=j+1; line[i]!='\"'; i++);
    i++;
    for (j=i; line[j]!='\"'; j++);
    ficheroTest = new String (line,i,j-i);

    /*Getting the path and base name of the results files*/
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    token = tokens.nextToken();

    /*Getting the names of output files*/
    line = token.getBytes();
    for (i=0; line[i]!='\"'; i++);
    i++;
    for (j=i; line[j]!='\"'; j++);
    ficheroSalida[0] = new String (line,i,j-i);
    for (i=j+1; line[i]!='\"'; i++);
    i++;
    for (j=i; line[j]!='\"'; j++);
    ficheroSalida[1] = new String (line,i,j-i);

    /*Getting the seed*/
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    semilla = Long.parseLong(tokens.nextToken().substring(1));

    /*Getting the number of neighbors*/
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    kSMOTE = Integer.parseInt(tokens.nextToken().substring(1));

    /*Getting the type of SMOTE algorithm*/
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    token = tokens.nextToken();
    token = token.substring(1);
    if (token.equalsIgnoreCase("both")) ASMO = 0;
    else if (token.equalsIgnoreCase("minority")) ASMO = 1;
    else ASMO = 2;

    /*Getting the type of balancing in SMOTE*/
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    token = tokens.nextToken();
    token = token.substring(1);
    if (token.equalsIgnoreCase("YES")) balance = true;
    else balance = false;

    /*Getting the quantity of smoting*/
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    smoting = Double.parseDouble(tokens.nextToken().substring(1));

    /*Getting the type of distance function*/
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    distanceEu = tokens.nextToken().substring(1).equalsIgnoreCase("Euclidean")?true:false;
  }
}
