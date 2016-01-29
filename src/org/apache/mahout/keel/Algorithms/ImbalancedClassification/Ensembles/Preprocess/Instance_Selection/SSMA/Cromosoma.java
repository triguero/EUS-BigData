package org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Instance_Selection.SSMA;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.mahout.keel.Algorithms.Preprocess.Basic.KNN;
import org.apache.mahout.keel.Algorithms.Sarah.AUC.CalculateAUC;
import org.apache.mahout.keel.Algorithms.Sarah.AUC.PosProb;
import org.core.Randomize;

public class Cromosoma implements Comparable<Cromosoma> {
	
	/*Cromosome data structure*/
    boolean cuerpo[];

    /*Index for nearest neighbours*/
    int vecinos[][];

    /*Useful data for cromosomes*/
    double fitness;
    double fitnessAUC;
    boolean evaluado;
    boolean valido;

    double selValue;

    /**
     * Builder. Construct a random chromosome of specified size
     *
     * @param K Number of neighbors of the KNN algorithm
     * @param size Size of the chromosome
     * @param dMatrix Distance matrix
     * @param datos Reference to the training set
     * @param real  Reference to the training set (real valued)
     * @param nominal  Reference to the training set (nominal valued)	 
     * @param nulo  Reference to the training set (null values)	 	 
     * @param distanceEu True= Euclidean distance; False= HVDM
     * @param posIndices
     * @param negIndices
     */
    public Cromosoma (int K, int size, double dMatrix[][], 
            double datos[][], double real[][], int nominal[][],
            boolean nulo[][], boolean distanceEu, int[] posIndices,
            int[] negIndices) {

            double u;
            //int i, j;

            cuerpo = new boolean[size];
            vecinos = new int[size][K];
            for(int i = 0; i < vecinos.length; i++){
              Arrays.fill(vecinos[i], -1);
            }
            for (int i=0; i<size; i++) {
                    u = Randomize.Rand();
                    if (u < 0.5) {
                            cuerpo[i] = false;
                    } else {
                            cuerpo[i] = true;
                    }
            }
            evaluado = false;
            valido = true;

            for (int i=0; i<size; i++) {
                    for (int j=0; j<K; j++) {
                            vecinos[i][j] = obtenerCercano(vecinos[i],j,
                                    dMatrix, i, datos, real, nominal, 
                                    nulo, distanceEu, 
                                    posIndices, negIndices);
                    }
            }

    }//end-method

    /**
     * Builder. Copies a chromosome of specified size
     *
     * @param K Number of neighbors of the KNN algorithm
     * @param size Size of the chromosome
     * @param a Chromosome to copy
     */
    public Cromosoma (int K, int size, Cromosoma a) {

        cuerpo = new boolean[size];
        vecinos = new int[size][K];
        for(int i = 0; i < vecinos.length; i++){
              Arrays.fill(vecinos[i], -1);
        }
        for (int i=0; i<cuerpo.length; i++) {
            cuerpo[i] = a.getGen(i);
            for (int j=0; j<K; j++) {
                    vecinos[i][j] = a.getVecino(i,j);
            }
        }
        fitness = a.getFitness();
        fitnessAUC = a.getFitnessAUC();
        selValue = a.getSel();
        evaluado = true;
        valido = true;

    }//end-method

    /**
     * Builder. Creates a chromosome from two parents
     *
     * @param K Number of neighbors of the KNN algorithm
     * @param a First chromosome
     * @param b Second chromosome
     * @param pCross Probability of crossing
     * @param size Size of the chromosome
     * @param minSize
     * @param majSize
     */
    public Cromosoma (int K, Cromosoma a, Cromosoma b, 
            double pCross, int size, int minSize, int majSize) {

        cuerpo = new boolean[size];
        vecinos = new int[size][K];
        for(int i = 0; i < vecinos.length; i++){
              Arrays.fill(vecinos[i], -1);
        }

        /*
         * We order the indices of the positive instances.
         * Those set to 1 in parent with highest fitness are
         * considered first, in a random order and then we continue
         * with those set to 0, also in a random order.
         */
         Cromosoma fittest;
         if(a.getFitness() > b.getFitness()){
             fittest = a;
         } else {
             fittest = b;
         }

         // Shuffle
         int[] baraje = new int[minSize];
         int pos;
         int tmp;
         for (int j=0; j<minSize; j++)
              baraje[j] = j;
         for (int j=0; j<minSize; j++) {
              pos = Randomize.Randint (j, minSize-1);
              tmp = baraje[j];
              baraje[j] = baraje[pos];
              baraje[pos] = tmp;
          }

         int[] sortedIndices = new int[minSize];
         int front = 0;
         int back = minSize - 1;
         for(int j=0; j < baraje.length; j++){
             if(fittest.getGen(baraje[j])){ // Gene set to 1
                 sortedIndices[front] = baraje[j];
                 front++;
             } else { // Gene set to 0
                 sortedIndices[back] = baraje[j];
                 back--;
             }
         }

         // Majority part
        int majInChild = 0;
        for (int j=minSize; j<minSize + majSize; j++) {
              if (Randomize.Rand() < pCross) {
                    cuerpo[j] = b.getGen(j);
              } else {
                    cuerpo[j] = a.getGen(j);
              }            

              if(cuerpo[j]){
                  majInChild++;
              }
        }

        // Minority part
        int minInChild = 0;
        for (int j=0; j<sortedIndices.length; j++) {
              int index = sortedIndices[j];

              /*
               * Minority genes are set to 1 (provided at least one of the
               * parents has this gene set to 1) for as long as there are 
               * fewer active minority than majority genes.
               */
              if((minInChild < majInChild) 
                      && (a.getGen(index) || b.getGen(index))){
                  cuerpo[index] = true;
                  minInChild++;
              } else {
                  if (Randomize.Rand() < pCross) {
                    cuerpo[index] = b.getGen(index);
                  } else {
                    cuerpo[index] = a.getGen(index);
                  } 
              }

        }

        evaluado = false;
        valido = true;

    }//end-method

    /**
     * Mutation operator
     *
     * @param K Number of neighbors of the KNN algorithm
     * @param pMut Mutation probability 
     * @param dMatrix Distance matrix
     * @param datos Reference to the training set
     * @param real  Reference to the training set (real valued)
     * @param nominal  Reference to the training set (nominal valued)	 
     * @param nulo  Reference to the training set (null values)	 	 
     * @param distanceEu True= Euclidean distance; False= HVDM
     * @param posIndices
     * @param negIndices
     */
    public void mutation (int K, double pMut, double dMatrix[][], 
            double datos[][], double real[][], int nominal[][], 
            boolean nulo[][], boolean distanceEu, int[] posIndices, 
            int[] negIndices) {

            int i, j;

            // Class distr in S
            int nMin = 0;
            int nMaj = 0;
            for(i = 0; i < posIndices.length; i++){
                if(cuerpo[i]){
                    nMin++;
                }
            }
            int offset = posIndices.length;
            for(i = 0; i < negIndices.length; i++){
                if(cuerpo[offset + i]){
                    nMaj++;
                }
            }

            boolean switchClasses = nMin > nMaj;

            // Inverse of the IR of S
            int majority = Math.max(nMaj, nMin);
            int minority = Math.min(nMaj, nMin);

            double invIRS;
            if(minority == 0){
                invIRS = 0;
            } else {
                invIRS = (double) minority / majority;
            }

            double pSmall = invIRS * pMut;
            double pLarge = (2.0 - invIRS) * pMut;

            // Part 1
            for (i=0; i<posIndices.length; i++) {
                if(switchClasses){  // This is the majority class
                  if (cuerpo[i]) {
                        if (Randomize.Rand() < pLarge) {
                          cuerpo[i] = false;
                        }
                  } else {
                        if (Randomize.Rand() < pSmall) {
                          cuerpo[i] = true;
                        }
                  }
                } else {   // This is the minority class
                   if (cuerpo[i]) {
                        if (Randomize.Rand() < pSmall) {
                          cuerpo[i] = false;
                        }
                  } else {
                        if (Randomize.Rand() < pLarge) {
                          cuerpo[i] = true;
                        }
                  }                       
                }
            }

            // Part 2
            for (i=offset; i<offset + negIndices.length; i++) {
                if(switchClasses){  // This is the minority class
                      if (cuerpo[i]) {
                        if (Randomize.Rand() < pSmall) {
                          cuerpo[i] = false;
                        }
                      } else {
                        if (Randomize.Rand() < pLarge) {
                          cuerpo[i] = true;
                        }
                  }                       
                } else {           // This is the majority class
                      if (cuerpo[i]) {
                        if (Randomize.Rand() < pLarge) {
                          cuerpo[i] = false;
                        }
                      } else {
                        if (Randomize.Rand() < pSmall) {
                          cuerpo[i] = true;
                        }
                      }                        
                }

            }

            for (i=0; i<cuerpo.length; i++) {
                  for (j=0; j<K; j++) {
                          vecinos[i][j] = 
                                  obtenerCercano(vecinos[i],j,dMatrix, 
                                  i, datos, real, nominal, nulo, 
                                  distanceEu, posIndices, negIndices);
                  }
            }

    }//end-method


    /**
     * Obtain the nearest neighbour given a mask (cromosome)
     *
     * @param vecinos Array of neighbors
     * @param J instance to search
     * @param dMatrix Distance matrix
     * @param index Index of the chromosome of reference
     * @param datos Reference to the training set
     * @param real  Reference to the training set (real valued)
     * @param nominal  Reference to the training set (nominal valued)	 
     * @param nulo  Reference to the training set (null values)	 	 
     * @param distanceEu True= Euclidean distance; False= HVDM
     * @param posIndices
     * @param negIndices
     *
     * @return Nearest instance to J
     */
    public int obtenerCercano (int vecinos[], int J,
            double dMatrix[][], int index, double datos[][], 
            double real[][], int nominal[][], boolean nulo[][], 
            boolean distanceEu, int[] posIndices, int[] negIndices) {

            double minDist;
            int minPos, i, j;
            double dist;
            boolean perfect, cont;

            int offset = posIndices.length;
            int actualIndex;

            int refIndex;
            if(index < posIndices.length){ // Determine the index in the dataset
                refIndex = posIndices[index];
            } else {
                refIndex = negIndices[index - offset];
            }

            if (dMatrix == null) {
              perfect = false;
              i = 0;
              do {
                    for ( ; i < cuerpo.length && !cuerpo[i]; i++);
                    cont = true;
                    for (j=0; j<J && cont; j++) {
                      if (vecinos[j] == i) {
                            cont = false;
                            i++;
                      }
                    }
                    perfect = cont;
              } while (!perfect);
              minPos = i;
              if (minPos == cuerpo.length)
                    return -1;
              if(minPos < posIndices.length){    // Find element represented by this gene
                  actualIndex = posIndices[minPos];
              } else {
                  actualIndex = negIndices[minPos - offset];
              }

              minDist = KNN.distancia(datos[refIndex],real[refIndex], 
                      nominal[refIndex], nulo[refIndex], datos[actualIndex], 
                      real[actualIndex], nominal[actualIndex], nulo[actualIndex], 
                      distanceEu);
              for (i=minPos+1; i<cuerpo.length; i++) {
                    if (cuerpo[i]) {
                      cont = true;
                      for (j=0; j<J && cont; j++) {
                            if (vecinos[j] == i) {
                              cont = false;
                            }
                      }
                      if (cont) {
                            if(i < posIndices.length){    // Find element represented by this gene
                                  actualIndex = posIndices[i];
                            } else {
                                  actualIndex = negIndices[i - offset];
                            }
                            dist = KNN.distancia(datos[refIndex],real[refIndex], 
                                    nominal[refIndex], nulo[refIndex], 
                                    datos[actualIndex], real[actualIndex], 
                                    nominal[actualIndex], nulo[actualIndex], 
                                    distanceEu);
                            if (minDist > dist) {
                              minPos = i;
                              minDist = dist;
                            }
                      }
                    }
              }
            } else {
              perfect = false;
              i = 0;
              do {
                    for (; i < cuerpo.length && !cuerpo[i]; i++);
                    cont = true;
                    for (j=0; j<J && cont; j++) {
                      if (vecinos[j] == i) {
                            cont = false;
                            i++;
                      }
                    }
                    perfect = cont;
              } while (!perfect);
              minPos = i;
              if (minPos == cuerpo.length)
                    return -1;
              if(minPos < posIndices.length){    // Find element represented by this gene
                  actualIndex = posIndices[minPos];
              } else {
                  actualIndex = negIndices[minPos - offset];
              }
              minDist = dMatrix[refIndex][actualIndex];
              for (i=minPos+1; i<cuerpo.length; i++) {
                    if (cuerpo[i]) {
                      cont = true;
                      for (j=0; j<J && cont; j++) {
                            if (vecinos[j] == i) {
                              cont = false;
                            }
                      }
                      if (cont) {
                            // Find element represented by this gene
                            if(i < posIndices.length){  
                                actualIndex = posIndices[i];
                            } else {
                                actualIndex = negIndices[i - offset];
                            }
                            if (minDist > dMatrix[refIndex][actualIndex]) {
                              minPos = i;
                              minDist = dMatrix[refIndex][actualIndex];
                            }
                      }
                    }
              }
            }

            if (minPos == cuerpo.length){
                return -1;
            } else {
                return minPos;
            }

    }//end-method

    /**
     * Get the value of a gene
     *
     * @param indice Index of the gene
     *
     * @return Value of the especified gene
     */
    public boolean getGen (int indice) {
            return cuerpo[indice];

    }//end-method


    /**
     * Get the j-neighbour of a given instance
     *
     * @param indicei Instance to search
     * @param indicej Order of the neighbor
     *
     * @return Index to the neighbor found
     */
    public int getVecino (int indicei, int indicej) {
                    return vecinos[indicei][indicej];

    }//end-method

    /**
     * Get the quality of a chromosome
     *
     * @return Quality of the chromosome
     */
    public double getSel () {
            return selValue;

    }//end-method

    /**
     * Get the fitness of a chromosome
     *
     * @return Fitness of the chromosome
     */
    public double getFitness () {
            return fitness;

    }//end-method

    /**
     * Get the accuracy fitness of a chromosome
     *
     * @return Accuracy fitness  of the chromosome
     */
    public double getFitnessAUC () {
            return fitnessAUC;

    }//end-method

    /**
     * Performs a full evaluation of a chromosome
     *
     * @param nClases Number of clases
     * @param K Number of neighbors of the KNN algorithm
     * @param origIR
     * @param posClass
     * @param P
     * @param posSize
     * @param negSize
     */
    public void evaluacionCompleta (int nClases, int K,  
            double origIR, int posClass, 
            double P, int posSize, int negSize, boolean useFscore, 
            ArrayList<Integer> elsToEvaluatePos, ArrayList<Integer>  elsToEvaluateNeg) {

            int votos[];
            int claseObt=0, maxValue;
            int claseReal;
            double prob;

            votos = new int[nClases];

            // Class distr in S
            int nMin = 0;
            int nMaj = 0;
            for(int i = 0; i < posSize; i++){
                if(cuerpo[i]){
                    nMin++;
                }
            }
            for(int i = 0; i < negSize; i++){
                if(cuerpo[posSize + i]){
                    nMaj++;
                }
            }

            // Evaluation measures
            int TP = 0;
            int TN = 0;
            int FP = 0;
            PosProb[] valsForAUC = new PosProb[elsToEvaluatePos.size() + elsToEvaluateNeg.size()];


            //positive
            for(int lala = 0; lala < elsToEvaluatePos.size(); lala++){            	
            	 int i = elsToEvaluatePos.get(lala);            	
                  // kNN classifier
                  Arrays.fill(votos,0);
                  int realK = numberOfValidNeighbors(vecinos[i]);
                  for (int j=0; j<realK; j++) {
                      if(vecinos[i][j] < posSize){
                          votos[posClass]++;
                      } else {
                          votos[posClass ^ 1]++;
                      }
                  }
                  maxValue = 0;
                  claseObt = -1;
                  for (int j=0; j<nClases; j++) {
                        if (votos[j] > maxValue) {
                          maxValue = votos[j];
                          claseObt = j;
                        }
                  }
               claseReal = posClass;
                if(claseObt != -1){  // Classification successful
                  if (claseReal == claseObt && claseReal == posClass){
                      TP++;
                  } else if(claseReal == claseObt){
                      TN++;
                  } else if(claseReal != posClass){
                      FP++;
                  }
                }

                  prob = 0.0;
                  if(realK > 0){
                      prob = (double) votos[posClass] / realK;
                  }                      
                  valsForAUC[lala] = new PosProb(claseReal == posClass, prob);

            }
           //negative
            for(int lala = 0; lala < elsToEvaluateNeg.size(); lala++){
            	
            	int i = posSize + elsToEvaluateNeg.get(lala);
            	
                  // kNN classifier
                  Arrays.fill(votos,0);
                  int realK = numberOfValidNeighbors(vecinos[i]);
                  for (int j=0; j<realK; j++) {
                      if(vecinos[i][j] < posSize){
                          votos[posClass]++;
                      } else {
                          votos[posClass ^ 1]++;
                      }
                  }
                  maxValue = 0;
                  claseObt = -1;
                  for (int j=0; j<nClases; j++) {
                        if (votos[j] > maxValue) {
                          maxValue = votos[j];
                          claseObt = j;
                        }
                  }

               claseReal = posClass ^ 1;


                if(claseObt != -1){  // Classification successful
                  if (claseReal == claseObt && claseReal == posClass){
                      TP++;
                  } else if(claseReal == claseObt){
                      TN++;
                  } else if(claseReal != posClass){
                      FP++;
                  }
                }


                  prob = 0.0;
                  if(realK > 0){
                      prob = (double) votos[posClass] / realK;
                  }                      
                  valsForAUC[elsToEvaluatePos.size() + lala] 
                		  = new PosProb(claseReal == posClass, prob);

            }
            
            double auc = CalculateAUC.calculate(valsForAUC);                

            double TPrate =  (double) TP / posSize;
            double TNrate = (double) TN / negSize;
            double gmean = Math.sqrt(TPrate * TNrate);
            
            // use F-score instead of
            double recall =  TPrate;
            double precision = (double) TP / (TP + FP);
            double f_score = 2 * recall * precision / (recall + precision);

            // Inverse of the IR of S
            int majority = Math.max(nMaj, nMin);
            int minority = Math.min(nMaj, nMin);

            double invIRS;
            if(minority == 0){
                invIRS = 0;
            } else {
                invIRS = (double) minority / majority;
            }

            double valueForSel;

            // Fitness evaluation
            if(nMaj == 0){
                
                if(useFscore){
                    fitness = f_score - P;
                    valueForSel = f_score; 
                } else {
                    fitness = gmean - P;
                    valueForSel = gmean;
                }
            } else {
                
                if(useFscore){
                    fitness = f_score - Math.abs(1 - invIRS) * P;
                    valueForSel = f_score;
                } else {
                    fitness = gmean - Math.abs(1 - invIRS) * P;
                    valueForSel = gmean;
                }
            }

            evaluado = true;

            // Calculating the value for Sel(.)
            double denom = valueForSel + invIRS;
            if(denom == 0){ // both equal zero, so harmonic mean is also zero
                selValue = 0;
            } else {
                selValue = (2.0 * valueForSel * invIRS) / denom; 
            }

            fitnessAUC = auc;

    }//end-method

    /**
     * Tests if the chromosome is valid
     *
     * @return True if the chromosome is valid. False, if not.
     */	
    public boolean esValido () {
            return valido;

    }//end-method

    /**
     * Marks a chromosome for deletion
     */	
    public void borrar () {
            valido = false;

    }//end-method

    /**
     * Set the value of a gene
     *
     * @param pos Index of the gene
     * @param valor Value to set
     */
    public void setGen (int pos, boolean valor) {
            cuerpo[pos] = valor;

    }//end-method

    /**
     * Tests if the chromosome is already evaluated
     *
     * @return True if the chromosome is already evaluated. False, if not.
     */	
    public boolean estaEvaluado () {

            return evaluado;

    }//end-method

    /**
     * Count the number of genes set to 1
     *
     * @return Number of genes set to 1 in the chromosome
     */	
    public int genesActivos () {
            int i, suma = 0;

            for (i=0; i<cuerpo.length; i++) {
              if (cuerpo[i]) suma++;
            }

            return suma;

    }//end-method

    /**
     * Performs the local search procedure of SSMA
     *
     * @param nClases Number of clases
     * @param K Number of neighbors of the KNN algorithm
     * @param clases Output attribute of the instances
     * @param dMatrix Distance matrix
     * @param umbral Current threshold
     * @param datos Reference to the training set
     * @param real  Reference to the training set (real valued)
     * @param nominal  Reference to the training set (nominal valued)	 
     * @param nulo  Reference to the training set (null values)	 	 
     * @param distanceEu True= Euclidean distance; False= HVDM
     * @param posIndices
     * @param negIndices
     * @param posClass
     * @param origIR
     * @param P
     *
     * @return Amount of evaluations spent
     */
public double optimizacionLocal (int nClases, int K, int clases[], 
        double dMatrix[][], double umbral, double datos[][], 
        double real[][], int nominal[][], boolean nulo[][], 
        boolean distanceEu, int[] posIndices, int[] negIndices, int posClass, 
        double origIR, double P, boolean useFscore, 
        ArrayList<Integer>  elsToEvaluatePos, ArrayList<Integer>  elsToEvaluateNeg) {

              int pos, tmp;
              double evaluaciones = 0;
              double ev;

              int offset = posIndices.length;

              // Class distr in S
              int nPos = 0;
              int nNeg = 0;
              for(int i = 0; i < offset; i++){
                    if(cuerpo[i]){
                        nPos++;
                    } 
              }
              for(int i = 0; i < negIndices.length; i++){
                    if(cuerpo[offset + i]){
                        nNeg++;
                    }
              }

              if(nPos == nNeg && nPos == 0){  
                  /*
                   * IR of S is 1, but S is empty.
                   * The optimization is achieved by adding a random 
                   * element of each class.
                   */
                  int posPos = Randomize.Randint (0, offset-1);
                  cuerpo[posPos] = true;
                  int posNeg = Randomize.Randint (offset, cuerpo.length-1);
                  cuerpo[posNeg] = true;
                  return 0.0;
              }

              if(nPos == nNeg){   // IR of S is 1, no optimization
                  return 0.0;
              }

              boolean switchClasses = nPos > nNeg;

              // Genes whose value will be changed
              int[] toBeChanged;
              int majority;
              int minority;
              int posClassS;
              if(switchClasses){

                  majority = nPos;
                  minority = nNeg;
                  posClassS = posClass ^ 1;

                  /*
                   * Set positive instance to 0 and negative instances to 1.
                   */
                  toBeChanged = new int[nPos + negIndices.length - nNeg];
                  int l = 0;
                  for(int i = 0; i < offset; i++){
                      if(cuerpo[i]){
                          toBeChanged[l] = i;
                          l++;
                      }
                  }
                  for(int i = offset; i < cuerpo.length; i++){
                      if(!cuerpo[i]){
                          toBeChanged[l] = i;
                          l++;
                      }
                  }
              } else {

                  majority = nNeg;
                  minority = nPos;
                  posClassS = posClass;

                  /*
                   * Set negative instance to 0 and positive instances to 1.
                   */
                  toBeChanged = new int[nNeg + posIndices.length - nPos];
                  int l = 0;
                  for(int i = offset; i < cuerpo.length; i++){
                      if(cuerpo[i]){
                          toBeChanged[l] = i;
                          l++;
                      }
                  }

                  for(int i = 0; i < offset; i++){
                      if(!cuerpo[i]){
                          toBeChanged[l] = i;
                          l++;
                      }
                  }
              }

              // Shuffle
              for (int j=0; j<toBeChanged.length; j++) {
                      pos = Randomize.Randint (j, toBeChanged.length-1);
                      tmp = toBeChanged[j];
                      toBeChanged[j] = toBeChanged[pos];
                      toBeChanged[pos] = tmp;
              }

              // Consider all instances that can be changed
              boolean cont = true;
              int c = 0;
              while(cont && c < toBeChanged.length){
                  int el = toBeChanged[c];
                  ev = evaluacionParcial(nClases, K, clases, el, 
                              dMatrix, umbral, datos, real, nominal, nulo, 
                              distanceEu, posIndices, negIndices, posClass, 
                              origIR, P, useFscore,
                              elsToEvaluatePos,elsToEvaluateNeg);
                  evaluaciones += Math.abs(ev);
                  if(ev >= 0){  // Change was made, check IR
                      if((el < offset && posClassS == posClass) 
                              || (el >= offset && posClassS != posClass)){
                          minority++;
                      } else {
                          majority--;
                      }
                      cont = majority > minority;
                  }
                  c++;
              }
              return evaluaciones;

    }//end-method

    /**
     * Performs ta partial evaluation
     *
     * @param nClases Number of clases
     * @param K Number of neighbors of the KNN algorithm
     * @param clases Output attribute of the instances
     * @param ref Instance adjusted
     * @param dMatrix Distance matrix
     * @param umbral Current threshold
     * @param datos Reference to the training set
     * @param real  Reference to the training set (real valued)
     * @param nominal  Reference to the training set (nominal valued)	 
     * @param nulo  Reference to the training set (null values)	 	 
     * @param distanceEu True= Euclidean distance; False= HVDM
     * @param posIndices
     * @param negIndices
     * @param posClass
     * @param origIR
     * @param P
     *
     * @return Amount of evaluations spent
     */
    public double evaluacionParcial (int nClases, int K, int clases[], 
            int ref, double dMatrix[][], double umbral, double datos[][], 
            double real[][], int nominal[][], boolean nulo[][], 
            boolean distanceEu, int[] posIndices, int[] negIndices, 
            int posClass, double origIR, double P, boolean useFscore, 
            ArrayList<Integer>  elsToEvaluatePos, ArrayList<Integer>  elsToEvaluateNeg) {

          int vecinosTemp[][];
          double ganancia = 0; //an instance just been dropped
          int votos[];
          double probNuevo, probAnterior;
          PosProb[] valsAUCAnterior = new PosProb[cuerpo.length];
          PosProb[] valsAUCNuevo = new PosProb[cuerpo.length];
          boolean isPositive;

          votos = new int[nClases];
          vecinosTemp = new int[cuerpo.length][K];
          for(int i = 0; i < vecinosTemp.length; i++){
              Arrays.fill(vecinosTemp[i], -1);
          }

          cuerpo[ref] = !cuerpo[ref]; // Change value of this gene
          for (int i=0; i<cuerpo.length; i++) {
                  for (int j=0; j<K; j++) {
                          vecinosTemp[i][j] = 
                                  obtenerCercano(vecinosTemp[i],
                                  j, dMatrix, i, datos, real, 
                                  nominal, nulo, distanceEu,
                                  posIndices, negIndices);
                  }

                  isPositive = i < posIndices.length;

                  // Anterior
                  Arrays.fill(votos, 0);
                  int realKAnterior = numberOfValidNeighbors(vecinos[i]);
                  for (int j = 0; j < realKAnterior; j++) {
                      if(vecinos[i][j] < posIndices.length){
                          votos[posClass]++;
                      } else {
                          votos[posClass ^ 1]++;
                      }
                  }

                  probAnterior = 0.0;
                  if (realKAnterior > 0){
                      probAnterior = (double) votos[posClass] / realKAnterior;
                  }
                  valsAUCAnterior[i] = 
                          new PosProb(isPositive, probAnterior);

                  // Nuevo
                  Arrays.fill(votos, 0);
                  int realKNuevo = numberOfValidNeighbors(vecinosTemp[i]);
                  for (int j = 0; j < realKNuevo; j++) {
                      if(vecinosTemp[i][j] < posIndices.length){
                          votos[posClass]++;
                      } else {
                          votos[posClass ^ 1]++;
                      }
                  }

                  probNuevo = 0.0;
                  if(realKNuevo > 0){
                      probNuevo = (double) votos[posClass] / realKNuevo;
                  }
                  valsAUCNuevo[i] = 
                          new PosProb(isPositive, probNuevo);

          }

          double aucAnterior = CalculateAUC.calculate(valsAUCAnterior);
          double aucNuevo = CalculateAUC.calculate(valsAUCNuevo);
          ganancia = aucNuevo - aucAnterior;

          if (Math.round(ganancia) >= (double)umbral) {
              for (int i=0; i<cuerpo.length; i++) {
                      for (int j=0; j<K; j++) {
                              vecinos[i][j] = vecinosTemp[i][j];	    		  
                      }
              }
              evaluacionCompleta (nClases, K, origIR, 
                      posClass, P, 
                      posIndices.length, negIndices.length, useFscore,
                      elsToEvaluatePos, elsToEvaluateNeg);
              return 1.0;
          } else {
              cuerpo[ref] = !cuerpo[ref]; // Undo the change
              return -1.0;
          }

    }//end-method


    /**
     * Compare to Method
     *
     * @param o1 Chromosome to compare
     *
     * @return Relative order between the chromosomes
     */
    public int compareTo (Cromosoma o1) {
            double valor1 = this.selValue;
            double valor2 = ((Cromosoma)o1).selValue;
            if (valor1 > valor2)
                    return -1;
            else if (valor1 < valor2)
                    return 1;
            else return 0;

    }//end-method


    /**
     * Test if two chromosome differ in only one gene
     *
     * @param a Chromosome to compare
     *
     * @return Position of the difference, if only one is found. Otherwise, -1
     */
    public int differenceAtOne (Cromosoma a) {

            int i;
            int cont = 0, pos = -1;

            for (i=0; i<cuerpo.length && cont < 2; i++)
              if (cuerpo[i] != a.getGen(i)) {
                    pos = i;
                    cont++;
              }

            if (cont >= 2)
              return -1;
            else return pos;

    }//end-method


    /**
     * To String Method
     *
     * @return String representation of the chromosome
     */
    @Override
    public String toString() {

            int i;

            String temp = "[";
            for (i=0; i<cuerpo.length; i++)
              if (cuerpo[i])
                    temp += "1";
              else
                    temp += "0";
            temp += ", " + String.valueOf(fitness) + ", " 
                    + String.valueOf(genesActivos()) + "]";

            return temp;
    }//end-method

    /**
     * This method counts how many nearest neighbors have been determined,
     * which can be less than K, when there are too few active genes.
     */
    private int numberOfValidNeighbors(int[] theVecinos){
        int c = 0;
        while(c < theVecinos.length && theVecinos[c] != -1){
            c++;
        }
        return c;
    }
    

}
