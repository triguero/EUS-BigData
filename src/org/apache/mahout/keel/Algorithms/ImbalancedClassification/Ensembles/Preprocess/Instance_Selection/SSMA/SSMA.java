package org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Instance_Selection.SSMA;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.StringTokenizer;


import org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.parseParameters;
import org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Basic.Metodo;
import org.apache.mahout.keel.Algorithms.Preprocess.Basic.KNN;
import org.apache.mahout.keel.Algorithms.Preprocess.Basic.OutputIS;
import org.apache.mahout.keel.Dataset.InstanceSet;
import org.core.Fichero;
import org.core.Randomize;

public class SSMA extends Metodo{
	
	/*Own parameters of the algorithm*/
	private long semilla;
	private int tamPoblacion;
	private double nEval;
	private double pCross;
	private double pMut;
	private int kNeigh;
        

    private double penal;    
        
    private int mu;
        
    private boolean useFscore;
    
    private int[] selected;
    private int nWindowPos;
    private int nWindowNeg;


	/**
	 * Builder with a script file (configuration file)
	 * @param ficheroScript
	 */
	public SSMA(String ficheroScript,  InstanceSet IS) {
		super(ficheroScript, IS);
		
	}

	/**
	 * Executes the algorithm
	 */
	public void runAlgorithm () {
		
		int i, j, l;
		int nSel = 0;
		Cromosoma poblacion[];
		double ev = 0;
		double dMatrix[][];
		int sel1, sel2, comp1, comp2;
		Cromosoma hijos[];
		double umbralOpt;
		boolean veryLarge;
		double GAeffort=0, LSeffort=0, temporal;
		double fAcierto=0, fReduccion=0;
		int contAcierto=0, contReduccion=0;
		int nClases;

		long tiempo = System.currentTimeMillis();

		/*Getting the number of different classes*/
		nClases = 0;
		for (i=0; i<clasesTrain.length; i++)
		  if (clasesTrain[i] > nClases)
			nClases = clasesTrain[i];
		nClases++;
                
        /*
         * Determine minority and majority class (binary) and IR 
         * of the original set.
         */
        int[] classDistr = new int[nClases];
        for (i = 0; i < clasesTrain.length; i++){
            classDistr[clasesTrain[i]]++;
        }
        int posClass;
        if(classDistr[0] < classDistr[1]){
            posClass = 0;
        } else {
            posClass = 1;
        }
        double origIR = (double) classDistr[posClass ^ 1]/ classDistr[posClass];
        
        


        /*
         * Order the dataset
         */
        int[] posIndices = new int[classDistr[posClass]];
        int[] negIndices = new int[classDistr[posClass ^ 1]];
        int posJ = 0;
        int negJ = 0;
        for(i = 0; i < clasesTrain.length; i++){
            if(clasesTrain[i] == posClass){
                posIndices[posJ] = i;
                posJ++;
            } else if(clasesTrain[i] == (posClass ^ 1)){
                negIndices[negJ] = i;
                negJ++;
            }
        }
        
        // Preparation for windowing        
        if (nWindowPos < 0){
            nWindowPos = 1;
        }
        if ((nWindowNeg < 0) && (posIndices.length != 0)){
            nWindowNeg = (int) origIR;
        } else if (posIndices.length == 0){
            nWindowNeg = 1;
        }
        
        ArrayList<Integer> indexPos = new ArrayList<Integer>();
        for(int el = 0; el < posIndices.length; el++){
        	indexPos.add((Integer) posIndices[el]);
        }
        ArrayList<Integer> indexNeg = new ArrayList<Integer>();
        for(int el = 0; el < negIndices.length; el++){
        	indexNeg.add((Integer) negIndices[el]);
        }
        
        ArrayList<Integer> indexPosStrata[] = new ArrayList[nWindowPos];
        ArrayList<Integer> indexNegStrata[] = new ArrayList[nWindowNeg];
        
        int strata = 0;
        int aux;
        // positive
        for (i = 0; i < nWindowPos; i++){
            indexPosStrata[i] = new ArrayList<Integer>();
        }
        while (!indexPos.isEmpty()) {
            aux = Randomize.Randint(0, indexPos.size());
            indexPosStrata[strata].add(aux); // use index within posIndices, not actual index in the dataset,
                                             // because this is also how Cromosoma uses them.
            indexPos.remove(aux);
            strata = (strata + 1) % nWindowPos;
        }
        // negative
        for (i = 0; i < nWindowNeg; i++)
            indexNegStrata[i] = new ArrayList<Integer>();
        while (!indexNeg.isEmpty()) {
            aux = Randomize.Randint(0, indexNeg.size());
            indexNegStrata[strata].add(aux); // use index within negIndices, not actual index in the dataset,
                                             // because this is also how Cromosoma uses them.
            indexNeg.remove(aux);
            strata = (strata + 1) % nWindowNeg;
        }        
        // End preparation


         veryLarge = datosTrain.length > 6000;

		if (veryLarge == false) {
		  /*Construct a distance matrix of the instances*/
		  dMatrix = new double[datosTrain.length][datosTrain.length];
		  for (i = 0; i < dMatrix.length; i++) {
			for (j = i + 1; j < dMatrix[i].length; j++) {
			  dMatrix[i][j] = KNN.distancia(datosTrain[i], 
                                  realTrain[i], nominalTrain[i],
                                  nulosTrain[i], datosTrain[j], 
                                  realTrain[j], nominalTrain[j], 
                                  nulosTrain[j], distanceEu);
			}

		  }
		  for (i = 0; i < dMatrix.length; i++) {
			dMatrix[i][i] = Double.POSITIVE_INFINITY;
		  }
		  for (i = 0; i < dMatrix.length; i++) {
			for (j = i - 1; j >= 0; j--) {
			  dMatrix[i][j] = dMatrix[j][i];
			}
		  }
		} else {
		  dMatrix = null;
		}

		/*Random initialization of the population*/
		Randomize.setSeed (semilla);
		poblacion = new Cromosoma[tamPoblacion];
		for (i=0; i<tamPoblacion; i++)
		  poblacion[i] = new Cromosoma (kNeigh, datosTrain.length, 
                          dMatrix, datosTrain, realTrain, nominalTrain, 
                          nulosTrain, distanceEu, posIndices, negIndices);
		

		
		// select strata to evaluate
		int strataPos = 0;
        int strataNeg = 0;
        strataPos = (strataPos + 1) % nWindowPos;
        strataNeg = (strataNeg + 1) % nWindowNeg;

		/*Initial evaluation of the population*/
		for (i=0; i<tamPoblacion; i++) {
		  poblacion[i].evaluacionCompleta(nClases, kNeigh,
                          origIR, posClass, 
                          penal, posIndices.length, 
                          negIndices.length, useFscore, 
                          indexPosStrata[strataPos], indexNegStrata[strataNeg]);


        }

		umbralOpt = 0.0;
                
        int itsWithoutReplacement = 0;

		/*Until stop condition*/
		while (ev < nEval) {
			
	      strataPos = (strataPos + 1) % nWindowPos;
          strataNeg = (strataNeg + 1) % nWindowNeg;
			


          int fittest = getFittest(poblacion);
		  if (fAcierto >= (double)poblacion[fittest].getFitnessAUC()) {
			contAcierto++;
		  } else {
			contAcierto=0;
		  }
		  fAcierto = (double)poblacion[fittest].getFitnessAUC();

		  if (fReduccion >= (1.0-((double)poblacion[fittest].genesActivos()/(double)datosTrain.length))*100.0) {
			contReduccion++;
		  } else {
			contReduccion=0;
		  }
		  fReduccion = (1.0-((double)poblacion[fittest].genesActivos()/(double)datosTrain.length))*100.0;

		  if (contReduccion >= 10 || contAcierto >= 10){
			if (Randomize.Randint(0,1)==0) {
			  if (contAcierto >= 10) {
				contAcierto = 0;
				umbralOpt += 0.001;
			  } else {
				contReduccion = 0;
				umbralOpt -= 0.001;
			  }
			} else {
			  if (contReduccion >= 10) {
				contReduccion = 0;
				umbralOpt -= 0.001;
			  } else {
				contAcierto = 0;
				umbralOpt += 0.001;
			  }
			}
		  }

		  /*Binary tournament selection*/
		  comp1 = Randomize.Randint(0,tamPoblacion-1);
		  do {
			comp2 = Randomize.Randint(0,tamPoblacion-1);
		  } while (comp2 == comp1);

		  if (poblacion[comp1].getSel() > poblacion[comp2].getSel())
			sel1 = comp1;
		  else sel1 = comp2;
		  comp1 = Randomize.Randint(0,tamPoblacion-1);
		  do {
			comp2 = Randomize.Randint(0,tamPoblacion-1);
		  } while (comp2 == comp1);
		  if (poblacion[comp1].getSel() > poblacion[comp2].getSel())
			sel2 = comp1;
		  else
			sel2 = comp2;
		  



		  hijos = new Cromosoma[2];
		  hijos[0] = new Cromosoma (kNeigh, poblacion[sel1], 
                          poblacion[sel2], pCross,datosTrain.length, 
                          posIndices.length, negIndices.length);
		  hijos[1] = new Cromosoma (kNeigh, poblacion[sel2], 
                          poblacion[sel1], pCross,datosTrain.length,
                          posIndices.length, negIndices.length);
		  hijos[0].mutation (kNeigh, pMut, dMatrix, datosTrain, 
                          realTrain, nominalTrain, nulosTrain, distanceEu,
                          posIndices, negIndices);
		  hijos[1].mutation (kNeigh, pMut, dMatrix, datosTrain, 
                          realTrain, nominalTrain, nulosTrain, distanceEu,
                          posIndices, negIndices);
		  


		  /*Evaluation of offspring*/
		  hijos[0].evaluacionCompleta(nClases, kNeigh,
                          origIR, posClass, 
                          penal, posIndices.length, 
                          negIndices.length, useFscore, 
                          indexPosStrata[strataPos], indexNegStrata[strataNeg]); 

		  hijos[1].evaluacionCompleta(nClases, kNeigh,
                          origIR, posClass, 
                          penal, posIndices.length, 
                          negIndices.length, useFscore, 
                          indexPosStrata[strataPos], indexNegStrata[strataNeg]);

		  
		  ev+=2;
		  GAeffort += 2;
		  
		  
		  // also evaluate and sort current population, to be on same scale
		  for(int el = 0; el < poblacion.length; el++){
			  poblacion[el].evaluacionCompleta(nClases, kNeigh,
                      origIR, posClass, 
                      penal, posIndices.length, 
                      negIndices.length, useFscore, 
                      indexPosStrata[strataPos], indexNegStrata[strataNeg]);

		  }
		  Arrays.sort(poblacion);
		  ev += poblacion.length;
		  
		  
		  temporal = ev;
		  if (hijos[0].getFitness() > poblacion[tamPoblacion-1].getFitness() 
                          || Randomize.Rand() < 0.0625) {
			  ev += hijos[0].optimizacionLocal(nClases, kNeigh, 
                                  clasesTrain,dMatrix,umbralOpt, datosTrain, 
                                  realTrain, nominalTrain, nulosTrain, 
                                  distanceEu, posIndices, negIndices,
                                  posClass, origIR, penal, useFscore, 
                                  indexPosStrata[strataPos], indexNegStrata[strataNeg]);   

		  }
		  
		  if (hijos[1].getFitness() > poblacion[tamPoblacion-1].getFitness() 
                          || Randomize.Rand() < 0.0625) {
			  ev += hijos[1].optimizacionLocal(nClases, kNeigh, 
                                  clasesTrain,dMatrix,umbralOpt, datosTrain, 
                                  realTrain, nominalTrain, nulosTrain, 
                                  distanceEu, posIndices, negIndices,
                                  posClass, origIR, penal, useFscore, 
                                  indexPosStrata[strataPos], indexNegStrata[strataNeg]);

		  }

		  LSeffort += (ev - temporal);

		  /*
           * Replacement
           */
          Arrays.sort(poblacion);
          if(itsWithoutReplacement > mu){ // No replacement occurred for too many iterations
                /*Replace the two worst*/
                poblacion[tamPoblacion-1] = new Cromosoma (kNeigh, datosTrain.length, hijos[0]);
                poblacion[tamPoblacion-2] = new Cromosoma (kNeigh, datosTrain.length, hijos[1]);
                itsWithoutReplacement = 0;

          } else {
                /*
                 * We consider the current worst individuals (based
                 * on Sel(.)). From among the two constructed
                 * children and two selected individuals, the 
                 * ones with the highest fitness are retained.
                 */           
                int bestNew;
                int worstNew;
                if(hijos[0].getFitness() > hijos[1].getFitness()){
                    bestNew = 0;
                    worstNew = 1;
                } else {
                    bestNew = 1;
                    worstNew = 0;
                }

                int bestOld;
                int worstOld;
                if(poblacion[tamPoblacion-2].getFitness() > 
                        poblacion[tamPoblacion-1].getFitness()){
                    bestOld = tamPoblacion-2;
                    worstOld = tamPoblacion-1;
                } else {
                    bestOld = tamPoblacion-1;
                    worstOld = tamPoblacion-2;
                }

                if(hijos[bestNew].getFitness() 
                        < poblacion[worstOld].getFitness()){ // No replacement
                    itsWithoutReplacement++;
                } else if(hijos[worstNew].getFitness() 
                        > poblacion[bestOld].getFitness()){  // Both children are included                                
                    poblacion[tamPoblacion-1] = 
                            new Cromosoma (kNeigh, datosTrain.length, 
                            hijos[0]);
                    poblacion[tamPoblacion-2] = 
                            new Cromosoma (kNeigh, datosTrain.length, 
                            hijos[1]);
                    itsWithoutReplacement = 0;
                } else { // One individual is replaced
                    poblacion[worstOld] = 
                            new Cromosoma (kNeigh, datosTrain.length, 
                            hijos[bestNew]);
                   itsWithoutReplacement = 0;
                }

          }

		} // end while

        int index = getFittest(poblacion);        

        nSel = poblacion[index].genesActivos();
        selected = new int[nSel];
        
        /*Building of S set from the best chromosome obtained*/
        l = 0; 

        for (i=0; i<posIndices.length; i++) {
          if (poblacion[index].getGen(i)) { //the instance must be copied to the solution
                int realIndex = posIndices[i];
                selected[l] = realIndex;
                l++;
          }
        }

        for (i=0; i<negIndices.length; i++) {
          if (poblacion[index].getGen(posIndices.length + i)) { //the instance must be copied to the solution
                int realIndex = negIndices[i];
                selected[l] = realIndex;
                l++;
          }
        }


		System.out.println("SSMA_Imb "+ relation + " " 
                        + (double)(System.currentTimeMillis()-tiempo)/1000.0 + "s");

		
	}//end-method 
	

        
   /**
	* It reads the configuration file for performing the EUS-CHC method
	*/
   public void readConfiguration(String ficheroScript) {
     		
    		
    		parseParameters param = new parseParameters();
    		param.parseConfigurationFile(ficheroScript);
    		ficheroTraining = param.getTrainingInputFile();
    		ficheroTest = param.getTestInputFile();
    		ficheroSalida = new String[2];
    		ficheroSalida[0] = param.getTrainingOutputFile();
    		ficheroSalida[1] = param.getTestOutputFile();
    		int i = 0;
    		semilla = Long.parseLong(param.getParameter(i++));
    		tamPoblacion = Integer.parseInt(param.getParameter(i++));
    		nEval = Integer.parseInt(param.getParameter(i++));
    		pCross = Double.parseDouble(param.getParameter(i++));
    		pMut = Double.parseDouble(param.getParameter(i++));
    		kNeigh = Integer.parseInt(param.getParameter(i++));
    		distanceEu = param.getParameter(i++).equalsIgnoreCase("Euclidean") ? true : false;
    		penal = Double.parseDouble(param.getParameter(i++));
    		mu = Integer.parseInt(param.getParameter(i++));
    		useFscore = param.getParameter(i++).equalsIgnoreCase("Yes") ? true : false;
                    
            nWindowPos = Integer.parseInt(param.getParameter(i++));
            nWindowNeg = Integer.parseInt(param.getParameter(i++));
	}
        
    /**
     * Find the chromosome with the highest fitness.
     */
    private int getFittest(Cromosoma[] poblacion) {
        int index = 0;
        double best = poblacion[0].getFitness();
        
        for(int i = 1; i < poblacion.length; i++){
            if(poblacion[i].getFitness() > best){
                index = i;
                best = poblacion[i].getFitness();
            }
        }
        
        return index;
    }
    
    public int[] getSelected(){
    	return selected;
    }

}
