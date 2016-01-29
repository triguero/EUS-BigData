package org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Instance_Selection.ENNTh;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.StringTokenizer;


import org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.parseParameters;
import org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Basic.Metodo;
import org.apache.mahout.keel.Algorithms.Preprocess.Basic.KNN;
import org.apache.mahout.keel.Algorithms.Preprocess.Basic.OutputIS;
import org.apache.mahout.keel.Algorithms.Preprocess.Basic.Referencia;
import org.apache.mahout.keel.Dataset.InstanceSet;
import org.core.Fichero;
import org.core.Randomize;

public class ENNTh extends Metodo{
	
	/*Own parameters of the algorithm*/
	  private int k;
	  private double mu;
	  private int version;
	  
	  private int[] selected;


		/**
		 * Builder with a script file (configuration file)
		 * @param ficheroScript
		 */
		public ENNTh(String ficheroScript, InstanceSet IS) {
			super(ficheroScript, IS);

		}

	  
	  public void runAlgorithm () {

	    int i, j, l;
	    int nClases;
	    int claseObt;
	    boolean marcas[];
	    int nSel;    
	    int vecinos[];
	    double prob[];
	    double sumProb;
	    double maxProb;
	    int pos;

	    long tiempo = System.currentTimeMillis();

	    /*Inicialization of the flagged instances vector for a posterior copy*/
	    marcas = new boolean[datosTrain.length];
	    for (i=0; i<datosTrain.length; i++){
	      marcas[i] = true;
	    }
	    nSel = datosTrain.length;

	    /*Getting the number of differents classes*/
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
	    

	    
	    vecinos = new int[k];
	    double[][] priors = new double[datosTrain.length][nClases];
	    
	    // Determine the probabilities p_+ and p_- for all instances
	    if(version <= 3){
	        for (i=0; i<datosTrain.length; i++) {    
	            if(clasesTrain[i] == posClass){
	                priors[i][posClass] = 1.0;
	                priors[i][posClass ^ 1] = 0.0;
	            } else {
	              KNN.evaluacionKNN2(k, datosTrain, realTrain, nominalTrain, nulosTrain, 
	                      clasesTrain, datosTrain[i], realTrain[i], nominalTrain[i], 
	                      nulosTrain[i], nClases, distanceEu, vecinos);
	              int[] ks = new int[nClases];
	              for(j = 0; j < vecinos.length; j++){
	                  if(vecinos[j] >= 0){
	                      ks[clasesTrain[vecinos[j]]]++;
	                  }              
	              }
	              priors[i][posClass] = ks[posClass]/ k;
	              priors[i][posClass ^ 1] = ks[posClass ^ 1]/ k;
	            }
	        }   
	    } else {
	        for (i=0; i<datosTrain.length; i++) {    
	                priors[i][clasesTrain[i]] = 1.0;
	                priors[i][clasesTrain[i] ^ 1] = 0.0;
	        } 
	    }
	    
	    
	    
	    vecinos = new int[k];
	    prob = new double[nClases];
	    
	    // Index lists with elements marked for removal
	    ArrayList<Referencia> remove_positive = new ArrayList<Referencia>();
	    ArrayList<Referencia> remove_negative = new ArrayList<Referencia>();  

	    /*
	     * Body of the algorithm. 
	     * For each instance in T, search the correspond class conform his mayority 
	     * from the nearest neighborhood. 
	     * Is it is positive, the instance is selected.
	     */
	    for (i=0; i<datosTrain.length; i++) {    	
	      KNN.evaluacionKNN2(k, datosTrain, realTrain, nominalTrain, nulosTrain, 
	              clasesTrain, datosTrain[i], realTrain[i], nominalTrain[i], 
	              nulosTrain[i], nClases, distanceEu, vecinos);
	      
	      Arrays.fill(prob, 0.0);
	      for (j=0; j<vecinos.length; j++){
	    	  if (vecinos[j]>=0) {
	              prob[posClass] += priors[vecinos[j]][posClass] 
	                      / (1.0 + KNN.distancia(datosTrain[i], realTrain[i], 
	                      nominalTrain[i], nulosTrain[i], datosTrain[vecinos[j]], 
	                      realTrain[vecinos[j]], nominalTrain[vecinos[j]], 
	                      nulosTrain[vecinos[j]], distanceEu));
	              prob[posClass ^ 1] += priors[vecinos[j]][posClass ^ 1] 
	                      / (1.0 + KNN.distancia(datosTrain[i], realTrain[i], 
	                      nominalTrain[i], nulosTrain[i], datosTrain[vecinos[j]], 
	                      realTrain[vecinos[j]], nominalTrain[vecinos[j]], 
	                      nulosTrain[vecinos[j]], distanceEu));
	    	  }
	      }
	      sumProb = 0.0;
	      for (j=0; j<prob.length; j++) {
	    	  sumProb += prob[j];
	      }
	      for (j=0; j<prob.length; j++) {
	    	  prob[j] /= sumProb;
	      }
	      
	      maxProb = prob[0];
	      pos = 0;
	      for (j=1; j<prob.length; j++) {
	    	  if (prob[j] > maxProb) {
	              maxProb = prob[j];
	              pos = j;
	    	  }
	      }
	      
	      
	      
	      claseObt = pos;
	      
	      if(version == 3 || version == 6){
	        if(claseObt != clasesTrain[i]){
	            if(clasesTrain[i] == posClass){
	                remove_positive.add(new Referencia(i, 
	                        prob[posClass ^ 1] - prob[posClass]));
	            } else {
	                remove_negative.add(new Referencia(i, 
	                        prob[posClass] - prob[posClass ^ 1]));
	            }
	        } else if (clasesTrain[i] != posClass 
	                && prob[posClass] > mu * prob[posClass ^ 1]){
	            remove_negative.add(new Referencia(i, 
	                        prob[posClass] - mu * prob[posClass ^ 1]));
	        }
	      } else if((version == 1 || version == 4) && claseObt != clasesTrain[i]){
	          if(clasesTrain[i] == posClass){
	             remove_positive.add(new Referencia(i, 
	                        prob[posClass ^ 1] - prob[posClass]));
	          } else {
	              remove_negative.add(new Referencia(i, 
	                        prob[posClass] - prob[posClass ^ 1]));
	          }
	      } else if((version == 2 || version == 5) 
	                            && (claseObt != clasesTrain[i] || maxProb <= mu)){
	          if(clasesTrain[i] == posClass){
	             remove_positive.add(new Referencia(i, 
	                        prob[posClass ^ 1] - prob[posClass]));
	          } else {
	              remove_negative.add(new Referencia(i, 
	                        prob[posClass] - prob[posClass ^ 1]));
	          }
	      }
	    }
	    
	    
	    /*
	     * Determine IR of S.
	     */
	    int nNeg = classDistr[posClass ^ 1] - remove_negative.size();
	    int nPos = classDistr[posClass] - remove_positive.size();
	    
	      System.out.println("nPos = " + nPos + ", nNeg = " + nNeg);
	    
	    if(nPos == 0 && nNeg == 0){ // everything removed
	            
	        // Select one random element of each class
	        do {
	            pos = Randomize.Randint(0,datosTrain.length-1);
	        } while (clasesTrain[pos] != posClass);

	        int neg;
	        do {
	            neg = Randomize.Randint(0,datosTrain.length-1);
	        } while (clasesTrain[neg] == posClass);

	        for(i = 0; i < marcas.length; i++){
	            marcas[i] = (i == pos || i == neg);
	        }
	        nSel = 2;

	    } else if(nPos > nNeg){ // Marks of negative instances need to be ignored

	         // Remove the appropriate positive instances
	        for(int s = 0; s < remove_positive.size(); s++){
	            marcas[remove_positive.get(s).entero] = false;
	            nSel--;
	        }

	        // Sort negative elements according to decreasing removal scores
	        Collections.sort(remove_negative); 

	        // Remove marked negative instances, ignoring the final part
	        for(int s = 0; s < classDistr[posClass ^ 1] - nPos; s++){
	            marcas[remove_negative.get(s).entero] = false;
	            nSel--;
	        }

	    } else if(nPos == 0 || ((double) nNeg / nPos) > origIR){

	         // Remove the appropriate negative instances
	        for(int s = 0; s < remove_negative.size(); s++){
	            marcas[remove_negative.get(s).entero] = false;
	            nSel--;
	        }

	        // Sort postive elements according to decreasing removal scores
	        Collections.sort(remove_positive); 

	        // Some marks of positive instances need to be ignored
	        int nRetain = (int) Math.ceil(nNeg / origIR);
	        for(int s = 0; s < Math.min(classDistr[posClass] - nRetain, 
	                remove_positive.size()); s++){
	            marcas[remove_positive.get(s).entero] = false;
	            nSel--;
	        }            
	    } else { // No problem, all marked instances can be removed.

	        // Positive instances
	        for(int s = 0; s < remove_positive.size(); s++){
	            marcas[remove_positive.get(s).entero] = false;
	            nSel--;
	        }

	        // Negative instances
	        for(int s = 0; s < remove_negative.size(); s++){
	            marcas[remove_negative.get(s).entero] = false;
	            nSel--;
	        }

	    } 
	    
	    
	    
	    selected = new int[nSel];
	    for (i=0, l=0; i<datosTrain.length; i++) {
	      if (marcas[i]) { //the instance will be copied to the solution	        
	        selected[l] = i;
	        l++;
	      }
	    }

	    System.out.println("ENNTh_Imb "+ relation + " " 
	            + (double)(System.currentTimeMillis()-tiempo)/1000.0 + "s");

	  }
 
	  
	  
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
    		k = Integer.parseInt(param.getParameter(i++));
    		distanceEu = param.getParameter(i++).equalsIgnoreCase("Euclidean") ? true : false;
    		mu = Double.parseDouble(param.getParameter(i++));
    		version = Integer.parseInt(param.getParameter(i++));
		}
	  
	public int[] getSelected(){
		return selected;
	}

}
